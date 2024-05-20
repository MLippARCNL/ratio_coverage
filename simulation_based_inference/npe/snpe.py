import time
import pickle
from typing import Any, Callable, Dict, Optional, Tuple, Union
from collections import OrderedDict
from pathlib import Path
from copy import deepcopy
from tqdm import tqdm
import copy
from .discriminator import Discriminator1DLipschitzWasserstein

import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import nn, ones, Tensor, save, load
from torch.utils import data
from torch.nn import Module
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.distributions import Distribution

from sbi.utils import x_shape_from_simulation, test_posterior_net_for_multi_d_x
from sbi.inference import SNPE
from sbi.types import TensorboardSummaryWriter


class CustomSNPE(SNPE):
    def __init__(self,
                 prior: Union[Distribution],
                 density_estimator: Union[str, Callable] = "maf",
                 device: str = "cpu",
                 logging_level: Union[int, str] = "WARNING",
                 summary_writer: Optional[TensorboardSummaryWriter] = None,
                 show_progress_bars: bool = True,
                 gamma: float = 1
                 ):
        super().__init__(
            prior=prior,
            density_estimator=density_estimator,
            device=device,
            logging_level=logging_level,
            summary_writer=summary_writer,
            show_progress_bars=show_progress_bars,
        )
        self.gamma = gamma

    def train(self,
              training_batch_size: int = 50,
              learning_rate: float = 5e-4,
              validation_fraction: float = 0.1,
              max_num_epochs: int = 2 ** 31 - 1,
              clip_max_norm: Optional[float] = 5.0,
              calibration_kernel: Optional[Callable] = None,
              resume_training: bool = False,
              discard_prior_samples: bool = False,
              retrain_from_scratch: bool = False,
              show_train_summary: bool = False,
              dataloader_kwargs: Optional[dict] = None,
              path_checkpoint=None,
              **kwargs
              ) -> nn.Module:
        assert kwargs.__len__() == 0
        # Load data from most recent round.
        self._round = max(self._data_round_index)

        # Calibration kernels proposed in Lueckmann, Gonçalves et al., 2017.
        if calibration_kernel is None:
            calibration_kernel = lambda x: ones([len(x)], device=self._device)

        # Starting index for the training set (1 = discard round-0 samples).
        start_idx = int(discard_prior_samples and self._round > 0)

        # For non-atomic loss, we can not reuse samples from previous rounds
        # as of now. SNPE-A can, by construction of the algorithm, only use
        # samples from the last round. SNPE-A is the only algorithm that has
        # an attribute `_ran_final_round`, so this is how we check for whether
        # or not we are using SNPE-A.
        if self.use_non_atomic_loss or hasattr(self, "_ran_final_round"):
            start_idx = self._round

        # Set the proposal to the last proposal that was passed by the user.
        # For atomic SNPE, it does not matter what the proposal is. For
        # non-atomic SNPE, we only use the latest data that was passed, i.e.
        # the one from the last proposal.
        proposal = self._proposal_roundwise[-1]

        # train_loader, val_loader = self.get_dataloaders(
        #     starting_round=start_idx,
        #     training_batch_size=training_batch_size,
        #     validation_fraction=validation_fraction,
        #     validation_inputs=validation_inputs,
        #     validation_outputs=validation_outputs,
        #     resume_training=resume_training,
        #     dataloader_kwargs=dataloader_kwargs,
        # )

        train_loader, val_loader = self.get_dataloaders(
            start_idx,
            training_batch_size,
            validation_fraction,
            resume_training,
            dataloader_kwargs=dataloader_kwargs,
        )

        # First round or if retraining from scratch:
        # Call the `self._build_neural_net` with the rounds' thetas and xs as
        # arguments, which will build the neural network. This is passed into
        # NeuralPosterior, to create a neural posterior which can `sample()`
        # and `log_prob()`. The network is accessible via `.net`.
        if self._neural_net is None or retrain_from_scratch:
            # Get theta,x to initialize NN
            theta, x, _ = self.get_simulations(starting_round=start_idx)
            # Use only training data for building the neural net (z-scoring
            # transforms)
            print(
                f"Theta and x shapes: {theta[self.train_indices].shape}, "
                f"{x[self.train_indices].shape}"
            )
            self._neural_net = self._build_neural_net(
                theta[self.train_indices].to("cpu"),
                x[self.train_indices].to("cpu"),
            )

            self._x_shape = x_shape_from_simulation(x.to("cpu"))

            test_posterior_net_for_multi_d_x(
                self._neural_net,
                theta.to("cpu"),
                x.to("cpu"),
            )

            del theta, x

        # Move entire net to device for training.
        self._neural_net.to(self._device)

        if not resume_training:
            self.optimizer = torch.optim.Adam(
                list(self._neural_net.parameters()), lr=learning_rate
            )
            self.epoch, self._val_log_prob = 0, float("-Inf")

        # Checkpoint loading if path provided.
        if path_checkpoint is not None:
            load_checkpoint(
                model=self._neural_net,
                checkpoint=torch.load(
                    path_checkpoint + "/checkpoint.pt",
                    map_location=torch.device("cpu"),
                ),
                load_cp_continue=True,
                optimizer=self.optimizer,
            )
            self._best_state_dict = torch.load(
                os.path.join(path_checkpoint, "checkpoint_best.pt")
            )

        """
        # We don't want early stopping.
        while self.epoch < max_num_epochs and not self._converged(
            self.epoch, stop_after_epochs
        ):
        """
        # Load validation losses of the previous stage (if previous stage
        # exists):
        if (path_checkpoint is not None) and os.path.exists(os.path.join(path_checkpoint, "val_losses.npy")):
            self.val_losses = np.load(os.path.join(path_checkpoint, "val_losses.npy")).tolist()
            min_val_loss = min(self.val_losses)
        else:
            self.val_losses = []
            min_val_loss = float("inf")
        while self.epoch < max_num_epochs:
            # Train for a single epoch.
            self._neural_net.train()
            train_log_probs_sum = 0
            epoch_start_time = time.time()
            for batch in train_loader:
                self.optimizer.zero_grad()
                # Get batches on current device.
                theta_batch, x_batch = (
                    batch[0].to(self._device),
                    batch[1].to(self._device),
                )
                train_losses = self._loss(
                    theta_batch,
                    x_batch,
                    None,
                    proposal,
                    calibration_kernel,
                    force_first_round_loss=True,
                )
                train_loss = torch.mean(train_losses)
                train_log_probs_sum -= train_losses.sum().item()

                train_loss.backward()
                if clip_max_norm is not None:
                    clip_grad_norm_(
                        self._neural_net.parameters(), max_norm=clip_max_norm
                    )
                self.optimizer.step()

            train_log_prob_average = train_log_probs_sum / (
                    len(train_loader) * train_loader.batch_size  # type: ignore
            )
            self._summary["training_log_probs"].append(train_log_prob_average)

            # Calculate validation performance.
            self._neural_net.eval()
            val_log_prob_sum = 0
            val_loss_sum = 0

            with torch.no_grad():
                for batch in val_loader:
                    theta_batch, x_batch = (
                        batch[0].to(self._device),
                        batch[1].to(self._device)  # ,
                        # batch[2].to(self._device),
                    )
                    # Take negative loss here to get validation log_prob.
                    val_losses = self._loss(
                        theta_batch,
                        x_batch,
                        None,
                        proposal,
                        calibration_kernel,
                        force_first_round_loss=True,
                    )
                    val_loss_sum += torch.mean(val_losses).item()
                    val_log_prob_sum -= val_losses.sum().item()

            # Take mean over all validation samples.
            self._val_log_prob = val_log_prob_sum / (
                    len(val_loader) * val_loader.batch_size  # type: ignore
            )
            self._val_loss = val_loss_sum / len(val_loader)  # type: ignore
            self.val_losses.append(self._val_loss)

            if self._val_loss < min_val_loss:
                min_val_loss = self._val_loss
                self._best_state_dict = self._estimator_cpu_state_dict()

            # Log validation log prob for every epoch.
            self._summary["validation_log_probs"].append(self._val_log_prob)
            self._summary["epoch_durations_sec"].append(
                time.time() - epoch_start_time
            )
            self.epoch += 1
            self._maybe_show_progress(self._show_progress_bars, self.epoch)

        """
        self._report_convergence_at_end(
            self.epoch, stop_after_epochs, max_num_epochs
        )
        """

        # Update summary.
        self._summary["epochs_trained"].append(self.epoch)
        self._summary["best_validation_log_prob"].append(
            np.max(self._summary["validation_log_probs"])
        )

        # Update description for progress bar.
        if show_train_summary:
            print(self._describe_round(self._round, self._summary))

        # Avoid keeping the gradients in the resulting network, which can
        # cause memory leakage when benchmarking.
        self._neural_net.zero_grad(set_to_none=True)

        return deepcopy(self._neural_net)

    @torch.no_grad()
    def _estimator_cpu_state_dict(self):
        # Check if we're training a Data Parallel model.
        self._neural_net.eval()
        self._neural_net = self._neural_net.cpu()
        if isinstance(self._neural_net, torch.nn.DataParallel):
            state_dict = deepcopy(self._neural_net.module.state_dict())
        else:
            state_dict = deepcopy(self._neural_net.state_dict())
        # Move back to the original device
        self._neural_net = self._neural_net.to(self._device)

        return state_dict

    @torch.no_grad()
    def save_checkpoint(self, path: Path):
        save(self._best_state_dict, path)


class SNPE_TV(CustomSNPE):
    """
        discriminator depends on (theta, x)
    """

    def __init__(self,
                 prior: Union[Distribution],
                 density_estimator: Union[str, Callable] = "maf",
                 device: str = "cpu",
                 logging_level: Union[int, str] = "WARNING",
                 summary_writer: Optional[TensorboardSummaryWriter] = None,
                 show_progress_bars: bool = True,
                 gamma: float = 1,
                 discriminator: Callable = Discriminator1DLipschitzWasserstein,
                 discriminator_dim: int = 10,
                 alternate_training: bool = False,
                 gradients_path: str | Path = None,
                 *,
                 f=1,
                 preTrain=0
                 ):
        super().__init__(
            prior=prior,
            density_estimator=density_estimator,
            device=device,
            logging_level=logging_level,
            summary_writer=summary_writer,
            show_progress_bars=show_progress_bars,
            gamma=gamma,
        )

        self.gradients_path = gradients_path if isinstance(gradients_path, Path) else Path(gradients_path) if isinstance(gradients_path, str) else None
        if self.gradients_path is not None:
            if not self.gradients_path.exists():
                self.gradients_path.touch()
            assert self.gradients_path.is_file()
        self.f = f
        self._discriminator = None
        self._build_discriminator = discriminator
        self._dim_discriminator = discriminator_dim
        self.train = self._train if not alternate_training else self._train_alternate
        self.preTrain = preTrain

    def save_gradients(self, param):
        if self.gradients_path is not None:
            _data = []
            for p in param:
                if (p.requires_grad) and p.grad is not None:
                    _data.append(copy.deepcopy(p.grad))
            with self.gradients_path.open(mode='a+b') as f:
                pickle.dump(_data, f)

    def _loss(self,
              theta: Tensor,
              x: Tensor,
              masks: Tensor,
              proposal: Optional[Any],
              calibration_kernel: Callable,
              force_first_round_loss: bool = False,
              ) -> Tensor:

        # return - self.gamma * self._loss_discriminator(theta, x)

        log_prob = self._neural_net.log_prob(theta, x)

        if self.gamma > 0:
            print(-(calibration_kernel(x) * log_prob).mean().item(), - (self.gamma * self._loss_discriminator(theta, x)).mean().item())
            return -(calibration_kernel(x) * log_prob) - self.gamma * self._loss_discriminator(theta, x)
        else:
            return -(calibration_kernel(x) * log_prob)

    def _loss_discriminator(self,
                            theta: Tensor,
                            x: Tensor,
                            ) -> Tensor:

        theta_hat = self._neural_net.sample(1, context=x).squeeze(1)
        assert theta.shape == theta_hat.shape

        l1 = self._discriminator(torch.cat([theta, x], 1))
        l2 = self._discriminator(torch.cat([theta_hat, x], 1))

        return -torch.abs(l1.mean() - l2.mean())

    def _train(self,
               training_batch_size: int = 50,
               learning_rate_Adam: float = 5e-4,
               learning_rate_SGD: float = 5e-4,
               validation_fraction: float = 0.1,
               max_num_epochs: int = 2 ** 31 - 1,
               clip_max_norm: Optional[float] = 5.0,
               calibration_kernel: Optional[Callable] = None,
               resume_training: bool = False,
               discard_prior_samples: bool = False,
               retrain_from_scratch: bool = False,
               show_train_summary: bool = False,
               dataloader_kwargs: Optional[dict] = None,
               path_checkpoint=None,
               force_first_round_loss=True,
               **kwargs
               ) -> nn.Module:
        print('### train normally ###')
        assert kwargs.__len__() == 0
        assert force_first_round_loss
        # Load data from most recent round.
        self._round = max(self._data_round_index)

        # Calibration kernels proposed in Lueckmann, Gonçalves et al., 2017.
        if calibration_kernel is None:
            calibration_kernel = lambda x: ones([len(x)], device=self._device)

        # Starting index for the training set (1 = discard round-0 samples).
        start_idx = int(discard_prior_samples and self._round > 0)

        # For non-atomic loss, we can not reuse samples from previous rounds
        # as of now. SNPE-A can, by construction of the algorithm, only use
        # samples from the last round. SNPE-A is the only algorithm that has
        # an attribute `_ran_final_round`, so this is how we check for whether
        # or not we are using SNPE-A.
        if self.use_non_atomic_loss or hasattr(self, "_ran_final_round"):
            start_idx = self._round

        # Set the proposal to the last proposal that was passed by the user.
        # For atomic SNPE, it does not matter what the proposal is. For
        # non-atomic SNPE, we only use the latest data that was passed, i.e.
        # the one from the last proposal.
        proposal = self._proposal_roundwise[-1]

        train_loader, val_loader = self.get_dataloaders(
            start_idx,
            training_batch_size,
            validation_fraction,
            resume_training,
            dataloader_kwargs=dataloader_kwargs,
        )

        # First round or if retraining from scratch:
        # Call the `self._build_neural_net` with the rounds' thetas and xs as
        # arguments, which will build the neural network. This is passed into
        # NeuralPosterior, to create a neural posterior which can `sample()`
        # and `log_prob()`. The network is accessible via `.net`.
        if self._neural_net is None or retrain_from_scratch:
            # Get theta,x to initialize NN
            theta, x, _ = self.get_simulations(starting_round=start_idx)
            # Use only training data for building the neural net (z-scoring
            # transforms)
            print(
                f"Theta and x shapes: {theta[self.train_indices].shape}, "
                f"{x[self.train_indices].shape}"
            )
            self._neural_net = self._build_neural_net(
                theta[self.train_indices].to("cpu"),
                x[self.train_indices].to("cpu"),
            )
            self._discriminator = self._build_discriminator(self._dim_discriminator).to("cpu")

            self._x_shape = x_shape_from_simulation(x.to("cpu"))

            test_posterior_net_for_multi_d_x(
                self._neural_net,
                theta.to("cpu"),
                x.to("cpu"),
            )

            del theta, x

        # Move entire net to device for training.
        self._neural_net.to(self._device)
        self._discriminator.to(self._device)

        if not resume_training:
            self.optimizer = torch.optim.Adam(
                list(self._neural_net.parameters()), lr=learning_rate_Adam
            )
            self.optimizer_d = torch.optim.Adam(
                list(self._discriminator.parameters()), lr=learning_rate_Adam
            )
            # self.optimizer = torch.optim.SGD(
            #     list(self._neural_net.parameters()), lr=learning_rate_SGD
            # )
            # self.optimizer_d = torch.optim.SGD(
            #     list(self._discriminator.parameters()), lr=learning_rate_SGD
            # )

            self.epoch, self._val_log_prob = 0, float("-Inf")

        # Checkpoint loading if path provided.
        if path_checkpoint is not None:
            load_checkpoint(
                model=self._neural_net,
                checkpoint_path=Path(path_checkpoint) / "checkpoint.pt",
                load_cp_continue=True,
                optimizer=self.optimizer,
            )
            load_checkpoint(
                model=self._discriminator,
                checkpoint_path=Path(path_checkpoint) / "checkpoint_d.pt",
                load_cp_continue=True,
                optimizer=self.optimizer_d,
            )
            self._best_state_dict = torch.load(
                os.path.join(path_checkpoint, "checkpoint_best.pt")
            )

        """
        # We don't want early stopping.
        while self.epoch < max_num_epochs and not self._converged(
            self.epoch, stop_after_epochs
        ):
        """
        # Load validation losses of the previous stage (if previous stage
        # exists):
        if (path_checkpoint is not None) and os.path.exists(os.path.join(path_checkpoint, "val_losses.npy")):
            self.val_losses = np.load(os.path.join(path_checkpoint, "val_losses.npy")).tolist()
            min_val_loss = min(self.val_losses)
        else:
            self.val_losses = []
            min_val_loss = float("inf")

        while self.epoch < max_num_epochs:
            epoch_start_time = time.time()

            self._callback()

            # Train for a single epoch the discriminator
            if self.gamma > 0:
                self._neural_net.eval()
                self._discriminator.train()
                train_log_probs_sum = 0
                for _ in range(self.f):
                    for batch in train_loader:
                        self.optimizer_d.zero_grad()
                        # Get batches on current device.
                        theta_batch, x_batch = (
                            batch[0].to(self._device),
                            batch[1].to(self._device),
                        )
                        train_losses = self._loss_discriminator(
                            theta_batch,
                            x_batch,
                        )
                        train_loss = torch.mean(train_losses)
                        train_log_probs_sum -= train_losses.sum().item()

                        train_loss.backward()
                        if clip_max_norm is not None:
                            clip_grad_norm_(self._discriminator.parameters(), max_norm=clip_max_norm)
                        self.save_gradients(self._discriminator.parameters())
                        self.optimizer_d.step()

            # renormalize lipschitz
            if hasattr(self._discriminator, 'renormalize'):
                self._discriminator.renormalize()
            else:
                print('### no "renormalize()"')

            # Train for a single epoch the estimator
            self._neural_net.train()
            self._discriminator.eval()
            train_log_probs_sum = 0
            for batch in train_loader:
                self.optimizer.zero_grad()
                # Get batches on current device.
                theta_batch, x_batch = (
                    batch[0].to(self._device),
                    batch[1].to(self._device),
                )
                train_losses = self._loss(
                    theta_batch,
                    x_batch,
                    None,
                    proposal,
                    calibration_kernel,
                    force_first_round_loss=True,
                )
                train_loss = torch.mean(train_losses)
                train_log_probs_sum -= train_losses.sum().item()

                train_loss.backward()
                if clip_max_norm is not None:
                    clip_grad_norm_(self._neural_net.parameters(), max_norm=clip_max_norm)
                self.save_gradients(self._neural_net.parameters())
                self.optimizer.step()

            train_log_prob_average = train_log_probs_sum / (
                    len(train_loader) * train_loader.batch_size  # type: ignore
            )
            self._summary["training_log_probs"].append(train_log_prob_average)

            # Calculate validation performance.
            self._neural_net.eval()
            val_log_prob_sum = 0
            val_loss_sum = 0

            with torch.no_grad():
                for batch in val_loader:
                    theta_batch, x_batch = (
                        batch[0].to(self._device),
                        batch[1].to(self._device)  # ,
                        # batch[2].to(self._device),
                    )
                    # Take negative loss here to get validation log_prob.
                    val_losses = self._loss(
                        theta_batch,
                        x_batch,
                        None,
                        proposal,
                        calibration_kernel,
                        force_first_round_loss=True,
                    )
                    val_loss_sum += torch.mean(val_losses).item()
                    val_log_prob_sum -= val_losses.sum().item()
            assert val_loss_sum is not float('nan')

            print(f'# val: {val_loss_sum:.5f}')

            # Take mean over all validation samples.
            self._val_log_prob = val_log_prob_sum / (
                    len(val_loader) * val_loader.batch_size  # type: ignore
            )
            self._val_loss = val_loss_sum / len(val_loader)  # type: ignore
            self.val_losses.append(self._val_loss)

            if self._val_loss < min_val_loss:
                min_val_loss = self._val_loss
                self._best_state_dict = self._estimator_cpu_state_dict()

            # Log validation log prob for every epoch.
            self._summary["validation_log_probs"].append(self._val_log_prob)
            self._summary["epoch_durations_sec"].append(
                time.time() - epoch_start_time
            )
            self.epoch += 1
            self._maybe_show_progress(self._show_progress_bars, self.epoch)

        """
        self._report_convergence_at_end(
            self.epoch, stop_after_epochs, max_num_epochs
        )
        """

        # Update summary.
        self._summary["epochs_trained"].append(self.epoch)
        self._summary["best_validation_log_prob"].append(
            np.max(self._summary["validation_log_probs"])
        )

        # Update description for progress bar.
        if show_train_summary:
            print(self._describe_round(self._round, self._summary))

        # Avoid keeping the gradients in the resulting network, which can
        # cause memory leakage when benchmarking.
        self._neural_net.zero_grad(set_to_none=True)

        return deepcopy(self._neural_net)

    def _train_alternate(self,
                         training_batch_size: int = 50,
                         learning_rate_Adam: float = 5e-4,
                         learning_rate_SGD: float = 5e-4,
                         validation_fraction: float = 0.1,
                         max_num_epochs: int = 2 ** 31 - 1,
                         clip_max_norm: Optional[float] = 5.0,
                         calibration_kernel: Optional[Callable] = None,
                         resume_training: bool = False,
                         discard_prior_samples: bool = False,
                         retrain_from_scratch: bool = False,
                         show_train_summary: bool = False,
                         dataloader_kwargs: Optional[dict] = None,
                         path_checkpoint=None,
                         force_first_round_loss=True,
                         **kwargs
                         ) -> nn.Module:
        print('### train alternatively ###')
        assert kwargs.__len__() == 0
        assert force_first_round_loss
        # Load data from most recent round.
        self._round = max(self._data_round_index)

        # Calibration kernels proposed in Lueckmann, Gonçalves et al., 2017.
        if calibration_kernel is None:
            calibration_kernel = lambda x: ones([len(x)], device=self._device)

        # Starting index for the training set (1 = discard round-0 samples).
        start_idx = int(discard_prior_samples and self._round > 0)

        # For non-atomic loss, we can not reuse samples from previous rounds
        # as of now. SNPE-A can, by construction of the algorithm, only use
        # samples from the last round. SNPE-A is the only algorithm that has
        # an attribute `_ran_final_round`, so this is how we check for whether
        # or not we are using SNPE-A.
        if self.use_non_atomic_loss or hasattr(self, "_ran_final_round"):
            start_idx = self._round

        # Set the proposal to the last proposal that was passed by the user.
        # For atomic SNPE, it does not matter what the proposal is. For
        # non-atomic SNPE, we only use the latest data that was passed, i.e.
        # the one from the last proposal.
        proposal = self._proposal_roundwise[-1]

        train_loader, val_loader = self.get_dataloaders(
            start_idx,
            training_batch_size,
            validation_fraction,
            resume_training,
            dataloader_kwargs=dataloader_kwargs,
        )

        # First round or if retraining from scratch:
        # Call the `self._build_neural_net` with the rounds' thetas and xs as
        # arguments, which will build the neural network. This is passed into
        # NeuralPosterior, to create a neural posterior which can `sample()`
        # and `log_prob()`. The network is accessible via `.net`.
        if self._neural_net is None or retrain_from_scratch:
            # Get theta,x to initialize NN
            theta, x, _ = self.get_simulations(starting_round=start_idx)
            # Use only training data for building the neural net (z-scoring
            # transforms)
            print(
                f"Theta and x shapes: {theta[self.train_indices].shape}, "
                f"{x[self.train_indices].shape}"
            )
            self._neural_net = self._build_neural_net(
                theta[self.train_indices].to("cpu"),
                x[self.train_indices].to("cpu"),
            )
            self._discriminator = self._build_discriminator(self._dim_discriminator).to("cpu")

            self._x_shape = x_shape_from_simulation(x.to("cpu"))

            test_posterior_net_for_multi_d_x(
                self._neural_net,
                theta.to("cpu"),
                x.to("cpu"),
            )

            del theta, x

        # Move entire net to device for training.
        self._neural_net.to(self._device)
        self._discriminator.to(self._device)

        if not resume_training:
            self.optimizer = torch.optim.Adam(
                list(self._neural_net.parameters()), lr=learning_rate_Adam
            )
            self.optimizer_d = torch.optim.Adam(
                list(self._discriminator.parameters()), lr=learning_rate_Adam
            )
            # self.optimizer = torch.optim.SGD(
            #     list(self._neural_net.parameters()), lr=learning_rate_SGD
            # )
            # self.optimizer_d = torch.optim.SGD(
            #     list(self._discriminator.parameters()), lr=learning_rate_SGD
            # )

            self.epoch, self._val_log_prob = 0, float("-Inf")

        # Checkpoint loading if path provided.
        if path_checkpoint is not None:
            load_checkpoint(
                model=self._neural_net,
                checkpoint_path=Path(path_checkpoint) / "checkpoint.pt",
                load_cp_continue=True,
                optimizer=self.optimizer,
            )
            load_checkpoint(
                model=self._discriminator,
                checkpoint_path=Path(path_checkpoint) / "checkpoint_d.pt",
                load_cp_continue=True,
                optimizer=self.optimizer_d,
            )
            self._best_state_dict = torch.load(
                os.path.join(path_checkpoint, "checkpoint_best.pt")
            )

        """
        # We don't want early stopping.
        while self.epoch < max_num_epochs and not self._converged(
            self.epoch, stop_after_epochs
        ):
        """
        # Load validation losses of the previous stage (if previous stage
        # exists):
        if (path_checkpoint is not None) and os.path.exists(os.path.join(path_checkpoint, "val_losses.npy")):
            self.val_losses = np.load(os.path.join(path_checkpoint, "val_losses.npy")).tolist()
            min_val_loss = min(self.val_losses)
        else:
            self.val_losses = []
            min_val_loss = float("inf")

        while self.epoch < max_num_epochs:
            epoch_start_time = time.time()

            self._callback()

            train_log_probs_sum = 0
            for batch in train_loader:
                # renormalize lipschitz
                if hasattr(self._discriminator, 'renormalize'):
                    self._discriminator.renormalize()
                else:
                    print('### no "renormalize()"')

                # Train for a single epoch the estimator
                self._neural_net.train()
                self._discriminator.eval()
                self.optimizer.zero_grad()
                # Get batches on current device.
                theta_batch, x_batch = (
                    batch[0].to(self._device),
                    batch[1].to(self._device),
                )
                train_losses = self._loss(
                    theta_batch,
                    x_batch,
                    None,
                    proposal,
                    calibration_kernel,
                    force_first_round_loss=True,
                )
                train_loss = torch.mean(train_losses)
                train_log_probs_sum -= train_losses.sum().item()

                train_loss.backward()
                if clip_max_norm is not None:
                    clip_grad_norm_(self._neural_net.parameters(), max_norm=clip_max_norm)
                self.save_gradients(self._neural_net.parameters())
                self.optimizer.step()

                # Train for a single epoch the discriminator
                if self.gamma > 0:
                    self._neural_net.eval()
                    self._discriminator.train()
                    for _ in range(self.f):
                        self.optimizer_d.zero_grad()
                        theta_batch, x_batch = (
                            batch[0].to(self._device),
                            batch[1].to(self._device),
                        )
                        train_losses = self._loss_discriminator(
                            theta_batch,
                            x_batch,
                        )
                        train_loss = torch.mean(train_losses)

                        train_loss.backward()
                        if clip_max_norm is not None:
                            clip_grad_norm_(self._discriminator.parameters(), max_norm=clip_max_norm)
                        self.save_gradients(self._discriminator.parameters())
                        self.optimizer_d.step()

            train_log_prob_average = train_log_probs_sum / (
                    len(train_loader) * train_loader.batch_size  # type: ignore
            )
            self._summary["training_log_probs"].append(train_log_prob_average)

            # Calculate validation performance.
            self._neural_net.eval()
            val_log_prob_sum = 0
            val_loss_sum = 0

            with torch.no_grad():
                for batch in val_loader:
                    theta_batch, x_batch = (
                        batch[0].to(self._device),
                        batch[1].to(self._device)  # ,
                        # batch[2].to(self._device),
                    )
                    # Take negative loss here to get validation log_prob.
                    val_losses = self._loss(
                        theta_batch,
                        x_batch,
                        None,
                        proposal,
                        calibration_kernel,
                        force_first_round_loss=True,
                    )
                    val_loss_sum += torch.mean(val_losses).item()
                    val_log_prob_sum -= val_losses.sum().item()
            assert val_loss_sum is not float('nan')

            print(f'# val: {val_loss_sum:.5f}')

            # Take mean over all validation samples.
            self._val_log_prob = val_log_prob_sum / (
                    len(val_loader) * val_loader.batch_size  # type: ignore
            )
            self._val_loss = val_loss_sum / len(val_loader)  # type: ignore
            self.val_losses.append(self._val_loss)

            if self._val_loss < min_val_loss:
                min_val_loss = self._val_loss
                self._best_state_dict = self._estimator_cpu_state_dict()

            # Log validation log prob for every epoch.
            self._summary["validation_log_probs"].append(self._val_log_prob)
            self._summary["epoch_durations_sec"].append(
                time.time() - epoch_start_time
            )
            self.epoch += 1
            self._maybe_show_progress(self._show_progress_bars, self.epoch)

        """
        self._report_convergence_at_end(
            self.epoch, stop_after_epochs, max_num_epochs
        )
        """

        # Update summary.
        self._summary["epochs_trained"].append(self.epoch)
        self._summary["best_validation_log_prob"].append(
            np.max(self._summary["validation_log_probs"])
        )

        # Update description for progress bar.
        if show_train_summary:
            print(self._describe_round(self._round, self._summary))

        # Avoid keeping the gradients in the resulting network, which can
        # cause memory leakage when benchmarking.
        self._neural_net.zero_grad(set_to_none=True)

        return deepcopy(self._neural_net)

    def _callback(self):
        if self.preTrain > 0:
            if self.epoch == 0:
                self.gamma_old = self.gamma
                self.gamma = 0
            elif self.epoch > self.preTrain:
                self.gamma = self.gamma_old


class SNPE_CDF(SNPE_TV):
    """
        discriminator depends on p
    """

    def __init__(self,
                 prior: Union[Distribution],
                 density_estimator: Union[str, Callable] = "maf",
                 device: str = "cpu",
                 logging_level: Union[int, str] = "WARNING",
                 summary_writer: Optional[TensorboardSummaryWriter] = None,
                 show_progress_bars: bool = True,
                 gamma: float = 1,
                 discriminator: Callable = Discriminator1DLipschitzWasserstein,
                 discriminator_dim: int = 1,
                 alternate_training: bool = False,
                 gradients_path: str | Path = None,
                 *,
                 f=1,
                 preTrain=0
                 ):
        assert discriminator_dim == 1
        super().__init__(
            prior=prior,
            density_estimator=density_estimator,
            device=device,
            logging_level=logging_level,
            summary_writer=summary_writer,
            show_progress_bars=show_progress_bars,
            gamma=gamma,
            discriminator=discriminator,
            discriminator_dim=1,
            alternate_training=alternate_training,
            gradients_path=gradients_path,
            f=f,
            preTrain=preTrain
        )

    def _loss_discriminator(self,
                            theta: Tensor,
                            x: Tensor,
                            ) -> Tensor:
        _, p_hat = (x.squeeze(1) for x in self._neural_net.sample_and_log_prob(1, context=x))
        log_prob = self._neural_net.log_prob(theta, x)
        assert log_prob.shape == p_hat.shape

        l1 = self._discriminator(p_hat[..., None]).squeeze(1)
        l2 = self._discriminator(log_prob[..., None]).squeeze(1)
        assert l1.shape == p_hat.shape
        assert l2.shape == log_prob.shape

        return -torch.abs(l1.mean() - l2.mean())


class SNPE_CDF2(SNPE_TV):
    """
        discriminator depends on (p,x)
    """

    def __init__(self,
                 prior: Union[Distribution],
                 density_estimator: Union[str, Callable] = "maf",
                 device: str = "cpu",
                 logging_level: Union[int, str] = "WARNING",
                 summary_writer: Optional[TensorboardSummaryWriter] = None,
                 show_progress_bars: bool = True,
                 gamma: float = 1,
                 discriminator: Callable = Discriminator1DLipschitzWasserstein,
                 discriminator_dim: int = 9,
                 alternate_training: bool = False,
                 gradients_path: str | Path = None,
                 *,
                 f=1,
                 preTrain=0
                 ):
        super().__init__(
            prior=prior,
            density_estimator=density_estimator,
            device=device,
            logging_level=logging_level,
            summary_writer=summary_writer,
            show_progress_bars=show_progress_bars,
            gamma=gamma,
            discriminator=discriminator,
            discriminator_dim=discriminator_dim,
            alternate_training=alternate_training,
            gradients_path=gradients_path,
            f=f,
            preTrain=preTrain
        )

    def _loss_discriminator(self,
                            theta: Tensor,
                            x: Tensor,
                            ) -> Tensor:
        _, p_hat = (x.squeeze(1) for x in self._neural_net.sample_and_log_prob(1, context=x))
        log_prob = self._neural_net.log_prob(theta, x)
        assert log_prob.shape == p_hat.shape

        l1 = self._discriminator(torch.cat([p_hat[..., None], x], 1)).squeeze(1)
        l2 = self._discriminator(torch.cat([log_prob[..., None], x], 1)).squeeze(1)

        assert l1.shape == p_hat.shape
        assert l2.shape == log_prob.shape

        return -torch.abs(l1 - l2).mean()


@torch.no_grad()
def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, load_cp_continue, optimizer=None):
    """
    Load an existing checkpoint of the model to continue training or for
    evaluation.

    :param torch.nn.Module model: Model loaded to continue training or for
        evaluation
    :param checkpoint_path: Checkpoint for continuing to train or for
        evaluation
    :type checkpoint_path: Path
    :param bool load_cp_continue: Whether the checkpoint is loaded to continue
        training
    :param optimizer: Optimizer that was used
    :type optimizer: torch.optim or None
    """
    checkpoint = load(checkpoint_path, map_location=torch.device("cpu"))
    if isinstance(checkpoint, OrderedDict):
        model.load_state_dict(state_dict=checkpoint)
    else:
        model.load_state_dict(state_dict=checkpoint["state_dict"])

    if load_cp_continue:
        assert optimizer is not None, (
            f"When checkpoint is loaded to continue training, optimizer "
            f"cannot  be {optimizer}"
        )
        optimizer.load_state_dict(state_dict=checkpoint["optimizer"])
    print("--> The checkpoint of the model is being loaded")


@torch.no_grad()
def highest_density_level(pdf, alpha, bias=0.0, min_epsilon=10e-17, region=False):
    # Check if a proper bias has been specified.
    if bias >= alpha:
        raise ValueError("The bias cannot be larger or equal to the specified alpha level.")
    # Detect numpy type
    if type(pdf).__module__ != np.__name__:
        pdf = pdf.cpu().clone().numpy()
    else:
        pdf = np.array(pdf)
    total_pdf = pdf.sum()
    pdf /= total_pdf
    # Compute highest density level and the corresponding mask
    n = len(pdf)
    optimal_level = pdf.max().item()
    epsilon = 10e-02
    while epsilon >= min_epsilon:
        area = float(0)
        while area <= (alpha + bias):
            # Compute the integral
            m = (pdf >= optimal_level).astype(np.float32)
            area = np.sum(m * pdf)
            # Compute the error and apply gradient descent
            optimal_level -= epsilon
        optimal_level += 2 * epsilon
        epsilon /= 10
    optimal_level *= total_pdf
    if region:
        return optimal_level, torch.from_numpy(m)
    else:
        return optimal_level


@torch.no_grad()
def get_log_prob_for_context(r, inputs, context=None, embedded_context=None, ensemble=False):
    if not ensemble:
        if embedded_context is None:
            embedded_context = r._embedding_net(context.unsqueeze(0))
        noise, logabsdet = r._transform(
            inputs,
            context=embedded_context.expand(
                len(inputs), *[-1] * (embedded_context.ndim - 1)
            ),
        )
        log_prob = r._distribution.log_prob(
            noise,
            context=embedded_context.expand(
                len(inputs), *[-1] * (embedded_context.ndim - 1)
            ),
        )
        return log_prob + logabsdet
    else:
        posteriors = [
            get_log_prob_for_context(
                flow,
                inputs,
                context,
                embedded_context[idx]
                if embedded_context is not None
                else None,
            )
            for idx, flow in enumerate(r.flows)
        ]
        log_sum_exp = torch.stack(posteriors, dim=0).logsumexp(dim=0)
        return log_sum_exp - torch.tensor(
            float(len(r.flows)), device=log_sum_exp.device
        )


@torch.no_grad()
def compute_log_posterior(r, observable, extent, resolution=100, return_grid=False, batch_size=None):
    # Prepare grid
    epsilon = 0.00001
    # Account for half-open interval of uniform
    # prior, ``accelerator`` is just the device:

    p1 = torch.linspace(extent[0], extent[1] - epsilon, resolution)
    p2 = torch.linspace(extent[2], extent[3] - epsilon, resolution)
    g1, g2 = torch.meshgrid(p1.view(-1), p2.view(-1), indexing='ij')
    # Vectorize
    inputs = torch.cat([g1.reshape(-1, 1), g2.reshape(-1, 1)], dim=1)

    embeded_observable = r._embedding_net(observable)

    if batch_size is None:
        log_posterior = get_log_prob_for_context(r, inputs=inputs, embedded_context=embeded_observable, ensemble=False)
    else:
        log_posterior = torch.empty(inputs.shape[0])
        for b in range(0, inputs.shape[0], batch_size):
            cur_inputs = inputs[b: b + batch_size]
            log_posterior[b: b + batch_size] = get_log_prob_for_context(r, inputs=cur_inputs, embedded_context=embeded_observable, ensemble=False)

    log_posterior = log_posterior.view(resolution, resolution).cpu()

    if return_grid:
        return log_posterior, p1.cpu(), p2.cpu()
    else:
        return log_posterior


@torch.no_grad()
def plot_posterior(p1, p2, pdf, nominal, mean_1, mean_2, index, outputdir: Path, extent):
    p1 = p1.cpu()
    p2 = p2.cpu()
    pdf = pdf.cpu()
    nominal = nominal.cpu()
    mean_1 = mean_1.cpu()
    mean_2 = mean_2.cpu()
    g1, g2 = torch.meshgrid(p1.view(-1), p2.view(-1), indexing='ij')
    plt.pcolormesh(
        g1, g2, pdf, antialiased=True, edgecolors="face", shading="auto"
    )
    plt.set_cmap("viridis_r")
    plt.colorbar()
    plt.plot(nominal[0, 0], nominal[0, 1], "*", color="k")
    plt.hlines(mean_2, extent[0], extent[1])
    plt.vlines(mean_1, extent[2], extent[3])

    if outputdir is not None:
        plt.savefig(outputdir / "posterior_{}.pdf".format(index))
    else:
        plt.show()
    plt.close()


@torch.no_grad()
def estimate_coverage(r: Module, inputs: Tensor, outputs: Tensor, outputdir: Path, extent: list, alphas=[0.05]):
    n = len(inputs)
    covered = [0 for _ in alphas]
    sizes = [[] for _ in range(len(alphas))]
    bias = [0.0, 0.0]
    bias_square = [0.0, 0.0]
    variance = [0.0, 0.0]

    if outputs is not None:
        outputdir.mkdir(exist_ok=True)

    resolution = 90  # this is BNRE (and CalNRE) hyperparameter, kept for now
    return_grid = True

    length_1 = (extent[1] - extent[0]) / resolution
    length_2 = (extent[3] - extent[2]) / resolution

    for index in tqdm(range(n), "Coverages evaluated"):
        # Prepare setup
        nominal = inputs[index]
        nominal = nominal.squeeze().unsqueeze(0)

        observable = outputs[index].squeeze().unsqueeze(0)
        with torch.no_grad():
            pdf, p1, p2 = compute_log_posterior(
                r=r,
                extent=extent,
                observable=observable,
                resolution=resolution,
                return_grid=return_grid,
            )
            pdf = pdf.exp()
            nominal_pdf = r.log_prob(inputs=nominal, context=observable).exp().cpu()

        for i, alpha in enumerate(alphas):
            level, mask = highest_density_level(pdf, alpha, region=True)
            sizes[i].append(mask.sum() / np.prod(mask.shape))
            if nominal_pdf >= level:
                covered[i] += 1

        pdf = pdf / (length_1 * length_2 * pdf.sum())
        margin_1 = pdf.sum(dim=1) * length_2
        margin_2 = pdf.sum(dim=0) * length_1
        mean_1 = (margin_1 * length_1 * p1).sum()
        mean_2 = (margin_2 * length_2 * p2).sum()
        bias[0] += torch.abs((mean_1 - nominal[0, 0]).cpu().float())
        bias[1] += torch.abs((mean_2 - nominal[0, 1]).cpu().float())
        bias_square[0] += (mean_1 - nominal[0, 0]).cpu().float() ** 2
        bias_square[1] += (mean_2 - nominal[0, 1]).cpu().float() ** 2
        variance[0] += (margin_1 * length_1 * (p1 - mean_1) ** 2).sum().cpu().float()
        variance[1] += (margin_2 * length_2 * (p2 - mean_2) ** 2).sum().cpu().float()

        if index < 20:
            plot_posterior(
                p1=p1,
                p2=p2,
                pdf=pdf,
                nominal=nominal,
                mean_1=mean_1,
                mean_2=mean_2,
                index=index,
                outputdir=outputdir,
                extent=extent,
            )

    return [x / n for x in covered], sizes, [x / n for x in bias], [x / n for x in variance], [x / n for x in bias_square]
