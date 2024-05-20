sys.path.append(r".")

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import zuko
import math
from itertools import islice
from tqdm import tqdm
import lampe
from lampe.data import JointLoader, H5Dataset, JointDataset
from lampe.inference import NPE, NPELoss
from lampe.diagnostics import expected_coverage_mc, expected_coverage_ni
from lampe.plots import nice_rc, corner, mark_point, coverage_plot
from lampe.utils import GDStep
import numpy as np
from simulation_based_inference.npe.npe import *
from simulation_based_inference.nre.nre import my_coverage_plot, my_expected_coverage
from utility.misc import *
from torch.autograd import Variable
import sbibm
from sbibm.metrics import c2st
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, callback
from threading import Thread, Event
import time
import pdb
from pathlib import Path
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi
from sbi.utils import posterior_nn
from sbi.inference.posteriors import DirectPosterior
from sbi.neural_nets.flow import build_maf, build_nsf
from torch.distributions.multivariate_normal import MultivariateNormal as Normal
import tempfile
import datetime
from functools import partial
import pickle
import json
import plotly.express as px


# torch.set_num_threads(64)


class Uniform(torch.distributions.uniform.Uniform):

    def __init__(self, lower, upper):
        super(Uniform, self).__init__(lower, upper)

    def log_prob(self, sample):
        return super(Uniform, self).log_prob(sample).mean()


class my_SBI_wrapper():

    def __init__(self, nn, x):
        self.nn = nn
        self.context = x

    def log_prob(self, theta):
        if len(theta.shape) == 2:
            return self.nn.log_prob(theta, context=self.context.tile((theta.size(0),)).reshape(theta.size(0), -1))
        return self.nn.log_prob(theta.unsqueeze(0), context=self.context.unsqueeze(0))

    def sample(self, shape):
        return self.nn.sample(shape[0], context=self.context.unsqueeze(0))[0]


def build_embedding():
    hidden = 64
    latent = 10
    return torch.nn.Sequential(
        torch.nn.Linear(task.dim_data, hidden),
        torch.nn.SELU(),
        torch.nn.Linear(hidden, hidden),
        torch.nn.SELU(),
        torch.nn.Linear(hidden, hidden),
        torch.nn.SELU(),
        torch.nn.Linear(hidden, latent))


def slcp(theta):
    batch = theta.size(0)
    # Mean
    mean = torch.tensor([0.7, -2.9]).tile(batch).view(-1, 2)

    # Covariance
    s1 = theta[:, 0] ** 2
    s2 = theta[:, 1] ** 2
    rho = torch.distributions.uniform.Uniform(-3.0, 3.0).sample((batch,)).tanh()

    cov = torch.stack(
        [
            s1 ** 2,
            rho * s1 * s2,
            rho * s1 * s2,
            s2 ** 2,
        ]
    ).transpose(1, 0).reshape(-1, 2, 2)

    normal = torch.distributions.MultivariateNormal(mean, cov)

    # x = torch.distributions.multivariate_normal.MultivariateNormal(_mu, _sigma).sample((4,)).transpose(0, -2).flatten(-2)

    return normal.sample((4,)).transpose(1, 0).flatten(-2)


def _slcp(inputs):
    samples = []

    for _input in inputs:
        mean = torch.tensor([_input[0], _input[1]])
        scale = 1.0
        s_1 = _input[2] ** 2
        s_2 = _input[3] ** 2
        rho = _input[4].tanh()
        covariance = torch.tensor([
            [scale * s_1 ** 2, scale * rho * s_1 * s_2],
            [scale * rho * s_1 * s_2, scale * s_2 ** 2]])
        normal = Normal(mean, covariance)
        x_out = normal.sample(torch.Size([4])).view(1, -1)
        samples.append(x_out.view(1, -1))

    return torch.cat(samples, dim=0)


# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"

from sbi.utils.get_nn_models import posterior_nn
from zuko.distributions import BoxUniform

LOWER = -3 * torch.ones(5)
UPPER = 3 * torch.ones(5)
# prior = BoxUniform(LOWER, UPPER)
# simulator = slcp
# LOWER = -3 * torch.ones(5).float()
# UPPER = 3 * torch.ones(5).float()
# prior = Uniform(LOWER, UPPER)
# simulator = _slcp
config = {}

config['validation_size'] = 2 ** 11
config['learning_rate_SGD'] = 0.1
config['learning_rate_Adam'] = 0.001
config['batch_size'] = 512
config['num_rounds'] = 10
config['restore_best_dict'] = True
config['hidden_features'] = 64
config['num_transforms'] = 3
config['epochs'] = 200
config['preTrain'] = 0
config['gamma'] = 00.0
config['f'] = 0

smoothing = 2

fig_path = Path(r".\data")
assert fig_path.exists()

config['fig_path'] = str(fig_path.resolve())

H5Dataset.__enter__ = lambda self: self
H5Dataset.__exit__ = lambda self, type, value, traceback: self.file.__exit__(type, value, traceback)

task = sbibm.get_task("slcp")
# task = sbibm.get_task("two_moons")
# task = sbibm.get_task("gaussian_mixture")
observation = task.get_observation(num_observation=1)
reference_samples = task.get_reference_posterior_samples(num_observation=1)
config['task'] = task.name

prior = task.get_prior_dist()
simulator = task.get_simulator()

print('>>>', task.dim_data, task.dim_parameters)
print('>>>', json.dumps(config, indent=4))

LOAD_DICT = True

for _iter in range(1):
    start_time = time.time()
    with tempfile.TemporaryDirectory() as tmpdirname:
        loader = JointLoader(prior, simulator, batch_size=config['batch_size'], vectorized=True)
        H5Dataset.store(loader, Path(tmpdirname) / 'test.h5', size=2 ** 11, overwrite=True)
        with H5Dataset(Path(tmpdirname) / 'test.h5') as testset:
            for budget in 2 ** np.array([10, 16]):  # 10, 12, 14, 16
                print('Starting', budget, config['batch_size'], _iter + 1, ':', datetime.datetime.now())
                density_estimator = posterior_nn(
                    model="nsf",  # "nsf", "maf"
                    hidden_features=config['hidden_features'],
                    num_transforms=config['num_transforms'],
                    embedding_net=build_embedding(),
                )

                inference = SNPE_CDF2(f=config['f'],
                                      prior=prior,
                                      density_estimator=density_estimator,
                                      summary_writer=None,
                                      device=device,
                                      show_progress_bars=True,
                                      gamma=config['gamma'],
                                      alternate_training=True,
                                      discriminator_dim=task.dim_data + task.dim_parameters,
                                      discriminator=Discriminator1DLipschitzWasserstein,
                                      preTrain=config['preTrain']
                                      )

                _resume_training = False
                theta_train, x_train = simulate_for_sbi(simulator, prior, num_simulations=int(budget + config['validation_size']))
                inference.append_simulations(
                    theta=theta_train,
                    x=x_train,
                    data_device="cpu",
                    # proposal=prior
                )
                # for _i in range(config['num_rounds']):
                if not LOAD_DICT:
                    with tqdm(range(config['num_rounds']), unit='rounds') as tq, redirect(), Timer('total Training'):
                        for _i in tq:
                            inference.train(
                                learning_rate_SGD=config['learning_rate_SGD'] * config["batch_size"] / 32,
                                learning_rate_Adam=config['learning_rate_Adam'] * config["batch_size"] / 32,
                                max_num_epochs=config['epochs'] - 1,
                                training_batch_size=config["batch_size"],
                                validation_fraction=config['validation_size'] / float(budget + config['validation_size']),
                                show_train_summary=False,
                                resume_training=_resume_training,
                                clip_max_norm=2.0,
                                force_first_round_loss=True,
                            )
                            _resume_training = True
                else:
                    if not (fig_path / f'{budget}_{config["batch_size"]}_dict ({_iter + 1}).pickle').exists():
                        raise Exception(f"cannot LOAD_DICT: " + str(fig_path / f'{budget}_{config["batch_size"]}_dict ({_iter + 1}).pickle'))
                    print('starting "LOAD_DICT"', datetime.datetime.now())
                    with (fig_path / f'{budget}_{config["batch_size"]}_dict ({_iter + 1}).pickle').open("rb") as file:
                        inference._best_state_dict = pickle.load(file)
                    _theta = prior.sample((config["batch_size"],))
                    inference._neural_net = inference._build_neural_net(_theta.to("cpu"), simulator(_theta).to("cpu"))

                if hasattr(inference, "_best_state_dict"):
                    if not LOAD_DICT:
                        print('starting "SAVE_DICT"', datetime.datetime.now())
                        with (fig_path / f'{budget}_{config["batch_size"]}_dict ({_iter + 1}).pickle').open("wb") as file:
                            if config['restore_best_dict']:
                                pickle.dump(inference._best_state_dict, file)
                            else:
                                pickle.dump(inference._estimator_cpu_state_dict(), file)
                        with (fig_path / f'{budget}_{config["batch_size"]}_dict ({_iter + 1}).json').open("w") as f:
                            json.dump(config, f)
                    if config['restore_best_dict'] or LOAD_DICT:
                        inference._neural_net.load_state_dict(state_dict=inference._best_state_dict)
                else:
                    print_error("NO BEST DICT")

                with torch.no_grad(), Timer(f"starting 'posterior' {datetime.datetime.now()}"):
                    posterior = inference.build_posterior(inference._neural_net)
                    posterior.sample = partial(posterior.sample, show_progress_bars=False)

                fig2 = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])
                fig2.write_image(fig_path / f'opfer.pdf')
                time.sleep(1)

                if False:
                    with torch.no_grad(), Timer(f"starting 'posterior_samples' {datetime.datetime.now()}"):
                        posterior_samples = posterior.set_default_x(observation).sample(sample_shape=(10_000,))
                        c2st_accuracy = c2st(reference_samples, posterior_samples)
                        print(c2st_accuracy)

                    fig = my_corner_plot(posterior_samples, legend='model', smooth=smoothing)  # , domain=(LOWER, UPPER)
                    
                    fig.update_xaxes(minor=dict(ticklen=0, tickcolor="black"), tickfont=dict(color="rgba(0,0,0,0)", size=1), title_standoff=5)
                    fig.update_yaxes(minor=dict(ticklen=0, tickcolor="black"), tickfont=dict(color="rgba(0,0,0,0)", size=1), title_standoff=1)

                    fig.update_layout(
                        boxmode='group',
                        boxgap=0.0,
                        boxgroupgap=0.0,
                        margin=dict(l=0, r=0, b=0, t=0, pad=0),fig = my_corner_plot(reference_samples, legend='truth', figure=fig, smooth=smoothing, color=plotly.colors.DEFAULT_PLOTLY_COLORS[1])
                    # fig.update_layout(title_text=f'c2st={c2st_accuracy[0]:.3f}')
                    # plt.savefig(fig_path / f'{budget}_{config["batch_size"]}_sbi ({_iter + 1}).pdf')
                        legend=dict(
                            yanchor="top",
                            y=0.99,
                            xanchor="right",
                            x=0.99
                        )
                    )
                    fig.show()
                    fig.write_image(fig_path / f'{budget}_{config["batch_size"]}_sbi ({_iter + 1}).pdf')

                # Save validation losses (per stage):
                # np.save(os.path.join(outputdir, "val_losses.npy"), inference.val_losses)
                # plt.figure()
                # plt.plot(inference._summary["validation_log_probs"])

                """ ------------ SBI plotting coverage ------------ """
                # Save checkpoint corresponding to latest model:
                # inference.save_checkpoint(path=outputdir)
                # Save model parameters of best model:
                # torch.save(inference._best_state_dict,os.path.join(outputdir, "checkpoint_best.pt"))
                # confidence_levels = np.linspace(0.05, 0.95, 19)
                # theta_test, x_test = simulate_for_sbi(simulator, prior, num_simulations=2 ** 11)
                # coverages, contour_sizes, bias, variance, bias_square = estimate_coverage(r=inference._neural_net, inputs=theta_test, outputs=x_test, outputdir=fig_path,
                #                                                                           extent=[LOWER[0].item(), UPPER[0].item(), LOWER[1].item(), UPPER[1].item(), ],
                #                                                                           alphas=confidence_levels, )
                # coverage_plot(confidence_levels, coverages, legend='NPE')
                # plt.title(str(budget))
                # plt.savefig(fig_path / f'{budget}_sbi.pdf')
                with torch.no_grad(), Timer(f"starting 'coverages' {datetime.datetime.now()}"):
                    if not isinstance(testset, JointDataset):
                        testset = testset.to_memory()
                    samples = torch.stack([posterior.set_default_x(_x).sample((1024,)) for _x in testset.x])
                    confidence_levels, coverages = my_expected_coverage(lambda theta, x: posterior.set_default_x(x).log_prob(theta), samples, testset)
                    # confidence_levels, coverages = expected_coverage_mc(lambda px: posterior.set_default_x(px), testset, device='cpu')
                    # confidence_levels, coverages = expected_coverage_ni(lambda theta, px: posterior.set_default_x(px).log_prob(theta), testset, (LOWER, UPPER), device='cpu')

                fig = my_coverage_plot(confidence_levels.numpy(), coverages.numpy(), legend='classical coverage',
                                       figure=make_subplots(cols=1, rows=1, column_titles=['classical coverage']))

                fig.update_traces(showlegend=False)
                fig.update_layout(height=200, width=200,
                                  boxmode='group',
                                  boxgap=0.0,
                                  boxgroupgap=0.0,
                                  margin=dict(l=00, r=0, b=0, t=0, pad=0)
                                  )
                # plt.savefig(fig_path / f'{budget}_{config["batch_size"]}_lampe ({_iter + 1}).pdf')
                fig.show()
                fig.write_image(fig_path / f'{budget}_{config["batch_size"]}_lampe ({_iter + 1}).pdf')

                np.savez(fig_path / f'{budget}_{config["batch_size"]}_lampe ({_iter + 1}).npz', confidence_levels.numpy(), coverages.numpy())

    print("\n--- %s seconds ---\n" % (time.time() - start_time))

plt.show()
input('...')
