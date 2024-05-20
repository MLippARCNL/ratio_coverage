sys.path.append(r".rc")

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import zuko
import math
from itertools import islice
from tqdm import tqdm
import lampe
from lampe.data import JointLoader, H5Dataset
from lampe.inference import NPE, NPELoss
from lampe.diagnostics import expected_coverage_mc, expected_coverage_ni
from lampe.plots import nice_rc, corner, mark_point, coverage_plot
from lampe.utils import GDStep
import numpy as np
from simulation_based_inference.npe.npe import *
from simulation_based_inference.nre.nre import *
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


if __name__ == "__main__":
    import sbibm
    fig_path = Path(r".\data")
    assert fig_path.exists()

    budget = 2 ** 10  # 10, (12, 14,) 16
    batch_size = 512
    _iter = 0
    hidden_features = 64
    num_transforms = 3

    if not (fig_path / f'{budget}_{batch_size}_dict ({_iter + 1}).pickle').exists():
        raise Exception(f"cannot LOAD_DICT: {(fig_path / f'{budget}_{batch_size}_dict ({_iter + 1}).pickle')}")

    if (fig_path / f'{budget}_{batch_size}_dict ({_iter + 1}).json').exists():
        with (fig_path / f'{budget}_{batch_size}_dict ({_iter + 1}).json').open("r") as f:
            config = json.loads(f.read())
    else:
        print('Information missing:')
        _task = input('Task:')
        _f = int(input('f:'))
        _gamma = float(input('gamma:'))
        config = dict(task=_task, f=_f, gamma=_gamma, fig_path=str((fig_path / f'{budget}_{batch_size}_dict ({_iter + 1}).pickle').resolve()))
    print(json.dumps(config, indent=4))

    task = sbibm.get_task(config['task'])
    observation = task.get_observation(num_observation=1)
    reference_samples = task.get_reference_posterior_samples(num_observation=1)

    prior = task.get_prior_dist()
    simulator = task.get_simulator()
    domain = (task.prior_params['low'], task.prior_params['high'])
    print('>>>', task.dim_data, task.dim_parameters)

    print('Starting', budget, batch_size, _iter + 1, ':', datetime.datetime.now())
    density_estimator = posterior_nn(
        model="nsf",
        hidden_features=hidden_features,
        num_transforms=num_transforms,
        embedding_net=build_embedding(),
    )
    inference = SNPE_CDF2(f=config['f'],
                          prior=prior,
                          density_estimator=density_estimator,
                          summary_writer=None,
                          device="cpu",
                          show_progress_bars=True,
                          gamma=config['gamma'],
                          alternate_training=True,
                          # discriminator_dim=task.dim_data + 1,
                          discriminator_dim=task.dim_data + task.dim_parameters,
                          # gradients_path=fig_path / f'{budget}_{batch_size}_grad ({_iter + 1}).npy',
                          # discriminator=Discriminator1DHeavyTails
                          )
    print('starting "LOAD_DICT"', datetime.datetime.now())
    with (fig_path / f'{budget}_{batch_size}_dict ({_iter + 1}).pickle').open("rb") as file:
        inference._best_state_dict = pickle.load(file)
    _theta = prior.sample((batch_size,))
    inference._neural_net = inference._build_neural_net(_theta.to("cpu"), simulator(_theta).to("cpu"))
    inference._neural_net.load_state_dict(state_dict=inference._best_state_dict)
    inference._neural_net.eval()
    posterior = inference.build_posterior(inference._neural_net, prior=prior)
    posterior.sample = partial(posterior.sample, show_progress_bars=False)

    test_classification(prior, simulator, posterior, pdf_path=fig_path / f'{budget}_{batch_size}_NRE ({_iter + 1}).pdf', title='', legend='none' if budget != 2**16 else 'outside')

    print('PRAISE THE SUN', budget, batch_size, _iter + 1, ':', datetime.datetime.now())
    input('...')
