import matplotlib.pyplot as plt
import plotly.colors
import torch
import time
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Distribution
import torch.nn.functional as F
from pyknos.nflows.flows import Flow

import zuko
from zuko.utils import broadcast

from sbi.inference.posteriors.direct_posterior import DirectPosterior

from lampe.data import JointLoader, JointDataset
from lampe.inference import NRE, NRELoss, MetropolisHastings
from lampe.plots import nice_rc, corner, mark_point, coverage_plot
from lampe.utils import GDStep, gridapply
from lampe.diagnostics import expected_coverage_mc, expected_coverage_ni

from pathlib import Path
from typing import *
import pickle
from itertools import islice
from tqdm import tqdm

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from functools import partial

from utility import Timer


class my_NRELoss(nn.Module):

    def __init__(self, estimator: nn.Module):
        super().__init__()

        self.estimator = estimator

    def forward(self, theta: Tensor, theta_prime: Tensor, x: Tensor) -> Tensor:
        assert theta.shape == theta_prime.shape
        log_r, log_r_prime = self.estimator(
            torch.stack((theta, theta_prime)),
            x,
        )

        l1 = -F.logsigmoid(-log_r).mean()  # == 0
        l0 = -F.logsigmoid(log_r_prime).mean()  # == 1
        # => log (theta / theta_prime)

        return (l1 + l0) / 2


def my_expected_TV(g: Callable[[Tensor, Tensor], Tensor], P: JointDataset, Q: JointDataset, n: int = 1024, ) -> Tensor:
    with torch.no_grad():
        return 0.5 * torch.abs(torch.sign(g(P.theta, P.x)).mean() - torch.sign(g(Q.theta, Q.x)).mean())


def my_expected_KL(g: Callable[[Tensor, Tensor], Tensor], P: JointDataset, n: int = 1024, ) -> Tensor:
    with torch.no_grad():
        return -g(P.theta, P.x).mean()


def my_expected_coverage(log_p: Callable[[Tensor, Tensor], Tensor], samples: Iterable[Tensor], pairs: Iterable[Tuple[Tensor, Tensor]], n: int = 1024, ) -> Tuple[
    Tensor, Tensor]:
    ranks = []

    from torchdata.datapipes.map import SequenceWrapper

    with torch.no_grad():
        for (theta, x), sample in tqdm(SequenceWrapper(pairs).zip(SequenceWrapper(samples)), unit='pair'):
            mask = log_p(theta, x) < log_p(sample, x)
            rank = mask.sum() / mask.numel()

            ranks.append(rank)

    ranks = torch.stack(ranks).cpu()
    ranks = torch.cat((ranks, ranks.new_tensor((0.0, 1.0))))

    return (
        torch.sort(ranks).values,
        torch.linspace(0, 1, len(ranks)),
    )


def my_coverage_plot(levels: Tensor, coverages: Tensor, color: str = None, legend: str = None, figure: go.Figure = None, **kwargs) -> go.Figure:
    r"""
        Copied from lampe.plots.coverage_plot and made into plotly plots
    """

    levels, coverages = torch.as_tensor(levels), torch.as_tensor(coverages)

    # Figure
    if figure is None:
        figure = go.Figure()

    # Plot
    if next(figure.select_traces(selector=dict(name='ideal_diagonal', type='scatter', mode='lines'), **kwargs), None) is None:
        figure.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(color='black', dash='dash', width=1), mode='lines', name='ideal_diagonal', showlegend=False), **kwargs)

    figure.add_trace(go.Scatter(x=levels, y=coverages, line=dict(color=color), mode='lines', name=legend), **kwargs)

    # for xaxes in figure.select_xaxes(**kwargs):
    #     xaxes.update(dict(range=[-0.1, 1.1], minor=dict(showgrid=True), title=r'Credible level'))
    # for yaxes in figure.select_yaxes(**kwargs):
    #     yaxes.update(dict(minor=dict(showgrid=True), title=r'Expected coverage'))

    figure.update_xaxes(range=[-0.1, 1.1], minor=dict(showgrid=True), title=r'Credible level', **kwargs)
    figure.update_yaxes(minor=dict(showgrid=True), title=r'Expected coverage', **kwargs)

    return figure


def test_classification(prior: Distribution, simulator: Callable, posterior: DirectPosterior, pdf_path: Path = None, epochs: int = 32, title='',
                        legend: Literal['inside', 'outside', 'none'] = 'inside'):
    theta = prior.sample()
    x = simulator(theta)

    posterior_model: Flow = posterior.posterior_estimator

    loader = JointLoader(prior, simulator, batch_size=256, vectorized=True)

    estimator = NRE(theta_dim=theta.numel(), x_dim=x.numel(), hidden_features=[64] * 5, activation=nn.ELU)

    loss = my_NRELoss(estimator)
    optimizer = optim.AdamW(estimator.parameters(), lr=1e-3)
    step = GDStep(optimizer, clip=1.0)  # gradient descent step with gradient clipping

    estimator.train()

    with tqdm(range(epochs), unit='epoch') as tq, Timer("Training NRE"):
        for _ in tq:
            losses = torch.stack([
                step(loss(theta, posterior_model.sample(1, context=x).squeeze(1), x))
                for theta, x in islice(loader, 512)
            ])  # => log (model_posterior / true_posterior)

            tq.set_postfix(loss=losses.mean().item())

    estimator.eval()

    theta = prior.sample((1024,))
    x = simulator(theta)
    testset = JointDataset(theta, x)

    with torch.no_grad(), Timer("Sample from posterior model"):
        samples = torch.stack([posterior.set_default_x(_x).sample((1024,)) for _x in x])

    with torch.no_grad(), Timer("Calculating Coverage"):
        npe_levels, npe_coverages = my_expected_coverage(lambda theta, x: posterior.set_default_x(x).log_prob(theta), samples, testset)

        nre_levels, nre_coverages = my_expected_coverage(estimator, samples, testset)

        TV_metric = my_expected_TV(estimator, testset, JointDataset(samples, x))
        KL_metric = my_expected_KL(estimator, testset)

    fig = make_subplots(cols=1, rows=1, horizontal_spacing=0.01, column_titles=[f'{title}TV={float(TV_metric):.3f} LK={float(KL_metric):.3f}'])

    fig = my_coverage_plot(npe_levels, npe_coverages, legend='class.', figure=fig, col=1, row=1, color=plotly.colors.DEFAULT_PLOTLY_COLORS[1])
    fig = my_coverage_plot(nre_levels, nre_coverages, legend='ratio', figure=fig, col=1, row=1, color=plotly.colors.DEFAULT_PLOTLY_COLORS[0])

    fig.update_yaxes(title_text='', showticklabels=False, col=2, row=1)

    if legend == 'none':
        fig.update_traces(showlegend=False)
    elif legend == 'inside':
        fig.update_layout(
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ))
    fig.update_layout(height=200, width=300 if legend == 'outside' else 200,
                      # title_text=f'{title}TV={float(TV_metric):.3f} LK={float(KL_metric):.3f}',
                      boxmode='group',
                      boxgap=0.0,
                      boxgroupgap=0.0,
                      margin=dict(l=00, r=0, b=0, t=0, pad=0),
                      )

    if pdf_path is not None:
        fig2 = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])
        fig2.write_image(pdf_path)

        time.sleep(1)
        fig.write_image(pdf_path)
    fig.show()
