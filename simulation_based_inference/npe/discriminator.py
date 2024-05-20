import numpy as np
import plotly
import torch
import torch.nn as nn
import torchvision
from torch import Tensor
from torch.nn.modules.module import T
from torch.optim import Optimizer
from torch.nn.utils.parametrizations import spectral_norm
from lampe.utils import GDStep
from utility import print_warn, print_success, print_color
from typing import *

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class my_GDStep():
    """
        Custom GDStep used with Discriminator1DLipschitz which ensures 1-lipschitz by rescaling the weights.
        Use my_GDStep.do_it_once=True to print gradients once.
    """

    def __init__(self, optimizer: Optimizer, disc: nn.Module, clip: float = None):
        self.do_it_once = True
        self.discriminator = disc
        self.optimizer = optimizer
        self.parameters = [
            p
            for group in optimizer.param_groups
            for p in group['params']
        ]
        self.clip = clip
        if not isinstance(self.discriminator, Discriminator1DLipschitz):
            print('####')
            print_warn(f'\tWrong discriminator! need Discriminator1DLipschitz and found {type(self.discriminator).__name__} for gradient normalization')
            print('####')
            self.do_norm = False
        else:
            print_success(f'\tfound Discriminator1DLipschitz for gradient normalization')
            self.do_norm = True

    def __call__(self, loss: torch.Tensor) -> torch.Tensor:
        if loss.isfinite().all():
            self.optimizer.zero_grad()
            loss.backward()

            if self.do_it_once:
                if hasattr(self.discriminator, 'print_grad'):
                    self.discriminator.print_grad()
                    self.do_it_once = False
                else:
                    print_warn(f'{type(self.discriminator).__name__} does not have `print_grad()` function')

            if self.clip is None:
                self.optimizer.step()
            else:
                norm = nn.utils.clip_grad_norm_(self.parameters, self.clip)
                if norm.isfinite():
                    self.optimizer.step()

        # if self.do_norm:
        #     if hasattr(self.discriminator, 'renormalize'):
        #         self.discriminator.renormalize()
        #         self.discriminator.print_weight()
        #     else:
        #         print_warn(f'{type(self.discriminator).__name__} does not have `renormalize()` function')

        return loss.detach()


class Discriminator1DLipschitzWasserstein(Discriminator1DLipschitz):
    """
         Basic Discriminator1D with renormalization function and tanh final activation.
    """

    def __init__(self, theta_dim):
        super().__init__(theta_dim)
        print('### Discriminator1DLipschitzWasserstein')
        self.layers = [
            nn.Linear(theta_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            # nn.Sigmoid(),
            nn.Tanh(),
        ]
        self.linear_relu_stack = nn.Sequential(*self.layers)


class wrappe_distribution(torch.distributions.distribution.Distribution):
    def __init__(self, fnc):
        self.fnc = fnc

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        return self.fnc(num_samples=int(torch.Size(sample_shape).numel()))



""" PLOTING namespace """


def credible_levels(hist: Tensor, creds: Tensor) -> Tensor:
    x, _ = torch.sort(hist.flatten(), descending=True)
    cdf = x.cumsum(dim=0)
    idx = torch.searchsorted(cdf, creds * cdf[-1])

    return x[idx]


def gaussian_blur(img: Tensor, sigma: float = 1.0) -> Tensor:
    size = 2 * int(3 * sigma) + 1

    k = np.arange(size) - size / 2
    k = np.exp(-(k ** 2) / (2 * sigma ** 2))
    k = k / np.sum(k)

    smooth = lambda x: np.convolve(x, k, mode='same')

    for i in range(len(img.shape)):
        img = np.apply_along_axis(smooth, i, img)

    return torch.as_tensor(img)


def my_corner_plot(
        data: Tensor,
        weights: Tensor = None,
        domain: Tuple[Tensor, Tensor] = None,
        bins: Union[int, Sequence[int]] = 64,
        creds: Sequence[float] = (0.6827, 0.9545, 0.9973),
        color: Union[str, tuple] = None,
        smooth: float = 0,
        figure: go.Figure = None,
        legend: str = None,
        **kwargs,
) -> go.Figure:
    D = data.shape[-1]

    if type(bins) is int:
        bins = [bins] * D

    if domain is None:
        lower, upper = data.min(dim=0).values, data.max(dim=0).values
    else:
        lower, upper = map(torch.as_tensor, domain)

    bins = [
        torch.linspace(lower[i], upper[i], bins[i] + 1)
        for i in range(D)
    ]

    hists: List[List[Tensor]] = [[torch.empty(0) for _ in range(D)] for _ in range(D)]

    for i in range(D):
        for j in range(i + 1):
            if i == j:
                hist, _ = torch.histogram(
                    data[..., i],
                    bins=bins[i],
                    density=True,
                    weight=weights,
                )
            else:
                hist, _ = torch.histogramdd(
                    torch.stack([data[..., i], data[..., j]]).T,
                    bins=(bins[i], bins[j]),
                    density=True,
                    weight=weights,
                )

            hists[i][j] = hist

    creds = torch.as_tensor(creds).sort(descending=True)[0]
    creds = torch.cat((creds, torch.zeros(1)))
    # levels = (creds - creds.min()) / (creds.max() - creds.min())
    # levels = (levels[:-1] + levels[1:]) / 2

    if color is None:
        color = plotly.colors.DEFAULT_PLOTLY_COLORS[0]
    _r, _g, _b = plotly.colors.unlabel_rgb(color)

    if figure is None:
        figure = make_subplots(rows=D, cols=D, horizontal_spacing=0.004, vertical_spacing=0.004, shared_xaxes=True)
        figure.update_traces(diagonal_visible=True, showupperhalf=False)
        figure.update_layout(width=400, height=400)
        for ci in range(1,len(creds)):
            figure.add_trace(go.Contour(z=[[0]],
                                        showscale=False,
                                        showlegend=True,
                                        visible=True,
                                        hoverinfo=None,
                                        name=f'{creds[ci-1] * 100:.1f} %',
                                        line=dict(color=f"rgba({0}, {0}, {0}, {float(ci / len(creds))})", width=0),
                                        fillcolor=f"rgba({0}, {0}, {0}, {float(ci / len(creds))})",
                                        contours=dict(type="constraint",
                                                      operation="][",
                                                      value=[1, 2],
                                                      coloring="none")
                                        ), row=2, col=1)

    for i in range(D):
        for j in range(D):
            hist = hists[i][j]

            if j > i:
                continue

            if hist is None:
                continue

            if smooth > 0:
                hist = gaussian_blur(hist, smooth)

            ## Draw
            x, y = bins[j], bins[i]
            x = (x[1:] + x[:-1]) / 2
            y = (y[1:] + y[:-1]) / 2

            if j + 1 > 1:
                figure.update_yaxes(title_text='', showticklabels=False, row=i + 1, col=j + 1)
            elif i == j:
                figure.update_yaxes(title_text=r'$\theta_' + str(i) + '$', showticklabels=False, row=i + 1, col=j + 1)
            else:
                figure.update_yaxes(title_text=r'$\theta_' + str(i) + '$', row=i + 1, col=j + 1)
            if i + 1 < D:
                figure.update_xaxes(title_text='', showticklabels=False, row=i + 1, col=j + 1)
            else:
                figure.update_xaxes(title_text=r'$\theta_' + str(j) + '$', row=i + 1, col=j + 1)

            if i == j:
                figure.add_trace(go.Scatter(x=x, y=hist, line=dict(color=color, width=1), showlegend=i == 0, name=legend), row=i + 1, col=j + 1)

            else:
   
                levels = torch.unique(credible_levels(hist, creds))
                for ci in range(len(levels) - 1):
                    figure.add_trace(go.Contour(z=hist,
                                                x=x,
                                                y=y,
                                                showscale=False,
                                                showlegend=False,
                                                line=dict(width=1, color=f"rgba({_r}, {_g}, {_b}, {float(ci / len(levels))})"),
                                                fillcolor=f"rgba({_r}, {_g}, {_b}, {float(ci / len(levels))})",
                                                # fig.data[2].fillcolor = 'rgba' + fig.data[0].line.color[3:-1] + ', ' + str(transparency) + ')'
                                                contours=dict(type="constraint",
                                                              operation="][",
                                                              value=[levels[ci], levels[ci + 1]],
                                                              coloring="none")
                                                ), row=i + 1, col=j + 1)
   

    return figure
