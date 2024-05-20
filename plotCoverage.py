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
from lampe.data import JointLoader, H5Dataset
from lampe.inference import NPE, NPELoss
from lampe.diagnostics import expected_coverage_mc, expected_coverage_ni
from lampe.plots import nice_rc, corner, mark_point, coverage_plot
from lampe.utils import GDStep
import numpy as np
from simulation_based_inference.npe.npe import *
from utility.misc import *
from torch.autograd import Variable
from functools import partial
from typing import Callable

from torch.distributions.transforms import AffineTransform
from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal
from torch.distributions.distribution import Distribution
from torch.distributions.categorical import Categorical
from torch.distributions.mixture_same_family import MixtureSameFamily

from plotly.subplots import make_subplots
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, callback

plt.rcParams.update(nice_rc(latex=False))  # nicer plot settings


def getNormalMixture(p1, p2, n=20):
    mix = Categorical(torch.as_tensor([1 for _ in range(n)]))
    if len(torch.as_tensor(p1).shape) == 1:
        p1 = p1[0]
    comp = Normal(torch.linspace(-4, 4, n), torch.as_tensor([p2 for _ in range(n)]))
    return MixtureSameFamily(mix, comp)


def calc(dist_ref: Distribution, dist_cmp: Distribution, *, prior: Distribution = None, use_ratio=False, n=1024, iter=1024):
    ranks = [torch.as_tensor(0.)]
    # ranks2 = [torch.as_tensor(0.)]

    for _ in tqdm(range(iter), unit='pair'):
        pri = prior.sample((1,)) if prior is not None else torch.as_tensor([0.0])
        theta = torch.square(pri) + dist_ref.sample((1,))
        samples = torch.square(pri) + dist_cmp.sample((n,))
        if use_ratio:
            mask = (dist_cmp.log_prob(theta) - dist_ref.log_prob(theta)) < (dist_cmp.log_prob(samples) - dist_ref.log_prob(samples))
        else:
            mask = dist_cmp.log_prob(theta) < dist_cmp.log_prob(samples)
        rank = mask.sum() / mask.numel()

        ranks.append(rank)
    ranks = torch.stack(ranks).cpu()
    ranks = torch.cat((ranks, ranks.new_tensor((0.0, 1.0))))

    return (
        torch.sort(ranks).values,
        torch.linspace(0, 1, len(ranks)),
    )


def calc2(dist_r: Callable, dist_c: Callable, prior: Distribution, *, use_ratio=False, n=1024, iter=1024):
    ranks = [torch.as_tensor(0.)]
    # ranks2 = [torch.as_tensor(0.)]

    for _ in tqdm(range(iter), unit='pair'):
        pri = prior.sample((1,)) if prior is not None else torch.as_tensor([0.0])
        dist_ref = dist_r(pri)
        dist_cmp = dist_c(pri)
        theta = dist_ref.sample((1,))
        samples = dist_cmp.sample((n,))
        if use_ratio:
            mask = (dist_ref.log_prob(theta) - dist_cmp.log_prob(theta)) < (dist_ref.log_prob(samples) - dist_cmp.log_prob(samples))
        else:
            mask = dist_cmp.log_prob(theta) < dist_cmp.log_prob(samples)
        rank = mask.sum() / mask.numel()

        ranks.append(rank)
    ranks = torch.stack(ranks).cpu()
    ranks = torch.cat((ranks, ranks.new_tensor((0.0, 1.0))))

    return (
        torch.sort(ranks).values,
        torch.linspace(0, 1, len(ranks)),
    )


x = torch.linspace(-5, 5, 1000)

ref_dist = torch.distributions.normal.Normal(loc=0, scale=1)
cmp_dist = torch.distributions.normal.Normal(loc=0.1, scale=0.5)
# cmp_dist = torch.distributions.uniform.Uniform(low=0.5, high=1)

ref = torch.exp(ref_dist.log_prob(x))
compare = torch.exp(cmp_dist.log_prob(x))

idx = ((x > -1).long() * (x < 1).long()).type(torch.bool)

npe_levels, npe_coverages = calc(cmp_dist, ref_dist)
npe_levels2, npe_coverages2 = calc(cmp_dist, ref_dist, use_ratio=True)

fig = make_subplots(rows=1, cols=4, column_titles=['p(x|theta)', 'coverage', 'ratio-coverage', 'p(x|theta=0)'], horizontal_spacing=0.1, vertical_spacing=0.01)

fig.update_layout(title="p(x|theta): x ~ 𝒩(theta+mean, scale), q(x|theta): x ~ 𝒩(theta+mean, scale)+𝒩(-(theta+mean), scale)")

fig.add_trace(go.Scatter(y=ref, x=x, name=f'ref', line=dict(color='rgb(31,119,180)')), 1, 4)
fig.add_trace(go.Scatter(y=compare, x=x, name=f'cmp'), 1, 4)
fig.add_trace(go.Scatter(y=ref[idx], x=x[idx], name=f'ref', fillcolor='rgba(31,119,180, 0.3)', mode='none', showlegend=False), 1, 4)
fig.add_trace(go.Scatter(y=compare[idx], x=x[idx], name=f'ref', fillcolor='rgba(255,127,14, 0.3)', fillpattern=dict(shape='x'), mode='none', showlegend=False), 1, 4)

fig.add_trace(go.Scatter(y=np.linspace(0, 1, 1000), x=np.linspace(0, 1, 1000), mode='lines', line=dict(color='black', dash='dash'), showlegend=False), 1, 2)
fig.add_trace(go.Scatter(y=np.linspace(0, 1, 1000), x=np.linspace(0, 1, 1000), mode='lines', line=dict(color='black', dash='dash'), showlegend=False), 1, 3)
fig.add_trace(go.Scatter(y=npe_coverages, x=npe_levels, name=f'coverage', line=dict(color='rgb(31,119,180)'), mode='lines', showlegend=False), 1, 2)
fig.add_trace(go.Scatter(y=npe_coverages2, x=npe_levels2, name=f'ratio coverage', line=dict(color='rgb(31,119,180)'), mode='lines', showlegend=False), 1, 3)

theta = torch.linspace(-2, 2, 100)
x = torch.linspace(0, 1, 1000)
_styling = dict(colorscale=[[0, 'rgba(31,119,180,0.5)'], [1, 'rgba(31,119,180,0.5)']], contours=dict(coloring='lines'), line=dict(dash='solid', width=2))
fig.add_trace(go.Contour(z=torch.stack([torch.exp(Normal(loc=t, scale=1).log_prob(x)) for t in theta]), x=x, y=theta, **_styling), 1, 1)
_styling.update(dict(colorscale=[[0, 'rgba(255,127,14,0.5)'], [1, 'rgba(255,127,14,0.5)']]))
fig.add_trace(go.Contour(z=torch.stack([torch.exp(Normal(loc=t, scale=1).log_prob(x)) for t in theta]), x=x, y=theta, **_styling), 1, 1)
fig.add_trace(go.Scatter(y=np.zeros_like(theta), x=x, mode='lines', line=dict(color='black', dash='dot'), showlegend=False), 1, 1)
# fig.update_xaxes(showticklabels=False, row=1, col=1)
# fig.update_yaxes(showticklabels=False, row=1, col=1)
fig.update_xaxes(range=[-0.05, 1.05], title=dict(text="Credible level"), row=1, col=2)
fig.update_yaxes(range=[-0.05, 1.05], title=dict(text="Expected Coverage"), row=1, col=2)
fig.update_xaxes(range=[-0.05, 1.05], title=dict(text="Credible level"), row=1, col=3)
fig.update_yaxes(range=[-0.05, 1.05], title=dict(text="Expected Coverage"), row=1, col=3)
fig.update_xaxes(title=dict(text="x"), row=1, col=1)
fig.update_yaxes(title=dict(text="theta"), row=1, col=1)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
external_script = ['https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-MML-AM_CHTML']

app = Dash(__name__, external_stylesheets=external_stylesheets, external_scripts=external_script)

app.layout = html.Div([
    dcc.Graph(figure=fig, id='my-figure', config={'toImageButtonOptions': {'format': 'svg', 'filename': f'time_measurement'}}),
    html.Div(id='slider-output-container'),
    dcc.Dropdown({'normal.Normal': 'Normal', 'uniform.Uniform': 'Unifrom', 'beta.Beta': 'Beta', 'binomial.Binomial': 'Binominal'}, 'normal.Normal', id='dist-dropdown'),
    html.Div(children='mean:', id='param1-container'),
    dcc.Slider(-10, 10, 0.01, value=0, id='param1-value', marks=None, tooltip={"placement": "bottom", "always_visible": True}),
    html.Div(children='scale:', id='param2-container'),
    dcc.Slider(-10, 10, 0.01, value=1, id='param2-value', marks=None, tooltip={"placement": "bottom", "always_visible": True}),
    html.Div(children='n:', id='param3-container'),
    dcc.Slider(1, int(2 ** 14), 1, value=1024, id='param3-value', marks=None, tooltip={"placement": "bottom", "always_visible": True}),
    html.Div(children='#peaks:', id='param4-container'),
    dcc.Slider(2, 30, 1, value=2, id='param4-value', marks=None, tooltip={"placement": "bottom", "always_visible": True}),
])


@callback(
    Output('slider-output-container', 'children'),
    Output('param1-container', 'children'),
    Output('param2-container', 'children'),
    Output('my-figure', 'figure'),
    Input('param1-value', 'value'),
    Input('param2-value', 'value'),
    Input('param3-value', 'value'),
    Input('param4-value', 'value'),
    Input('dist-dropdown', 'value'),
    Input('my-figure', 'figure'))
def update_output(p1, p2, n, num_peaks, dist_name, figure):
    global getNormalMixture
    p1 = float(p1)
    p2 = float(p2)
    x = torch.linspace(-5, 5, 1000)
    ref_dist = torch.distributions.normal.Normal(loc=p1, scale=p2)
    # transforms = [torch.distributions.AffineTransform(loc=1, scale=1)]
    # ref_dist = torch.distributions.TransformedDistribution(ref_dist, transforms)

    dist_name = dist_name.split('.')

    cmp_dist = getattr(getattr(torch.distributions, dist_name[0]), dist_name[1])
    # cmp_dist = cmp_dist(p1, p2)

    # def getUniform(*args, **kwargs):
    #     return Uniform(-5, 5, validate_args=False)
    #
    # getNormalMixture = getUniform

    getNormalMixture = partial(getNormalMixture, n=num_peaks)

    if p1 != -p1:
        cmp_dist = getNormalMixture(p1, p2)
    else:
        cmp_dist = cmp_dist(loc=p1, scale=p2)

    # cmp_dist = torch.distributions.uniform.Uniform(low=0.5, high=1)

    ref = torch.exp(ref_dist.log_prob(x))
    compare = torch.exp(cmp_dist.log_prob(x))

    idx = ((x > -1).long() * (x < 1).long()).type(torch.bool)

    # npe_levels, npe_coverages = calc(ref_dist, cmp_dist, n=n)
    # npe_levels2, npe_coverages2 = calc(ref_dist, cmp_dist, use_ratio=True, n=n)
    npe_levels, npe_coverages = calc2(partial(Normal, scale=p2), partial(getNormalMixture, p2=p2), Uniform(-2 + p1, 2 + p1), n=n)
    npe_levels2, npe_coverages2 = calc2(partial(Normal, scale=p2), partial(getNormalMixture, p2=p2), Uniform(-2 + p1, 2 + p1), use_ratio=True, n=n)

    figure['data'][0]['y'] = ref
    figure['data'][1]['y'] = compare

    figure['data'][2]['x'] = x[idx]
    figure['data'][2]['y'] = ref[idx]
    figure['data'][3]['x'] = x[idx]
    figure['data'][3]['y'] = compare[idx]

    figure['data'][6]['x'] = npe_levels
    figure['data'][6]['y'] = npe_coverages
    figure['data'][7]['x'] = npe_levels2
    figure['data'][7]['y'] = npe_coverages2

    figure['data'][8]['x'] = x
    figure['data'][8]['y'] = torch.linspace(-2, 2, 100)
    figure['data'][8]['z'] = torch.stack([torch.exp(Normal(t, p2).log_prob(x)) for t in torch.linspace(-2 + p1, 2 + p1, 100)])

    figure['data'][9]['x'] = x
    figure['data'][9]['y'] = torch.linspace(-2, 2, 100)
    figure['data'][9]['z'] = torch.stack([torch.exp(getNormalMixture(t, p2).log_prob(x)) for t in torch.linspace(-2 + p1, 2 + p1, 100)])

    figure['data'][10]['x'] = x
    figure['data'][10]['y'] = np.zeros_like(x)

    return (
        'You have selected: ...'.format(),
        'mean:',
        'scale:',
        figure
    )


app.run(debug=True)
