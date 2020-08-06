import matplotlib.pyplot as plt
import matplotlib.figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mamosa.autotrack.ProbabilisticTracking import MAPTracker
from mamosa.utils.plot_utils import set_int_axis_ticks
import numpy as np
from typing import Tuple


def plot_marginal_likelihoods(
    tracker: MAPTracker,
    iline: int,
    horizon_n: int = 0,
    relative=True,
    xticks=None,
    figsize: Tuple[float, float] = (6.4, 4.8),
) -> matplotlib.figure.Figure:
    """Plot marginal likelihoods for horizon depths.

    Args:
        tracker: MAPTracker instance.
        iline: Which iline to plot marginal likelihoods for.
        horizon_n: Horizon number.
        relative: Plot marginals likelihoods relatively to the largest one. I.e. the
            likelihoods are scaled to [0, 1].
        xticks: X-axis ticks.
        figsize: Figure size.

    Returns:
        Matplotlib figure object.

    """
    likelihoods = tracker.get_marginal_likelihoods(horizon_n, iline, relative=relative)
    fig = plt.figure(figsize=figsize)
    set_int_axis_ticks()
    if xticks is not None:
        plt.xticks(xticks)
    if relative is False:
        # TODO: Log colorscale instead
        likelihoods = np.log10(likelihoods)
        likelihoods[likelihoods == 0] = likelihoods.max()
    im = plt.imshow(likelihoods.T, cmap="Greys")
    plt.xlabel("$x$")
    plt.ylabel("Two-way travel time $s$")
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", "5%", pad="3%")
    plt.colorbar(im, cax=cax)
    return fig


def plot_posterior_marginals(
    tracker: MAPTracker,
    iline: int,
    horizon_n: int = 0,
    figsize: Tuple[float, float] = (6.4, 4.8),
):
    """Plot posterior marginals for horizon depths.

    Args:
        tracker: MAPTracker instance.
        iline: Which iline to plot marginal likelihoods for.
        horizon_n: Horizon number.
        figsize: Figure size.

    Returns:
        Matplotlib figure object.

    """
    marginals = tracker.get_marginal_posteriors(iline, horizon_n)
    fig = plt.figure(figsize=figsize)
    set_int_axis_ticks()
    im = plt.imshow(marginals.T, cmap="Greys")
    plt.xlabel("$x$")
    plt.ylabel("Two-way travel time $s$")
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", "5%", pad="3%")
    plt.colorbar(im, cax=cax)
    return fig
