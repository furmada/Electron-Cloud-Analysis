from eca01 import *
import numpy as np
import scipy
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, Normalize

plt.rcParams['savefig.format'] = 'svg'
plt.rcParams['figure.autolayout'] = False
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10

def _axes(size: tuple[int, int] | None | Axes) -> Axes:
    if type(size) == tuple:
        plt.figure(figsize=size)
    if not isinstance(size, Axes):
        return plt.gca()
    return size

def model_plot(model: ECModel, fits: Iterable[Fit], size: tuple[int, int] | None | Axes = (10, 5), log: tuple[bool, bool] = (False, False), show_error: bool = True, refit: bool = False):
    """
    Plot the raw data and each Fit if it succeeded.
    """
    ax = _axes(size)
    if len(list(fits)) == 0:
        step = model.time_to_index(model.t_offs)
        ax.scatter((model.time[::step] - model.t_offs) / model.b_spac, model.N_electrons[::step], s=1, c="k", label="Data")
    else:
        for fit in fits:
            result = fit.fit(model, refit=refit)
            T, F = fit.select_data(model)
            B = (T - model.t_offs) / model.b_spac
            if not result is None:
                ax.plot(B, fit.fit_function(model)(T), label="Fit: {}".format(fit.name))
                if show_error:
                    ax.fill_between(
                        B,
                        fit.fit_function(model, result[0] - result[1])(T),
                        fit.fit_function(model, result[0] + result[1])(T),
                        alpha=0.2, label="Err: {}".format(fit.name))
            else:
                print("Fitting failed: {}".format(getattr(model, "_"+fit.name+"_fitting_error")))
            ax.scatter(B, F, s=1, label="Data: {}".format(fit.name))
    if log[0]: ax.set_xscale("log")
    if log[1]: ax.set_yscale("log")
    if not size is None:
        ax.set_xlabel("Bunch Passage")
        ax.set_ylabel("Electron Linear Density")
        ax.set_title("SEY={:.2f}, I={:.2E}, B={}".format(model.del_max, model.fact_beam, model.B_multip))
        ax.legend()

def versus_plot(db: SimDB, paramA: str, paramB: str, colorBy: str | None = None,
                size: tuple[int, int] | None = (8, 6), dot_size: int = 10, log: tuple[bool, bool] = (False, False), colormap: None | Colormap = None, **restrict):
    """
    Produce a plot of param B vs. paramA, optionally colored using param colorBy, under restrict conditions.
    """
    ax = _axes(size)
    fig = plt.gcf()
    table = db.versus(
        [paramA] + ([] if colorBy is None else [paramA]),
        [paramB] + ([] if colorBy is None else [colorBy]),
        **restrict
    )
    ax.scatter(
        table[0][0],
        table[0][1],
        c=None if colorBy is None else table[1][1],
        s=dot_size,
        cmap=plt.cm.viridis if colormap is None else colormap
    )
    if not colorBy is None:
        fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(vmin=min(table[1][1]), vmax=max(table[1][1])), cmap=plt.cm.viridis if colormap is None else colormap), ax=ax)
    if log[0]: ax.set_xscale("log")
    if log[1]: ax.set_yscale("log")
    if not size is None:
        ax.set_xlabel(paramA)
        ax.set_ylabel(paramB)
        ax.set_title("{}({}){}".format(paramB, paramA, "by " + colorBy if not colorBy is None else ""))

