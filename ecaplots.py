from eca import *
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, Normalize
import numpy as np
from typing import Iterable

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

def model_plot(model: ECModel, fits: Iterable[Fit], size: tuple[int, int] | None | Axes = (10, 5),
                 log: tuple[bool, bool] = (False, False), show_error: bool = True, refit: bool = False,
                 central_density: bool | None = False, fit_maxX: float = -1.0, label: str | None = None,
                 train: int | list[int] | np.ndarray | None = None):
    """
    Plot the raw data and each Fit if it succeeded.
    """
    ax = _axes(size)
    fits_list = list(fits)

    # Extract the raw continuous data for the specified train(s) if needed
    if not (central_density is None):
        step = max(1, model.time_to_index(model.t_offs))
        idx_base = np.arange(0, len(model.time), step)
        if train is not None:
            if isinstance(train, (int, np.integer)):
                trains = [train]
            else:
                trains = train
            mask = np.zeros(len(model.time), dtype=bool)
            for t in trains:
                try:
                    mask[model.train_to_index(t)] = True
                except IndexError:
                    pass # Train index might be out of bounds, skip it gracefully
            plot_idx = idx_base[mask[idx_base]]
        else:
            plot_idx = idx_base
        T_raw = model.time[plot_idx]
        B_raw = (T_raw - model.t_offs) / model.b_spac
        if central_density:
            F_raw = model.central_density[plot_idx]
        else:
            F_raw = model.N_electrons[plot_idx]
    if len(fits_list) == 0:
        if not (central_density is None):
            if fit_maxX > 0 and len(B_raw) > 0:
                start_b = B_raw[0]
                valid = B_raw <= start_b + fit_maxX
                B_raw, F_raw = B_raw[valid], F_raw[valid]
            ax.scatter(B_raw, F_raw, s=1, label=("" if label is None else label + " ") + "Data")
    else:
        for fit in fits_list:
            result = fit.fit(model, refit=refit)
            if central_density is None:
                T = np.arange(model.t_offs, fit_maxX * model.b_spac if fit_maxX > 0 else model.time[-1])
                F = np.zeros_like(T)
            else:
                T, F = fit.select_data(model)
            
            if fit_maxX > 0 and len(T) > 0:
                valid = T <= T[0] + fit_maxX * model.b_spac
                T, F = T[valid], F[valid]
                
            if len(T) > 0:
                start_t = T[0]
                end_t = max(T[-1], start_t + fit_maxX * model.b_spac if fit_maxX > 0 else T[-1])
            else:
                start_t = 0
                end_t = fit_maxX * model.b_spac if fit_maxX > 0 else model.time[-1]

            Tsmooth = np.linspace(start_t, end_t, max(100, int((end_t - start_t) / model.b_spac) + 1))
            #B = (T - model.t_offs) / model.b_spac
            B = ((T + model.train_times[fit._mget("train", model, -1)]) - model.t_offs) / model.b_spac
            Bsmooth = (Tsmooth + model.train_times[fit._mget("train", model, -1)] - model.t_offs) / model.b_spac
            if not (result is None):
                ax.plot(Bsmooth, fit.fit_function(model)(Tsmooth), label=("" if label is None else label + " ") + fit.name + " Fit")
                if show_error:
                    ax.fill_between(
                        Bsmooth,
                        fit.fit_function(model, result[0] - result[1])(Tsmooth),
                        fit.fit_function(model, result[0] + result[1])(Tsmooth),
                        alpha=0.2, label=("" if label is None else label + " ") + fit.name + " Fit Error")
            else:
                err_msg = getattr(model, "_"+fit.name+"_fitting_error", "Unknown error")
                print("Fitting failed: {}".format(err_msg))
                
            if not (central_density is None):
                ax.scatter(B, F, s=20, marker="x", label="Fit Points: {}".format(fit.name))
                
        # Overlay the continuous raw data for the specified train
        if not (central_density is None):
            if fit_maxX > 0 and len(B_raw) > 0:
                start_b = B_raw[0]
                valid = B_raw <= start_b + fit_maxX
                B_raw, F_raw = B_raw[valid], F_raw[valid]
            ax.scatter(B_raw, F_raw, s=1, alpha=0.5, label=("" if label is None else label + " ") + "Raw Data")

    if log[0]: ax.set_xscale("log")
    if log[1]: ax.set_yscale("log")
    if not size is None:
        ax.set_xlabel("Bunch Passage")
        ax.set_ylabel("N. Electrons in Centre / m" if central_density == True else "N. Electrons / m")
        ax.set_title("SEY={:.2f}, I={:.2E}, B={}".format(model.del_max, model.fact_beam, model.B_multip))
        ax.legend()

def versus_plot(db: SimDB, paramA: str, paramB: str, colorBy: str | None = None, size: tuple[int, int] | None = (8, 6), 
                dot_size: int = 10, log: tuple[bool, bool] = (False, False), colormap: None | Colormap = None, **restrict) -> tuple:
    """
    Produce a plot of param B vs. paramA, optionally colored using param colorBy, under restrict conditions.
    """
    ax = _axes(size)
    fig = plt.gcf()
    
    if colorBy is None:
        # When no coloring variable is specified, just get X and Y data
        table = db.versus(paramA, paramB, **restrict)
        ax.scatter(
            table[0],
            table[1],
            s=dot_size,
            cmap=plt.cm.viridis if colormap is None else colormap
        )
    else:
        # When coloring by a variable, get X, Y, and color data
        table = db.versus(
            [paramA, paramA],
            [paramB, colorBy],
            **restrict
        )
        ax.scatter(
            table[0][0],
            table[0][1],
            c=table[1][1],
            s=dot_size,
            cmap=plt.cm.viridis if colormap is None else colormap
        )
        fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(vmin=min(table[1][1]), vmax=max(table[1][1])), cmap=plt.cm.viridis if colormap is None else colormap), ax=ax)
    
    if log[0]: ax.set_xscale("log")
    if log[1]: ax.set_yscale("log")
    if not size is None:
        ax.set_xlabel(paramA)
        ax.set_ylabel(paramB)
        ax.set_title("{}({}){}".format(paramB, paramA, " by " + colorBy if colorBy is not None else ""))

    return table if colorBy is None else (table[0][0], table[0][1], table[1][1])

def histogram_plot(db: SimDB, param: str, bins: int | str = 'auto', 
                   size: tuple[int, int] | None | Axes = (8, 6), 
                   log_y: bool = False, color: str = 'C0', **restrict):
    """
    Produce a histogram of a specified parameter across the database, under restrict conditions.
    """
    ax = _axes(size)
    # Extract the values for the requested parameter
    results = db.where(**restrict)
    values = [r[param] for r in results if param in r]
    # Filter out non-numeric values or NaNs to prevent plotting errors
    valid_values = [v for v in values if isinstance(v, (int, float, np.number)) and not (np.isnan(v) or np.isinf(v))]
    if not valid_values:
        ax.text(0.5, 0.5, f"No valid numeric data for {param}", ha='center', va='center')
        return None
    n, bins_out, patches = ax.hist(valid_values, bins=bins, color=color, edgecolor='black', alpha=0.7)
    if log_y:
        ax.set_yscale("log")
    if not isinstance(size, Axes):
        ax.set_xlabel(param)
        ax.set_ylabel("Count")
        ax.set_title(f"Histogram of {param}")
    return n, bins_out, patches