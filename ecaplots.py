from eca import *
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

def model_plot(model: ECModel, fits: Iterable[Fit], size: tuple[int, int] | None | Axes = (10, 5),
                 log: tuple[bool, bool] = (False, False), show_error: bool = True, refit: bool = False,
                 central_density: bool | None = False, fit_maxX: float = -1.0, label: str | None = None):
    """
    Plot the raw data and each Fit if it succeeded.
    """
    ax = _axes(size)
    if len(list(fits)) == 0:
        step = model.time_to_index(model.t_offs)
        if not central_density is None:
            if central_density:
                B = (model.time[::step] - model.t_offs) / model.b_spac
                F = model.central_density[::step]
            else:
                B = (model.time[::step] - model.t_offs) / model.b_spac
                F = model.N_electrons[::step]
            if fit_maxX > 0:
                B, F = B[B <= fit_maxX], F[B <= fit_maxX]
            ax.scatter(B, F, s=1, label=("" if label is None else label + " ") + "Data")
    else:
        for fit in fits:
            result = fit.fit(model, refit=refit)
            if central_density is None:
                T = np.arange(model.t_offs, fit_maxX * model.b_spac)
                F = np.zeros_like(T)
            else:
                T, F = fit.select_data(model)
            if fit_maxX > 0:
                T, F = T[T <= fit_maxX * model.b_spac], F[T <= fit_maxX * model.b_spac]
            Tsmooth = np.linspace(0, max(fit_maxX * model.b_spac, T[-1]), int(max(fit_maxX, T[-1] / model.b_spac))+1)          
            B = (T - model.t_offs) / model.b_spac
            Bsmooth = (Tsmooth - model.t_offs) / model.b_spac
            if not (result is None):
                ax.plot(Bsmooth, fit.fit_function(model)(Tsmooth), label=("" if label is None else label + " ") + fit.name + " Fit")
                if show_error:
                    ax.fill_between(
                        Bsmooth,
                        fit.fit_function(model, result[0] - result[1])(Tsmooth),
                        fit.fit_function(model, result[0] + result[1])(Tsmooth),
                        alpha=0.2, label=("" if label is None else label + " ") + fit.name + " Fit Error")
            else:
                print("Fitting failed: {}".format(getattr(model, "_"+fit.name+"_fitting_error")))
            if not (central_density is None):
                ax.scatter(B, F, s=1, label="Data: {}".format(fit.name))
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