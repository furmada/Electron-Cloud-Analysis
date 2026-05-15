from eca import *
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, Normalize
import matplotlib.gridspec as gridspec
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
                 train: int | str | None = "All"):
    """
    Plot the raw data and each Fit if it succeeded.
    Confines fit functions to the trains they were evaluated on.
    """
    ax = _axes(size)
    fits_list = list(fits)

    is_all_trains = train is None or str(train).lower() == "all"
    plot_train_idx = None if is_all_trains else int(train)

    # --- 1. Plot Raw Data ---
    if central_density is not None:
        step = max(1, model.time_to_index(model.t_offs))
        
        if is_all_trains:
            plot_idx = np.arange(0, model.cutoff, step)
        else:
            try:
                plot_idx = model.train_to_index(plot_train_idx)[::step]
            except IndexError:
                plot_idx = np.arange(0, model.cutoff, step) # Fallback if out of bounds

        T_raw = model.time[plot_idx]
        B_raw = (T_raw - model.t_offs) / model.b_spac
        F_raw = model.central_density[plot_idx] if central_density else model.N_electrons[plot_idx]

        # Constrain X-axis view if focused on a specific train
        if fit_maxX > 0 and len(B_raw) > 0 and not is_all_trains:
            valid = B_raw <= (B_raw[0] + fit_maxX)
            B_raw, F_raw = B_raw[valid], F_raw[valid]

        alpha = 0.5 if fits_list else 1.0
        ax.scatter(B_raw, F_raw, s=1, alpha=alpha, label=f"{label + ' ' if label else ''}Raw Data")

    # --- 2. Plot Fits ---
    for fit in fits_list:
        result = fit.fit(model, refit=refit)
        fit_train = fit._mget("train", model, -1)
        
        T_fit_pts, F_fit_pts = fit.select_data(model)
        if len(T_fit_pts) == 0:
            continue

        # Evaluate fit function over the exact span of the selected data points
        T_max = T_fit_pts[-1]
        if fit_maxX > 0:
            T_max = min(T_max, fit_maxX * model.b_spac)

        T_smooth = np.linspace(0, T_max, max(100, int(T_max / model.b_spac) + 1))
        
        try:
            train_start_time = model.train_times[fit_train]
        except IndexError:
            train_start_time = model.train_times[-1]

        # Map relative time back to absolute bunch passage for plotting
        B_smooth = (T_smooth + train_start_time - model.t_offs) / model.b_spac
        B_fit_pts = (T_fit_pts + train_start_time - model.t_offs) / model.b_spac

        # Render fit curve only if we plot all, or if its train matches the requested view
        num_trains = len(model.train_times)
        actual_fit_train = fit_train if fit_train >= 0 else num_trains + fit_train
        
        if is_all_trains or plot_train_idx == actual_fit_train:
            if result is not None:
                fit_fn = fit.fit_function(model)
                ax.plot(B_smooth, fit_fn(T_smooth), label=f"{label + ' ' if label else ''}{fit.name} Fit")
                
                if show_error:
                    ax.fill_between(
                        B_smooth,
                        fit.fit_function(model, result[0] - result[1])(T_smooth),
                        fit.fit_function(model, result[0] + result[1])(T_smooth),
                        alpha=0.2
                    )
            else:
                print(f"Fitting failed: {getattr(model, '_' + fit.name + '_fitting_error', 'Unknown error')}")

            if central_density is not None:
                ax.scatter(B_fit_pts, F_fit_pts, s=20, marker="x", label=f"Fit Points: {fit.name}")

    if log[0]: ax.set_xscale("log")
    if log[1]: ax.set_yscale("log")
    ax.set_xlabel("Bunch Passage")
    ax.set_ylabel("N. Electrons in Centre / m" if central_density else "N. Electrons / m")
    ax.set_title(f"SEY={model.del_max:.2f}, I={model.fact_beam:.2E}, B={model.B_multip}")
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

def instability_grid_plot(db: SimDB, size: tuple[int, int] | None = (14, 16), save_path: str | None = None, **restrict):
    """
    Replicates the 8-panel grid from 001c_full_sims_bunch_evolution_rms.py.
    Accepts a SimDB (InstabilityDB) and a restrict query.
    Plots the first 5 matching simulations by default to avoid clutter, 
    or all if fewer than 5.
    """
    # Fetch models matching the query
    results = db.where(**restrict)
    if len(results) == 0:
        print("No simulations found matching the restrict query.")
        return None
    # Limit to first 5 for clarity, unless fewer exist
    display_count = min(5, len(results))
    models = [InstabilityModel(db.db, r) for r in results[:display_count]]
    fig = plt.figure(figsize=size)
    gs = gridspec.GridSpec(4, 2)
    # Define axes
    ax_cent_x = fig.add_subplot(gs[0])
    ax_cent_y = fig.add_subplot(gs[1], sharex=ax_cent_x)
    ax_eps_x = fig.add_subplot(gs[2], sharex=ax_cent_x)
    ax_eps_y = fig.add_subplot(gs[3], sharex=ax_cent_x)
    ax_mp = fig.add_subplot(gs[4], sharex=ax_cent_x)
    ax_sz = fig.add_subplot(gs[5], sharex=ax_cent_x)
    ax_dummy1 = fig.add_subplot(gs[6], sharex=ax_cent_x)
    ax_dummy2 = fig.add_subplot(gs[7], sharex=ax_cent_x)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(models)))
    
    for i, model in enumerate(models):
        c = colors[i]
        turns = np.arange(model.n_turns)
        # 1. Centroids (Normalized)
        if len(model.mean_x) > 0:
            sigma_x = np.std(model.mean_x) if np.std(model.mean_x) > 0 else 1e-9
            norm_x = (model.mean_x - np.mean(model.mean_x)) / sigma_x
            ax_cent_x.plot(turns, norm_x, color=c, label=f'Sim {i}')
        if len(model.mean_y) > 0:
            sigma_y = np.std(model.mean_y) if np.std(model.mean_y) > 0 else 1e-9
            norm_y = (model.mean_y - np.mean(model.mean_y)) / sigma_y
            ax_cent_y.plot(turns, norm_y, color=c)
        # 2. Emittance (Normalized)
        if len(model.epsn_x) > 0 and model.epsn_x[0] != 0:
            ax_eps_x.plot(turns, model.epsn_x / model.epsn_x[0], color=c)
        if len(model.epsn_y) > 0 and model.epsn_y[0] != 0:
            ax_eps_y.plot(turns, model.epsn_y / model.epsn_y[0], color=c)
        # 3. MP Count & Bunch Length
        if len(model.macroparticlenumber) > 0:
            ax_mp.plot(turns, model.macroparticlenumber, color=c)
        if len(model.sigma_z) > 0:
            # Convert to ns: sigma_z (m) / c * 4 * 1e9
            c_light = 299792458.0
            ax_sz.plot(turns, model.sigma_z * 4 / c_light * 1e9, color=c)

    # Labels
    ax_cent_x.set_ylabel('Norm. Centroid X')
    ax_cent_y.set_ylabel('Norm. Centroid Y')
    ax_eps_x.set_ylabel('Norm. Emittance X')
    ax_eps_y.set_ylabel('Norm. Emittance Y')
    ax_mp.set_ylabel('Macroparticles')
    ax_sz.set_ylabel('Bunch Length (4σ) [ns]')
    for ax in [ax_cent_x, ax_cent_y, ax_eps_x, ax_eps_y, ax_mp, ax_sz]:
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.ticklabel_format(style='sci', scilimits=(0,0), axis='y')
    ax_mp.set_xlabel('Turns')
    ax_sz.set_xlabel('Turns')
    ax_cent_y.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig

def growth_rate_vs_density_plot(db: SimDB, density_key: str = 'init_unif_edens_dip', 
                                growth_key: str = 'growth_rate_mode', 
                                size: tuple[int, int] | None = (8, 6), 
                                save_path: str | None = None, **restrict):
    """
    Plots Growth Rate vs. Electron Density.
    Accepts a SimDB and a restrict query.
    """
    ax = _axes(size)
    results = db.where(**restrict)
    densities = []
    rates = []
    for r in results:
        if density_key in r and growth_key in r:
            d = r[density_key]
            g = r[growth_key]
            if isinstance(d, (int, float, np.number)) and isinstance(g, (int, float, np.number)) and not np.isnan(g):
                densities.append(d)
                rates.append(g)
            
    densities = np.array(densities)
    rates = np.array(rates)
    if len(densities) > 0:
        ax.semilogy(densities, rates, 'o-', linewidth=2, markersize=8)
        ax.set_xlabel('Electron Density [$e^-/m^3$]')
        ax.set_ylabel('Growth Rate [turn$^{-1}$]')
        ax.grid(True, linestyle='dashed')
        ax.set_xlim(left=0)
    else:
        ax.text(0.5, 0.5, "No valid data found", ha='center', va='center')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=220)
    return ax

def blowup_time_vs_strength_plot(db: SimDB, strength_key: str = 'strength_factor', 
                                 turns_key: str = 'blowup_turn_first', 
                                 size: tuple[int, int] | None = (12, 10), 
                                 save_path: str | None = None, **restrict):
    """
    Plots Blow-up Time vs. Strength Factor.
    Accepts a SimDB and a restrict query.
    """
    ax = _axes(size)
    results = db.where(**restrict)
    strengths = []
    turns = []
    for r in results:
        if strength_key in r and turns_key in r:
            s = r[strength_key]
            t = r[turns_key]
            if isinstance(s, (int, float, np.number)) and isinstance(t, (int, float, np.number)) and not np.isnan(t):
                strengths.append(s)
                turns.append(t)    
    strengths = np.array(strengths)
    turns = np.array(turns)
    
    if len(strengths) > 0:
        ax.semilogy(strengths, turns, 'o-', linewidth=3.5, markersize=10)
        ax.axhline(y=20000, color='darkgreen', linestyle='--', linewidth=2, label='Limit')
        ax.set_xlabel('Strength Factor')
        ax.set_ylabel('Blow-up Time [Turns]')
        ax.grid(True, linestyle='dashed')
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No valid data found", ha='center', va='center')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    return ax