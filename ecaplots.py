from eca import *
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, Normalize
import matplotlib.gridspec as gridspec
from scipy import signal as sp_signal
import numpy as np
from typing import Iterable

# Global Matplotlib Configuration
plt.rcParams['savefig.format'] = 'svg'
plt.rcParams['figure.autolayout'] = False
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10

def _axes(size: tuple[int, int] | None | Axes) -> Axes:
    """Helper to handle polymorphic axes and figure sizing inputs."""
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
    cmap_to_use = plt.cm.viridis if colormap is None else colormap
    
    if colorBy is None:
        table = db.versus(paramA, paramB, **restrict)
        ax.scatter(
            table[0],
            table[1],
            s=dot_size,
            cmap=cmap_to_use
        )
    else:
        table = db.versus(
            [paramA, paramA],
            [paramB, colorBy],
            **restrict
        )
        
        color_data = np.array(table[1][1])
        vmin = np.nanmin(color_data) if len(color_data) > 0 else 0
        vmax = np.nanmax(color_data) if len(color_data) > 0 else 1
        
        ax.scatter(
            table[0][0],
            table[0][1],
            c=color_data,
            s=dot_size,
            cmap=cmap_to_use
        )
        fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap_to_use), ax=ax)
    
    if log[0]: ax.set_xscale("log")
    if log[1]: ax.set_yscale("log")
    if size is not None:
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
    results = db.where(**restrict)
    values = [r[param] for r in results if param in r]
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

def instability_grid_plot(db: SimDB, size: tuple[int, int] | None = (14, 16), 
                          save_path: str | None = None, max_plots: int = 5,
                          colormap: None | Colormap = None,
                          **restrict):
    """
    Replicates the 8-panel grid with a fixed colorbar layout.
    """
    results = db.where(**restrict)
    if len(results) == 0:
        print("No simulations found matching the restrict query.")
        return None

    display_count = min(max_plots, len(results))
    models = [InstabilityModel(db.db, r) for r in results[:display_count]]
    
    if colormap is None:
        colormap = plt.cm.viridis
    
    has_slice_data = any(len(m.slice_mean_x) > 0 for m in models)
    
    n_rows = 3 if has_slice_data else 2
    fig = plt.figure(figsize=size)
    gs = gridspec.GridSpec(n_rows, 3, figure=fig, width_ratios=[1, 1, 0.15]) 
    
    # --- Create Axes ---
    ax_cent_x = fig.add_subplot(gs[0, 0])
    ax_cent_y = fig.add_subplot(gs[0, 1], sharex=ax_cent_x, sharey=ax_cent_x)
    
    if has_slice_data:
        ax_rms_x = fig.add_subplot(gs[1, 0], sharex=ax_cent_x)
        ax_rms_y = fig.add_subplot(gs[1, 1], sharex=ax_cent_x)
        row_emittance = 2
        axes = {
            'cent_x': ax_cent_x, 'cent_y': ax_cent_y,
            'rms_x': ax_rms_x, 'rms_y': ax_rms_y,
        }
    else:
        ax_rms_x = None
        ax_rms_y = None
        row_emittance = 1
        axes = {
            'cent_x': ax_cent_x, 'cent_y': ax_cent_y,
        }
    
    ax_eps_x = fig.add_subplot(gs[row_emittance, 0], sharex=ax_cent_x)
    ax_eps_y = fig.add_subplot(gs[row_emittance, 1], sharex=ax_cent_x)
    axes['eps_x'] = ax_eps_x
    axes['eps_y'] = ax_eps_y
    
    # --- Plot Data ---
    colors = colormap(np.linspace(0, 1, display_count))
    
    for i, model in enumerate(models):
        c = colors[i]
        n_turns = len(model.mean_x) if len(model.mean_x) > 0 else 0
        if n_turns == 0:
            continue
        
        turns = np.arange(n_turns)
        
        # 1. Centroids
        if len(model.mean_x) > 0 and len(model.sigma_x) > 0:
            sigma_x_init = model.sigma_x[0] if model.sigma_x[0] != 0 and np.isfinite(model.sigma_x[0]) else 1e-9
            mask_x = np.isfinite(model.mean_x) & np.isfinite(model.sigma_x)
            if np.any(mask_x):
                centroid_x = model.mean_x - np.mean(model.mean_x[mask_x])
                norm_x = centroid_x[mask_x] / sigma_x_init
                ax_cent_x.plot(turns[mask_x], norm_x, color=c, label=f'Sim {i}' if i == 0 else "")
        
        if len(model.mean_y) > 0 and len(model.sigma_y) > 0:
            sigma_y_init = model.sigma_y[0] if model.sigma_y[0] != 0 and np.isfinite(model.sigma_y[0]) else 1e-9
            mask_y = np.isfinite(model.mean_y) & np.isfinite(model.sigma_y)
            if np.any(mask_y):
                centroid_y = model.mean_y - np.mean(model.mean_y[mask_y])
                norm_y = centroid_y[mask_y] / sigma_y_init
                ax_cent_y.plot(turns[mask_y], norm_y, color=c, 
                              label=f'Sim {i}: σ_y={model.sigma_y[0]:.2e}' if i == 0 else "")
        
        # 2. Charge-Weighted RMS
        if has_slice_data and ax_rms_x is not None:
            if len(model.charge_weighted_rms_x) > 0:
                rms_x = model.charge_weighted_rms_x
                if len(rms_x) > 20:
                    rms_x = sp_signal.savgol_filter(sp_signal.convolve(rms_x, sp_signal.windows.boxcar(20), mode='same') / 20, 21, 3)
                mask_rms_x = np.isfinite(rms_x)
                if np.any(mask_rms_x):
                    ax_rms_x.plot(turns[mask_rms_x], rms_x[mask_rms_x], color=c)
            
            if len(model.slice_mean_y) > 0:
                mean_y_slice = model.slice_mean_y
                weights = model.slice_n_macroparticles
                if mean_y_slice.shape == weights.shape:
                    weighted_mean = np.sum(mean_y_slice * weights, axis=0) / (np.sum(weights, axis=0) + 1e-12)
                    rms_y = np.sqrt(np.sum((mean_y_slice - weighted_mean)**2 * weights, axis=0) / (np.sum(weights, axis=0) + 1e-12))
                    if len(rms_y) > 20:
                        rms_y = sp_signal.savgol_filter(sp_signal.convolve(rms_y, sp_signal.windows.boxcar(20), mode='same') / 20, 21, 3)
                    mask_rms_y = np.isfinite(rms_y)
                    if np.any(mask_rms_y):
                        ax_rms_y.plot(turns[mask_rms_y], rms_y[mask_rms_y], color=c)
        
        # 3. Normalized Emittance
        if len(model.epsn_x) > 0 and model.epsn_x[0] != 0:
            mask_eps_x = np.isfinite(model.epsn_x) & (model.epsn_x > 0)
            if np.any(mask_eps_x):
                eps_x_norm = model.epsn_x[mask_eps_x] / model.epsn_x[0]
                ax_eps_x.plot(turns[mask_eps_x], eps_x_norm, color=c)
        
        if len(model.epsn_y) > 0 and model.epsn_y[0] != 0:
            mask_eps_y = np.isfinite(model.epsn_y) & (model.epsn_y > 0)
            if np.any(mask_eps_y):
                eps_y_norm = model.epsn_y[mask_eps_y] / model.epsn_y[0]
                ax_eps_y.plot(turns[mask_eps_y], eps_y_norm, color=c)

    # --- Formatting ---
    labels = {
        'cent_x': r'Horizontal Centroid / $\sigma_x$',
        'cent_y': r'Vertical Centroid / $\sigma_y$',
        'rms_x': 'Charge-weighted RMS X [a.u.]',
        'rms_y': 'Charge-weighted RMS Y [a.u.]',
        'eps_x': r'Norm. Emittance X / $\epsilon_{x,0}$',
        'eps_y': r'Norm. Emittance Y / $\epsilon_{y,0}$',
    }
    
    for key, ax in axes.items():
        if ax is None: continue
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')
        if key in labels:
            ax.set_ylabel(labels[key], fontsize=11)
        ax.set_xlabel('Turns', fontsize=11)

    sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=0, vmax=display_count-1))
    sm.set_array([])
    
    cbar_ax = fig.add_subplot(gs[:, 2]) 
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Simulation Index', fontsize=11)
    
    title_parts = []
    if len(results) > 0:
        r = results[0]
        if 'init_unif_edens_dip' in r:
            title_parts.append(f"Density: {r['init_unif_edens_dip']:.2e}")
        if 'element' in r:
            title_parts.append(f"Element: {r['element']}")
    
    if title_parts:
        fig.suptitle("Instability Evolution: " + " | ".join(title_parts), fontsize=14, y=0.995)
        
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def growth_rate_vs_density_plot(db: SimDB, density_key: str = 'init_unif_edens_dip', 
                                growth_key: str = 'growth_rate_mode', 
                                size: tuple[int, int] | None = (8, 6), 
                                save_path: str | None = None, 
                                colormap: None | Colormap = None,
                                **restrict):
    """
    Plots Growth Rate vs. Electron Density.
    """
    ax = _axes(size)
    results = db.where(**restrict)
    densities = []
    rates = []
    
    for r in results:
        if density_key in r and growth_key in r:
            d = r[density_key]
            g = r[growth_key]
            if isinstance(d, (int, float, np.number)) and isinstance(g, (int, float, np.number)):
                if np.isfinite(g) and np.isfinite(d) and d > 0 and g > 0:
                    densities.append(d)
                    rates.append(g)
    
    densities = np.array(densities)
    rates = np.array(rates)
    
    if len(densities) > 0:
        sort_idx = np.argsort(densities)
        densities = densities[sort_idx]
        rates = rates[sort_idx]
        
        if colormap is None:
            colormap = plt.cm.viridis
        
        ax.semilogy(densities, rates, 'o-', linewidth=2.5, markersize=8, 
                   color=colormap(0.7), label='Growth Rate')
        ax.set_xlabel(r'Electron Density [$e^-/m^3$]', fontsize=12)
        ax.set_ylabel(r'Growth Rate [turn$^{-1}$]', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_xlim(left=min(densities)*0.9)
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No valid data found", ha='center', va='center', 
               transform=ax.transAxes)
        ax.set_xlabel(density_key)
        ax.set_ylabel(growth_key)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=220)
    
    return ax


def blowup_time_vs_strength_plot(db: SimDB, strength_key: str = 'strength_factor', 
                                 turns_key: str = 'blowup_turn_first', 
                                 size: tuple[int, int] | None = (12, 10), 
                                 save_path: str | None = None,
                                 colormap: None | Colormap = None,
                                 **restrict):
    """
    Plots Blow-up Time vs. Strength Factor.
    """
    ax = _axes(size)
    results = db.where(**restrict)
    strengths = []
    turns = []
    
    for r in results:
        if strength_key in r and turns_key in r:
            s = r[strength_key]
            t = r[turns_key]
            if isinstance(s, (int, float, np.number)) and isinstance(t, (int, float, np.number)):
                if np.isfinite(t) and np.isfinite(s) and s > 0:
                    strengths.append(s)
                    turns.append(t)
    
    strengths = np.array(strengths)
    turns = np.array(turns)
    
    if len(strengths) > 0:
        sort_idx = np.argsort(strengths)
        strengths = strengths[sort_idx]
        turns = turns[sort_idx]
        
        if colormap is None:
            colormap = plt.cm.viridis
        
        ax.semilogy(strengths, turns, 'o-', linewidth=3.5, markersize=10, 
                   color=colormap(0.7), label='Blowup Time')
        
        max_turns = np.nanmax(turns) if len(turns) > 0 else 20000
        limit = 20000
        if limit < max_turns:
            ax.axhline(y=limit, color='darkgreen', linestyle='--', linewidth=2.5, label='Target Limit')
            ax.legend(fontsize=11)
        
        ax.set_xlabel('Strength Factor', fontsize=12)
        ax.set_ylabel('Blow-up Time [Turns]', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_xlim(left=min(strengths)*0.9)
    else:
        ax.text(0.5, 0.5, "No valid data found", ha='center', va='center', 
               transform=ax.transAxes)
        ax.set_xlabel(strength_key)
        ax.set_ylabel(turns_key)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    
    return ax


def intrabunch_mode_heatmap(db: SimDB, model_idx: int = 0, 
                            size: tuple[int, int] | None = (10, 6),
                            save_path: str | None = None,
                            colormap: None | Colormap = None,
                            **restrict):
    """
    Plots the intra-bunch mode structure (Slice vs Turn) heatmap.
    """
    results = db.where(**restrict)
    if len(results) == 0:
        print("No simulations found.")
        return None
    
    if model_idx >= len(results):
        model_idx = len(results) - 1
    
    model = InstabilityModel(db.db, results[model_idx])
    
    if len(model.slice_mean_x) == 0:
        print("No slice data available for this simulation.")
        return None
    
    if colormap is None:
        colormap = plt.cm.RdBu_r
    
    fig, ax = plt.subplots(figsize=size)
    
    if model.slice_mean_x.ndim != 2:
        print("Slice data has unexpected shape.")
        return None
    
    signal = model.slice_mean_x
    
    if len(model.slice_n_macroparticles) > 0 and model.slice_n_macroparticles.shape == signal.shape:
        weights = model.slice_n_macroparticles
        signal = signal * weights / (np.mean(weights) + 1e-12)
    
    signal_max = np.max(np.abs(signal))
    signal_norm = signal / signal_max if signal_max > 0 else signal
    
    n_slices, n_turns = signal.shape
    z_axis = np.linspace(-1, 1, n_slices)  
    t_axis = np.arange(n_turns)
    
    im = ax.pcolormesh(t_axis, z_axis, signal_norm, shading='auto', cmap=colormap)
    
    ax.set_xlabel('Turns', fontsize=12)
    ax.set_ylabel('Longitudinal Position (Normalized)', fontsize=12)
    ax.set_title(f'Intra-bunch Mode Structure (Sim {model_idx})', fontsize=13)
    
    plt.colorbar(im, ax=ax, label='Normalized Signal')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    
    return fig

def instability_mode_evolution_plot(db: SimDB, model_idx: int = 0, 
                                    size: tuple[int, int] | None = (12, 8),
                                    save_path: str | None = None,
                                    colormap: None | Colormap = None,
                                    **restrict):
    """
    Plots the evolution of intra-bunch modes over time (Spectrogram).
    Highlights the dominant mode and its growth trajectory.
    """
    results = db.where(**restrict)
    if len(results) == 0:
        print("No simulations found matching the query.")
        return None
    
    if model_idx >= len(results):
        model_idx = len(results) - 1
    
    model = InstabilityModel(db.db, results[model_idx])
    
    power = model.slice_fft_power
    if len(power) == 0:
        print("No slice data or FFT calculation failed.")
        return None
    
    dominant_idx = model.dominant_mode_idx
    growth_rate = getattr(model, 'dominant_mode_growth_rate', np.nan)
    
    if colormap is None:
        colormap = plt.cm.viridis
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=size, gridspec_kw={'height_ratios': [3, 1]})
    
    turns = np.arange(power.shape[1])
    freqs = np.arange(power.shape[0]) / 64.0 
    
    power_norm = np.log10(power + 1e-12)
    
    im = ax1.pcolormesh(turns, freqs, power_norm.T[:-1, :-1], shading='auto', cmap=colormap)
    ax1.set_ylabel('Fractional Tune (Mode Index / 64)', fontsize=12)
    ax1.set_xlabel('Turns', fontsize=12)
    ax1.set_title(f'Mode Evolution (Sim {model_idx}) - Growth Rate: {growth_rate:.4f} ' r'turn$^{-1}$', fontsize=13)
    
    if 0 < dominant_idx < len(freqs):
        ax1.axhline(y=freqs[dominant_idx], color='white', linestyle='--', 
                    linewidth=2, label=f'Dominant Mode (idx={dominant_idx})')
        ax1.legend(loc='upper right')
    
    plt.colorbar(im, ax=ax1, label='Log Power (a.u.)')
    ax1.grid(True, linestyle='--', alpha=0.3, axis='x')
    
    if dominant_idx > 0:
        dom_power = power[dominant_idx, :]
        if len(dom_power) > 20:
            dom_power_smooth = sp_signal.savgol_filter(dom_power, 21, 3)
        else:
            dom_power_smooth = dom_power
            
        ax2.semilogy(turns, dom_power_smooth, 'r-', linewidth=2, label='Dominant Mode Power')
        ax2.set_xlabel('Turns', fontsize=12)
        ax2.set_ylabel('Mode Power (a.u.)', fontsize=12)
        ax2.set_title(f'Dominant Mode Growth (Idx={dominant_idx})', fontsize=12)
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, "No dominant mode detected", ha='center', va='center', transform=ax2.transAxes)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off')

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig