"""
Synchrotron Radiation Flux analysis tools for use with PyECLOUD and Electron Cloud Analysis
@author Adam Furman
@email adam.furman@cern.ch
"""

import warnings
from typing import Iterable
import numpy as np
import scipy.io as sio
from scipy.optimize import minimize, curve_fit
from scipy.ndimage import gaussian_filter
from scipy.stats import binned_statistic_2d
from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Patch
from os.path import join

# Machine information tools

def load_absorbers(path: str) -> np.ndarray:
    """Load a file containing locations of synchrotron radiation absorbers."""
    return np.loadtxt(path, delimiter=",", skiprows=1).T[1:2]

def load_elements(path: str, sort_by: str = "start Z (m)") -> tuple[list[dict], dict]:
    """
    Load a CSV file containing information about the machine.
    """
    headers = []
    elements = []
    lists = {}
    with open(path, "r") as f:
        for i, raw in enumerate(f.readlines()):
            data = raw.rstrip("\n").split(",")
            if i == 0:
                headers = data
                lists = {h: [] for h in headers}
            else:
                element = {}
                for header, value in zip(headers, data):
                    v = None
                    if value != "":
                        try: v = float(value)
                        except: v = value
                    element[header] = v
                    lists[header].append(v)
                elements.append(element)
    order = np.argsort(np.array(lists[sort_by]))
    lists = {h: np.array(l)[order] for h, l in lists.items()}
    return list(sorted(elements, key=lambda e: e[sort_by])), lists

# Flux data parsing tools

def read_flux_data(flux_files: list[str] | str | tuple) -> np.ndarray:
    """
    Read (several) flux files, containing (X, Y, Z, F) data, and join them into one array.
    """
    x_coord, y_coord, z_coord, F_value = np.array([]), np.array([]), np.array([]), np.array([])
    for flux_file in ((flux_files,) if type(flux_files) == str else flux_files):
        flux_data = np.loadtxt(flux_file, skiprows=1).T
        x_coord = np.concatenate((x_coord, flux_data[0] / 100), axis=None) # Given in cm, we want m
        y_coord = np.concatenate((y_coord, flux_data[1] / 100), axis=None)
        z_coord = np.concatenate((z_coord, flux_data[2] / 100), axis=None)
        F_value = np.concatenate((F_value, flux_data[3]), axis=None)
    order = np.argsort(z_coord).ravel()
    return np.array([x_coord.ravel()[order], y_coord.ravel()[order], z_coord.ravel()[order], F_value.ravel()[order]])

def unbend_data(points, elements):
    """
    Unbends planar (XZ) tube data using local reference trajectory offsets.
    
    Parameters:
    - points: (3, N) array of (x, y, z)
    - bending_elements: List of tuples (z_ref, x_ref, theta_ref, s_ref)
    
    Returns:
    - unbent_points: (3, N) array with X-Z plane straightened
    """
    bending_elements = np.array([
        (e["start Z (m)"], e["start X (m)"], e["start Theta [rad]"], e["start s [m]"]) 
        for e in elements
    ])
    # Convert list of tuples to numpy arrays for vectorized interpolation
    z_ref_vals = bending_elements[:, 0]
    x_ref_vals = bending_elements[:, 1]
    theta_vals = bending_elements[:, 2]
    s_ref_vals = bending_elements[:, 3]
    
    # 1. Interpolate the reference trajectory for every point's global Z
    # Note: interp expects x-coordinates (z_ref_vals) to be increasing.
    x_ref_interp = np.interp(points[2, :], z_ref_vals, x_ref_vals)
    theta_interp = np.interp(points[2, :], z_ref_vals, theta_vals)
    s_ref_interp = np.interp(points[2, :], z_ref_vals, s_ref_vals)
    z_ref_interp = np.interp(points[2, :], z_ref_vals, z_ref_vals) # usually just points[2,:]
    
    # 2. Calculate local offsets from the reference trajectory (Translation)
    dx = points[0, :] - x_ref_interp
    dz = points[2, :] - z_ref_interp
    
    # 3. Rotate the local offset vector by -theta (Rotation)
    # This aligns the local cross-section with the new straight Z-axis
    c = np.cos(-theta_interp)
    s = np.sin(-theta_interp)
    
    # x_unbent is the transverse distance from the design orbit
    x_unbent = dx * c - dz * s
    # z_unbent is the path length (s) plus the longitudinal projection of the offset
    z_unbent = s_ref_interp + (dx * s + dz * c)
    y_unbent = points[1, :]
    
    return np.vstack((x_unbent, y_unbent, z_unbent))

def _max_inscribed_ellipse(X: np.ndarray, Y: np.ndarray) -> tuple[np.number, np.number]:
    """
    Compute the maximum-area ellipse inscribed in a nearly convex shape.
    The shape and ellipse are assumed centered at the origin.
    """
    points = np.column_stack((X, Y))
    # Initial guess
    r0 = np.min(np.linalg.norm(points, axis=1))
    r_max = np.max(np.linalg.norm(points, axis=1))
    x0 = np.array([0.9 * r0, 0.9 * r0])
    bounds = [(0.9*r0, r_max), (0.9*r0, r_max)]
    def objective(params):
        a, b = params
        v = 1.0 - (points[:, 0]**2 / a**2 + points[:, 1]**2 / b**2)
        inside_penalty = np.sum(np.maximum(v, 0.0)**2)
        return np.log((r_max-r0)/(a*b)) + (1e5*inside_penalty)**2
    result = minimize(
        objective,
        x0,
        method='SLSQP',
        bounds=bounds,
        options={'ftol': 1e-9, 'maxiter': 1000}
    )
    if not result.success:
        raise RuntimeError("Ellipse optimization failed: " + result.message)
    a, b = result.x * (1 - 100*np.sum(np.maximum(1.0 - (points[:, 0]**2 / result.x[0]**2 + points[:, 1]**2 / result.x[1]**2), 0.0)))
    return a, b

def make_matfiles(Z: float, zslice: np.ndarray, output_dir: str = "output", k_pe_sts: Iterable[float] = [1e-6, 1e-5, 1e-4, 1e-3]) -> tuple[list[str], list[str]]:
    """
    Take the "zslice" containing points (X, Y, Flux) - the output of slice_flux_data, and output the PyECLOUD .mat files.
    """
    x_b, y_b, f_b = zslice[0], zslice[1], zslice[2]
    # Compute the segment lengths
    seg_len = np.sqrt(np.diff(x_b, append=x_b[0])**2 + np.diff(y_b, append=y_b[0])**2)
    # Compute the length-weighted flux distribution
    cumdist = np.cumsum((f_b * seg_len) / (np.sum(f_b * seg_len)))
    # Fit ellipse
    x_sem_ellip_insc, y_sem_ellip_insc = _max_inscribed_ellipse(x_b, y_b)
    print("Chamber with {} vertices. Circumference {:.2f}, weighted flux sum={:.2f}. x_semi={:.4f}, y_semi={:.4f}".format(
        x_b.size, np.sum(seg_len), np.sum(cumdist), x_sem_ellip_insc, y_sem_ellip_insc))
    chm_files = []
    photo_files = []
    # Output the vertex and photoemission files
    for k_pe_st in k_pe_sts:
        output_chm = "chm_Z_{:.2f}_k_pe_st_{:.2E}.mat".format(Z, k_pe_st)
        output_pho = "pem_Z_{:.2f}_k_pe_st_{:.2E}.mat".format(Z, k_pe_st)
        vtx_dict = {
            "Vx": x_b,
            "Vy": y_b,
            "x_sem_ellip_insc": round(float(x_sem_ellip_insc), 4),
            "y_sem_ellip_insc": round(float(y_sem_ellip_insc), 4),
        }
        photo_dict = {
            "phem_cdf": cumdist,
            "k_pe_st": k_pe_st,
            **vtx_dict
        }
        chm_file = join(output_dir, output_chm)
        photo_file = join(output_dir, output_pho)
        sio.savemat(chm_file, vtx_dict, oned_as='row')
        sio.savemat(photo_file, photo_dict, oned_as='row')
        print("-> Z={:.2f} k_pe_st={:.2E}, wrote {}, {}".format(Z, k_pe_st, chm_file, photo_file))
        chm_files.append(chm_file)
        photo_files.append(photo_file)
    return (chm_files, photo_files)

def create_flux_map_adaptive(z: np.ndarray, theta: np.ndarray, flux: np.ndarray, bins: tuple[int, int] = (1000, 400), sigma: float = 1.0):
    """
    Create an interpolated flux map from unbent flux data.
    """
    # 1. Aggregate flux sum and point counts per bin
    flux_sum, z_edges, t_edges, _ = binned_statistic_2d(
        z, theta, flux, statistic='sum', bins=bins
    )
    counts, _, _, _ = binned_statistic_2d(
        z, theta, flux, statistic='count', bins=bins
    )

    if sigma > 0:
        # 2. Prepare padding for Theta periodicity (axis 1)
        pad_width = min(int(3 * sigma), 1)
        # Pad both the sum and the counts
        sum_padded = np.pad(flux_sum, ((0, 0), (pad_width, pad_width)), mode='wrap')
        counts_padded = np.pad(counts, ((0, 0), (pad_width, pad_width)), mode='wrap')
        
        # 3. Smooth both arrays independently
        # This smears the flux and the 'weights' (counts) across empty bins
        smoothed_sum = gaussian_filter(sum_padded, sigma=sigma)
        smoothed_counts = gaussian_filter(counts_padded, sigma=sigma)
        
        # 4. Divide smoothed sum by smoothed counts (Normalized Convolution)
        with np.errstate(divide='ignore', invalid='ignore'):
            grid_padded = smoothed_sum / smoothed_counts
            
        # 5. Clean up and crop
        grid_padded = np.nan_to_num(grid_padded, nan=0.0)
        final_grid = grid_padded[:, pad_width:-pad_width]
        
        return final_grid, z_edges, t_edges
        
    # Fallback for sigma=0
    with np.errstate(divide='ignore', invalid='ignore'):
        grid = np.nan_to_num(flux_sum / counts)
    return grid, z_edges, t_edges

def plot_flux_with_lattice(ze, te, render, absorbers, elements):
    """
    Plot a flux distribution together with lattice elements.
    """
    # Set up a figure with two rows: flux map (top) and lattice bar (bottom)
    fig = plt.figure(figsize=(20, 6))
    gs = gridspec.GridSpec(2, 2, height_ratios=[8, 1], width_ratios=[40, 1], wspace=0.01, hspace=0.01)
    gs.update(hspace=0.05) # Join the plots closely
    
    ax_flux = plt.subplot(gs[0, 0])
    ax_lattice = plt.subplot(gs[1, 0], sharex=ax_flux)
    ax_cbar = plt.subplot(gs[0, 1])
    
    # 1. Plot the Smoothed Flux Map
    im = ax_flux.pcolormesh(ze, te, render.T, cmap="hot", shading="auto")
    plt.colorbar(im, cax=ax_cbar, label='$\\log_{10}(\\text{Flux})$')
    ax_flux.set_ylabel("Angle $\\theta$ [rad]")
    ax_flux.set_title("Synchrotron Radiation Flux Map from {}".format(", ".join(FFILES)))
    ax_flux.set_xlim(ze.min(), ze.max())
    
    # 2. Plot the Lattice Elements as colored blocks
    for e in elements:
        s_start = e['start s [m]']
        length = e['length (m)']
        
        # Color logic based on your twiss.csv columns
        if e['DIPOLE: Bending angle [rad]'] is not None:
            color = 'royalblue'
        elif e['QUADRUPOLE: Gradient (T/m)'] is not None:
            color = 'crimson'
        else:
            color = 'lightgray' # Drifts/others
        ax_lattice.add_patch(plt.Rectangle((s_start, 0), length, 1, color=color, linewidth=0))

    # Formatting the element bar
    ax_lattice.set_ylim(0, 1)
    ax_lattice.set_yticks([])
    ax_lattice.set_xlabel("Longitudinal Position $z$ [m]")
    ax_lattice.vlines(absorbers, 0, 1, colors="k", linestyles="dashed")

    # Add a legend for the element types
    legend_elements = [
        Patch(facecolor='royalblue', label='Dipole'),
        Patch(facecolor='crimson', label='Quad'),
        Patch(facecolor='lightgray', label='Drift/Other'),
        Patch(facecolor='black', label="Absorbers")
    ]
    ax_lattice.legend(handles=legend_elements, loc='upper center', 
                      bbox_to_anchor=(0.5, -0.8), ncol=4, frameon=False)

    plt.show()