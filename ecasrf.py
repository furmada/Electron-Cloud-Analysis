"""
Synchrotron Radiation Flux analysis tools for use with PyECLOUD and Electron Cloud Analysis
@author Adam Furman
@email adam.furman@cern.ch
"""

import warnings
from typing import Iterable
import numpy as np
import scipy.io as sio
from scipy.stats import iqr
from scipy.signal import find_peaks, savgol_filter, peak_widths
from scipy.optimize import minimize, curve_fit
from scipy.ndimage import gaussian_filter1d, maximum_filter1d
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
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

def slice_flux_data(flux_files: list[str] | str | tuple, dZ: float, num_points: int = 121, verbose=True) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Open the flux file, and cut it into dZ[m] sized sections, binning
    angularly into up to num_points segments.
    """
    # Read the data
    x_coord, y_coord, z_coord, F_value = read_flux_data(flux_files)
    # Set up outputs
    Z_starts = np.arange(np.min(z_coord), np.max(z_coord), dZ)
    slices = []
    for z in Z_starts:
        # Find the range of entries in this slice
        z_range = np.bitwise_and(z_coord >= z, z_coord < z + dZ)
        # Extract and center the coordinates
        x = x_coord[z_range] - np.mean(x_coord[z_range])
        y = y_coord[z_range] - np.mean(y_coord[z_range])
        f = F_value[z_range]
        # Compute the angles of each coordinate and sort starting at 0 angle
        angle = np.arctan2(y, x)
        angle[angle < 0] += 2*np.pi
        order = np.argsort(angle)
        angle = angle[order]
        x, y, f = x[order], y[order], f[order]
        # Bin points and fluxes by angle
        dTheta = 2*np.pi/(num_points+1)
        x_b, y_b, f_b = [], [], []
        for i in range(num_points+1):
            theta = dTheta * i
            angle_range = angle - theta - dTheta / 2
            angle_range[angle_range < 0] += 2*np.pi
            points = np.argwhere(angle_range <= dTheta / 2)
            if np.sum(points) > 0:
                x_b.append(np.mean(x[points]))
                y_b.append(np.mean(y[points]))
                f_b.append(np.mean(f[points]))
        slices.append(np.array([x_b, y_b, f_b]))
        if verbose: print("{:<5.2f}".format(z), end="\r")
    if verbose: print("Done.    ")
    return Z_starts, slices

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

# Flux data postprocessing tools

def dedensify_envelope(x: np.ndarray, y: np.ndarray, f: np.ndarray, target: int = 100, window_size: int = 5, curvature_weight: float = 3.0) -> np.ndarray:
    """
    De-densify a cross-section while preserving maximum flux values via an upper envelope.
    Improves on point distribution by mathematically computing geometric curvature and
    sampling densely in high-curvature regions.
    """
    # 1. Center and sort the points by angle
    cx, cy = np.mean(x), np.mean(y)
    x_c, y_c = x - cx, y - cy
    angle = np.arctan2(y_c, x_c)
    angle[angle < 0] += 2 * np.pi
    order = np.argsort(angle)
    x_s, y_s, f_s = x[order], y[order], f[order]
    # Ensure the window size isn't larger than our dataset
    win_size = min(window_size, len(f_s) // 2)
    # 2. Compute the upper envelope of the flux
    # We use mode='wrap' because the beam pipe cross-section is a closed periodic loop
    f_env = maximum_filter1d(f_s, size=win_size, mode='wrap')
    # Apply a light Gaussian filter to the envelope to prevent unnatural stepped plateaus
    f_env = gaussian_filter1d(f_env, sigma=1.0, mode='wrap')
    # 3. Compute true geometric curvature
    # Use Gaussian derivatives for smooth x', y', x'', y'' over the parametric curve
    sigma = 2.0
    dx = gaussian_filter1d(x_s, sigma=sigma, order=1, mode='wrap')
    dy = gaussian_filter1d(y_s, sigma=sigma, order=1, mode='wrap')
    ddx = gaussian_filter1d(x_s, sigma=sigma, order=2, mode='wrap')
    ddy = gaussian_filter1d(y_s, sigma=sigma, order=2, mode='wrap')
    # Geometric curvature formula: |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
    speed_sq = dx**2 + dy**2
    kappa = np.abs(dx * ddy - dy * ddx) / (speed_sq**(1.5) + 1e-12)
    # Normalize curvature between 0 and 1
    kappa_norm = (kappa - np.min(kappa)) / (np.max(kappa) - np.min(kappa) + 1e-12)
    # 4. Compute cumulative sampling density over the arc length
    # Calculate differential arc length (ds) between adjacent points
    ds = np.sqrt(np.diff(x_s, append=x_s[0])**2 + np.diff(y_s, append=y_s[0])**2)
    # Density function: base density (1.0) + weighted curvature
    density = 1.0 + curvature_weight * kappa_norm
    # Integrate density with respect to arc length to get the Cumulative Distribution
    dC = density * ds
    C = np.insert(np.cumsum(dC), 0, 0)[:-1] 
    # 5. Interpolate to extract target points
    # Extend arrays by 1 point at the end to cleanly handle the wrap-around interpolation
    C_ext = np.append(C, C[-1] + dC[-1])
    x_ext = np.append(x_s, x_s[0])
    y_ext = np.append(y_s, y_s[0])
    f_ext = np.append(f_env, f_env[0])
    # Create evenly spaced target integration values
    C_targets = np.linspace(0, C_ext[-1], target, endpoint=False)
    # Interpolate geometry and envelope flux
    x_new = interp1d(C_ext, x_ext)(C_targets)
    y_new = interp1d(C_ext, y_ext)(C_targets)
    f_new = interp1d(C_ext, f_ext)(C_targets)    
    return np.array([x_new, y_new, f_new])

def slice_and_dedensify(flux_files: list[str] | str | tuple, dZ: float, num_points: int = 121, verbose=True) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Open the flux file, and cut it into dZ[m] sized sections, binning
    angularly into up to num_points segments.
    """
    # Read the data
    x_coord, y_coord, z_coord, F_value = read_flux_data(flux_files)
    # Set up outputs
    Z_starts = np.arange(np.min(z_coord), np.max(z_coord), dZ)
    slices = []
    for z in Z_starts:
        # Find the range of entries in this slice
        z_range = np.bitwise_and(z_coord >= z, z_coord < z + dZ)
        slices.append(dedensify_envelope(x_coord[z_range], y_coord[z_range], F_value[z_range], num_points))
        if verbose: print("{:<5.2f}".format(z), end="\r")
    if verbose: print("Done.    ")
    return Z_starts, slices

def slice_average(slices: list[np.ndarray]) -> np.ndarray:
    """
    Compute the average profile (X, Y, F) from a collection of slices.
    Assumes all slices have been dedensified to the same target number of points
    and share the same angular starting point.
    """
    if not slices:
        raise ValueError("The input list of slices is empty.")
    # Safety check: Ensure all slices have the exact same shape
    # (e.g., 3 rows for X, Y, F, and N columns for the target points)
    base_shape = slices[0].shape
    for i, s in enumerate(slices):
        if s.shape != base_shape:
            raise ValueError(f"Shape mismatch at slice {i}: expected {base_shape}, got {s.shape}. "
                             "Ensure all slices in the collection were dedensified to the same target.")
    # Stack the list of 2D arrays into a single 3D NumPy array.
    # Resulting shape will be (num_slices, 3, num_points)
    stacked_slices = np.stack(slices)
    # Calculate the mean along the first axis (across all slices).
    # This simultaneously averages the X, Y, and F coordinates for each point index.
    averaged_slice = np.mean(stacked_slices, axis=0)
    return averaged_slice

def _find_periodic_peaks_uniform(signal, distance=None, prominence=None, width=None):
    """
    Wrapper for find_peaks that handles periodic boundary conditions via padding.
    Assumes 'signal' is uniformly sampled.
    """
    n = len(signal)
    # Pad to handle periodicity: [end...start] + [original] + [end...start]
    # Pad size should be large enough to catch wide peaks near boundaries
    pad_size = max(n // 4, 20) 
    wrapped_signal = np.concatenate([signal[-pad_size:], signal, signal[:pad_size]])
    
    if distance is None:
        distance = max(10, n // 10)
    
    peaks, properties = find_peaks(
        wrapped_signal,
        distance=distance,
        prominence=prominence,
        width=width,
        rel_height=0.5
    )
    
    # Filter to original range [pad_size, pad_size + n)
    valid_mask = (peaks >= pad_size) & (peaks < pad_size + n)
    valid_peaks = peaks[valid_mask] - pad_size
    
    # Filter properties
    filtered_props = {}
    for key, val in properties.items():
        if isinstance(val, np.ndarray):
            filtered_props[key] = val[valid_mask]
        else:
            filtered_props[key] = val
            
    return valid_peaks, filtered_props

def find_asymmetries(x: np.ndarray, y: np.ndarray, f: np.ndarray, 
                     min_prominence_factor: float = 0.1, 
                     min_distance_factor: float = 0.05,
                     interp_points: int = 200) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Improved peak and valley detection with explicit interpolation to uniform grid.
    
    This ensures that width calculations (in radians) are accurate even if the 
    input data points are not perfectly equispaced in angle.
    """
    if len(f) < 20:
        return np.array([]), np.array([]), np.array([])

    # 1. Preprocessing: Calculate Angles
    # Using the specific centering logic from your snippet: -(x - cx)
    cx, cy = np.mean(x), np.mean(y)
    # Angle calculation matching your snippet: atan2(y-cy, -(x-cx))
    # This sets 0 radians at the positive X-axis (if x>cx, y=cy -> atan2(0, -pos) = pi? 
    # Wait: atan2(y, -x). If x>cx, -x is negative. atan2(0, neg) = pi.
    # If x<cx, -x is positive. atan2(0, pos) = 0.
    # This effectively rotates the phase by 180 degrees compared to standard atan2(y,x).
    # We will stick to your logic exactly.
    angle_raw = np.arctan2(y - cy, -(x - cx))
    
    # Normalize to [0, 2pi)
    angle = angle_raw.copy()
    angle[angle < 0] += 2 * np.pi
    
    # Sort by angle to prepare for interpolation
    sort_idx = np.argsort(angle)
    angle_sorted = angle[sort_idx]
    f_sorted = f[sort_idx]
    
    # Handle potential log issues
    f_safe = np.where(f_sorted < 1e-12, 1e-12, f_sorted)
    logF = np.log10(f_safe)
    
    # 2. Interpolation to Uniform Grid
    # Create a strictly uniform grid of angles from 0 to 2pi
    # We use 'cubic' or 'linear'. Cubic is smoother but can overshoot. Linear is safer for sharp peaks.
    # Given the physics, cubic spline is usually fine if the data is smooth enough.
    # We add a small epsilon to the end to ensure the wrap-around is handled if we were doing cyclic interp,
    # but here we just map 0->2pi.
    new_angles = np.linspace(0, 2 * np.pi, interp_points, endpoint=False)
    
    # Interpolate
    # Note: Since the data is periodic, we should ideally handle the wrap-around in interpolation.
    # However, since we are just mapping 0->2pi and the data is likely continuous there, 
    # standard interpolation works if the first and last points are close in value.
    # To be safe, we can append the first point to the end for interpolation continuity.
    angle_interp = np.concatenate([angle_sorted, angle_sorted[0:1] + 2*np.pi])
    logF_interp = np.concatenate([logF, logF[0:1]])
    
    # Create interpolation function
    # 'kind'='cubic' provides smooth derivatives for better peak finding
    f_interp = interp1d(angle_interp, logF_interp, kind='cubic', fill_value="extrapolate")
    
    # Sample on the uniform grid
    uniform_logF = f_interp(new_angles)
    
    # 3. Smoothing (Optional but recommended for noisy data)
    # Window size relative to the new uniform grid
    win_size = int(min(max(5, interp_points * 0.05), interp_points // 2))
    if win_size % 2 == 0: win_size += 1
    smooth_logF = savgol_filter(uniform_logF, win_size, 3)
    
    # 4. Define Detection Parameters
    signal_range = np.max(smooth_logF) - np.min(smooth_logF)
    min_prominence = signal_range * min_prominence_factor
    # Distance in samples on the UNIFORM grid
    min_distance_samples = int(interp_points * min_distance_factor)
    
    # 5. Detect Peaks and Valleys
    peaks_idx, _ = _find_periodic_peaks_uniform(
        smooth_logF, 
        distance=min_distance_samples, 
        prominence=min_prominence
    )
    
    valleys_idx, _ = _find_periodic_peaks_uniform(
        -smooth_logF, 
        distance=min_distance_samples, 
        prominence=min_prominence
    )
    
    # 6. Calculate Widths
    # Since the grid is uniform, width in samples converts directly to radians
    # Resolution: 2pi / interp_points radians per sample
    rad_per_sample = 2 * np.pi / interp_points
    
    peak_widths_rad = np.zeros(len(peaks_idx))
    if len(peaks_idx) > 0:
        w_res = peak_widths(smooth_logF, peaks_idx, rel_height=0.5)
        peak_widths_rad = w_res[0] * rad_per_sample
        
    valley_widths_rad = np.zeros(len(valleys_idx))
    if len(valleys_idx) > 0:
        w_res = peak_widths(-smooth_logF, valleys_idx, rel_height=0.5)
        valley_widths_rad = w_res[0] * rad_per_sample

    # 7. Compile Results
    all_indices = np.concatenate([peaks_idx, valleys_idx])
    
    # Heights: Peaks are positive deviation, Valleys are negative deviation (depth)
    # For valleys, we calculate depth relative to the max of the smoothed signal
    median = np.median(smooth_logF)
    all_heights = np.concatenate([
        smooth_logF[peaks_idx] - median, 
        -smooth_logF[valleys_idx] + median
    ])
    
    all_widths = np.concatenate([peak_widths_rad, valley_widths_rad])
    
    # Sort by magnitude of deviation
    order = np.argsort(np.abs(all_heights))[::-1]
    
    # Map indices back to actual angles
    final_locations = new_angles[all_indices[order]]
    final_heights = all_heights[order]
    final_widths = all_widths[order]
    
    return final_locations, final_heights, final_widths