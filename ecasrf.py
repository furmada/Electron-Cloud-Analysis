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
from scipy.signal import find_peaks
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
        x_coord = np.concat((x_coord, flux_data[0] / 100)) # Given in cm, we want m
        y_coord = np.concat((y_coord, flux_data[1] / 100))
        z_coord = np.concat((z_coord, flux_data[2] / 100))
        F_value = np.concat((F_value, flux_data[3]))
    return np.array([x_coord.ravel(), y_coord.ravel(), z_coord.ravel(), F_value.ravel()])

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

def complete_slices(flux_files: list[str] | str | tuple, angle_gap: float = np.pi / 20) -> list[np.ndarray]:
    """
    Open the flux files, and find the densest collection of "complete" (e.g. 360 degrees of data) slices.
    """
    # Read the data
    x_coord, y_coord, z_coord, F_value = read_flux_data(flux_files)
    order = np.argsort(z_coord)
    x_coord, y_coord, z_coord, F_value = x_coord[order], y_coord[order], z_coord[order], F_value[order]
    # Find the unique z coordinates of the model
    uZ = np.sort(np.unique(z_coord))
    slices = [[] for _ in uZ]
    uzi = 0
    for i, z in enumerate(z_coord):
        for j in range(uzi, uZ.size):
            if z == uZ[j]:
                uzi = j
                break
        slices[uzi].append(i)
    # Each section represents a complete (angluar) collection of data points.
    csections = []
    lsection = np.array([], dtype=int)
    for i in range(uZ.size):
        lsection = np.concat((lsection, slices[i]), dtype=int).ravel()
        if lsection.size <= 2: continue
        theta = np.sort(np.atan2(y_coord[lsection] - np.mean(y_coord[lsection]), x_coord[lsection] - np.mean(x_coord[lsection])))
        if np.max(np.diff(theta)) < angle_gap:
            csections.append(lsection.copy())
            lsection = np.array([], dtype=int)
    return [np.array([x_coord[cs], y_coord[cs], z_coord[cs], F_value[cs]]) for cs in csections]

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

def _gauss(x, a, mu, sigma, c):
    """Gaussian profile for fitting the flux peak or valley."""
    return a * np.exp(-(x - mu)**2 / (2 * sigma**2)) + c

def fit_flux_feature(theta: np.ndarray, log_f: np.ndarray, mu_log: float) -> tuple[float, float]:
    """
    Factors out the Gaussian curve fitting. Uses scipy.signal.find_peaks to intelligently 
    locate the true structural peak/valley, avoiding single-facet noise spikes.
    """
    f_range = iqr(log_f)
    if f_range < 1:
        return 0, np.nan
    is_max = np.sum(log_f > mu_log + f_range) > np.sum(log_f < mu_log - f_range)
    bounds = ([0 if is_max else -10, 0.0, 0.01, np.min(log_f)], [10 if is_max else 0, 2*np.pi, np.inf, np.max(log_f)])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(_gauss, theta, log_f, 
                                p0=[0, np.pi, np.pi, mu_log], 
                                bounds=bounds,
                                maxfev=2000)
        peak_strength, peak_mu, peak_width, bg_c = popt
    except RuntimeError:
        # Fallback if the curve fit fails to converge
        peak_strength = 0
        peak_width = np.nan
        
    return peak_strength, peak_width

def extract_slice_metrics(x: np.ndarray, y: np.ndarray, f: np.ndarray) -> tuple[float, float, float, float]:
    """
    Computes total flux, asymmetry ratio, and Gaussian peak metrics (strength, width)
    by analyzing the flux over the angular profile of the chamber.
    """
    total_flux = np.sum(f)
    if total_flux <= 0:
        return 0.0, np.nan, np.nan, 0.0
    theta = np.arctan2(y, -x)
    theta[theta < 0] += 2 * np.pi
    order = np.argsort(theta)
    theta_sorted = theta[order]
    f_sorted = f[order]
    log_f = np.log10(f_sorted + 1e-12)
    peak_strength, peak_width = fit_flux_feature(theta_sorted, log_f, np.mean(log_f))    
    if np.isnan(peak_width):
        # Graceful fallback if the fit completely failed but flux exists
        asymmetry = 0
    else:
        # High amplitude (difference from mean) + highly localized (small width) = High Asymmetry.
        # The sign of peak_strength automatically dictates if it's a peak (+) or a valley (-).
        asymmetry = peak_strength / peak_width
    if asymmetry > 0:
        print("{:.2E}, {:.2f}, {:.2f}, {:.2f}".format(total_flux, peak_strength, peak_width, asymmetry))
    return total_flux, peak_strength, peak_width, asymmetry

def build_system_analysis(Zs: np.ndarray, slices: list[np.ndarray], elements: list[dict], absorbers: np.ndarray) -> dict:
    """
    Cross-references slice data with Twiss elements and absorbers to build a comprehensive 
    dictionary of numpy arrays for pattern analysis.
    """
    data: dict[str, list | np.ndarray] = {
        "Z": [], "Total_Flux": [], "Peak_Strength": [], "Peak_Width": [],
        "Asymmetry": [], "Element_Name": [], "Element_Type": [],
        "Mag_Field": [], "Dist_to_Absorber": []
    }
    for z, slc in zip(Zs, slices):
        x, y, f = slc[0], slc[1], slc[2]
        total_flux, peak_strength, peak_width, asymmetry = extract_slice_metrics(x, y, f)
        # Identify the element overlapping this Z position
        current_elem = "None"
        elem_type = "Drift/Unknown"
        mag_field = 0.0
        for e in elements:
            start_z = float(e.get("start Z (m)", 0))
            length = float(e.get("length (m)", 0))
            end_z = start_z + length
            if start_z <= z <= end_z:
                current_elem = e.get("Name", "Unknown")
                if e.get("DIPOLE: Field (T)") is not None:
                    elem_type = "Dipole"
                    mag_field = float(e["DIPOLE: Field (T)"])
                elif e.get("QUADRUPOLE: Gradient (T/m)") is not None:
                    elem_type = "Quadrupole"
                    mag_field = float(e["QUADRUPOLE: Gradient (T/m)"])
                break
        # Calculate distance to the most recent upstream absorber
        past_absorbers = absorbers[absorbers <= z]
        dist_to_abs = z - past_absorbers[-1] if len(past_absorbers) > 0 else np.nan
        data["Z"].append(z)
        data["Total_Flux"].append(total_flux)
        data["Peak_Strength"].append(peak_strength)
        data["Peak_Width"].append(peak_width)
        data["Asymmetry"].append(asymmetry)
        data["Element_Name"].append(current_elem)
        data["Element_Type"].append(elem_type)
        data["Mag_Field"].append(mag_field)
        data["Dist_to_Absorber"].append(dist_to_abs)
    # Convert lists to numpy arrays for easy indexing
    for key in data:
        data[key] = np.array(data[key])
    return data