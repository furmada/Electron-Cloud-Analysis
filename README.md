# Electron Cloud Analysis (ECA)

Simulation management and analysis tools for PyECLOUD.

A comprehensive Python framework for organizing, analyzing, and fitting PyECLOUD electron cloud simulation data. ECA provides both programmatic APIs and a graphical interface for working with simulation datasets.

## Overview

ECA has been refactored into a modular package structure. The framework consists of several integrated modules organized under the `eca/` package:

- **`eca` package** - Core database, data modeling, and fitting functionality
  - `eca/__init__.py` - Database classes and core data models
  - `eca/fit.py` - Curve fitting models and utilities
  - `eca/plots.py` - Plotting and visualization utilities
  - `eca/run.py` - Job submission and execution management
  - `eca/instability.py` - Instability simulation support
  - `eca/synrad.py` - Synchrotron radiation analysis
- `**ecagui.py**` - Interactive graphical user interface for data exploration and analysis
- `**instabgui.py**` - Graphical interface for instability simulations

## Installation

Ensure you have Python 3.7+ with the following dependencies:

```bash
pip install numpy scipy tinydb matplotlib tkinter h5py
```

For remote job submission features, you'll also need SSH and SLURM/Condor access configured.

---

## Scientific Concepts

### Electron Cloud Phenomena

Electron cloud effects are a critical concern in particle accelerators, particularly in machines with positron or proton beams. When charged particles traverse a beam pipe, they can emit synchrotron radiation or undergo other processes that release electrons from the pipe walls. These secondary electrons can then be accelerated by the beam's electromagnetic fields, leading to a cascade process known as **multipacting**.

The Electron Cloud Analysis framework helps scientists and engineers:

- **Simulate** electron cloud buildup under various machine conditions
- **Analyze** the dynamics of electron cloud formation and evolution
- **Fit** theoretical models to simulation data to extract physical parameters
- **Visualize** results to understand dependencies on machine parameters
- **Compare** different configurations to optimize machine performance

### Key Physical Parameters

The framework tracks and analyzes numerous physical parameters that characterize electron cloud behavior:

- **Secondary Emission Yield (SEY)** - `del_max`: Maximum secondary emission yield, a critical parameter determining whether multipacting occurs
- **Beam Intensity** - `fact_beam`, `mean_intensity`: Number of particles in the beam, affecting electron production rates
- **Magnetic Field Configuration** - `B_multip`: Magnetic field strengths that influence electron motion and confinement
- **Beam Pipe Geometry** - `perimeter`, `area`: Physical dimensions of the vacuum chamber
- **Bunch Spacing** - `b_spac`: Time between consecutive bunches in the beam
- **Initial Electron Density** - `Ne_0`: Starting electron population in the simulation

### Theoretical Models

ECA implements several theoretical models to fit simulation data:

- **Furman Model (No Photoemission)** - Describes electron cloud buildup in the absence of photoemission effects, characterized by a saturation density `yc` and buildup rate `beta`
- **Furman Model with Magnetic Correction** - Extends the basic Furman model to account for magnetic field effects with correction parameters `b0` and `b1`
- **Furman Model with Photoemission** - Full model incorporating photoemission effects with parameters `yc`, `alpha` (photoemission rate), `beta` (multipacting rate), and `y0` (initial density)
- **Photo-Gamma Model** - Alternative parameterization for photoemission-dominated regimes

These models help extract meaningful physical insights from raw simulation data, allowing scientists to understand the underlying mechanisms driving electron cloud formation.

---

## Module Reference

### Core Database and Data Model (`eca` package)

The `eca` package provides the foundational classes for managing simulation data. All functionality is imported from the submodules when you use `from eca import ...`.

#### Database Management

`**DBFolder(path, **shared_properties)**`

- Represents a source folder containing non-instability PyECLOUD simulations
- Recursively scans for folders containing complete simulation parameter files
- `path` - Root directory to scan for simulations
- `shared_properties` - Meta-attributes inherited by all discovered simulations (e.g., `machine="LHC"`, `beam="proton"`)
- **Properties:**
  - `sim_folders` - List of folders with complete simulation outputs
  - `run_folders` - List of in-progress simulation folders
- **Methods:**
  - `config_files_for(path)` - Returns dictionary of configuration file paths for a simulation

`**InstabilityDBFolder(path, **shared_properties)**`

- Variant for instability simulations (PyHEADTAIL outputs)
- Scans for `.h5` data files and pyecloud_config subdirectories
- Compatible with `SimDB` for unified database operations

`**SimDB(db_file, db_folders=[], verbose=True)**`

- Main database class for querying and managing simulation records
- Stores all simulation metadata in a lightweight TinyDB database
- `db_file` - Path to JSON database file (or TinyDB instance)
- `db_folders` - List of `DBFolder` or `InstabilityDBFolder` instances to index
- **Methods:**
  - `where(**search)` - Query database with flexible filtering (see Querying section below)
  - `by(attr, **search)` - Return sorted results by attribute
  - `unique(attr, **search)` - Get unique values of an attribute across matching simulations
  - `closest(entry, **search)` - Find entries matching most properties with `entry`
  - `extract(attrs, **search)` - Create sub-database with only specified attributes
  - `extract_array(attrs, **search)` - Return NumPy array plus search results
  - `versus(attrX, attrY, nonexistant=np.nan, **search)` - Extract paired (X, Y) data for plotting
  - `align(common_attrs, discriminator_attr, **search)` - Find common attribute values across discriminator groups
  - `all_keys()` - Return sorted list of all keys in database

**Querying with `where()**`

The `where()` method supports multiple query formats to help you find simulations of interest:

```python
# Exact match
db.where(parameter=value)

# Multiple exact values
db.where(parameter=WhereIn(val1, val2, val3))

# Inverted (NOT) matching
db.where(parameter=WhereNot(val1, val2))

# Callable filter function
db.where(parameter=lambda val, doc: val > 100)

# Document-level expression (prefix with _)
db.where(_expr="Ne_0 > 1e10 and buildup == True")

# Direct TinyDB query
db.where(query=Query().fragment({"key": value}))

# By document ID
db.where(doc_id=123)
db.where(doc_id=WhereIn(1, 2, 3))
```

`**TemplateSim(path)**`

- Template-based simulation generator for creating parameter studies
- Loads existing simulation as a base for parameter variations
- **Methods:**
  - `spawn(path, **changes)` - Create new simulation folder with modified parameters
  - `generate_sweep(parameters, sweep, **constant)` - Generate parameter combinations for systematic studies

#### Data Modeling

`**ECModel(db, doc_id_or_document)**`

- Represents a single PyECLOUD simulation analysis
- Lazily loads simulation data and auto-computes derived properties
- Provides access to all simulation parameters and computed results
- **Properties (configurable):**
  - `area` - Beam chamber cross-sectional area [m²]
  - `buildup` - Boolean: multipacting buildup detected
  - `bunch_step` - Time indices per bunch spacing
  - `bunch_times` - Time of each bunch maximum [s]
  - `cutoff` - Last time index before simulation end
  - `half_bunch` - Gaussian bunch half-width [s]
  - `magnet` - Magnet type (0: drift, 2: dipole, 4: quadrupole, 6: sextupole)
  - `mean_intensity` - Mean beam intensity [particles/m]
  - `Ne_0` - Initial electron count [electrons/m]
  - `perimeter` - Beam chamber perimeter [m]
- **Data access properties (read-only):**
  - `data` - Loaded .mat file dictionary from PyECLOUD output
  - `time` - Time array from simulation [s]
  - `N_electrons` - Electron count time series [electrons/m]
  - `central_density` - Central density time series [electrons/m³]
  - `N_sec` - Secondary emission time series [electrons/m³/s]
  - `N_col` - Impact time series [electrons/m³/s]
  - `intensity` - Beam intensity time series [particles/m]
  - `smooth` - Smoothed electron count (convolution)
  - `smooth_diff` - Smoothed derivative of electron count
- **Methods:**
  - `time_to_index(t)` - Convert time(s) to data array indices
  - `train_to_index(train)` - Convert train number to time indices
  - `train_to_bunches(train)` - Get indices of bunches in a specific train

`**InstabilityModel(db, doc_id_or_document)**`

- Represents PyHEADTAIL instability simulation analysis
- Handles HDF5 data files from instability studies
- **Properties:**
  - `bunch_data` - Map of HDF5 bunch evolution data
  - `slice_data` - Map of HDF5 slice evolution data
  - `charge_weighted_rms_x` - Charge-weighted RMS in X direction
  - `intrabunch_activity` - Smoothed intra-bunch activity
  - `mean_x`, `mean_y` - Centroid positions
  - `sigma_x`, `sigma_y` - Beam sizes
  - `epsn_x`, `epsn_y` - Normalized emittances
- **Computed properties:**
  - `growth_rate_centroid` - Growth rate from centroid motion
  - `growth_rate_mode` - Growth rate from mode analysis
  - `dominant_mode_idx` - Index of dominant instability mode
  - `instability_threshold` - Threshold for instability onset

#### Curve Fitting (`eca.fit`)

ECA provides sophisticated curve fitting capabilities to extract physical parameters from simulation data.

`**DataSelector**`

- Base class for selecting fitting data from an ECModel
- **Methods:**
  - `select(model)` - Returns (x_data, y_data) tuple for fitting

`**BeforeBunchSelector(use_central_density=False)**`

- Selects data points immediately before each bunch passage
- Useful for analyzing pre-bunch electron cloud conditions

`**BunchAverageSelector(use_central_density=False)**`

- Selects averaged data during each bunch passage
- Robust against non-uniform bunch layouts

`**MaskDataSelector(base_selector, mask)**`

- Wraps an existing selector and applies a custom mask array
- Allows for custom data point selection criteria

`**Fit(name, function, variables, selector)**`

- Base fitting class implementing scipy curve_fit
- `name` - Identifier for fit parameters in database
- `function` - String expression of fit function (uses `np`, `scipy`)
- `variables` - Tuple of variable names: (x, param1, param2, ...)
- **Methods:**
  - `bound(dict)` - Set parameter bounds: `{"param": (lower, upper)}`
  - `initial(dict)` - Set initial guesses: `{"param": value}`
  - `fix(dict)` - Fix parameters: `{"param": value}`
  - `select_data(model)` - Get data for fitting
  - `scale_factor(model, xdata, ydata)` - Override for data scaling
  - `fit(model, refit=False)` - Perform fit, store results in model
  - `fit_function(model, params=None)` - Get callable fit function
  - `scrub(model)` - Remove fit parameters from model
- **Properties:**
  - `full_names` - Database entry names for parameters and errors

**Furman Model Fits**

`**FurmanNoPhotoFit(selector=None)**`

- Furman model without photoemission: `(y0*yc*exp(beta*yc*x)) / (yc + y0*(exp(beta*yc*x)-1))`
- **Parameters:**
  - `yc` - Critical saturation density [electrons/m³]
  - `beta` - Multipacting buildup rate [1/s]
  - `y0` - Initial density (fixed from data)
- Automatically detects buildup vs. decay and sets appropriate bounds/initial values
- For decay cases, uses simplified model: `y0 * exp(beta * yc * x)`

`**FurmanNPMCFit(db, selector=None)**`

- Furman model with magnetic correction: `(y0*yc*exp(yc*x*(b0+0.5*b1*x))) / (yc + y0*(exp(yc*x*(b0+0.5*b1*x))-1))`
- **Parameters:**
  - `yc` - Critical saturation density
  - `b0` - Primary magnetic correction factor
  - `b1` - Secondary magnetic correction factor (quadratic term)
  - `y0` - Initial density (fixed)
- Requires database for baseline comparison with non-magnetic cases
- Automatically uses closest non-magnetic simulation as reference

`**FurmanPhotoFit(selector=None)**`

- Full Furman model with photoemission effects
- **Parameters:**
  - `yc` - Critical saturation density
  - `alpha` - Photoemission rate coefficient
  - `beta` - Multipacting buildup rate
  - `y0` - Initial density (fixed)
- Uses stable delta reparameterization for numerical stability
- Automatically falls back to FurmanNoPhotoFit if photoemission is zero
- Only fits buildup simulations

`**PhotoGammaFit(selector=None)**`

- Alternative Furman-like photoemission fit
- **Parameters:**
  - `yc` - Critical saturation density
  - `beta` - Rate parameter
  - `gamma` - Photoemission strength parameter
  - `y0` - Initial density (fixed)

`**fit_all_where(db, fit, refit=False, verbose=True, **search)**`

- Batch fit utility
- Applies fit to all results matching search criteria
- Returns ECModel instances with fit results stored in database
- Progress tracking with success/failure statistics

#### Visualization (`eca.plots`)

`**model_plot(model, fits, size=(10, 5), log=(False, False), show_error=True, refit=False, central_density=False, fit_maxX=-1.0, label=None, train="All")**`

- Plot raw simulation data and fitted curves
- `model` - ECModel instance to plot
- `fits` - Iterable of Fit instances to overlay
- `size` - Tuple (width, height) or Axes object
- `log` - Tuple (log_x, log_y) for axis scaling
- `show_error` - Show error bands for fit parameters
- `central_density` - Plot central density instead of total electron count
- `fit_maxX` - Maximum x-axis value (in bunch passages)
- `label` - Legend label prefix
- `train` - Which train to plot ("All", or train index)

`**versus_plot(db, paramA, paramB, colorBy=None, size=(8, 6), dot_size=10, log=(False, False), colormap=None, **restrict)**`

- Create scatter plot of paramB vs paramA from database
- `db` - SimDB instance
- `colorBy` - Optional third parameter for point coloring
- `restrict` - Query parameters (passed to db.where())
- Returns tuple of plotted arrays

`**histogram_plot(db, param, bins='auto', size=(8, 6), log_y=False, color='C0', **restrict)**`

- Plot histogram of parameter distribution
- `bins` - Number of bins or 'auto' for automatic binning
- `log_y` - Use log scale on y-axis
- `color` - Bar color
- `restrict` - Query parameters (passed to db.where())

`**instability_grid_plot(db, size=(14, 16), save_path=None, max_plots=5, colormap=None, **restrict)**`

- Create comprehensive 8-panel grid for instability analysis
- Shows centroid motion, RMS evolution, and emittance growth
- Useful for comparing multiple instability simulations

#### Job Submission (`eca.run`)

ECA provides tools for submitting and managing simulations across different computing environments.

`**RunTarget(job_folders)**`

- Abstract base class for job submission targets
- `job_folders` - List of simulation directories to execute

`**RunLocal(job_folders, python="python", ecloud="./")**`

- Run PyECLOUD simulations on local machine
- `job_folders` - List of simulation directories to execute
- `python` - Python interpreter path
- `ecloud` - PyECLOUD root directory
- **Methods:**
  - `make_script(folder, output=None)` - Generate shell script for a simulation
  - `submit(verbose=True)` - Execute all simulations sequentially
  - `check_unfinished()` - Return list of incomplete simulations

`**RunSLURM(job_folders, python="python", ecloud="./", submission_name=None, remote_folder=None, username=None, hostname="localhost")**`

- Submit simulations via SLURM on remote machine
- Handles SSH connection, file transfer, and job collection
- `submission_name` - Custom name for submission batch
- `remote_folder` - Directory on remote for simulations
- `username` / `hostname` - SSH credentials
- **Methods:**
  - `submit(verbose=True)` - Upload jobs and submit to SLURM
  - `retrieve(verbose=True)` - Download completed simulations when ready
  - `compress(destination, folders)` - Static method to tar simulation folders

`**RunCondor(job_folders, python="python", ecloud="./", submission_name=None, remote_folder=None, username=None, hostname="lxplus.cern.ch")**`

- Submit simulations via HTCondor on lxplus
- Inherits from RunSLURM (includes retrieve functionality)
- Default remote folder: `/afs/cern.ch/work/{username[0]}/{username}/`
- Supports automatic EOS storage for results

`**RunCondorContainer(job_folders, container_path, submission_name=None, remote_folder=None, username=None, hostname="lxplus.cern.ch")**`

- Run PyECLOUD in Apptainer container on HTCondor
- Useful when PyECLOUD not installed on lxplus
- `container_path` - Full path to container image on CVMFS

---

### Graphical User Interfaces

#### ecagui.py - Main Analysis GUI

Launch the GUI with:

```bash
python ecagui.py
```

**Main Features**

**Database Management**

- Load existing database files or create new ones
- Clear and reset database state
- Save in-memory databases to disk
- Real-time status updates

**Overview Tab**

- Browse all simulations in loaded database
- View simulation paths and property statistics
- Identify parameters with multiple unique values
- Refresh view after database changes

**Filter Tab**

- Three filtering modes:
  - **Exact Match** - Select specific values for a property
  - **Condition** - Use operators (>, <, >=, <=, ==, !=) with numeric values
  - **Custom Expression** - Write Python expressions (e.g., `Ne_0 > 1e10 and buildup == True`)
- Combine multiple filters with AND logic
- Copy/paste filters as JSON for sharing and reproducibility
- Apply filters, reset to full database, or open filtered results in new window

**Fitting Tab**

- Select fit model: FurmanNoPhoto, FurmanNPMC, FurmanPhoto, or PhotoGamma
- Apply to filtered simulations or entire database
- Track fitting progress with status updates
- Refit all or only unfitted simulations
- View success/failure statistics

**Database Plots Tab**

- Create scatter plots from any two parameters
- Optionally color points by a third parameter
- Supports log scale on both axes
- Interactive matplotlib toolbar for zoom/pan

**Histogram Tab**

- Generate histograms of any parameter
- Configurable bin count or auto-binning
- Toggle log scale on y-axis
- Visualize parameter distributions

**Individual Plot Tab**

- Select simulations from table to plot
- Add custom columns to simulation list for easy identification
- Overlay multiple fit models on individual simulations
- Configure plot options:
  - Toggle between total and central density
  - Set maximum X-axis value
  - Show/hide fit error bands
  - Select specific trains to plot
- Interactive legend and matplotlib navigation

**Keyboard/Mouse Usage**

- **Right-click** on filter text area to copy/paste/clear filters
- **Click** simulation rows to select for individual plotting
- **Hold Ctrl/Cmd** to select multiple simulations for overlaid plots

#### instabgui.py - Instability Analysis GUI

Specialized interface for instability simulation analysis with similar functionality tailored for PyHEADTAIL outputs.

---

## Usage Examples

### Basic Database Operations

```python
from eca import DBFolder, SimDB, ECModel

# Create database from simulation folders
db_folder = DBFolder("/path/to/simulations", machine="LHC", beam="proton")
db = SimDB("simulations.json", [db_folder])

# Query database
results = db.where(machine="LHC", buildup=True)
print(f"Found {len(results)} buildup simulations")

# Get unique parameter values
sey_values = db.unique("del_max", machine="LHC")
print(f"SEY values: {sey_values}")

# Get simulations with specific SEY
high_sey_sims = db.where(del_max=1.5)
print(f"Simulations with SEY=1.5: {len(high_sey_sims)}")
```

### Analyzing Simulation Data

```python
from eca import ECModel

# Load a specific simulation
model = ECModel(db, db.where(del_max=1.5)[0].doc_id)

# Access physical properties
print(f"Beam pipe area: {model.area:.4f} m²")
print(f"Beam pipe perimeter: {model.perimeter:.4f} m")
print(f"Initial electron density: {model.Ne_0:.2e} electrons/m")
print(f"Mean beam intensity: {model.mean_intensity:.2e} particles/m")
print(f"Multipacting buildup detected: {model.buildup}")

# Access time series data
print(f"Time array length: {len(model.time)} points")
print(f"Electron count range: {model.N_electrons.min():.2e} to {model.N_electrons.max():.2e}")
```

### Fitting Simulations

```python
from eca import FurmanNoPhotoFit, FurmanPhotoFit, fit_all_where

# Create fit model
fit = FurmanNoPhotoFit()

# Apply to all simulations with multipacting
fit_all_where(db, fit, buildup=True, verbose=True)

# Access fit results for a specific simulation
model = ECModel(db, db.where(buildup=True, furman_np_success=True)[0].doc_id)
print(f"Critical density (yc): {model.furman_np_yc:.2e} electrons/m³")
print(f"Buildup rate (beta): {model.furman_np_beta:.4f}")
print(f"Fit error (yc): ±{model.furman_np_yc_err:.2e}")

# Try photoemission fit for comparison
photo_fit = FurmanPhotoFit()
photo_fit.fit(model)
if model.furman_p_success:
    print(f"Photoemission alpha: {model.furman_p_alpha:.4e}")
```

### Comparing Different Configurations

```python
from eca import FurmanNoPhotoFit, fit_all_where

# Fit all simulations
fit = FurmanNoPhotoFit()
fit_all_where(db, fit, verbose=True)

# Extract yc vs SEY relationship
results = db.where(furman_np_success=True)
yc_values = [r['furman_np_yc'] for r in results]
sey_values = [r['del_max'] for r in results]

# Find correlation
import numpy as np
correlation = np.corrcoef(yc_values, sey_values)[0,1]
print(f"Correlation between yc and SEY: {correlation:.3f}")
```

### Plotting Results

```python
from eca.plots import versus_plot, model_plot, histogram_plot
import matplotlib.pyplot as plt

# Plot relationship between parameters
versus_plot(db, "del_max", "furman_np_yc", colorBy="buildup", size=(10, 6))
plt.title("Saturation Density vs SEY")
plt.show()

# Plot individual simulation with fit
model = ECModel(db, db.where(del_max=1.5)[0].doc_id)
fits = [FurmanNoPhotoFit()]
model_plot(model, fits, central_density=False, fit_maxX=100)
plt.title("Electron Cloud Buildup - SEY=1.5")
plt.show()

# Histogram of critical densities
histogram_plot(db, "furman_np_yc", bins=20, log_y=True)
plt.title("Distribution of Critical Densities")
plt.show()
```

### Running Simulations

```python
from eca import TemplateSim
from eca.run import RunLocal, RunSLURM, RunCondor

# Create parameter sweep using template
template = TemplateSim("/path/to/template_simulation")

# Generate sweep over SEY and beam intensity
configs = TemplateSim.generate_sweep(
    parameters=["del_max", "fact_beam"],
    sweep=[[0.5, 1.0, 1.5, 2.0], [1e11, 1e12]],
    filling_pattern_file="pattern.txt"
)

# Create simulation folders
for i, config in enumerate(configs):
    output_dir = f"/simulations/sweep_{i:03d}"
    template.spawn(output_dir, **config)

# Run locally
runner = RunLocal([f"/simulations/sweep_{i:03d}" for i in range(len(configs))], 
                  ecloud="/opt/pyecloud")
runner.submit()

# Or submit to SLURM
# runner = RunSLURM(
#     [f"/simulations/sweep_{i:03d}" for i in range(len(configs))],
#     python="/usr/bin/python3",
#     ecloud="/home/user/pyecloud",
#     hostname="compute.example.com"
# )
# runner.submit()
# runner.retrieve()  # Download when done

# Or submit to HTCondor at CERN
# runner = RunCondor(
#     [f"/simulations/sweep_{i:03d}" for i in range(len(configs))],
#     ecloud="/afs/cern.ch/user/a/afurman/pyecloud"
# )
# runner.submit()
```

### Advanced Analysis: Parameter Dependence Studies

```python
from eca import DBFolder, SimDB, FurmanNoPhotoFit, fit_all_where
import numpy as np

# Load database with multiple parameter variations
db_folder = DBFolder("/simulations/parameter_study", machine="LHC")
db = SimDB("study.json", [db_folder])

# Fit all simulations
fit = FurmanNoPhotoFit()
fit_all_where(db, fit, verbose=True)

# Study dependence of yc on SEY and magnetic field
sey_values = db.unique("del_max")
b_field_values = db.unique("B_multip")

for sey in sey_values:
    for b_field in b_field_values:
        results = db.where(del_max=sey, B_multip=b_field, furman_np_success=True)
        if results:
            avg_yc = np.mean([r['furman_np_yc'] for r in results])
            avg_beta = np.mean([r['furman_np_beta'] for r in results])
            print(f"SEY={sey:.2f}, B={b_field}: yc={avg_yc:.2e}, beta={avg_beta:.4f}")
```

### Creating Custom Fit Models

```python
from eca import Fit, BunchAverageSelector

class CustomFit(Fit):
    """Custom fit model for specific analysis needs."""
    EXPRESSION = "a * (1 - np.exp(-b * x)) + c * x"
    
    def __init__(self):
        super().__init__(
            "custom",
            self.EXPRESSION,
            ("x", "a", "b", "c"),
            BunchAverageSelector()
        )
    
    def fit(self, model, refit=False):
        # Custom initialization based on model properties
        self.initial({"a": model.Ne_0, "b": 0.1, "c": 0.01})
        self.bound({"a": (0, model.Ne_0 * 10), "b": (0, 10), "c": (0, 1)})
        return super().fit(model, refit)

# Use custom fit
custom_fit = CustomFit()
custom_fit.fit(model)
```

### Database Alignment and Comparison

```python
# Find common parameter values across simulation groups
common_vals, dbs_by_magnet = db.align(
    common_attrs=["del_max", "k_pe_st"],
    discriminator_attr="magnet"
)

for magnet_type, sub_db in zip(common_vals, dbs_by_magnet):
    print(f"Magnet type {magnet_type}: {len(sub_db.where())} matching simulations")
    
    # Analyze each group
    results = sub_db.where(furman_np_success=True)
    if results:
        avg_yc = np.mean([r['furman_np_yc'] for r in results])
        print(f"  Average yc: {avg_yc:.2e}")
```

### Working with Instability Simulations

```python
from eca import InstabilityDBFolder, SimDB
from eca.instability import InstabilityModel

# Create instability database
db_folder = InstabilityDBFolder("/path/to/instability_simulations")
db = SimDB("instability.json", [db_folder])

# Load instability model
imodel = InstabilityModel(db, db.where()[0].doc_id)

# Access instability properties
print(f"Number of turns: {imodel.n_turns}")
print(f"Number of slices: {imodel.n_slices}")
print(f"Growth rate (centroid): {imodel.growth_rate_centroid:.4e}")
print(f"Growth rate (mode): {imodel.growth_rate_mode:.4e}")
print(f"Dominant mode index: {imodel.dominant_mode_idx}")

# Access bunch and slice data
print(f"Mean X available: {len(imodel.mean_x)} points")
print(f"Charge-weighted RMS X: {imodel.charge_weighted_rms_x[:5]}")
```

---

## Advanced Topics

### Understanding the Furman Model

The Furman model describes electron cloud buildup as a saturation process. The basic model without photoemission has the form:

```
N(x) = (y0 * yc * exp(beta * yc * x)) / (yc + y0 * (exp(beta * yc * x) - 1))
```

Where:

- `x` is the number of bunch passages (or time in appropriate units)
- `y0` is the initial electron density
- `yc` is the saturation density (critical density)
- `beta` is the buildup rate coefficient

**Physical Interpretation:**

- **yc (Critical Density)**: The maximum electron density that can be achieved. This represents the point where electron production (from secondary emission) balances electron loss (from absorption on the pipe walls).
- **beta (Buildup Rate)**: How quickly the electron cloud builds up. Higher values indicate faster buildup, which can be driven by higher SEY or more favorable geometric/magnetic conditions.
- **y0 (Initial Density)**: The starting electron density, typically determined by initial conditions or previous bunch passages.

For **decay** (non-buildup) cases, a simplified exponential decay model is used:

```
N(x) = y0 * exp(beta * yc * x)
```

### Photoemission Effects

When photoemission is significant (e.g., from synchrotron radiation), the model includes an additional term:

```
N(x) = 0.5*(yc + sqrt(yc² + (4*alpha/beta))*tanh((x/2)*sqrt(beta*(4*alpha + (yc²)*beta))) - arctanh((yc - 2*y0)*sqrt(beta / (4*alpha + (yc²)*beta))))
```

Where:

- `alpha` is the photoemission rate coefficient, related to the photoemission yield and photon flux

**Physical Interpretation:**

- **alpha (Photoemission Rate)**: Represents the rate at which new electrons are created by photoemission processes. This is particularly important in machines with high-energy beams where synchrotron radiation is significant.

### Practical Considerations for Scientists

1. **SEY Threshold**: For multipacting to occur, the SEY (del_max) must typically exceed ~1.0. Values below this threshold generally result in electron cloud decay rather than buildup.
2. **Saturation Density**: The critical density `yc` is a key parameter for understanding the maximum electron cloud density your machine configuration can support. Higher `yc` values indicate more severe electron cloud effects.
3. **Buildup Rate**: The `beta` parameter indicates how quickly the electron cloud develops. High buildup rates can lead to rapid onset of electron cloud effects, potentially affecting beam stability.
4. **Magnetic Field Dependence**: Different magnet types have different effects on electron cloud formation. Dipole fields, for example, can confine electrons differently than quadrupole fields.
5. **Photoemission Importance**: In high-energy machines, photoemission from synchrotron radiation can be a significant source of electrons. The `alpha` parameter helps quantify this contribution.

### Best Practices for Analysis

1. **Start with Database Creation**: Always begin by creating a comprehensive database of your simulations using `DBFolder` and `SimDB`.
2. **Use Filtering Effectively**: The `where()` method with `WhereIn` and `WhereNot` allows you to efficiently select simulations of interest.
3. **Fit Systematically**: Use `fit_all_where()` to apply fits to many simulations at once, ensuring consistent analysis.
4. **Visualize Results**: Use the plotting functions to understand parameter dependencies and identify trends.
5. **Check Fit Quality**: Always examine fit success rates and error bars to ensure reliable results.
6. **Compare Configurations**: Use the alignment and comparison tools to understand how different machine configurations affect electron cloud behavior.

---

## References

- **PyECLOUD**: Electron cloud simulation framework developed at CERN
- **PyHEADTAIL**: Instability simulation code for particle accelerators
- **TinyDB**: Lightweight document database used internally for simulation metadata
- **Matplotlib**: Visualization library for plots and figures
- **SciPy**: Scientific computing library with curve_fit optimization
- **NumPy**: Fundamental package for numerical computing in Python

## Authors

**Adam Furman**  
[adam.furman@cern.ch](mailto:adam.furman@cern.ch)

For use with PyECLOUD at CERN.

---

*Last updated: June 2026*