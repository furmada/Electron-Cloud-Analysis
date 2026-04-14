# Electron Cloud Analysis (ECA)

Simulation management and analysis tools for PyECLOUD.

A comprehensive Python framework for organizing, analyzing, and fitting PyECLOUD electron cloud simulation data. ECA provides both programmatic APIs and a graphical interface for working with simulation datasets.

## Overview

ECA consists of several integrated modules:

- **eca.py** - Core database, data modeling, and fitting functionality
- **ecagui.py** - Interactive graphical user interface for data exploration and analysis
- **ecaplots.py** - Plotting and visualization utilities
- **ecarun.py** - Job submission and execution management for various computing environments

## Installation

Ensure you have Python 3.7+ with the following dependencies:
```bash
pip install numpy scipy tinydb matplotlib tkinter
```

For remote job submission features, you'll also need SSH and SLURM/Condor access configured.

---

## Module Reference

### eca.py - Core Functionality

#### Database Management

**`DBFolder(path, **shared_properties)`**
- Represents a source folder containing non-instability PyECLOUD simulations
- Recursively scans for folders containing complete simulation parameter files
- `path` - Root directory to scan for simulations
- `shared_properties` - Meta-attributes inherited by all discovered simulations
- Properties:
  - `sim_folders` - List of folders with complete simulation outputs
  - `run_folders` - List of in-progress simulation folders
- Methods:
  - `config_files_for(path)` - Returns dictionary of configuration file paths for a simulation

**`InstabilityDBFolder(path, **shared_properties)`**
- Variant for instability simulations (PyHEADTAIL outputs)
- Scans for `.h5` data files and pyecloud_config subdirectories
- Compatible with `SimDB` for unified database operations

**`SimDB(db_file, db_folders=[], verbose=True)`**
- Main database class for querying and managing simulation records
- `db_file` - Path to JSON database file (or TinyDB instance)
- `db_folders` - List of `DBFolder` instances to index
- Methods:
  - `where(**search)` - Query database with flexible filtering (see Querying section)
  - `by(attr, **search)` - Return sorted results by attribute
  - `unique(attr, **search)` - Get unique values of an attribute
  - `closest(entry, **search)` - Find entries matching most properties with `entry`
  - `extract(attrs, **search)` - Create sub-database with only specified attributes
  - `extract_array(attrs, **search)` - Return NumPy array plus search results
  - `versus(attrX, attrY, nonexistant=np.nan, **search)` - Extract paired (X, Y) data
  - `align(common_attrs, discriminator_attr, **search)` - Find common attribute values across discriminator groups
  - `all_keys()` - Return sorted list of all keys in database

**Querying with `where()`**

The `where()` method supports multiple query formats:

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

**`TemplateSim(path)`**
- Template-based simulation generator
- Loads existing simulation as a base for parameter variations
- Methods:
  - `spawn(path, **changes)` - Create new simulation folder with modified parameters
  - `generate_sweep(parameters, sweep, **constant)` - Generate parameter combinations for sweeping

#### Data Modeling

**`ECModel(db, doc_id_or_document)`**
- Represents a single PyECLOUD simulation analysis
- Lazily loads simulation data and auto-computes derived properties
- Properties (read/write):
  - `area` - Beam chamber cross-sectional area
  - `buildup` - Boolean: multipacting buildup detected
  - `bunch_step` - Time indices per bunch spacing
  - `bunch_times` - Time of each bunch maximum
  - `cutoff` - Last time index before simulation end
  - `half_bunch` - Gaussian bunch half-width
  - `magnet` - Magnet type (0: drift, 2: dipole, 4: quadrupole, 6: sextupole)
  - `mean_intensity` - Mean beam intensity
  - `Ne_0` - Initial electron count
  - `perimeter` - Beam chamber perimeter
- Data access properties (read-only):
  - `data` - Loaded .mat file dictionary
  - `time` - Time array from simulation
  - `N_electrons` - Electron count time series
  - `central_density` - Central density time series
  - `N_sec` - Secondary emission time series
  - `N_col` - Impact time series
  - `intensity` - Beam intensity time series
  - `smooth` - Smoothed electron count (convolution)
  - `smooth_diff` - Smoothed derivative of electron count
- Methods:
  - `time_to_index(t)` - Convert time(s) to data array indices
  - `gen_*` methods - Generators for computed properties (called automatically)

**`InstabilityModel(db, doc_id_or_document)`**
- Represents PyHEADTAIL instability simulation analysis
- Properties:
  - `data_files` - Map of HDF5 file contents and structure
  - `bunch_data_file` - Path to bunch evolution output
  - `slice_data_file` - Path to slice evolution output
- Methods:
  - `get_data(prop, filename=None, group=None)` - Extract data from HDF5 file

#### Curve Fitting

**`DataSelector`**
- Base class for selecting fitting data from an ECModel
- Methods:
  - `select(model)` - Returns (x_data, y_data) tuple

**`BeforeBunchSelector(use_central_density=False)`**
- Selects data points immediately before each bunch passage

**`BunchAverageSelector(use_central_density=False)`**
- Selects averaged data during each bunch passage

**`Fit(name, function, variables, selector)`**
- Base fitting class implementing scipy curve_fit
- `name` - Identifier for fit parameters in database
- `function` - String expression of fit function (uses `np`, `scipy`)
- `variables` - Tuple of variable names: (x, param1, param2, ...)
- Methods:
  - `bound(dict)` - Set bounds: `{"param": (lower, upper)}`
  - `initial(dict)` - Set initial guesses: `{"param": value}`
  - `fix(dict)` - Fix parameters: `{"param": value}`
  - `select_data(model)` - Get data for fitting
  - `scale_factor(model, xdata, ydata)` - Override for data scaling
  - `fit(model, refit=False)` - Perform fit, store results in model
  - `fit_function(model, params=None)` - Get callable fit function
  - `scrub(model)` - Remove fit parameters from model
- Properties:
  - `full_names` - Database entry names for parameters and errors

**`FurmanNoPhotoFit(selector=None)`**
- Furman model without photoemission: `(y0*yc*exp(beta*yc*x)) / (yc + y0*(exp(beta*yc*x)-1))`
- Parameters: `yc` (critical density), `beta` (multipacting), `y0` (initial, fixed)
- Automatically detects buildup and sets bounds/initial values

**`FurmanNPMCFit(db, selector=None)`**
- Furman model with magnetic correction: `(y0*yc*exp(yc*x*(b0+0.5*b1*x))) / (yc + y0*(exp(...)-1))`
- Parameters: `yc`, `b0`, `b1` (magnetic correction), `y0` (fixed)
- Requires database for baseline comparison

**`FurmanPhotoFit(selector=None)`**
- Furman full model with photoemission
- Parameters: `yc`, `alpha`, `beta`, `y0` (fixed)
- Only fits buildup simulations

**`fit_all_where(db, fit, refit=False, verbose=True, **search)`**
- Batch fit utility
- Applies fit to all results matching search criteria
- Returns ECModel instances with fit results stored in database

#### Utilities

**`smooth(array, window)`**
- Smooths NumPy array using convolution
- Preserves initial value and handles array boundaries

**`WhereIn(*items, str_match=True)`**
- Query helper for matching multiple values
- Handles type coercion (int vs float vs numpy types)

**`WhereNot(*items)`**
- Inverted WhereIn - matches values NOT in the set

**`SynchedEntry(db, doc_id, synch=[])`**
- Base class for auto-syncing object properties to database
- Marked attributes automatically update database on change
- Methods:
  - `set_sync(attr)` - Mark attribute for syncing
  - `ensure(attr, generator)` - Create attribute if missing
  - `detach()` / `attach()` - Pause/resume database synchronization

---

### ecagui.py - Graphical User Interface

Launch the GUI with:
```bash
python ecagui.py
```

#### Main Features

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
- Select fit model: FurmanNoPhoto, FurmanNPMC, or FurmanPhoto
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
- Interactive legend and matplotlib navigation

#### Keyboard/Mouse Usage
- **Right-click** on filter text area to copy/paste/clear filters
- **Click** simulation rows to select for individual plotting
- **Hold Ctrl/Cmd** to select multiple simulations for overlaid plots

---

### ecaplots.py - Visualization Utilities

**`model_plot(model, fits, size=(10, 5), log=(False, False), show_error=True, refit=False, central_density=False, fit_maxX=-1.0, label=None)`**
- Plot raw simulation data and fitted curves
- `model` - ECModel instance
- `fits` - Iterable of Fit instances to overlay
- `size` - Tuple (width, height) or Axes object
- `log` - Tuple (log_x, log_y) for axis scaling
- `show_error` - Show error bands for fit parameters
- `central_density` - Plot central density instead of total electron count
- `fit_maxX` - Maximum x-axis value (in bunch passages)
- `label` - Legend label prefix

**`versus_plot(db, paramA, paramB, colorBy=None, size=(8, 6), dot_size=10, log=(False, False), colormap=None, **restrict)`**
- Create scatter plot of paramB vs paramA from database
- `db` - SimDB instance
- `colorBy` - Optional third parameter for point coloring
- `restrict` - Query parameters (passed to db.where())
- Returns tuple of plotted arrays

**`histogram_plot(db, param, bins='auto', size=(8, 6), log_y=False, color='C0', **restrict)`**
- Plot histogram of parameter distribution
- `bins` - Number of bins or 'auto' for automatic binning
- `log_y` - Use log scale on y-axis
- `color` - Bar color
- `restrict` - Query parameters (passed to db.where())

---

### ecarun.py - Job Submission

#### Local Execution

**`RunLocal(job_folders, python="python", ecloud="./")`**
- Run PyECLOUD simulations on local machine
- `job_folders` - List of simulation directories to execute
- `python` - Python interpreter path
- `ecloud` - PyECLOUD root directory
- Methods:
  - `make_script(folder, output=None)` - Generate shell script for a simulation
  - `submit(verbose=True)` - Execute all simulations sequentially

#### SLURM Submission

**`RunSLURM(job_folders, python="python", ecloud="./", submission_name=None, remote_folder=None, username=None, hostname="localhost")`**
- Submit simulations via SLURM on remote machine
- Handles SSH connection, file transfer, and job collection
- `submission_name` - Custom name for submission batch
- `remote_folder` - Directory on remote for simulations
- `username` / `hostname` - SSH credentials
- Methods:
  - `submit(verbose=True)` - Upload jobs and submit to SLURM
  - `retrieve(verbose=True)` - Download completed simulations when ready
  - `compress(destination, folders)` - Static method to tar simulation folders

#### HTCondor Submission (lxplus.cern.ch)

**`RunCondor(job_folders, python="python", ecloud="./", submission_name=None, remote_folder=None, username=None, hostname="lxplus.cern.ch")`**
- Submit simulations via HTCondor on lxplus
- Inherits from RunSLURM (includes retrieve functionality)
- Default remote folder: `/afs/cern.ch/work/{username[0]}/{username}/`

**`RunCondorContainer(job_folders, container_path="...", submission_name=None, remote_folder=None, username=None, hostname="lxplus.cern.ch")`**
- Run PyECLOUD in Apptainer container on HTCondor
- Useful when PyECLOUD not installed on lxplus
- `container_path` - Full path to container image on CVMFS

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
```

### Fitting Simulations

```python
from eca import FurmanNoPhotoFit, fit_all_where

# Create fit model
fit = FurmanNoPhotoFit()

# Apply to all simulations with multipacting
fit_all_where(db, fit, buildup=True, verbose=True)

# Access results
for result in db.where(buildup=True, furman_np_success=True):
    model = ECModel(db, result)
    print(f"yc={model.furman_np_yc:.2e}, beta={model.furman_np_beta:.4f}")
```

### Plotting Results

```python
from ecaplots import versus_plot, model_plot
import matplotlib.pyplot as plt

# Plot relationship between parameters
versus_plot(db, "del_max", "Ne_0", colorBy="buildup", size=(10, 6))
plt.show()

# Plot individual simulation with fit
model = ECModel(db, db.where(del_max=0.5)[0])
fits = [FurmanNoPhotoFit()]
model_plot(model, fits, central_density=False, fit_maxX=100)
plt.show()
```

### Running Simulations

```python
from ecarun import RunLocal, RunSLURM

# Run locally
runner = RunLocal(["/sim1", "/sim2", "/sim3"], ecloud="/opt/pyecloud")
runner.submit()

# Submit to SLURM
runner = RunSLURM(
    ["/sim1", "/sim2"],
    python="/usr/bin/python3",
    ecloud="/home/user/pyecloud",
    hostname="compute.example.com"
)
runner.submit()
runner.retrieve()  # Download when done
```

---

## Advanced Topics

### Creating Custom Fit Models

```python
from eca import Fit, BunchAverageSelector

class CustomFit(Fit):
    EXPRESSION = "a * x + b"
    
    def __init__(self):
        super().__init__(
            "custom",
            self.EXPRESSION,
            ("x", "a", "b"),
            BunchAverageSelector()
        )
    
    def fit(self, model, refit=False):
        # Custom initialization
        self.initial({"a": 1.0, "b": 0.0})
        self.bound({"a": (0, 10), "b": (-10, 10)})
        return super().fit(model, refit)
```

### Database Alignment and Comparison

```python
# Find common parameter values across simulation groups
common_vals, dbs_by_magnet = db.align(
    common_attrs=["del_max", "k_pe_st"],
    discriminator_attr="magnet"
)

for magnet_type, sub_db in zip(common_vals, dbs_by_magnet):
    print(f"Magnet type {magnet_type}: {len(sub_db.where())} matching sims")
```

### Parameter Sweeps

```python
from eca import TemplateSim

template = TemplateSim("/path/to/template")

# Generate sweep over SEY and beam intensity
configs = TemplateSim.generate_sweep(
    parameters=["del_max", "fact_beam"],
    sweep=[[0.1, 0.5, 1.0], [1e11, 1e12, 1e13]],
    filling_pattern_file="pattern.txt"
)

for i, config in enumerate(configs):
    output_dir = f"/simulations/sweep_{i:03d}"
    template.spawn(output_dir, **config)
```

---

## References

- PyECLOUD: Electron cloud simulation framework
- TinyDB: Lightweight document database used internally
- Matplotlib: Visualization library for plots
- SciPy: Scientific computing with curve_fit optimization

## Authors

**Adam Furman**  
adam.furman@cern.ch

For use with PyECLOUD at CERN.