
# Electron Cloud Analysis (EcA)

**Electron Cloud Analysis (EcA)** is a Python package designed for analyzing and visualizing data from PyECLOUD simulations. It provides tools for managing simulation data, fitting models, and plotting results.

## Description

EcA is a toolkit for working with PyECLOUD simulation data. It allows users to:

- **Build and manage a database** of PyECLOUD simulations.
- **Query and analyze** simulation data.
- **Fit models** to simulation data.
- **Visualize** simulation results and model fits.
- **Submit** new simulations to computing clusters.

The package is structured into three modules:

- `eca.py`: Core functionality for managing simulation data and fitting models.
- `ecaplots.py`: Functions for plotting simulation data and model fits.
- `ecarun.py`: Tools for running new simulations on computing clusters.
---

## Installation

To install EcA, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd eca
pip install -r requirements.txt
```

**Dependencies:**

- `numpy`
- `scipy`
- `matplotlib`
- `tinydb`

---

## Usage

### Building a Database

To build a database of PyECLOUD simulations, use the `DBFolder` and `SimDB` classes:

```python
from eca import DBFolder, SimDB

# Create a DBFolder to scan for simulation data
db_folder = DBFolder(path="/path/to/simulations")

# Create a SimDB instance
sim_db = SimDB(db_file="simulations_db.json", db_folders=[db_folder])
```

### Querying the Database

You can query the database using the `SimDB` class:

```python
# Query for simulations with specific properties
results = sim_db.where(property=value)
results = sim_db.where(property=lambda value, result: value > result["other_property"]
results = sim_db.where(_=lambda result: filter_fn(result)
```

### Fitting Models

To fit models to simulation data, use one of the `Fit` subclasses.
```python
fit_np = eca.FurmanNoPhotoFit()
fit_p = eca.FurmanPhotoFit()
```

### Plotting Results

To visualize simulation data and model fits, use the `model_plot` function:

```python
from ecaplots import model_plot
model_plot(model=model, fits=[fit], size=(10, 5), log=(False, False), show_error=True)
```

For a visualization of variable `x` vs `y` (optionally colored by `c`), use the `versus_plot` function:

```python
from ecaplots import versus_plot
ecp.versus_plot(db, "<x>", "<y>", "<c>", size=(10, 5), **args_to_where)
```
The `**args_to_where` are passed on to `db.where`, which can be used to filter results.

### Creating New Simulations

Creating new simulations requires one simulation as a "template". Unless explicitly changed, the variables
in your new simulation will mirror the values of the template.

```python
template = eca.TemplateSim("./configurations/new_template")
```

The template can be used to generate a parameter sweep of one or several variables:
```python
# Sweep in B, SEY, Intensity
fact_beam_vect = [2.150e+10, 6.020e+10, 9.890e+10, 1.376e+11, 1.763e+11, 2.150e+11, 2.365e11]
del_max_vect = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
B_multip_vect = [[round(0.005 + 0.005*b, 3)] for b in range(9)]
work_dir = "./pyecloud_jobs"

confs = template.generate_sweep(
    ["fact_beam", "del_max", "B_multip"],
    [fact_beam_vect, del_max_vect, B_multip_vect])
```

`confs` is a list of dictionaries that modify the default template values.

To spawn an actual simulation, use the `template.spawn` method:
```python
folders = []
for configuration in confs:
    f = work_dir + "B_{B_multip[0]}_I_{fact_beam:.2E}_S_{del_max:.2f}".format(**c)
    template.spawn(f, **c)
    folders.append(f)
```

*Note that the folder names for each simulation should be unique!*

Submit a list of simulation folders using a simulation `Runner`.
```python
runner = ecarun.RunLocal(folders, python=executable, ecloud="./")
runner = ecarun.RunSLURM(folders, python="<python executable>", ecloud="<PyECLOUD directory>", submission_name="<name>", username="<login>", hostname="<slurm host>")
runner = ecarun.RunCondor(folders, python="<python executable>", ecloud="<PyECLOUD directory>", submission_name="<name>", username="<login>", hostname="lxplus.cern.ch")
```

You can then submit the folders using `runner.submit()` and retrieve finished simulations using `runner.retrieve()`.

*Submitting and retrieving will execute shell commands on a remote machine. Ensure you understand the code before running at your own risk.*

It is recommended that you have password-free (e.g. SSH key) access to the `hostname` server.
