
# Electron Cloud Analysis (EcA)

**Electron Cloud Analysis (EcA)** is a Python package designed for analyzing and visualizing data from PyECLOUD simulations. It provides tools for managing simulation data, fitting models, and plotting results.

## Table of Contents

- [Description](#description)
- [Installation](#installation)
- [Usage](#usage)
- [Public Functions](#public-functions)
- 
---

## Description

EcA is a toolkit for working with PyECLOUD simulation data. It allows users to:

- **Build and manage a database** of PyECLOUD simulations.
- **Query and analyze** simulation data.
- **Fit models** to simulation data.
- **Visualize** simulation results and model fits.

The package is structured into two main modules:

- `eca.py`: Core functionality for managing simulation data and fitting models.
- `ecaplots.py`: Functions for plotting simulation data and model fits.

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

To fit models to simulation data, use one of the `Fit` subclasses:

### Plotting Results

To visualize simulation data and model fits, use the `model_plot` function:

```python
from ecaplots import model_plot

# Plot the model and fits
model_plot(model=model, fits=[fit], size=(10, 5), log=(False, False), show_error=True)
```

---

### `eca.py`

#### `DBFolder`

- **Description**: A source folder for building a database. Recursively searches the path for folders containing all required files corresponding to a finished simulation.
- **Parameters**:
  - `path`: The path to the folder to scan.
  - `**shared_properties`: Meta-attributes that all simulations found within this folder will inherit.

#### `SimDB`

- **Description**: A database of PyECLOUD simulations that can be searched and queried.
- **Parameters**:
  - `db_file`: The file path or `TinyDB` instance to use for the database.
  - `db_folders`: A list of `DBFolder` instances to add to the database.
  - `verbose`: If `True`, prints progress information.

### `ecaplots.py`

#### `model_plot`

- **Description**: Plots the raw data and each fit if it succeeded.
- **Parameters**:
  - `model`: The model to plot.
  - `fits`: An iterable of `Fit` instances to plot.
  - `size`: The size of the plot or an `Axes` object.
  - `log`: A tuple indicating whether to use a logarithmic scale for the x and y axes.
  - `show_error`: If `True`, shows the error range for each fit.
  - `refit`: If `True`, refits the model before plotting.
