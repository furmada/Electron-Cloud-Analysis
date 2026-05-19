"""
EcA - Electron Cloud Analysis v1
@author Adam Furman
@email adam.furman@cern.ch

For use with PyECLOUD.
"""
from typing import Callable, Iterable
from functools import partial as functools_partial
from itertools import product

import numpy as np
import scipy
from tinydb import JSONStorage, TinyDB, Query
from tinydb.storages import MemoryStorage, JSONStorage
from tinydb.table import Document
from tinydb.middlewares import CachingMiddleware
from shutil import copy2
from glob import glob

import os

def unique_with_tolerance(arr, tol):
    arr = np.sort(arr)
    # Find indices where the difference between consecutive elements exceeds tolerance
    split_indices = np.where(np.diff(arr) > tol)[0] + 1
    # Split the sorted array at those indices and take the first element of each group
    return np.array([group[0] for group in np.split(arr, split_indices)])

class DBFolder(object):
    """
    A source folder for building a database, containing non-instability simulations.
    Recursively searches the path for folders containing all FILENAMES
    corresponding to a finished simulation.
    -> shared_properties are meta-attributes that all simulations found
        within this DBFolder will inherit.
    """
    FILENAMES: dict[str, str] = {
        "data":       "Pyecltest.mat",
        "sey":        "secondary_emission_parameters.input",
        "machine":    "machine_parameters.input",
        "simulation": "simulation_parameters.input",
        "beam":       "beam.beam"
    }

    def __init__(self, path: str, **shared_properties):
        self.path = path
        self.shared_properties = shared_properties
        self._scan()

    def _scan(self):
        self.sim_folders = set()
        self.run_folders = set()
        for r, _, files in os.walk(self.path):
            count = sum([(1 if f in files else 0) for f in DBFolder.FILENAMES.values()])
            if count == len(DBFolder.FILENAMES):
                self.sim_folders.add(r)
            elif count == len(DBFolder.FILENAMES) - 1:
                self.run_folders.add(r)
        self.sim_folders = list(sorted(self.sim_folders))
        self.run_folders = list(sorted(self.run_folders))

    def config_files_for(self, path: str) -> dict:
        """
        Return a dictionary of configuration filepaths for a simulation folder.
        """
        return {k: os.path.join(path, f) for k, f in DBFolder.FILENAMES.items()}

class InstabilityDBFolder(DBFolder):
    """
    A source folder for building a database, contianing instability simulations.
    Recursively searches the path for folders containing all FILENAMES
    corresponding to a finished simulation.
    -> shared_properties are meta-attributes that all simulations found
        within this DBFolder will inherit.
    """
    CONFIG_FOLDER = "pyecloud_config"
    FILENAMES: dict[str, str] = {
        "sey":        "secondary_emission_parameters.input",
        "machine":    "machine_parameters.input",
        "simulation": "simulation_parameters.input"
    }
    INSTAB_CONF_FILE = "Simulation_parameters.py"

    def __init__(self, path: str, **shared_properties):
        super().__init__(path, **shared_properties)

    def _scan(self):
        self.sim_folders = set()
        self.run_folders = set()
        for r, folders, files in os.walk(self.path):
            if InstabilityDBFolder.CONFIG_FOLDER in folders:
                count = sum(
                    [(1 if os.path.exists(os.path.join(r, InstabilityDBFolder.CONFIG_FOLDER, f)) else 0)
                    for f in InstabilityDBFolder.FILENAMES.values()])
                if count == len(InstabilityDBFolder.FILENAMES):
                    if sum([1 if f.endswith(".h5") else 0 for f in files]) > 0:
                        self.sim_folders.add(r)
                    else:
                        self.run_folders.add(r)
        self.sim_folders = list(sorted(self.sim_folders))
        self.run_folders = list(sorted(self.run_folders))

    def config_files_for(self, path: str) -> dict:
        """
        Return a dictionary of configuration filepaths for a simulation folder.
        """
        base = {k: os.path.join(path, InstabilityDBFolder.CONFIG_FOLDER, f) for k, f in InstabilityDBFolder.FILENAMES.items()}
        base["instability"] = os.path.join(path, InstabilityDBFolder.INSTAB_CONF_FILE)
        return base

class TemplateSim(object):
    """
    A single (existing) simulation folder that can be used as a template
    to generate new simulations by changing parameters.
    """
    def __init__(self, path: str):
        self.path = os.path.normpath(os.path.abspath(path))
        self.property_map = {}
        self.defaults = {}
        for filename in DBFolder.FILENAMES.values():
            if filename != DBFolder.FILENAMES["data"]:
                try:
                    for param, value in SimDB.parse_input_file(os.path.join(self.path, filename)).items():
                        self.property_map[param] = filename
                        self.defaults[param] = value
                except:
                    raise ValueError("The configuration file {} does not exist in {}, or it is malformed.".format(filename, self.path))
        self.resources = []
        for item in os.listdir(self.path):
            if os.path.isfile(os.path.join(self.path, item)) and not item in DBFolder.FILENAMES.values():
                self.resources.append(item)
    
    def spawn(self, path: str, **changes) -> str:
        """Spawn a new simulation based on the template, with the changes applied."""
        path = os.path.normpath(os.path.abspath(path))
        if os.path.exists(path) and os.path.exists(os.path.join(path, DBFolder.FILENAMES["data"])):
            raise ValueError("The desired destination {} already contains an output data file!".format(path))
        if not os.path.exists(path):
            os.mkdir(path)

        values = {**self.defaults, **changes}
        for filename in DBFolder.FILENAMES.values():
            if filename != DBFolder.FILENAMES["data"]:
                this_file = [prop for prop in values.keys() if prop in self.property_map and self.property_map[prop] == filename]
                contents = "#PyECLOUD Configuration from Template {}\n\n".format(self.path)
                for prop in sorted(this_file):
                    value = values[prop]
                    if type(value) == str and os.path.exists(value) and os.path.isfile(value):
                        value = os.path.normpath(os.path.abspath(value))
                        if os.path.commonpath((value, self.path)) != self.path:
                            # This is a filepath argument within the template. We also need this file.
                            rebase = os.path.join(path, os.path.basename(value))
                            copy2(value, rebase)
                            value = os.path.basename(rebase)
                    if prop == "logfile_path": value = "log.txt"
                    elif prop == "progress_path": value = "progress"
                    elif prop == "stopfile": value = "stop"
                    contents += "{} = {}\n".format(prop, ("\"" + value + "\"" if type(value) == str else value) if not isinstance(value, np.ndarray) else value.tolist())
                with open(os.path.join(path, filename), "w") as f:
                    f.write(contents)
        for resource in self.resources:
            copy2(os.path.join(self.path, resource), os.path.join(path, resource))
        return path

    @staticmethod
    def generate_sweep(parameters: Iterable, sweep: Iterable, **constant) -> list[dict]:
        """
        Generate one set of changes for every combination of values for all parameters to be swept.
        Supports two modes of specification:
        1. Independent parameters:
           parameters=['A', 'B']
           sweep=[[1, 2], ['x', 'y']]
           Result: A=1,B=x; A=1,B=y; A=2,B=x; A=2,B=y
        2. Linked (co-varying) parameters:
           parameters=[('A', 'B')]  # Tuple indicates they vary together
           sweep=[[(1, 'a'), (2, 'b')]] # Values are tuples matching the parameter tuple order
           Result: A=1,B=a; A=2,B=b
        Mixed usage is supported:
           parameters=['C', ('A', 'B')]
           sweep=[[10, 20], [(1, 'a'), (2, 'b')]]
           Result: C=10,A=1,B=a; C=10,A=2,B=b; C=20,A=1,B=a; C=20,A=2,B=b
        """
        param_list = list(parameters)
        sweep_list = list(sweep)
        if len(param_list) != len(sweep_list):
            raise ValueError("The number of parameters must match the number of sweep value lists.")
        dimensions = []
        for i, param in enumerate(param_list):
            values = list(sweep_list[i])
            if isinstance(param, tuple):
                # Linked parameters case
                if not all(isinstance(v, tuple) and len(v) == len(param) for v in values):
                    raise ValueError(
                        f"For linked parameters {param}, all values must be tuples of the same length."
                    )
                # Create a dimension where each item is a dict mapping param names to values
                dim = []
                for val_tuple in values:
                    dim.append(dict(zip(param, val_tuple)))
                dimensions.append(dim)
            else:
                # Independent parameter case
                # Create a dimension where each item is a dict with one key
                dim = [{param: v} for v in values]
                dimensions.append(dim)
        # Calculate total combinations
        total_combinations = np.prod([len(d) for d in dimensions])
        configurations = []
        # Use itertools.product to generate the Cartesian product of the dimensions
        # Each item in 'combo' is a tuple of dicts, e.g. ({'C': 10}, {'A': 1, 'B': 'a'})
        for combo in product(*dimensions):
            # Merge all dicts in the tuple into a single config dict
            merged_config = {}
            for d in combo:
                merged_config.update(d)
            # Add constant parameters
            final_config = {**constant, **merged_config}
            configurations.append(final_config)
        return configurations                

class SimDB(object):
    """
    A database of PyECLOUD simulations that can be searched and queried.
    """
    def __init__(self, db_file: str | TinyDB, db_folders=[], verbose=True):
        self.db = db_file if isinstance(db_file, TinyDB) else TinyDB(db_file, storage=CachingMiddleware(JSONStorage))
        self.verbose = verbose
        to_add = []
        for db_folder in db_folders:
            if not isinstance(db_folder, DBFolder):
                db_folder = DBFolder(db_folder)
            # Add all simulations to the database
            existing_paths = set(self.unique("path"))
            for sim_folder in db_folder.sim_folders:
                if not sim_folder in existing_paths:
                    # This is a new simulation, add it
                    to_add.append((sim_folder, db_folder))
        if len(to_add) > 0:
            if verbose: print("Found {} simulations to add to {}.".format(len(to_add), db_file))
            for i, entry in enumerate(to_add):
                if verbose: print("{:<3}/{}".format(i, len(to_add)), end="\r")
                self._add_single(verbose, *entry)
            if verbose: print("Done.      ")
            if isinstance(self.db.storage, CachingMiddleware):
                self.db.storage.flush()

    @staticmethod
    def parse_input_file(path: str) -> dict:
        """Read one of the PyECLOUD input files (python scripts)."""
        params = {}
        with open(path, "r") as f:
            exec(f.read().replace("import numpy as np", ""), {"np": np}, params)
        for k, v in params.items():
            if isinstance(v, np.ndarray):
                params[k] = v.tolist()
        return params

    @staticmethod
    def prepare(value):
        """Clean a value for storage in the db."""
        if isinstance(value, np.ndarray):
            # Cannot serialize ndarrays, so make it a list.
            return value.tolist()
        if isinstance(value, np.number):
            # Cannot serialize numpy numbers, so check and make it float or int
            return int(value) if isinstance(value, np.integer) else (complex(value) if np.iscomplex(value) else float(value))
        if isinstance(value, (bool, np.bool_)):
            # The numpy bool type is also not supported
            return bool(value)
        return value

    def _add_single(self, verbose: bool, sim_folder: str, db_folder: DBFolder) -> bool:
        """
        Add a single simulation to the database.
        Returns True if it was added, and False
        if simulation with identical parameters already exists.
        """
        parameters = {**db_folder.shared_properties}
        # Read all input parameter files
        for name, path in db_folder.config_files_for(sim_folder).items():
            if name != "data":
                parameters.update(**SimDB.parse_input_file(path))
        # Check if a simulation with these parameters exists
        search = self.db.search(Query().fragment(parameters))
        if len(search) == 0:
            # No existing entry with these parameters
            parameters["processed"] = False # Mark data file as un-processed
            parameters["path"] = sim_folder
            self.db.insert(parameters)
            return True
        elif verbose:
            print("Duplicate of {} found at {}".format(sim_folder, search[0]["path"]))
            for p, v in parameters.items():
                print(p, v, search[0][p])
            raise ValueError()
        return False

    def where(self, **search) -> list:
        """
        Search the database, returning an array of all entries matching the given values.
        The following are valid ways of providing search arguments:
        1. parameter = [ static value ] : matches a single value of a parameter
        2. parameter = f: Callable(value, result) : matches if f(value, result) is True
        3. _[any] = g: Callable(result) or str : matches if g(result) is True if g is a
                        Callable, or else compiles str and evaluates it, with local
                        variables initialized to the result.
        """
        if len(search) == 0:
            return self.db.all()
        # Searching by document ID requires special handling
        if "doc_id" in search:
            if isinstance(search["doc_id"], WhereIn):
                return [self.db.get(doc_id=int(i)) for i in search["doc_id"].match]
            elif isinstance(search["doc_id"], Callable):
                return [r for r in self.db.all() if search["doc_id"](r.doc_id, r)]
            return [self.db.get(doc_id=int(search["doc_id"]))]
        # We also support direct TinyDB queries
        if "query" in search and isinstance(search["query"], Query):
            return self.db.search(search["query"])
        non_dynamic = {}
        dynamic = {}
        for k, v in search.items():
            if type(v) == str and k.startswith("_"):
                comp = compile(v, "query_expr", "eval")
                context = {"np": np, "scipy": scipy, "ECModel": ECModel, "db": self}
                dynamic[k] = lambda r: eval(comp, context, {rk: rv for rk, rv in r.items()})
            elif isinstance(v, Callable):
                dynamic[k] = v
            elif isinstance(v, np.ndarray):
                non_dynamic[k] = v.tolist()
            elif isinstance(v, WhereIn):
                dynamic[k] = v.query
            else:
                non_dynamic[k] = v
        results = self.db.search(Query().fragment(non_dynamic))
        if len(dynamic) > 0:
            filtered_results = []
            for result in results:
                all_filters = True
                for k, filter_fn in dynamic.items():
                    try:
                        if not (filter_fn(result[k], result) if (k in result) else (filter_fn(result) if k.startswith("_") else False)):
                            all_filters = False
                            break
                    except NameError:
                        # Happens if we have a dynamic query and a result lacks a property
                        all_filters = False
                        break
                if all_filters:
                    filtered_results.append(result)
            return filtered_results
        return results

    def complementary(self, **search) -> dict:
        """
        Returns the query that will select all entries in the DB *not* selected by search.
        So, where(complementary(where(A == 1))) returns all entries where A != 1.
        """
        result_ids = set([r["doc_id"] for r in self.where(**search)])
        all_ids = [doc.doc_id for doc in self.db.all()]
        complement_ids = [i for i in all_ids if not i in result_ids]
        return {"doc_id": WhereIn(*complement_ids, str_match=False)}

    def by(self, attr: str, **search) -> list:
        """
        Like "where", but additionally sort the results by the values of the property "attr".
        """
        result = self.where(**search)
        return list(sorted(result, key=lambda r: r[attr]))

    def unique(self, attr: str, **search) -> list:
        """
        Returns the unique values of the property "attr" for the query (to "where").
        """
        result = self.where(**search)
        values = [r[attr] for r in result if attr in r]
        has_iterables = False
        for v in values:
            if type(v) != str and isinstance(v, Iterable):
                has_iterables = True
                break
        if has_iterables:
            unique = []
            for v in values:
                if not v in unique:
                    unique.append(v)
            return unique
        else: 
            return list(sorted(set(values)))

    def closest(self, entry: dict, **search) -> list[Document]:
        """
        Finds the entries in the database which matches the greatest number of properties with "entry".
        """
        all_entries = self.where(**search)
        if len(all_entries) == 0:
            return []
        matching = np.zeros(len(all_entries), dtype=int)
        match_on = list(entry.keys())
        for i, e in enumerate(all_entries):
            matching[i] = sum([1 for k in match_on if k in e and e[k] == entry[k]])
        return [all_entries[int(i)] for i in np.argwhere(matching == np.min(matching)).ravel().astype(int)]

    def extract(self, attrs: Iterable[str], **search) -> SimDB:
        """
        Create a sub-database, extracting the properties "attrs" from documents matching the query.
        Ignores missing properties.
        """
        if not isinstance(attrs, tuple):
            attrs = tuple(attrs)
        sdb = TinyDB(storage=MemoryStorage)
        for result in self.where(**search):
            sdb.insert({k: result[k] for k in attrs if k in result})
        return SimDB(sdb, [], self.verbose)

    def extract_array(self, attrs: Iterable[str], **search) -> tuple[np.ndarray, list]:
        """
        Like extract, but returns a ndarray as well as the search results.
        Throws an error if a result has any of the attrs missing, ensuring a non-ragged array
        """
        if not isinstance(attrs, tuple):
            attrs = tuple(attrs)
        results = self.where(**search)
        return np.array([[result[a] for a in attrs] for result in results]).T, results
    
    def versus(self, attrX: str | Iterable[str], attrY: str | Iterable[str], nonexistant: np.number | float | int = np.nan, **search) -> np.ndarray:
        """
        Extracts the paired data [X, Y] for each attrX, attrY given.
        If a result lacks the attribute, fill with "nonexistant", defaulting to NaN.
        The paired data will be sorted by each X.
        """
        attrX = list(attrX)
        attrY = list(attrY)
        results = self.where(**search)
        arr = np.zeros((len(attrX), 2, len(results)))
        for i, x, y in zip(range(len(attrX)), attrX, attrY):
            vals = np.array([(r[x] if x in r else nonexistant, r[y] if y in r else nonexistant) for r in results]).T
            arr[i,:,:] = vals[:,np.argsort(vals[0])]
        if arr.shape[0] == 1:
            return arr[0]
        return arr

    def align(self, common_attrs: Iterable[str], discriminator_attr: str, **search) -> tuple[list, list[SimDB]]:
        """
        Under the conditions of the search, find the values of each common_attr that are shared
        amongst all unique values of the discriminator_attr. Return a SimDB for each value
        of discriminator_attr, with results restricted to the common_attr and search.
        """
        common_values = [set() for _ in common_attrs]
        disc_values = self.unique(discriminator_attr, **search)
        for disc_value in disc_values:
            for a, attr in enumerate(common_attrs):
                attr_values = self.unique(attr, **{discriminator_attr: disc_value, **search})
                if len(common_values[a]) == 0:
                    common_values[a] = set(attr_values)
                else:
                    common_values[a] = common_values[a].intersection(set(attr_values))
        disc_dbs = []
        for disc_value in disc_values:
            disc_db = TinyDB(storage=MemoryStorage)
            disc_db.insert_multiple(self.db.search(
                lambda doc: doc.get(discriminator_attr) == disc_value and sum([0 if doc.get(attr) in common_values[a] else 1 for a, attr in enumerate(common_attrs)]) == 0))
            disc_dbs.append(SimDB(disc_db, [], self.verbose))
        return disc_values, disc_dbs

    def all_keys(self) -> list[str]:
        """
        Returns a sorted list of all keys present in the database.
        """
        keys = set()
        for result in self.where():
            for k in result.keys():
                keys.add(k)
        return list(sorted(keys))

# In eca.py, replace the WhereIn class with this improved version:

class WhereIn(object):
    """
    Addon to db.where functionality enabling matching to any one of a set of items.
    Now handles type coercion for better matching across different numeric types.
    """
    def __init__(self, *items, str_match=True):
        self.match = [tuple(i) if isinstance(i, list) or isinstance(i, np.ndarray) else i for i in items]
        if str_match:
            self.match_str = {str(item) for item in items}
    
    def query(self, value, _) -> bool:
        # First try exact match
        if isinstance(value, list) or isinstance(value, np.ndarray):
            value = tuple(value)
        if value in self.match:
            return True
        # Try numeric equivalence (handles int vs float vs numpy types)
        try:
            if isinstance(value, (int, float, np.number)):
                for match_val in self.match:
                    if isinstance(match_val, (int, float, np.number)):
                        if float(value) == float(match_val):
                            return True
        except (ValueError, TypeError):
            pass
        # Try string representation match
        if hasattr(self, "match_str") and str(value) in self.match_str:
            return True
        return False

class WhereNot(WhereIn):
    """
    Inverts the query.
    """
    def query(self, value, _) -> bool:
        return not super().query(value, _)


class SynchedEntry(object):
    """
    A SynchedEntry provides a way to automatically save data back to a TinyDB
    as it is changed by a Python object.
    Starting with an initial db entry (identified by a doc_id), Python properties
    can be marked to sync, meaning that changes to them will automatically be
    propagated to the database.
    """
    def __init__(self, db: TinyDB, doc_id: int, synch: Iterable[str] = []):
        # 1. Initialize core attributes using super() to bypass custom __setattr__
        super().__setattr__("db", db)
        super().__setattr__("id", doc_id)
        super().__setattr__("attached", True)
        super().__setattr__("_changes", {})
        super().__setattr__("_synced_attrs", set())
        for attr in synch:
            self.set_sync(attr)
    
    def detach(self):
        """Detach this object from the database, accumulating changes."""
        if self.attached:
            self.attached = False
            self._changes = {}

    def attach(self):
        """Reattach this object to the database, synchronizing the accumulated state."""
        if not self.attached:
            self.attached = True
            self.db.update({
                k: SimDB.prepare(v) for k, v in self._changes.items() 
                if not hasattr(self, "_" + k)
            }, doc_ids=[self.id])
            self._changes.clear()

    def _sync_get(self, attr: str):
        """Retrieve the property. Overridden by child classes (like ECModel) to generate if missing."""
        try:
            return getattr(self, "_" + attr)
        except AttributeError:
            raise AttributeError(f"'{self.__class__.__name__}' has no synced attribute '{attr}'")

    def _sync_set(self, attr: str, value):
        """Set the property, and update its value in the database."""
        if hasattr(self, "_" + attr):
            current_val = getattr(self, "_" + attr)
            if isinstance(value, np.ndarray):
                if isinstance(current_val, np.ndarray) and current_val.shape == value.shape and np.all(current_val == value):
                    return
            elif current_val == value:
                return
        super().__setattr__("_" + attr, value)
        if self.attached:
            self.db.update({attr: SimDB.prepare(value)}, doc_ids=[self.id])
        else:
            self._changes[attr] = value

    def _sync_delete(self, attr: str):
        """Delete the property and its corresponding value in the database."""
        if hasattr(self, "_" + attr):
            super().__delattr__("_" + attr)
        if self.attached:
            result = self.db.get(doc_id=self.id)
            if not result is None:
                self.db.update({k: v for k, v in result.items() if k != attr}, doc_ids=[self.id])
        else:
            self._changes.pop(attr, None)

    def set_sync(self, attr: str):
        """Register an attribute to be intercepted and synced."""
        self._synced_attrs.add(attr)

    def ensure(self, attr: str, gen: Callable) -> bool:
        """Ensure that the attribute exists and is synced, calling gen() to make it if it does not."""
        self.set_sync(attr)
        if not hasattr(self, "_" + attr):
            self._sync_set(attr, gen())
            return True
        return False

    def __getattr__(self, name: str):
        """Intercepts attribute access only if the attribute is NOT found normally."""
        if "_synced_attrs" in self.__dict__ and name in self._synced_attrs:
            return self._sync_get(name)
        return object.__getattribute__(self, name)

    def __setattr__(self, name: str, value):
        """Intercepts ALL attribute assignments."""
        if "_synced_attrs" in self.__dict__ and name in self._synced_attrs:
            self._sync_set(name, value)
        else:
            super().__setattr__(name, value)

    def __delattr__(self, name: str):
        """Intercepts ALL attribute deletions."""
        if "_synced_attrs" in self.__dict__ and name in self._synced_attrs:
            self._sync_delete(name)
        else:
            super().__delattr__(name)

def smooth(array: np.ndarray, window: int | np.integer) -> np.ndarray:
    """Smooths a NumPy array using convolution, preserving the initial value."""
    smoothed = np.zeros_like(array)
    if window > array.size / 2: window = array.size // 10
    for i in range(window):
        smoothed[i] = np.mean(array[:i+1])*(1 - (i/window)) + np.mean(array[i+1:i+1+window])*(i/window)
    try:
        smoothed[window:-window+1] = np.convolve(array[window:], np.ones(window) / window, "valid")
    except ValueError:
        return array
    for i in range(len(smoothed)-window+1, len(smoothed)):
        smoothed[i] = np.mean(array[i-window//2:])
    return smoothed

class ECModel(SynchedEntry):
    """
    An ECModel corresponds to the analysis of one PyECloud data file.
    """
    PROPERTIES: set[str] = {
        "area",             # The area of the beam chamber
        "buildup",          # True if there is multipacting buildup of EC
        "bunch_step",       # The number of indexes corresponding to b_spac
        "bunch_times",      # Times of each bunch maximum (t_off + n * b_spac)
        "cutoff",           # Index into data arrays corresponding to the last bunch passage
        "half_bunch",       # True (cut-off) half-width of bunch
        "magnet",           # The type of magnetic section (0: drift, 2: dipole, 4: quadrupole, 6: sextupole)
        "mean_intensity",   # The mean beam intensity
        "Ne_0",             # Initial number of electrons
        "perimeter",        # The perimeter of the beam chamber
        "train_times"      # Times of the start of each bunch train
    }

    def __init__(self, db: TinyDB | SimDB, doc: int | Document):
        result = doc if isinstance(doc, Document) else (db if isinstance(db, TinyDB) else db.db).get(doc_id=doc)
        if result is None: raise KeyError("The provided doc_id={} does not exist in the database!".format(doc))
        super().__init__(db if isinstance(db, TinyDB) else db.db, result.doc_id, ECModel.PROPERTIES)
        self.doc_id = result.doc_id
        # Populate this object with information from the database entry
        for k, v in result.items():
            self.set_sync(k)
            if isinstance(v, list) and len(v) > 0 and not type(v[0]) == str:
                setattr(self, "_"+k, np.array(v, dtype=type(v[0])))
            else:
                setattr(self, "_"+k, v)

    @property
    def data(self):
        if not hasattr(self, "_data"):
            self._data = scipy.io.loadmat(os.path.join(self.path, DBFolder.FILENAMES["data"]))
        return self._data

    @property
    def time(self):
        return self.data["t"].ravel()

    @property
    def N_electrons(self):
        return self.data["Nel_timep"].ravel()

    @property
    def central_density(self):
        return self.data["cen_density"].ravel()

    @property
    def N_sec(self):
        return self.data["Nel_emit_time"].ravel()

    @property
    def N_col(self):
        return self.data["Nel_imp_time"].ravel()

    @property
    def intensity(self):
        return self.data["lam_t_array"].ravel()

    @property
    def smooth(self):
        if not hasattr(self, "_smooth"):
            conv_window = int(2*self.bunch_step)
            self._smooth = smooth(self.N_electrons, conv_window)
        return self._smooth

    @property
    def smooth_diff(self):
        if not hasattr(self, "_smooth_diff"):
            conv_window = int(2*self.bunch_step)
            self._smooth_diff = smooth(np.diff(self.smooth, n=1), conv_window)
        return self._smooth_diff

    def _sync_get(self, attr: str):
        if not hasattr(self, "_"+attr):
            if hasattr(self, "gen_"+attr):
                # Call the attribute's generator
                getattr(self, "gen_"+attr)()
            else:
                raise ValueError("Requested synced property {} for which no generator is defined!".format(attr))
        return super()._sync_get(attr)

    def time_to_index(self, t: np.ndarray | np.number | float) -> np.ndarray | np.integer:
        """
        Convert time(s) to corresponding index(es) in the time array (floor-wise).
        """
        return np.floor(t / (self.time[1] - self.time[0])).astype(int)

    def train_to_index(self, train: int | np.integer) -> np.ndarray:
        """
        Convert a train number to its corresponding time indexes in the time array.
        """
        if train < 0:
            train = len(self.train_times) + train
        return np.arange(int(self.time_to_index(self.train_times[train])), int(self.time_to_index(self.train_times[train+1]) if train < len(self.train_times) - 1 else self.cutoff))

    def train_to_bunches(self, train: int | np.integer) -> np.ndarray:
        """
        Convert a train number to its corresponding bunch passage indexes.
        """
        train_lengths = np.diff(self.train_times, append=self.time[self.cutoff])
        return np.argwhere(np.bitwise_and(self.bunch_times >= self.train_times[train], self.bunch_times < self.train_times[train] + train_lengths[train])).ravel().astype(int)

    def gen_area(self):
        """
        Calculates the area of the beam chamber.
        """
        self.gen_perimeter()

    def gen_buildup(self):
        """
        True if multipacting buildup is detected
        """
        self.buildup = np.mean(self.smooth_diff[:self.cutoff:self.bunch_step]) > 0

    def gen_bunch_step(self):
        """
        The number of indexes corresponding to b_spac
        """
        self.bunch_step = self.time_to_index(self.b_spac)

    def gen_bunch_times(self):
        """
        The times corresponding to maximum beam intensity.
        """
        self.bunch_times = self.t_offs + (self.b_spac * np.argwhere(self.filling_pattern_file == 1).ravel())

    def gen_cutoff(self):
        """
        The index into the data arrays corresponding to the last bunch passage
        """
        self.cutoff = min(min(self.time.size, self.N_electrons.size)-1, self.time_to_index(self.bunch_times[-1]))

    def gen_half_bunch(self):
        """
        The true half-width of the simulated Gaussian bunch.
        """
        self.half_bunch = self.t_offs - self.time[np.argwhere(self.intensity[:self.time_to_index(self.t_offs)] > 0).ravel()[0]]

    def gen_magnet(self):
        """
        The type of magnetic section.
        """
        if len(self.B_multip) == 1 and self.B_multip[0] == 0:
            self.magnet = 0
            return
        self.magnet = 2 * len(self.B_multip)

    def gen_mean_intensity(self):
        """
        The mean beam intensity.
        """
        start = np.argwhere(self.intensity[:self.time_to_index(self.t_offs)] > 0).ravel()[0]
        self.mean_intensity = np.mean(self.intensity[start:start+self.time_to_index(self.b_spac)])

    def gen_Ne_0(self):
        """
        The initial amount of electrons - note that we do not use N_electrons[0] - instead, we take the amount at the point
        before the start of the first bunch passage.
        """
        self.Ne_0 = self.N_electrons[np.argwhere(self.intensity[:self.time_to_index(self.t_offs)] > 0).ravel()[0]]
    
    def gen_perimeter(self):
        """
        The length of the perimeter of the beam chamber.
        """
        chamber_path = os.path.join(self.path, self.filename_chm)
        if not os.path.exists(chamber_path):
            raise IOError("There is not a chamber definition at: {}".format(chamber_path))
        chamber_data = scipy.io.loadmat(chamber_path)
        Vx, Vy = chamber_data["Vx"].ravel(), chamber_data["Vy"].ravel()
        Vx, Vy = Vx - np.mean(Vx), Vy - np.mean(Vy)
        angle = np.atan2(Vy, Vx)
        angle[angle < 0] += 2*np.pi
        order = np.argsort(angle)
        Vx, Vy = Vx[order], Vy[order]
        Vx, Vy = np.append(Vx, Vx[0]), np.append(Vy, Vy[0])
        self.perimeter = np.sum(
            np.sqrt(np.diff(Vx)**2 + np.diff(Vy)**2)
        )
        self.area = 0.5 * np.abs(
            np.sum(Vx[:-1] * Vy[1:] - Vx[1:] * Vy[:-1])
        )

    def gen_train_times(self):
        """
        The times of each bunch train.
        """
        gap_lens = unique_with_tolerance(np.diff(self.bunch_times, prepend=self.t_offs), self.t_offs)
        if len(np.where(gap_lens > self.t_offs)[0]) > 2:
            # We actually have trains
            where_gaps = np.argwhere(np.diff(self.bunch_times, prepend=self.t_offs) > np.max(gap_lens) - self.t_offs).ravel()
            self.train_times = np.concatenate((np.array([self.bunch_times[0]]), self.bunch_times[where_gaps]), axis=None)
        else:
            # No trains
            self.train_times = np.array([self.bunch_times[0]])


class InstabilityModel(SynchedEntry):
    """
    An InstabilityModel corresponds to one PyHEADTAIL instability simulation.
    Extends SynchedEntry to auto-save ONLY computed scalar properties to the DB.
    Large time-series arrays are loaded dynamically from .h5 files and cached in memory.
    """
    PROPERTIES: set[str] = {
        "n_turns",              # Total number of turns (scalar)
        "n_slices",             # Number of slices (scalar)
        "growth_rate_centroid", # Calculated growth rate (scalar)
        "growth_rate_mode",     # Calculated growth rate of dominant mode (scalar)
        "dominant_mode_idx",    # Index of dominant mode (scalar)
        "tune_centroid",        # Approximate tune (scalar)
        "blowup_turn_first",    # Turn where emittance crosses threshold (scalar)
        "max_emittance_ratio",
        "charge_weighted_rms_x",
        "intrabunch_activity",
        "growth_rate_r_squared",
        "instability_threshold",
        "dominant_mode_growth_rate",
        "dominant_mode_idx"
    }

    BUNCH_EVOLUTION_PATTERN = "bunch_evolution*.h5"
    SLICE_EVOLUTION_PATTERN = "slice_evolution*.h5"
    _h5 = None

    def __init__(self, db: TinyDB | SimDB, doc: int | Document):
        # Lazy load h5py
        if InstabilityModel._h5 is None:
            import h5py
            InstabilityModel._h5 = h5py
        result = doc if isinstance(doc, Document) else (db if isinstance(db, TinyDB) else db.db).get(doc_id=doc)
        if result is None: 
            raise KeyError(f"The provided doc_id={doc} does not exist in the database!")
        
        super().__init__(db if isinstance(db, TinyDB) else db.db, result.doc_id, InstabilityModel.PROPERTIES)
        self.doc_id = result.doc_id
        # Populate this object with information from the database entry
        for k, v in result.items():
            self.set_sync(k)
            if isinstance(v, list) and len(v) > 0 and not type(v[0]) == str:
                setattr(self, "_"+k, np.array(v, dtype=type(v[0])))
            else:
                setattr(self, "_"+k, v)
        
        # Initialize cache for dynamic data loading
        self._bunch_data_cache: dict[str, np.ndarray] = {}
        self._slice_data_cache: dict[str, np.ndarray] = {}
        self._bunch_data_loaded = False
        self._slice_data_loaded = False

    def _load_h5_files(self, file_pattern: str, key: str = 'Bunch') -> dict[str, np.ndarray]:
        """Improved version with better error handling"""
        pattern_path = os.path.join(self.path, file_pattern)
        files = sorted(glob(pattern_path))
        if not files:
            return {}
        
        data_dict = {}
        for f_path in files:
            try:
                with self._h5.File(f_path, 'r') as fid:
                    group = fid.get(key, fid)  # More robust key lookup
                    
                    for k in group.keys():
                        val = np.array(group[k])
                        if val.shape == ():
                            val = val.item()
                        
                        if k not in data_dict:
                            data_dict[k] = []
                        data_dict[k].append(val)
            except Exception as e:
                print(f"Warning: Failed to load {f_path}: {e}")
                continue
        
        # Final concatenation with proper shape checking
        final_data = {}
        for k, list_of_arrays in data_dict.items():
            if len(list_of_arrays) == 1:
                final_data[k] = list_of_arrays[0]
            else:
                # Check if all arrays have compatible shapes
                shapes = [arr.shape for arr in list_of_arrays if isinstance(arr, np.ndarray)]
                if len(shapes) > 0 and all(len(s) == len(shapes[0]) for s in shapes):
                    try:
                        final_data[k] = np.concatenate(list_of_arrays, axis=0)
                    except ValueError as e:
                        print(f"Warning: Could not concatenate {k} from {file_pattern}: {e}")
                        final_data[k] = list_of_arrays[0]  # Use first available
                else:
                    final_data[k] = list_of_arrays[0]
        
        return final_data

    @property
    def bunch_data(self) -> Dict[str, np.ndarray]:
        """
        Lazy-loaded property containing bunch evolution data (1D arrays over turns).
        Loads from bunch_evolution*.h5 files only once per instance and caches in memory.
        """
        if not self._bunch_data_loaded:
            self._bunch_data_cache = self._load_h5_files(self.BUNCH_EVOLUTION_PATTERN, key='Bunch')
            if self._bunch_data_cache:
                self.bunch_data_file = list(self._bunch_data_cache.keys())[0]
            self._bunch_data_loaded = True
            
        return self._bunch_data_cache

    @property
    def slice_data(self) -> Dict[str, np.ndarray]:
        """
        Lazy-loaded property containing slice evolution data (2D arrays: slices × turns).
        Loads from slice_evolution*.h5 files only once per instance and caches in memory.
        """
        if not self._slice_data_loaded:
            self._slice_data_cache = self._load_h5_files(self.SLICE_EVOLUTION_PATTERN, key='Slices')
            if self._slice_data_cache:
                self.slice_data_file = list(self._slice_data_cache.keys())[0]
            self._slice_data_loaded = True
            
        return self._slice_data_cache

    @property
    def mean_x(self) -> np.ndarray:
        """Bunch centroid X (1D array over turns)."""
        if 'mean_x' in self.bunch_data:
            return self.bunch_data['mean_x']
        # Fallback: average slice data if bunch data not available
        if 'mean_x' in self.slice_data:
            return np.mean(self.slice_data['mean_x'], axis=0)
        return np.array([])

    @property
    def mean_y(self) -> np.ndarray:
        """Bunch centroid Y (1D array over turns)."""
        if 'mean_y' in self.bunch_data:
            return self.bunch_data['mean_y']
        if 'mean_y' in self.slice_data:
            return np.mean(self.slice_data['mean_y'], axis=0)
        return np.array([])

    @property
    def epsn_x(self) -> np.ndarray:
        """Normalized emittance X (1D array over turns)."""
        if 'epsn_x' in self.bunch_data:
            return self.bunch_data['epsn_x']
        if 'epsn_x' in self.slice_data:
            return np.mean(self.slice_data['epsn_x'], axis=0)
        return np.array([])

    @property
    def epsn_y(self) -> np.ndarray:
        """Normalized emittance Y (1D array over turns)."""
        if 'epsn_y' in self.bunch_data:
            return self.bunch_data['epsn_y']
        if 'epsn_y' in self.slice_data:
            return np.mean(self.slice_data['epsn_y'], axis=0)
        return np.array([])

    @property
    def sigma_x(self) -> np.ndarray:
        """Bunch length X (1D array over turns)."""
        if 'sigma_x' in self.bunch_data:
            return self.bunch_data['sigma_x']
        if 'sigma_x' in self.slice_data:
            return np.mean(self.slice_data['sigma_x'], axis=0)  # Fixed
        return np.array([])

    @property
    def sigma_y(self) -> np.ndarray:
        """Bunch length Y (1D array over turns)."""
        if 'sigma_y' in self.bunch_data:
            return self.bunch_data['sigma_y']
        if 'sigma_y' in self.slice_data:
            return np.mean(self.slice_data['sigma_y'], axis=0)  # Fixed
        return np.array([])

    @property
    def sigma_z(self) -> np.ndarray:
        """Bunch length Z (1D array over turns)."""
        if 'sigma_z' in self.bunch_data:
            return self.bunch_data['sigma_z']
        if 'sigma_z' in self.slice_data:
            return np.mean(self.slice_data['sigma_z'], axis=0)
        return np.array([])

    @property
    def macroparticlenumber(self) -> np.ndarray:
        """Total macroparticle count (1D array over turns)."""
        if 'macroparticlenumber' in self.bunch_data:
            return self.bunch_data['macroparticlenumber']
        if 'n_macroparticles_per_slice' in self.slice_data:
            return np.sum(self.slice_data['n_macroparticles_per_slice'], axis=0)
        return np.array([])

    @property
    def slice_mean_x(self) -> np.ndarray:
        """Slice centroid X (2D array: slices × turns)."""
        if 'mean_x' in self.slice_data:
            return self.slice_data['mean_x']
        return np.array([])

    @property
    def slice_mean_y(self) -> np.ndarray:
        """Slice centroid Y (2D array: slices × turns)."""
        if 'mean_y' in self.slice_data:
            return self.slice_data['mean_y']
        return np.array([])

    @property
    def slice_epsn_x(self) -> np.ndarray:
        """Slice emittance X (2D array: slices × turns)."""
        if 'epsn_x' in self.slice_data:
            return self.slice_data['epsn_x']
        return np.array([])

    @property
    def slice_n_macroparticles(self) -> np.ndarray:
        """Macroparticles per slice (2D array: slices × turns)."""
        if 'n_macroparticles_per_slice' in self.slice_data:
            return self.slice_data['n_macroparticles_per_slice']
        return np.array([])

    @property
    def slice_fft_power(self) -> np.ndarray:
        """
        Computes the Power Spectral Density (PSD) of slice centroids.
        Returns a 2D array: (Frequency_Bins, Turns).
        Uses a sliding window approach to capture time-evolution of modes.
        
        Note: This is a heavy computation. Results are cached in memory but NOT synced to DB.
        """
        if not hasattr(self, '_slice_fft_cache'):
            if len(self.slice_mean_x) == 0 or len(self.slice_mean_x.shape) != 2:
                self._slice_fft_cache = np.array([])
                return self._slice_fft_cache
            
            # Transpose to (Turns, Slices) for easier windowing
            data = self.slice_mean_x.T  # Shape: (Turns, Slices)
            n_turns, n_slices = data.shape
            
            # Parameters for STFT-like analysis
            window_size = 64  # Must be power of 2 for efficiency
            hop_size = 10     # Steps between windows
            
            if n_turns < window_size:
                # Fallback: Single FFT over whole duration if too short
                fft_result = scipy.fft.rfft(data, axis=0)
                power = np.abs(fft_result)**2
                self._slice_fft_cache = power
                return self._slice_fft_cache

            # Initialize output: (Freq_Bins, Turns)
            n_freqs = window_size // 2 + 1
            n_output_turns = (n_turns - window_size) // hop_size + 1
            power_matrix = np.zeros((n_freqs, n_output_turns))
            
            # Sliding window FFT
            for i in range(n_output_turns):
                start = i * hop_size
                end = start + window_size
                segment = data[start:end, :] # (Window, Slices)
                
                # Apply Hanning window to reduce spectral leakage
                window = scipy.signal.windows.hann(window_size)
                windowed_seg = segment * window[:, np.newaxis]
                
                # FFT along the turn axis (axis=0)
                fft_seg = scipy.fft.rfft(windowed_seg, axis=0)
                power_seg = np.abs(fft_seg)**2
                
                # Average over slices to get global mode power (optional: keep slice-resolved)
                # Here we average to find global coherent modes
                power_matrix[:, i] = np.mean(power_seg, axis=1)
            
            self._slice_fft_cache = power_matrix
        
        return self._slice_fft_cache

    @property
    def dominant_mode_idx(self) -> int:
        """
        Identifies the index of the most unstable mode (highest growth rate).
        Uses the early phase of the simulation (first 20% or 2000 turns) to avoid saturation effects.
        """
        power = self.slice_fft_power
        if len(power) == 0:
            return 0
        
        n_turns = power.shape[1]
        if n_turns == 0:
            return 0
            
        # Define growth phase (avoid saturation)
        growth_end_idx = min(int(n_turns * 0.2), 2000)
        if growth_end_idx < 10:
            growth_end_idx = n_turns // 2
            
        growth_phase = power[:, :growth_end_idx]
        
        # Calculate growth rate for each frequency bin
        # Fit log(power) vs turn index
        turns = np.arange(growth_end_idx)
        growth_rates = np.zeros(growth_phase.shape[0])
        
        for i in range(growth_phase.shape[0]):
            # Avoid log(0)
            valid = growth_phase[i, :] > 1e-12
            if np.sum(valid) < 5:
                growth_rates[i] = -np.inf
                continue
            
            log_pow = np.log(growth_phase[i, valid])
            # Linear fit
            coeffs = np.polyfit(turns[valid], log_pow, 1)
            growth_rates[i] = coeffs[0]
        
        # Find the mode with the maximum positive growth rate
        # Ignore DC component (index 0)
        if len(growth_rates) > 1:
            max_idx = np.argmax(growth_rates[1:]) + 1
            if growth_rates[max_idx] > 0:
                return max_idx
        
        return 0

    @property
    def dominant_mode_freq(self) -> float:
        """
        Returns the fractional tune (frequency) of the dominant mode.
        """
        power = self.slice_fft_power
        if len(power) == 0:
            return 0.0
            
        n_turns = power.shape[1]
        freqs = scipy.fft.rfftfreq(self._slice_fft_window_size if hasattr(self, '_slice_fft_window_size') else 64) 
        # Note: scipy.fft.rfftfreq depends on window size. For simplicity, we approximate based on total turns if needed,
        # but strictly speaking, the frequency resolution is 1/window_size.
        # A more robust way is to map the index to the actual tune based on the window size used.
        
        # Re-calculate freqs based on the actual window used in slice_fft_power
        # We need to store the window size or recalculate. Let's assume standard 64 for now or derive from data.
        # Better: Store window size in the property logic.
        window_size = 64 # Match the window_size in slice_fft_power
        freqs = scipy.fft.rfftfreq(window_size)
        
        idx = self.dominant_mode_idx
        if idx < len(freqs):
            return float(freqs[idx])
        return 0.0

    def _sync_get(self, attr: str):
        if not hasattr(self, "_"+attr):
            if hasattr(self, "gen_"+attr):
                # Call the attribute's generator
                getattr(self, "gen_"+attr)()
            else:
                raise ValueError("Requested synced property {} for which no generator is defined!".format(attr))
        return super()._sync_get(attr)

    def gen_n_turns(self):
        """Total number of turns in the simulation."""
        if len(self.mean_x) > 0:
            self.n_turns = len(self.mean_x)
        else:
            self.n_turns = 0

    def gen_n_slices(self):
        """Number of slices in the simulation."""
        if len(self.slice_mean_x.shape) == 2:
            self.n_slices = self.slice_mean_x.shape[0]
        else:
            self.n_slices = 1

    def gen_growth_rate_centroid(self, fit_window: int | None = None):
        """
        Calculates the growth rate of the centroid oscillation.
        Uses log-amplitude linear regression on the exponential growth region.
        
        Parameters:
        -----------
        fit_window : int, optional
            Number of turns from the start to use for fitting. If None, uses entire dataset.
        """
        x = self.mean_x
        if len(x) < 10:
            self.growth_rate_centroid = np.nan
            return

        turns = np.arange(len(x))
        
        # Mask for valid data (avoid log(0))
        mask = np.abs(x) > 1e-12
        
        # Optional: limit to rising portion before saturation
        if fit_window is not None:
            mask = mask & (turns < fit_window)
        
        if np.sum(mask) < 10:
            self.growth_rate_centroid = np.nan
            return

        try:
            # Linear fit to log-amplitude
            coeffs = np.polyfit(turns[mask], np.log(np.abs(x[mask])), 1)
            growth_rate = coeffs[0]
            
            # Validate: check R² of fit
            fitted = np.polyval(coeffs, turns[mask])
            r_squared = 1 - np.sum((np.log(np.abs(x[mask])) - fitted)**2) / \
                        np.sum((np.log(np.abs(x[mask])) - np.mean(np.log(np.abs(x[mask]))))**2)
            
            # Store both growth rate and fit quality
            self.growth_rate_centroid = growth_rate
            self.growth_rate_r_squared = r_squared if r_squared > 0 else 0
        except Exception:
            self.growth_rate_centroid = np.nan
            self.growth_rate_r_squared = 0

    def gen_dominant_mode_idx(self):
        """Finds the index of the dominant frequency in the centroid FFT."""
        x = self.mean_x
        if len(x) < 2:
            self.dominant_mode_idx = 0
            return

        fft_x = scipy.fft.rfft(x)
        power = np.abs(fft_x[1:])**2 # Exclude DC
        
        if len(power) == 0:
            self.dominant_mode_idx = 0
            return

        idx = np.argmax(power) + 1
        self.dominant_mode_idx = idx

    def gen_tune_centroid(self):
        """Estimates the tune from the dominant frequency."""
        x = self.mean_x
        if len(x) < 2:
            self.tune_centroid = 0.0
            return

        freqs = scipy.fft.rfftfreq(len(x))
        idx = self.dominant_mode_idx
        if idx < len(freqs):
            self.tune_centroid = freqs[idx]
        else:
            self.tune_centroid = 0.0

    def gen_growth_rate_mode(self):
        """Estimates growth rate of the dominant mode (proxy with centroid rate)."""
        self.growth_rate_mode = self.growth_rate_centroid

    def gen_blowup_turn_first(self):
        """Finds the first turn where emittance exceeds 1.2x initial value."""
        epsn = self.epsn_x
        if len(epsn) == 0:
            self.blowup_turn_first = np.nan
            return

        initial = epsn[0]
        if initial == 0:
            self.blowup_turn_first = np.nan
            return

        threshold = initial * 1.2
        mask = epsn > threshold
        
        if np.any(mask):
            self.blowup_turn_first = float(np.argmax(mask))
        else:
            self.blowup_turn_first = np.nan

    def gen_max_emittance_ratio(self):
        """Max ratio of emittance to initial."""
        epsn = self.epsn_x
        if len(epsn) == 0:
            self.max_emittance_ratio = 1.0
            return
        
        initial = epsn[0]
        if initial == 0:
            self.max_emittance_ratio = np.nan
        else:
            self.max_emittance_ratio = float(np.max(epsn) / initial)

    def gen_charge_weighted_rms_x(self):
        """
        Calculates charge-weighted RMS of centroid X from slice data.
        Used for more accurate instability detection (as in reference code).
        """
        if 'mean_x' not in self.slice_data or 'n_macroparticles_per_slice' not in self.slice_data:
            self.charge_weighted_rms_x = np.array([])
            return
        mean_x = self.slice_data['mean_x']
        weights = self.slice_data['n_macroparticles_per_slice']
        # VALIDATE SHAPES
        if mean_x.ndim != 2 or weights.ndim != 2:
            self.charge_weighted_rms_x = np.array([])
            return
        if mean_x.shape != weights.shape:
            self.charge_weighted_rms_x = np.array([])
            return
        # Safe computation
        weighted_mean = np.sum(mean_x * weights, axis=0) / (np.sum(weights, axis=0) + 1e-12)
        rms = np.sqrt(np.sum((mean_x - weighted_mean)**2 * weights, axis=0) / (np.sum(weights, axis=0) + 1e-12))
        
        self.charge_weighted_rms_x = rms

    def gen_intrabunch_activity(self, window: int = 21):
        """
        Calculates intrabunch activity using Savitzky-Golay filter on slice data.
        Matches the reference code's approach.
        """
        if not hasattr(self, 'charge_weighted_rms_x') or len(self.charge_weighted_rms_x) == 0:
            self.intrabunch_activity = np.array([])
            return
        
        from scipy.signal import savgol_filter
        
        if len(self.charge_weighted_rms_x) <= window:
            self.intrabunch_activity = self.charge_weighted_rms_x
        else:
            self.intrabunch_activity = savgol_filter(self.charge_weighted_rms_x, window, 3)

    def gen_growth_rate_r_squared(self):
        """Calculate R² of growth rate fit."""
        self.gen_growth_rate_centroid()

    def gen_instability_threshold(self):
        """Binary flag: instability present (growth_rate > threshold)."""
        threshold = 0.001  # turn^-1, adjust based on machine
        self.instability_threshold = self.growth_rate_centroid > threshold

    def gen_dominant_mode_growth_rate(self):
        """
        Calculates the growth rate (turns^-1) of the dominant mode.
        Syncs this scalar to the DB.
        """
        power = self.slice_fft_power
        if len(power) == 0:
            self.dominant_mode_growth_rate = np.nan
            return

        n_turns = power.shape[1]
        growth_end_idx = min(int(n_turns * 0.2), 2000)
        if growth_end_idx < 10:
            growth_end_idx = n_turns // 2
            
        idx = self.dominant_mode_idx
        if idx == 0:
            self.dominant_mode_growth_rate = 0.0
            return

        mode_signal = power[idx, :growth_end_idx]
        turns = np.arange(growth_end_idx)
        
        valid = mode_signal > 1e-12
        if np.sum(valid) < 5:
            self.dominant_mode_growth_rate = np.nan
            return

        log_pow = np.log(mode_signal[valid])
        coeffs = np.polyfit(turns[valid], log_pow, 1)
        self.dominant_mode_growth_rate = float(coeffs[0])

    def run_analysis(self):
        """
        Convenience method to trigger all generators.
        Accessing properties triggers calculation and syncs results to DB.
        """
        for prop in self.PROPERTIES:
            getattr(self, "gen_"+prop)()
        return self

class DataSelector(object):
    """
    A DataSelector chooses data for fitting or other analysis from an ECModel.
    """
    def __init__(self, use_central_density: bool = False, use_train: int = -1):
        self.use_central_density = use_central_density
        self.use_train = use_train

    def select(self, model: ECModel) -> tuple[np.ndarray, np.ndarray]:
        train = model.train_to_index(self.use_train)
        return (model.time[train], model.central_density[train] if self.use_central_density else model.N_electrons[train])

class BeforeBunchSelector(DataSelector):
    """
    Selects the points immediately before each bunch passage as the fitting points.
    """
    def select(self, model: ECModel) -> tuple[np.ndarray, np.ndarray]:
        pre_bp = np.array(model.time_to_index(model.bunch_times[model.train_to_bunches(self.use_train)] - model.half_bunch))
        time = model.time[pre_bp]
        return (time - time[0], (model.central_density if self.use_central_density else model.N_electrons)[pre_bp])

class BunchAverageSelector(DataSelector):
    """
    Selects the points immediately before each bunch passage as the fitting points.
    """
    def select(self, model: ECModel) -> tuple[np.ndarray, np.ndarray]:
        valid = model.train_to_bunches(self.use_train)
        pre_bp = np.array(model.time_to_index(model.bunch_times[valid] - model.half_bunch))
        averages = np.zeros_like(pre_bp)
        for b, start in enumerate(pre_bp):
            averages[b] = np.mean(
                (model.central_density if self.use_central_density else model.N_electrons)[start:start+model.bunch_step])
        time = (model.bunch_times - model.half_bunch + (model.b_spac / 2))[valid]
        return (time - time[0], averages)

class Fit(object):
    """
    A Fit is the base class for applying a curve fit to an ECModel.
    """
    def __init__(self, name: str, function: str, variables: Iterable[str], selector: DataSelector):
        self.name = name
        self.function = function
        self.variables = tuple(variables)
        self.selector = selector
        self._reset_state()
        self._compiled_fn = compile(self.function, f"<{self.name}_fitfn>", "eval")

    def _reset_state(self):
        self.initial_guess = np.ones(shape=(len(self.variables)-1,))
        self.bounds_lower = np.full_like(self.initial_guess, -np.inf)
        self.bounds_upper = np.full_like(self.initial_guess, np.inf)
        self.fixed = np.full_like(self.initial_guess, False, dtype=bool)
        self.fixed_values = np.zeros_like(self.initial_guess)

    def bound(self, bound_info: dict):
        """Specify the bounds for each variable: (lower, upper) in the provided dict"""
        for v, b in bound_info.items():
            i = self.variables.index(v)
            if i == -1: raise KeyError(f"Bounds were provided for non-existent variable {v}")
            self.bounds_lower[i - 1] = b[0]
            self.bounds_upper[i - 1] = b[1]

    def initial(self, initial_info: dict):
        """Specify the initial guess for each variable: value"""
        for v, g in initial_info.items():
            i = self.variables.index(v)
            if i == -1: raise KeyError(f"An initial value was provided for non-existent variable {v}")
            self.initial_guess[i - 1] = g

    def fix(self, fix_info: dict):
        """Specify fixed values for each variable: value. They will not be optimized"""
        for v, f in fix_info.items():
            i = self.variables.index(v)
            if i == -1: raise KeyError(f"A fixed value was provided for non-existent variable {v}")
            self.fixed[i - 1] = True
            self.fixed_values[i - 1] = f

    def limit_estimate(self, xdata: np.ndarray, ydata: np.ndarray) -> float:
        """
        Estimate the mathematical limit of f(x) using a stabilized, wide-range
        linearized per-capita growth rate model across the active buildup phase.
        """
        if len(ydata) == 0:
            return 0.0
        if len(ydata) <= 5:
            return float(ydata[-1])

        # 1. Compute global numerical derivatives across the entire dataset
        dx = np.diff(xdata)
        dy = np.diff(ydata)
        
        # Avoid division by zero in x intervals
        dydx = dy / np.where(dx == 0, 1e-12, dx)
        y_mid = 0.5 * (ydata[:-1] + ydata[1:])

        # 2. Establish a wide, high-signal mask to filter out numerical noise.
        # We target the primary active growth zone (e.g., above 5% of max accumulation)
        # where the derivative is positive and well above the noise floor.
        max_y = np.max(ydata)
        max_dydx = np.max(dydx)
        
        mask = (
            (y_mid > max_y * 0.05) &      # Exclude noisy initial flat steps
            (dydx > max_dydx * 0.01) &    # Exclude flat saturation tails where dy/dx -> 0
            (y_mid < max_y * 0.95)        # Focus on the high-gradient region
        )

        # Ensure we have a statistically significant number of points to fit a line
        if np.sum(mask) >= 5:
            y_fit = y_mid[mask]
            z_fit = dydx[mask] / y_fit  # Per-capita growth rate: z = (dy/dx) / y
            
            try:
                # Linear regression over a wide, stable baseline: z = m * y + c
                m, c = np.polyfit(y_fit, z_fit, 1)
                if m < 0 and c > 0:
                    asymptotic_limit = -c / m
                    # Reject if the prediction is lower than what we've already recorded
                    if asymptotic_limit > max_y:
                        return float(asymptotic_limit)
            except (np.linalg.LinAlgError, ValueError):
                pass

        # 3. Robust Fallback: Moving slope mechanics if the global fit diverges
        smooth_diff = smooth(np.diff(ydata), max(ydata.size // 10, 5))
        last_window = max(ydata.size // 10, 5)
        mean_tail_slope = np.mean(smooth_diff[-last_window:])

        if mean_tail_slope >= ydata[-1] * 1e-2:
            # Curve is still aggressively climbing; project dynamically
            return float((3.0 + (np.argmax(smooth_diff) / smooth_diff.size)) * max_y)
        
        return float(min(np.mean(ydata[-5:-1]) + max(mean_tail_slope, 0), max_y))

    def _mget(self, attr: str, model: ECModel, default=False):
        return getattr(model, self.name+"_"+attr) if hasattr(model, "_"+self.name+"_"+attr) or hasattr(model, self.name+"_"+attr) else default
    
    def _mset(self, attr: str, value, model: ECModel):
        model.set_sync(self.name+"_"+attr)
        model._sync_set(self.name+"_"+attr, value)

    def _evaluate(self, code, names, f_mask, f_vals, x, *free_params):
        eval_context = {"np": np, "scipy": scipy, names[0]: x}
        free_ptr = 0
        for i, name in enumerate(names[1:]):
            if f_mask[i]:
                eval_context[name] = f_vals[i]
            else:
                eval_context[name] = free_params[free_ptr]
                free_ptr += 1
        return eval(code, eval_context)

    def _make_target(self):
        return functools_partial(self._evaluate, self._compiled_fn, self.variables, self.fixed.copy(), self.fixed_values.copy())

    @property
    def full_names(self):
        return np.array([(self.name+"_"+v, self.name+"_"+v+"_err") for v in self.variables[1:]], dtype=str)

    def select_data(self, model: ECModel) -> tuple[np.ndarray, np.ndarray]:
        return self.selector.select(model)

    def scale_factor(self, model: ECModel, xdata: np.ndarray, ydata: np.ndarray) -> tuple[float | int | np.number, float | int | np.number]:
        self._mset("scale_x", 1, model)
        self._mset("scale_y", 1, model)
        return (1, 1)

    def fit(self, model: ECModel, refit: bool = False) -> np.ndarray | None:
        if refit or (not self._mget("success", model)):
            try:
                xdata, ydata = self.select_data(model)
                scale_x, scale_y = self.scale_factor(model, xdata, ydata)
                result, covariance = scipy.optimize.curve_fit(
                    self._make_target(),
                    scale_x * xdata, scale_y * ydata,
                    p0=self.initial_guess[~self.fixed],
                    bounds=(self.bounds_lower[~self.fixed], self.bounds_upper[~self.fixed]),
                    loss="arctan",
                    maxfev=1e5,
                    nan_policy="omit",
                    ftol=1e-10
                )
                perr = np.sqrt(np.diag(covariance))
                f = 0 
                for i, v in enumerate(self.variables[1:]):
                    if self.fixed[i]:
                        self._mset(v, self.fixed_values[i], model)
                        self._mset(v+"_err", 0, model)
                    else:
                        self._mset(v, result[f], model)
                        self._mset(v+"_err", perr[f], model)
                        f += 1
                self._mset("success", True, model)
                self._mset("train", self.selector.use_train, model)
                for attr in ["_data", "_smooth", "_smooth_diff"]:
                    if hasattr(model, attr): delattr(model, attr)
            except (RuntimeError, ValueError) as e:
                self._mset("success", False, model)
                setattr(model, f"_{self.name}_fitting_error", str(e))
                return None
            self._reset_state()
        return np.array([(self._mget(v, model), self._mget(v+"_err", model)) for v in self.variables[1:]]).T

    def fit_function(self, model: ECModel, params: np.ndarray | None = None) -> Callable[..., float | np.number | np.ndarray]:
        if params is None:
            params = self.fit(model)
            if params is None: raise RuntimeError("Cannot return a function for a failed fit!")
            params = params[0]
        scale_x = self._mget("scale_x", model, default=1)
        scale_y = self._mget("scale_y", model, default=1)
        if scale_x == False or scale_y == False or scale_y == 0:
            scale_x, scale_y = self.scale_factor(model, *self.select_data(model))
        target = self._make_target()
        fn = lambda x: target(scale_x * x, *params[~self.fixed]) / scale_y
        setattr(fn, "variables", self.variables)
        setattr(fn, "parameters", params)
        setattr(fn, "model", self.name)
        return fn

    def scrub(self, model: ECModel):
        for v in self.variables[1:]:
            if hasattr(model, f"_{self.name}_{v}"): delattr(model, f"{self.name}_{v}")
            if hasattr(model, f"_{self.name}_{v}_err"): delattr(model, f"{self.name}_{v}_err")
        if hasattr(model, f"_{self.name}_success"): delattr(model, f"{self.name}_success")


class FurmanBaseFit(Fit):
    """
    Intermediate base class consolidating common scale factors for all Furman EC variations.
    """
    def scale_factor(self, model: ECModel, xdata: np.ndarray, ydata: np.ndarray) -> tuple[float | int | np.number, float | int | np.number]:
        scale_x = np.reciprocal(model.b_spac)
        scale_y = np.reciprocal(model.mean_intensity)
        self._mset("scale_x", scale_x, model)
        self._mset("scale_y", scale_y, model)
        return (scale_x, scale_y)


class FurmanNoPhotoFit(FurmanBaseFit):
    """
    "FURMAN" FULL MODEL (B) - NO PHOTOEMISSION
    """
    FURMAN_NO_PHOTO = """(y0 * yc * np.exp(beta * yc * x)) / (yc + y0 * (np.exp(beta * yc * x) - 1))"""

    def __init__(self, selector: None | DataSelector = None):
        super().__init__(
            "furman_np",
            FurmanNoPhotoFit.FURMAN_NO_PHOTO,
            ("x", "yc", "beta", "y0"),
            BunchAverageSelector() if selector is None else selector
        )

    def fit(self, model: ECModel, refit: bool = False) -> np.ndarray | None:
        xdata, ydata = self.select_data(model)
        _, scale_y = self.scale_factor(model, xdata, ydata)
        self.fix({"y0": ydata[0] * scale_y})
        
        if model.buildup:
            mslope = np.argmin(np.square(ydata - ((ydata[-1] - ydata[0])/2)))
            limit_est_scaled = self.limit_estimate(xdata, ydata) * scale_y
            
            # Open the upper bound dynamically to avoid bounding cliff constraints
            bounds = {
                "yc": (ydata[0] * scale_y, max(limit_est_scaled * 3.0, ydata[-1] * scale_y * 5.0)),
                "beta": (0, 3 * model.del_max)
            }
            self.initial({
                "yc": max(limit_est_scaled, ydata[-1] * scale_y * 1.05),
                "beta": min(1, abs(4 * np.mean(np.diff(ydata[mslope-2:mslope+2])) / ((ydata[-1]**2) * scale_y)))
            })
        else:
            bounds = {"yc": (0, np.inf), "beta": (-np.inf, 0)}
            self.initial({"yc": 2 * ydata[0] * scale_y, "beta": -0.1})
            
        self.bound(bounds)
        return super().fit(model, refit)


class FurmanNPMCFit(FurmanBaseFit):
    """
    "FURMAN" NO PHOTOEMISSION MODEL WITH MAGNETIC CORRECTION
    """
    FURMAN_NO_PHOTO_MC = "(y0 * yc * np.exp(yc*x*(b0 + 0.5*b1*x))) / (yc + y0 * (np.exp(yc*x*(b0 + 0.5*b1*x)) - 1))"

    def __init__(self, db: SimDB, selector: None | DataSelector = None):
        self.db = db
        super().__init__(
            "furman_npmc",
            FurmanNPMCFit.FURMAN_NO_PHOTO_MC,
            ("x", "yc", "b0", "b1", "y0"),
            BunchAverageSelector() if selector is None else selector
        )
    
    def fit(self, model: ECModel, refit: bool = False) -> np.ndarray | None:
        # Composition: Bootstrapping baseline parameters from standard NoPhoto fit execution
        np_result = FurmanNoPhotoFit(selector=self.selector).fit(model, refit=refit)
        if np_result is None:
            self._mset("success", False, model)
            setattr(model, f"_{self.name}_fitting_error", "The underlying NoPhoto fit failed.")
            return None
            
        if list(model.B_multip) == [0]:
            self.fix({"y0": model.furman_np_y0, "yc": model.furman_np_yc, "b1": 0})
            self.bound({"b0": (-2 * abs(model.furman_np_beta), 2 * abs(model.furman_np_beta))})
            self.initial({"b0": model.furman_np_beta})
        else:
            self.fix({"y0": model.furman_np_y0, "yc": model.furman_np_yc})
            base_search = self.db.where(doc_id=model.doc_id)[0]
            base_model = self.db.closest(base_search, photoemission=False, furman_np_success=True, B_multip=[0])
            
            if len(base_model) != 0:
                base_model = ECModel(model.db, base_model[0])
                base_beta = base_model.furman_np_beta if base_model.furman_np_beta > 0 else (model.furman_np_beta / 2)
                self.initial({"b0": base_beta, "b1": 0})
                self.bound({"b0": (0, 2 * base_beta), "b1": (-base_beta, base_beta)})
            else:
                self.initial({"b0": 0.1, "b1": 0})
                self.bound({"b0": (0, np.inf), "b1": (-np.inf, np.inf)})
                
        return super().fit(model, refit)


class FurmanPhotoFit(FurmanBaseFit):
    """
    "FURMAN" FULL MODEL (B) - WITH PHOTOEMISSION
    
    Self-contained dual-expression execution. Leaves the base class unmodified.
    """
    # Nice, human-readable textbook formula for reporting and fit_function evaluation
    FURMAN_PHOTO_READABLE = """0.5*(yc + np.sqrt(yc**2 + (4*alpha/beta))*np.tanh((x/2)*np.sqrt(beta*(4*alpha + (yc**2)*beta)) - np.arctanh((yc - 2*y0)*np.sqrt(beta / (4*alpha + (yc**2)*beta)))))"""

    # High-performance exponential form for numerical solver stability
    FURMAN_PHOTO_OPTIMIZED = (
        "(lambda D: "
        "  (lambda y_inf, y_neg: "
        "    (lambda E: "
        "      (y_inf * (y0 - y_neg) + y_neg * (y_inf - y0) * E) / "
        "      np.where(((y0 - y_neg) + (y_inf - y0) * E) == 0, 1e-12, (y0 - y_neg) + (y_inf - y0) * E)"
        "    )(np.exp(-beta * D * x))"
        "  )(0.5 * (yc + D), 0.5 * (yc - D))"
        ")(np.sqrt(yc**2 + 4.0 * alpha / np.where(beta == 0, 1e-12, beta)))"
    )

    def __init__(self, selector: None | DataSelector = None):
        # 1. Initialize standard behavior with the human-readable string
        super().__init__(
            "furman_p",
            FurmanPhotoFit.FURMAN_PHOTO_READABLE,
            ("x", "yc", "alpha", "beta", "y0"),
            BunchAverageSelector() if selector is None else selector
        )
        # 2. Compile the optimized version privately within this subclass
        self._compiled_opt_fn = compile(FurmanPhotoFit.FURMAN_PHOTO_OPTIMIZED, "<furman_p_opt>", "eval")
    
    def fit(self, model: ECModel, refit: bool = False) -> np.ndarray | None:
        if not model.buildup:
            self._mset("success", False, model)
            setattr(model, f"_{self.name}_fitting_error", "Cannot fit photoemission model for non-buildup simulation.")
            return None

        xdata, ydata = self.select_data(model)
        scale_x, scale_y = self.scale_factor(model, xdata, ydata)
        
        X = xdata * scale_x
        Y = ydata * scale_y
        self.fix({"y0": Y[0]})
        
        # --- Intelligent Parameter Extraction Track ---
        dX = np.diff(X)
        dY = np.diff(Y)
        dYdX = dY / np.where(dX == 0, 1e-12, dX)
        Y_mid = 0.5 * (Y[:-1] + Y[1:])
        
        mask = (Y_mid < np.max(Y) * 0.92) & (dYdX > np.max(dYdX) * 0.02)
        alpha_init = model.k_pe_st * scipy.constants.c * model.b_spac * 0.1
        beta_init = 0.1
        yc_init = self.limit_estimate(xdata, ydata) * scale_y
        
        if np.sum(mask) >= 6:
            try:
                a, b, c = np.polyfit(Y_mid[mask], dYdX[mask], 2)
                if a < 0:
                    beta_init = -a
                    yc_init = b / beta_init
                    alpha_init = max(c, 1e-8)
            except (np.linalg.LinAlgError, ValueError):
                pass

        np_result = FurmanNoPhotoFit(selector=self.selector).fit(model, refit=refit)
        if np_result is None:
            self._mset("success", False, model)
            setattr(model, f"_{self.name}_fitting_error", "The underlying NoPhoto fit failed.")
            return None
            
        yc_baseline, beta_baseline = np_result[0, 0], np_result[0, 1]
        alpha_max = model.k_pe_st * scipy.constants.c * model.b_spac

        # --- Dynamic Bound Pinning ---
        self.bound({
            "yc": (Y[0], max(yc_init * 3.0, yc_baseline * 3.0, np.max(Y) * 5.0)),
            "alpha": (0.0, max(alpha_max, alpha_init * 10.0, 1.0)),
            "beta": (0.0, max(beta_baseline * 5, beta_init * 5, 10))
        })
        
        self.initial({
            "yc": max(yc_init, yc_baseline, np.max(Y) * 1.05),
            "alpha": min(alpha_init, alpha_max * 0.9) if alpha_max > 0 else alpha_init,
            "beta": min(beta_init, beta_baseline)
        })
        
        # Cache the readable tracking function
        original_compiled_fn = self._compiled_fn
        # Force the base class to look at the robust exponential version during optimization
        self._compiled_fn = self._compiled_opt_fn
        
        try:
            # Executes Fit.fit() using the stable math block
            result = super().fit(model, refit)
        finally:
            # Unconditionally restore the nice textbook formula for fit_function reporting
            self._compiled_fn = original_compiled_fn
            
        return result

def fit_all_where(db: SimDB, fit: Fit, refit: bool = False, verbose: bool = True, **search):
    """
    For the database db, apply the Fit to each result of the search (using where()),
    returning an ECModel instance for each result.
    """
    if not refit:
        search = {**search, fit.name+"_success": False}
    results = db.where(**search)
    for i, result in enumerate(results):
        try:
            model = ECModel(db.db, result)
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                fit.fit(model, refit=refit)
            if isinstance(db.db.storage, CachingMiddleware):
                db.db.storage.flush()
            if verbose: print("{:<3}/{} {}".format(i, len(results), "S" if fit._mget("success", model) else "F"), end="\r")
        except:
            if verbose: print("Exceptional error in fit. A file may be missing.")
    if verbose: print("Done.        ")

if __name__ == "__main__":
    from sys import argv
    from json import loads
    if len(argv) <= 1:
        print("Usage: {} db_filename ...db_folder[={{params}}]".format(argv[0]))
    if len(argv) == 2:
        db = SimDB(argv[1])
        print("Database with {} entries.".format(len(db.db)))
        print("\n - " + "\n - ".join(db.all_keys()))
    else:
        if argv[2] == "interactive":
            import code
            v = {"db": SimDB(argv[1]), "ECModel": ECModel, "np": np, "scipy": scipy}
            code.interact(local=v, banner="Interactive Python prompt. 'db' is loaded.")
        else:
            db_filename = argv[1]
            db_finfo = argv[2:]
            db_folders = []
            for info in db_finfo:
                s = info.find("=")
                if s > -1:
                    path = info[:s]
                    params = loads(info[s+1:])
                    db_folders.append(DBFolder(path, **params))
                else:
                    db_folders.append(DBFolder(info))
            db = SimDB(db_filename, db_folders)