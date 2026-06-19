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

def realpath(path: str) -> str:
    return os.path.realpath(os.path.expanduser(path))

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
                    [1 if os.path.exists(os.path.join(r, InstabilityDBFolder.CONFIG_FOLDER, f)) else 0
                     for f in InstabilityDBFolder.FILENAMES.values()])
                if count == len(InstabilityDBFolder.FILENAMES):
                    if any(f.endswith(".h5") for f in files):
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
        self.path = realpath(path)
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
        path = realpath(path)
        if os.path.exists(path) and os.path.exists(os.path.join(path, DBFolder.FILENAMES["data"])):
            raise ValueError("The desired destination {} already contains an output data file!".format(path))
        if not os.path.exists(path):
            os.mkdir(path)

        values = {**self.defaults, **changes}
        skip_resource = set()
        for filename in DBFolder.FILENAMES.values():
            if filename != DBFolder.FILENAMES["data"]:
                this_file = [prop for prop in values.keys() if prop in self.property_map and self.property_map[prop] == filename]
                contents = "#PyECLOUD Configuration from Template {}\n\n".format(self.path)
                for prop in sorted(this_file):
                    value = values[prop]
                    if type(value) == str and os.path.exists(value) and os.path.isfile(value):
                        value = realpath(value)
                        if os.path.commonpath((value, self.path)) != self.path:
                            # This is a filepath argument within the template. We also need this file.
                            rebase = os.path.join(path, os.path.basename(value))
                            copy2(value, rebase)
                            value = os.path.basename(rebase)
                            if value in self.resources:
                                skip_resource.add(value)
                    if prop == "logfile_path": value = "log.txt"
                    elif prop == "progress_path": value = "progress"
                    elif prop == "stopfile": value = "stop"
                    contents += "{} = {}\n".format(prop, ("\"" + value + "\"" if type(value) == str else value) if not isinstance(value, np.ndarray) else value.tolist())
                with open(os.path.join(path, filename), "w") as f:
                    f.write(contents)
        for resource in self.resources:
            if resource not in skip_resource:
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
        self.db = db_file if isinstance(db_file, TinyDB) else TinyDB(realpath(db_file), storage=CachingMiddleware(JSONStorage))
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

    def insert(self, *documents: list[dict | Document]):
        """
        Manually insert document(s) into the database, for example when merging databases.
        The validity or completeness of inserted documents is not checked!
        """
        for document in documents:
            self.db.insert(document)

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
        if self.time.size <= 1:
            return 0
        return np.floor(t / (self.time[1] - self.time[0])).astype(int)

    def train_to_index(self, train: int | np.integer) -> np.ndarray:
        """
        Convert a train number to its corresponding time indexes in the time array.
        """
        if train < 0:
            train = len(self.train_times) + train
        return np.arange(int(self.time_to_index(self.train_times[train])), int(self.time_to_index(self.train_times[train+1]) if train < len(self.train_times) - 1 else self.cutoff))

    def train_to_bunches(self, train: int) -> np.ndarray:
        """
        Returns indices of bunches belonging to a specific train.
        Handles non-uniform bunch spacing by using time-windows.
        """
        if len(self.train_times) == 0:
            return np.array([], dtype=int)
        
        # Handle negative indexing
        if train < 0:
            train = len(self.train_times) + train
            
        start_t = self.train_times[train]
        # End of this train is the start of the next, or the end of the simulation
        end_t = self.train_times[train+1] if (train + 1) < len(self.train_times) else self.time[self.cutoff]
        
        # Filter bunch_times that fall within this specific window
        return np.where((self.bunch_times >= start_t) & (self.bunch_times < end_t))[0]

    def gen_area(self):
        """
        Calculates the area of the beam chamber.
        """
        self.gen_perimeter()

    def gen_buildup(self):
        """
        True if multipacting buildup is detected
        """
        try:
            self.buildup = np.mean(self.smooth_diff[:self.cutoff:self.bunch_step]) > 0
        except:
            self.buildup = False

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
        if self.time.size == 0 or self.time[-1] <= self.t_offs:
            self.half_bunch = self.t_offs
        else:
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
        Detects train starts by comparing total simulation time to pattern duration
        and identifying periodic repetitions in the filling pattern.
        """
        # 1. Physical Check: Do we even have enough time for multiple trains?
        # Total pattern duration = number of slots * bunch spacing
        pattern_duration = len(self.filling_pattern_file) * self.b_spac
        sim_duration = self.time[-1] - self.time[0]

        # If simulation is shorter than one pattern length, it's a single train
        if sim_duration <= 1.8 * pattern_duration:
            # Use the first filled bunch as the start
            filled_indices = np.where(self.filling_pattern_file == 1)[0]
            if len(filled_indices) > 0:
                self.train_times = np.array([self.t_offs + (filled_indices[0] * self.b_spac)])
            else:
                self.train_times = np.array([])
            return

        # 2. Multi-Train Logic: Segment by pattern repetitions
        # Identify which bunches (by index) are filled
        filled_slots = np.where(self.filling_pattern_file == 1)[0]
        if len(filled_slots) == 0:
            self.train_times = np.array([])
            return

        # The first train starts at the first filled slot
        first_train_start = self.t_offs + (filled_slots[0] * self.b_spac)
        
        # Calculate starts for subsequent trains based on pattern duration
        # We find how many repetitions occur in the total time
        num_repetitions = int(np.ceil(sim_duration / pattern_duration))
        self.train_times = np.array([
            first_train_start + (i * pattern_duration) 
            for i in range(num_repetitions)
        ])

class InstabilityModel(SynchedEntry):
    """
    An InstabilityModel corresponds to one PyHEADTAIL instability simulation.
    Extends SynchedEntry to auto-save ONLY computed scalar properties to the DB.
    Large time-series arrays are loaded dynamically or cached locally.
    """
    # Removed time-series arrays (charge_weighted_rms_x, intrabunch_activity) and duplicates
    PROPERTIES: set[str] = {
        "n_turns",              
        "n_slices",             
        "growth_rate_centroid", 
        "growth_rate_mode",     
        "dominant_mode_idx",    
        "tune_centroid",        
        "blowup_turn_first",    
        "max_emittance_ratio",
        "growth_rate_r_squared",
        "instability_threshold",
        "dominant_mode_growth_rate"
    }

    BUNCH_EVOLUTION_PATTERN = "bunch_evolution*.h5"
    SLICE_EVOLUTION_PATTERN = "slice_evolution*.h5"
    _h5 = None

    def __init__(self, db: TinyDB | SimDB, doc: int | Document):
        if InstabilityModel._h5 is None:
            import h5py
            InstabilityModel._h5 = h5py
        result = doc if isinstance(doc, Document) else (db if isinstance(db, TinyDB) else db.db).get(doc_id=doc)
        if result is None: 
            raise KeyError(f"The provided doc_id={doc} does not exist in the database!")
        
        super().__init__(db if isinstance(db, TinyDB) else db.db, result.doc_id, InstabilityModel.PROPERTIES)
        self.doc_id = result.doc_id
        
        for k, v in result.items():
            self.set_sync(k)
            if isinstance(v, list) and len(v) > 0 and not isinstance(v[0], str):
                setattr(self, "_"+k, np.array(v, dtype=type(v[0])))
            else:
                setattr(self, "_"+k, v)
        
        self._bunch_data_cache: dict[str, np.ndarray] = {}
        self._slice_data_cache: dict[str, np.ndarray] = {}
        self._bunch_data_loaded = False
        self._slice_data_loaded = False

    def _load_h5_files(self, file_pattern: str, key: str = 'Bunch') -> dict[str, np.ndarray]:
        pattern_path = os.path.join(self.path, file_pattern)
        files = sorted(glob(pattern_path))
        if not files:
            return {}
        
        data_dict = {}
        for f_path in files:
            try:
                with self._h5.File(f_path, 'r') as fid:
                    group = fid.get(key, fid)
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
        
        final_data = {}
        for k, list_of_arrays in data_dict.items():
            if len(list_of_arrays) == 1:
                final_data[k] = list_of_arrays[0]
            else:
                shapes = [arr.shape for arr in list_of_arrays if isinstance(arr, np.ndarray)]
                if len(shapes) > 0 and all(len(s) == len(shapes[0]) for s in shapes):
                    try:
                        final_data[k] = np.concatenate(list_of_arrays, axis=0)
                    except ValueError as e:
                        print(f"Warning: Could not concatenate {k} from {file_pattern}: {e}")
                        final_data[k] = list_of_arrays[0]
                else:
                    final_data[k] = list_of_arrays[0]
        return final_data

    @property
    def bunch_data(self) -> dict[str, np.ndarray]:
        if not self._bunch_data_loaded:
            self._bunch_data_cache = self._load_h5_files(self.BUNCH_EVOLUTION_PATTERN, key='Bunch')
            self._bunch_data_loaded = True
        return self._bunch_data_cache

    @property
    def slice_data(self) -> dict[str, np.ndarray]:
        if not self._slice_data_loaded:
            self._slice_data_cache = self._load_h5_files(self.SLICE_EVOLUTION_PATTERN, key='Slices')
            self._slice_data_loaded = True
        return self._slice_data_cache

    # --- Time-Series Vector Properties (Cached locally, excluded from DB sync) ---
    @property
    def charge_weighted_rms_x(self) -> np.ndarray:
        if not hasattr(self, '_charge_weighted_rms_x_cache'):
            if 'mean_x' not in self.slice_data or 'n_macroparticles_per_slice' not in self.slice_data:
                self._charge_weighted_rms_x_cache = np.array([])
            else:
                mean_x = self.slice_data['mean_x']
                weights = self.slice_data['n_macroparticles_per_slice']
                if mean_x.ndim != 2 or weights.ndim != 2 or mean_x.shape != weights.shape:
                    self._charge_weighted_rms_x_cache = np.array([])
                else:
                    weighted_mean = np.sum(mean_x * weights, axis=0) / (np.sum(weights, axis=0) + 1e-12)
                    self._charge_weighted_rms_x_cache = np.sqrt(
                        np.sum((mean_x - weighted_mean)**2 * weights, axis=0) / (np.sum(weights, axis=0) + 1e-12)
                    )
        return self._charge_weighted_rms_x_cache

    @property
    def intrabunch_activity(self) -> np.ndarray:
        if not hasattr(self, '_intrabunch_activity_cache'):
            rms = self.charge_weighted_rms_x
            window = 21
            if len(rms) == 0:
                self._intrabunch_activity_cache = np.array([])
            elif len(rms) <= window:
                self._intrabunch_activity_cache = rms
            else:
                self._intrabunch_activity_cache = scipy.signal.savgol_filter(rms, window, 3)
        return self._intrabunch_activity_cache

    # --- Standard Scalar/Array Getters ---
    @property
    def mean_x(self) -> np.ndarray:
        if 'mean_x' in self.bunch_data: return self.bunch_data['mean_x']
        if 'mean_x' in self.slice_data: return np.mean(self.slice_data['mean_x'], axis=0)
        return np.array([])

    @property
    def mean_y(self) -> np.ndarray:
        if 'mean_y' in self.bunch_data: return self.bunch_data['mean_y']
        if 'mean_y' in self.slice_data: return np.mean(self.slice_data['mean_y'], axis=0)
        return np.array([])

    @property
    def epsn_x(self) -> np.ndarray:
        if 'epsn_x' in self.bunch_data: return self.bunch_data['epsn_x']
        if 'epsn_x' in self.slice_data: return np.mean(self.slice_data['epsn_x'], axis=0)
        return np.array([])

    @property
    def epsn_y(self) -> np.ndarray:
        if 'epsn_y' in self.bunch_data: return self.bunch_data['epsn_y']
        if 'epsn_y' in self.slice_data: return np.mean(self.slice_data['epsn_y'], axis=0)
        return np.array([])

    @property
    def sigma_x(self) -> np.ndarray:
        if 'sigma_x' in self.bunch_data: return self.bunch_data['sigma_x']
        if 'sigma_x' in self.slice_data: return np.mean(self.slice_data['sigma_x'], axis=0)
        return np.array([])

    @property
    def sigma_y(self) -> np.ndarray:
        if 'sigma_y' in self.bunch_data: return self.bunch_data['sigma_y']
        if 'sigma_y' in self.slice_data: return np.mean(self.slice_data['sigma_y'], axis=0)
        return np.array([])

    @property
    def slice_mean_x(self) -> np.ndarray:
        return self.slice_data.get('mean_x', np.array([]))

    @property
    def slice_mean_y(self) -> np.ndarray:
        return self.slice_data.get('mean_y', np.array([]))

    @property
    def slice_fft_power(self) -> np.ndarray:
        """Computes the PSD matrix. Shape: (Frequency_Bins, Windows/Turns)"""
        if not hasattr(self, '_slice_fft_cache'):
            if len(self.slice_mean_x) == 0 or self.slice_mean_x.ndim != 2:
                self._slice_fft_cache = np.array([])
                return self._slice_fft_cache
            
            data = self.slice_mean_x.T  # Shape: (Turns, Slices)
            n_turns, n_slices = data.shape
            window_size = 64  
            hop_size = 10     
            
            if n_turns < window_size:
                # FIX: Average across slices to preserve a consistent (Freq_Bins, 1) output matrix shape
                fft_result = scipy.fft.rfft(data, axis=0)
                power = np.mean(np.abs(fft_result)**2, axis=1)[:, np.newaxis]
                self._slice_fft_cache = power
                return self._slice_fft_cache

            n_freqs = window_size // 2 + 1
            n_output_turns = (n_turns - window_size) // hop_size + 1
            power_matrix = np.zeros((n_freqs, n_output_turns))
            
            for i in range(n_output_turns):
                start = i * hop_size
                end = start + window_size
                segment = data[start:end, :]
                window = scipy.signal.windows.hann(window_size)
                windowed_seg = segment * window[:, np.newaxis]
                fft_seg = scipy.fft.rfft(windowed_seg, axis=0)
                power_matrix[:, i] = np.mean(np.abs(fft_seg)**2, axis=1)
            
            self._slice_fft_cache = power_matrix
        return self._slice_fft_cache

    @property
    def dominant_mode_freq(self) -> float:
        power = self.slice_fft_power
        if len(power) == 0:
            return 0.0
        freqs = scipy.fft.rfftfreq(64)
        idx = self.dominant_mode_idx  # Triggers dynamic sync lookup
        return float(freqs[idx]) if idx < len(freqs) else 0.0

    def _sync_get(self, attr: str):
        """Dynamic attribute tracker supporting arbitrary non-uniform metadata parameters."""
        if not hasattr(self, "_"+attr):
            if hasattr(self, "gen_"+attr):
                getattr(self, "gen_"+attr)()
            else:
                # Fallback gracefully if database property lacks a specific generator method
                return getattr(self, "_" + attr, None)
        return super()._sync_get(attr)

    # --- DB Synced Parameter Generators ---
    def gen_n_turns(self):
        self.n_turns = len(self.mean_x) if len(self.mean_x) > 0 else 0

    def gen_n_slices(self):
        self.n_slices = self.slice_mean_x.shape[0] if self.slice_mean_x.ndim == 2 else 1

    def gen_growth_rate_centroid(self, fit_window: int | None = None):
        x = self.mean_x
        if len(x) < 10:
            self.growth_rate_centroid, self.growth_rate_r_squared = np.nan, 0.0
            return
        turns = np.arange(len(x))
        mask = np.abs(x) > 1e-12
        if fit_window is not None:
            mask &= (turns < fit_window)
        if np.sum(mask) < 10:
            self.growth_rate_centroid, self.growth_rate_r_squared = np.nan, 0.0
            return
        try:
            coeffs = np.polyfit(turns[mask], np.log(np.abs(x[mask])), 1)
            fitted = np.polyval(coeffs, turns[mask])
            r_squared = 1 - np.sum((np.log(np.abs(x[mask])) - fitted)**2) / \
                        np.sum((np.log(np.abs(x[mask])) - np.mean(np.log(np.abs(x[mask]))))**2)
            self.growth_rate_centroid = coeffs[0]
            self.growth_rate_r_squared = max(0.0, r_squared)
        except Exception:
            self.growth_rate_centroid, self.growth_rate_r_squared = np.nan, 0.0

    def gen_dominant_mode_idx(self):
        """FIX: Moved the advanced spectrogram growth-fitting logic directly into the synced generator."""
        power = self.slice_fft_power
        if len(power) == 0 or power.shape[1] == 0:
            self.dominant_mode_idx = 0
            return

        n_windows = power.shape[1]
        growth_end_idx = min(int(n_windows * 0.2), 2000)
        if growth_end_idx < 10:
            growth_end_idx = n_windows // 2
            
        growth_phase = power[:, :growth_end_idx]
        turns = np.arange(growth_end_idx)
        growth_rates = np.zeros(growth_phase.shape[0])
        
        for i in range(growth_phase.shape[0]):
            valid = growth_phase[i, :] > 1e-12
            if np.sum(valid) < 5:
                growth_rates[i] = -np.inf
                continue
            coeffs = np.polyfit(turns[valid], np.log(growth_phase[i, valid]), 1)
            growth_rates[i] = coeffs[0]
        
        max_idx = 0
        if len(growth_rates) > 1:
            max_idx = np.argmax(growth_rates[1:]) + 1
            if growth_rates[max_idx] <= 0:
                max_idx = 0
        self.dominant_mode_idx = max_idx

    def gen_tune_centroid(self):
        x = self.mean_x
        if len(x) < 2:
            self.tune_centroid = 0.0
            return
        fft_x = scipy.fft.rfft(x)
        power = np.abs(fft_x[1:])**2
        idx = np.argmax(power) + 1 if len(power) > 0 else 0
        freqs = scipy.fft.rfftfreq(len(x))
        self.tune_centroid = freqs[idx] if idx < len(freqs) else 0.0

    def gen_growth_rate_mode(self):
        self.growth_rate_mode = self.growth_rate_centroid

    def gen_blowup_turn_first(self):
        epsn = self.epsn_x
        if len(epsn) == 0 or epsn[0] == 0:
            self.blowup_turn_first = np.nan
            return
        mask = epsn > (epsn[0] * 1.2)
        self.blowup_turn_first = float(np.argmax(mask)) if np.any(mask) else np.nan

    def gen_max_emittance_ratio(self):
        epsn = self.epsn_x
        if len(epsn) == 0 or epsn[0] == 0:
            self.max_emittance_ratio = np.nan
        else:
            self.max_emittance_ratio = float(np.max(epsn) / epsn[0])

    def gen_growth_rate_r_squared(self):
        if not hasattr(self, '_growth_rate_r_squared'):
            self.gen_growth_rate_centroid()

    def gen_instability_threshold(self):
        self.instability_threshold = bool(self.growth_rate_centroid > 0.001)

    def gen_dominant_mode_growth_rate(self):
        power = self.slice_fft_power
        idx = self.dominant_mode_idx
        if len(power) == 0 or idx == 0:
            self.dominant_mode_growth_rate = np.nan
            return
        n_windows = power.shape[1]
        growth_end_idx = min(int(n_windows * 0.2), 2000)
        if growth_end_idx < 10:
            growth_end_idx = n_windows // 2
        mode_signal = power[idx, :growth_end_idx]
        turns = np.arange(growth_end_idx)
        valid = mode_signal > 1e-12
        if np.sum(valid) < 5:
            self.dominant_mode_growth_rate = np.nan
            return
        coeffs = np.polyfit(turns[valid], np.log(mode_signal[valid]), 1)
        self.dominant_mode_growth_rate = float(coeffs[0])

    def run_analysis(self):
        for prop in self.PROPERTIES:
            getattr(self, "gen_"+prop)()
        return self

class DataSelector(object):
    """A DataSelector chooses data for fitting or other analysis from an ECModel."""
    def __init__(self, use_central_density: bool = False, use_train: int = -1):
        self.use_central_density = use_central_density
        self.use_train = use_train

    def select(self, model: "ECModel") -> tuple[np.ndarray, np.ndarray]:
        try:
            train = model.train_to_index(self.use_train)
            if len(train) == 0:
                return np.array([]), np.array([])
            
            y_src = model.central_density if self.use_central_density else model.N_electrons
            return (model.time[train], y_src[train])
        except (IndexError, ValueError):
            return np.array([]), np.array([])


class BeforeBunchSelector(DataSelector):
    """Selects the points immediately before each bunch passage as the fitting points."""
    def select(self, model: "ECModel") -> tuple[np.ndarray, np.ndarray]:
        valid = model.train_to_bunches(self.use_train)
        if len(valid) == 0:
            return np.array([]), np.array([])
            
        # Find index right before each bunch passes
        idx = model.time_to_index(model.bunch_times[valid] - model.half_bunch)
        idx = np.clip(idx, 0, model.time.size - 1)
        
        y_src = model.central_density if self.use_central_density else model.N_electrons
        return (model.time[idx] - model.time[idx[0]], y_src[idx])


class BunchAverageSelector(DataSelector):
    """Selects the average data points within each bunch interval, robust against non-uniform layouts."""
    def select(self, model: "ECModel") -> tuple[np.ndarray, np.ndarray]:
        valid = model.train_to_bunches(self.use_train)
        if len(valid) == 0:
            return np.array([]), np.array([])

        y_src = model.central_density if self.use_central_density else model.N_electrons
        
        # Convert absolute timestamps to raw array start indices
        starts = np.clip(model.time_to_index(model.bunch_times[valid] - model.half_bunch), 0, y_src.size - 1)
        
        # Non-Uniform Protection: Compute safe window lengths
        # If the next bunch arrives early, shrink the window to match the actual distance.
        max_step = model.bunch_step
        if len(starts) > 1:
            distances = np.diff(starts)
            steps = np.append(np.minimum(max_step, distances), max_step)
        else:
            steps = np.array([max_step])
            
        ends = np.clip(starts + steps, 0, y_src.size)

        # Calculate exact average time coordinates and values per slice
        t_out = np.array([np.mean(model.time[s:e]) for s, e in zip(starts, ends)])
        y_out = np.array([np.mean(y_src[s:e]) for s, e in zip(starts, ends)])

        return (t_out - t_out[0], y_out)

class MaskDataSelector(DataSelector):
    """Wraps an existing selector and applies a custom mask array."""
    def __init__(self, base_selector: DataSelector, mask: np.ndarray | Callable[[np.ndarray, np.ndarray], np.ndarray]):
        self.base = base_selector
        self.use_central_density = self.base.use_central_density
        self.use_train = self.base.use_train
        self.mask = mask
    
    def select(self, model: ECModel) -> tuple[np.ndarray, np.ndarray]:
        xdata, ydata = self.base.select(model)
        mask = self.mask(xdata, ydata) if callable(self.mask) else self.mask
        return (xdata[mask], ydata[mask])

class Fit(object):
    """A Fit is the base class for applying a curve fit to an ECModel."""
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
        for v, b in bound_info.items():
            if (i := self.variables.index(v)) != -1:
                self.bounds_lower[i - 1], self.bounds_upper[i - 1] = b[0], b[1]

    def initial(self, initial_info: dict):
        for v, g in initial_info.items():
            if (i := self.variables.index(v)) != -1:
                self.initial_guess[i - 1] = g

    def fix(self, fix_info: dict):
        for v, f in fix_info.items():
            if (i := self.variables.index(v)) != -1:
                self.fixed[i - 1], self.fixed_values[i - 1] = True, f

    def limit_estimate(self, xdata: np.ndarray, ydata: np.ndarray) -> float:
        """
        Estimates or constrains the mathematical saturation limit yc.
        
        Resolves late buildup starts by isolating the active signal window.
        Adapts structurally to both FurmanNoPhoto (geometric inversion) and 
        FurmanPhoto (quadratic Riccati root-finding) regimes.
        """
        if (n := len(ydata)) == 0: return 0.0
        max_y = float(np.max(ydata))
        if n <= 5 or max_y == 0: return max_y
        
        # 1. Compute derivatives globally for terminal tail checks
        dydx = np.diff(ydata) / np.where(np.diff(xdata) == 0, 1e-12, np.diff(xdata))
        tail_growth = np.mean(dydx[-3:])
        
        # Curve has naturally flattened or turned downward -> Already saturated
        if tail_growth <= 0.02 * np.max(dydx) or tail_growth <= 0:
            return max_y

        # 2. Isolate the Active Buildup Zone (Fixes the late-start edge case)
        # Define active zone where density has risen above a baseline floor (e.g., 2%)
        active_mask = ydata > 0.02 * max_y
        active_idx = np.where(active_mask)[0]
        
        if len(active_idx) < 5: 
            return max_y * 1.5  # Not enough active points to extract curvature reliably

        # Slice data down to the active horizon
        y_active = ydata[active_idx]
        x_active = xdata[active_idx]
        dydx_active = dydx[active_idx[:-1]]  # Align derivative length
        y_mid_active = 0.5 * (y_active[:-1] + y_active[1:])
        
        n_act = len(y_active)

        if getattr(self, 'photoem_flag', 0) > 0:
            # --- PHOTOEMISSION REGIME (Riccati Quadratic Fit) ---
            try:
                # Fit: dydx = p0*y^2 + p1*y + p2
                p = np.polyfit(y_mid_active, dydx_active, 2)
                
                # For a stable saturating curve, the y^2 coefficient must be negative
                if p[0] < 0:
                    roots = np.roots(p)
                    # We want the real root that represents forward saturation (yc > max_y)
                    valid_roots = roots[np.isreal(roots) & (roots > max_y)]
                    if len(valid_roots) > 0:
                        return float(np.clip(np.min(valid_roots), max_y, max_y * 5.0))
            except (np.linalg.LinAlgError, ValueError):
                pass
            
            return max_y * 2.0  # Photoemission fallback (generally higher/faster scaling)

        else:
            # --- NO-PHOTOEMISSION REGIME (3-Point Geometric Inversion) ---
            # Sample 3 perfectly evenly spaced points strictly *within* the active window
            k = (n_act - 1) // 2
            y1 = y_active[n_act - 1 - 2 * k]
            y2 = y_active[n_act - 1 - k]
            y3 = y_active[n_act - 1]
            
            denom = y1 * y3 - y2**2
            
            if denom < 0:  # Confirms curve is concave/decelerating toward a ceiling
                yc_alg = (2 * y1 * y2 * y3 - y2**2 * (y1 + y3)) / denom
                if yc_alg > max_y:
                    return float(min(yc_alg, max_y * 5.0))
                    
            return max_y * 1.5

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

    def scale_factor(self, model: ECModel, xdata: np.ndarray, ydata: np.ndarray) -> tuple[float, float]:
        self._mset("scale_x", 1, model)
        self._mset("scale_y", 1, model)
        return (1.0, 1.0)

    def fit(self, model: ECModel, refit: bool = False, xdata: np.ndarray | None = None, ydata: np.ndarray | None = None) -> np.ndarray | None:
        if refit or (not self._mget("success", model)):
            try:
                if xdata is None or ydata is None:
                    xdata, ydata = self.select_data(model)
                scale_x, scale_y = self.scale_factor(model, xdata, ydata)
                result, covariance = scipy.optimize.curve_fit(
                    self._make_target(), scale_x * xdata, scale_y * ydata,
                    p0=self.initial_guess[~self.fixed],
                    bounds=(self.bounds_lower[~self.fixed], self.bounds_upper[~self.fixed]),
                    loss="arctan", maxfev=int(1e5), nan_policy="omit", ftol=1e-10, xtol=1e-10, x_scale="jac"
                )
                perr = np.sqrt(np.diag(covariance))
                f = 0 
                for i, v in enumerate(self.variables[1:]):
                    if self.fixed[i]:
                        self._mset(v, self.fixed_values[i], model); self._mset(v+"_err", 0, model)
                    else:
                        self._mset(v, result[f], model); self._mset(v+"_err", perr[f], model)
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

    def fit_function(self, model: ECModel, params: np.ndarray | None = None) -> Callable[..., np.ndarray]:
        if params is None:
            if (p_fit := self.fit(model)) is None: raise RuntimeError("Cannot return a function for a failed fit!")
            params = p_fit[0]
        scale_x = self._mget("scale_x", model, default=1)
        scale_y = self._mget("scale_y", model, default=1)
        if not scale_x or not scale_y:
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
    """Intermediate base class consolidating common scale factors for all Furman variations."""
    def scale_factor(self, model: ECModel, xdata: np.ndarray, ydata: np.ndarray) -> tuple[float, float]:
        scale_x, scale_y = np.reciprocal(model.b_spac), np.reciprocal(model.mean_intensity)
        self._mset("scale_x", scale_x, model)
        self._mset("scale_y", scale_y, model)
        return (scale_x, scale_y)


class FurmanNoPhotoFit(FurmanBaseFit):
    """Logistic Buildup / Exponential Decay Variant assuming zero photoemission."""
    FURMAN_BUILDUP = """(y0 * yc * np.exp(beta * yc * x)) / (yc + y0 * (np.exp(beta * yc * x) - 1))"""
    FURMAN_DECAY = """y0 * np.exp(beta * yc * x)"""

    def __init__(self, selector: None | DataSelector = None):
        super().__init__("furman_np", self.FURMAN_BUILDUP, ("x", "yc", "beta", "y0"), BunchAverageSelector() if selector is None else selector)

    def fit(self, model: ECModel, refit: bool = False, xdata: np.ndarray | None = None, ydata: np.ndarray | None = None) -> np.ndarray | None:
        if xdata is None or ydata is None:
            xdata, ydata = self.select_data(model)
            
        if model.buildup:
            self.function = self.FURMAN_BUILDUP
            self._compiled_fn = compile(self.function, f"<{self.name}_fitfn>", "eval")
            scale_x, scale_y = self.scale_factor(model, xdata, ydata)
            self.fix({"y0": ydata[0] * scale_y})
            
            lim_est_scaled = self.limit_estimate(xdata, ydata) * scale_y
            max_y_scaled = np.max(ydata) * scale_y
            
            self.bound({
                "yc": (ydata[0] * scale_y, max(lim_est_scaled, max_y_scaled * 1.001)),
                "beta": (1e-6, 3 * model.del_max if getattr(model, "k_pe_st", 0) < 1e-6 else 10 * model.del_max)
            })
            mslope = np.argmin(np.square(ydata - ((ydata[-1] - ydata[0])/2)))
            self.initial({
                "yc": lim_est_scaled,
                "beta": min(1.0, abs(4 * np.mean(np.diff(ydata[mslope-2:mslope+2])) / ((ydata[-1]**2) * scale_y)))
            })
        else:
            if (max_idx := np.argmax(ydata)) > 0:
                xdata, ydata = xdata[max_idx:], ydata[max_idx:]
            if (drop := ydata[0] - (np.mean(ydata[-5:]) if len(ydata) >= 5 else ydata[-1])) > 0:
                mask = ydata >= (ydata[-1] + 0.15 * drop)
                xdata, ydata = xdata[mask], ydata[mask]
                
            xdata -= xdata[0]
            scale_x, scale_y = self.scale_factor(model, xdata, ydata)
            self.function = self.FURMAN_DECAY
            self._compiled_fn = compile(self.function, f"<{self.name}_fitfn>", "eval")
            self.fix({"y0": ydata[0] * scale_y})
            
            dx = (xdata[1] - xdata[0]) * scale_x if len(xdata) > 1 else 1.0
            gamma_est = (np.log(ydata[1]) - np.log(ydata[0])) / dx if len(xdata) > 1 and ydata[1] > 0 and ydata[0] > 0 else -0.1
            if gamma_est >= 0: gamma_est = -0.1
            
            self.bound({"yc": (-np.inf, -1e-12), "beta": (1e-6, np.inf)})
            self.initial({"yc": gamma_est, "beta": 1.0})
            
        return super().fit(model, refit, xdata, ydata)


class FurmanNPMCFit(FurmanBaseFit):
    """No Photoemission variant corrected for background magnetic multi-pole boundaries."""
    FURMAN_NO_PHOTO_MC = "(y0 * yc * np.exp(yc*x*(b0 + 0.5*b1*x))) / (yc + y0 * (np.exp(yc*x*(b0 + 0.5*b1*x)) - 1))"

    def __init__(self, db: SimDB, selector: None | DataSelector = None):
        self.db = db
        super().__init__("furman_npmc", self.FURMAN_NO_PHOTO_MC, ("x", "yc", "b0", "b1", "y0"), BunchAverageSelector() if selector is None else selector)
    
    def fit(self, model: ECModel, refit: bool = False, xdata: np.ndarray | None = None, ydata: np.ndarray | None = None) -> np.ndarray | None:
        if FurmanNoPhotoFit(selector=self.selector).fit(model, refit=refit) is None:
            self._mset("success", False, model)
            setattr(model, f"_{self.name}_fitting_error", "The underlying NoPhoto fit failed.")
            return None
            
        if list(model.B_multip) == [0]:
            self.fix({"y0": model.furman_np_y0, "yc": model.furman_np_yc, "b1": 0})
            self.bound({"b0": (-2 * abs(model.furman_np_beta), 2 * abs(model.furman_np_beta))})
            self.initial({"b0": model.furman_np_beta})
        else:
            self.fix({"y0": model.furman_np_y0, "yc": model.furman_np_yc})
            base_model = self.db.closest(self.db.where(doc_id=model.doc_id)[0], photoemission=False, furman_np_success=True, B_multip=[0])
            base_beta = ECModel(model.db, base_model[0]).furman_np_beta if base_model else (model.furman_np_beta / 2)
            base_beta = base_beta if base_beta > 0 else (model.furman_np_beta / 2)
            
            self.initial({"b0": base_beta, "b1": 0})
            self.bound({"b0": (0, 2 * base_beta), "b1": (-base_beta, base_beta)})
                
        return super().fit(model, refit, xdata, ydata)


class FurmanPhotoFit(FurmanBaseFit):
    """Full Model optimized via stable delta reparameterization to accommodate active photoemission."""
    FURMAN_PHOTO_READABLE = """0.5*(yc + np.sqrt(yc**2 + (4*alpha/beta))*np.tanh((x/2)*np.sqrt(beta*(4*alpha + (yc**2)*beta)) - np.arctanh((yc - 2*y0)*np.sqrt(beta / (4*alpha + (yc**2)*beta)))))"""

    def __init__(self, selector: None | DataSelector = None):
        super().__init__("furman_p", self.FURMAN_PHOTO_READABLE, ("x", "yc", "alpha", "beta", "y0"), BunchAverageSelector() if selector is None else selector)
    
    def fit(self, model, refit: bool = False, xdata: np.ndarray | None = None, ydata: np.ndarray | None = None) -> np.ndarray | None:
        if not model.buildup:
            self._mset("success", False, model)
            setattr(model, f"_{self.name}_fitting_error", "Cannot fit photoemission model for non-buildup simulation.")
            return None

        if not refit and self._mget("success", model):
            return np.array([(self._mget(v, model), self._mget(v+"_err", model)) for v in self.variables[1:]]).T

        if getattr(model, "k_pe_st", 0.0) <= 0:
            if (np_result := FurmanNoPhotoFit(selector=self.selector).fit(model, refit=refit)) is None:
                self._mset("success", False, model); setattr(model, f"_{self.name}_fitting_error", "Zero-photoemission NoPhoto fallback failed.")
                return None
            for i, var in enumerate(["yc", "beta", "y0"]):
                self._mset(var, np_result[0, i], model); self._mset(var+"_err", np_result[1, i], model)
            self._mset("scale_x", model.furman_np_scale_x, model); self._mset("scale_y", model.furman_np_scale_y, model)
            self._mset("alpha", 0.0, model); self._mset("alpha_err", 0.0, model); self._mset("success", True, model)
            return np.array([(self._mget(v, model), self._mget(v+"_err", model)) for v in self.variables[1:]]).T

        try:
            if xdata is None or ydata is None: xdata, ydata = self.select_data(model)
            scale_x, scale_y = self.scale_factor(model, xdata, ydata)
            X, Y = xdata * scale_x, ydata * scale_y
            
            if (np_result := FurmanNoPhotoFit(selector=self.selector).fit(model, refit=refit)) is None:
                self._mset("success", False, model); setattr(model, f"_{self.name}_fitting_error", "The underlying NoPhoto fit failed.")
                return None
                
            dYdX = np.diff(Y) / np.where(np.diff(X) == 0, 1e-12, np.diff(X))
            m0 = max(np.mean(dYdX[:max(1, len(dYdX)//20)]), 1e-12)
            
            yc_init = max(np_result[0, 0], self.limit_estimate(xdata, ydata) * scale_y)
            alpha_init = max(min(model.k_pe_st * scipy.constants.c * model.b_spac * 0.1, m0), 1e-8)
            beta_init = max(4.0 * (np.max(dYdX) - alpha_init) / (yc_init**2), 1e-8)
            delta_p0 = max(0.5 * (-yc_init + np.sqrt(yc_init**2 + 4.0 * alpha_init / beta_init)), 1e-8)

            def stable_delta_target(x, yc, delta, beta):
                D = yc + 2.0 * delta
                E = np.exp(-beta * D * x)
                denom = (Y[0] + delta) + (yc + delta - Y[0]) * E
                return ((yc + delta) * (Y[0] + delta) + (-delta) * (yc + delta - Y[0]) * E) / np.where(denom == 0, 1e-12, denom)

            result, covariance = scipy.optimize.curve_fit(
                stable_delta_target, X, Y, p0=[yc_init, delta_p0, beta_init],
                bounds=([Y[0], 1e-8, 0.0], [yc_init * 1.1, max(np.max(Y) * 5.0, 10.0), max(beta_init * 10.0, 10.0)]),
                loss="arctan", maxfev=int(1e5), nan_policy="omit", ftol=1e-12, xtol=1e-12, x_scale="jac"
            )
            
            yc_fit, delta_fit, beta_fit = result
            yc_err, delta_err, beta_err = np.sqrt(np.diag(covariance))
            
            alpha_fit = beta_fit * (yc_fit + delta_fit) * delta_fit
            jacobian = np.array([beta_fit * delta_fit, beta_fit * (yc_fit + 2.0 * delta_fit), (yc_fit + delta_fit) * delta_fit])
            alpha_err = np.sqrt(max(jacobian @ covariance @ jacobian.T, 0.0))

            self._mset("yc", yc_fit, model); self._mset("yc_err", yc_err, model)
            self._mset("alpha", alpha_fit, model); self._mset("alpha_err", alpha_err, model)
            self._mset("beta", beta_fit, model); self._mset("beta_err", beta_err, model)
            self._mset("y0", Y[0], model); self._mset("y0_err", 0, model)
            self._mset("success", True, model)
            self._mset("train", getattr(self.selector, 'use_train', True), model)
            
            for attr in ["_data", "_smooth", "_smooth_diff"]:
                if hasattr(model, attr): delattr(model, attr)
        except (RuntimeError, ValueError) as e:
            self._mset("success", False, model); setattr(model, f"_{self.name}_fitting_error", str(e))
            return None
            
        self._reset_state()
        return np.array([(self._mget(v, model), self._mget(v+"_err", model)) for v in self.variables[1:]]).T


class PhotoGammaFit(FurmanBaseFit):
    """
    Furman-like photoemission fit with different structure.

    """
    PHOTO_GAMMA = """(yc/2) + gamma*(( (2*gamma - yc + 2*y0)*np.exp(2*beta*gamma*x) - 1) / ( (2*gamma - yc + 2*y0)*np.exp(2*beta*gamma*x) + 1 ))"""

    def __init__(self, selector: None | DataSelector = None):
        super().__init__(
            "pgfit",
            PhotoGammaFit.PHOTO_GAMMA,
            ("x", "yc", "beta", "gamma", "y0"),
            BunchAverageSelector() if selector is None else selector
        )
    
    def fit(self, model, refit: bool = False, xdata: np.ndarray | None = None, ydata: np.ndarray | None = None) -> np.ndarray | None:
        if not refit and self._mget("success", model):
            return np.array([(self._mget(v, model), self._mget(v+"_err", model)) for v in self.variables[1:]]).T

        if xdata is None or ydata is None:
            xdata, ydata = self.select_data(model)
        scale_x, scale_y = self.scale_factor(model, xdata, ydata)
        self.fix({"y0": ydata[0] * scale_y})
        bounds = {
            "yc": (ydata[0] * scale_y, np.inf),
            "beta": (-np.inf, np.inf),
            "gamma": (0, np.inf)
        }
        self.initial({
            "yc": ydata[-1] * scale_y,
            "beta": 1.0 if model.buildup else -1.0,
            "gamma": 0
        })


def fit_all_where(db: SimDB, fit: Fit, refit: bool = False, verbose: bool = True, **search):
    """
    For the database db, apply the Fit to each result of the search (using where()),
    returning an ECModel instance for each result.
    """
    results = db.where(**search)
    for i, result in enumerate(results):
        if hasattr(result, fit.name+"_success") and (getattr(result, fit.name+"_success") == True) and (not refit):
            continue
        try:
            model = ECModel(db.db, result)
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                fit.fit(model, refit=refit)
            if isinstance(db.db.storage, CachingMiddleware):
                db.db.storage.flush()
            if verbose: print("{:<3}/{} {}".format(i, len(results), "S" if fit._mget("success", model) else "F"), end="\r")
        except Exception as e:
            if verbose:
                print(f"Exceptional error in fit of doc_id {model.doc_id}. A file may be missing:", e)
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