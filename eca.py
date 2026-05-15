"""
EcA - Electron Cloud Analysis v1
@author Adam Furman
@email adam.furman@cern.ch

For use with PyECLOUD.
"""
from typing import Callable, Iterable
from functools import partial as functools_partial

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
        Each parameter should correspond to one entry in sweep.
        Each entry in sweep should be a list of values that parameter can take.
        """
        parameters = list(parameters)
        sweep = list(sweep)
        lengths = [len(list(s)) for s in sweep]
        counters = np.zeros(len(parameters), dtype=int)

        configurations = []
        for _ in range(np.prod(np.array(lengths))):
            configurations.append({**constant, **{
                parameters[c]: sweep[c][counters[c]] for c in range(counters.size)
            }})
            
            j = len(counters) - 1
            counters[j] += 1
            while counters[j] >= lengths[j]:
                counters[j] = 0
                j -= 1
                counters[j] += 1
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
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

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
            self.train_times = np.concat((np.array([self.bunch_times[0]]), self.bunch_times[where_gaps]), axis=None)
        else:
            # No trains
            self.train_times = np.array([self.bunch_times[0]])

class InstabilityModel(SynchedEntry):
    """
    An InstabilityModel corresponds to one PyHEADTAIL instability simulation.
    Extends SynchedEntry to auto-save analysis results (growth rates, blow-up times) to the DB.
    Handles multi-partition .h5 files (e.g., bunch_evolution_00.h5, bunch_evolution_01.h5).
    """
    PROPERTIES: set[str] = {
        "data_files",           # Map of .h5 files found
        "bunch_data_file",      # Path to the primary bunch evolution file (merged)
        "slice_data_file",      # Path to the primary slice evolution file (merged)
        "n_turns",              # Total number of turns
        "growth_rate_centroid", # Calculated growth rate from centroid fit
        "growth_rate_mode",     # Calculated growth rate of dominant mode
        "dominant_mode_idx",    # Index of dominant mode in FFT
        "tune_centroid",        # Approximate tune from centroid FFT
        "blowup_turn_first",    # Turn where emittance crosses threshold
        "blowup_turn_last",     # Last turn of simulation
        "max_emittance_ratio",  # Max ratio of final/initial emittance
        "mean_x",               # Loaded mean_x array (cached)
        "mean_y",               # Loaded mean_y array
        "epsn_x",               # Loaded epsn_x array
        "epsn_y",               # Loaded epsn_y array
        "sigma_z",              # Loaded sigma_z array
        "macroparticlenumber",  # Loaded MP count
    }

    BUNCH_EVOLUTION_PATTERN = "bunch_evolution*.h5"
    SLICE_EVOLUTION_PATTERN = "slice_evolution*.h5"
    _h5 = None

    def __init__(self, db: TinyDB | SimDB, doc: int | Document):
        # Lazy load h5py to avoid import errors if not needed
        if InstabilityModel._h5 is None:
            import h5py
            InstabilityModel._h5 = h5py
        result = doc if isinstance(doc, Document) else (db if isinstance(db, TinyDB) else db.db).get(doc_id=doc)
        if result is None: 
            raise KeyError(f"The provided doc_id={doc} does not exist in the database!")
        super().__init__(db if isinstance(db, TinyDB) else db.db, result.doc_id, InstabilityModel.PROPERTIES)
        self.doc_id = result.doc_id
        # Populate from DB
        for k, v in result.items():
            self.set_sync(k)
            # Handle list conversion for numpy arrays if stored as lists
            if isinstance(v, list) and len(v) > 0 and not isinstance(v[0], str):
                setattr(self, "_" + k, np.array(v, dtype=type(v[0])))
            else:
                setattr(self, "_" + k, v)

    def _load_h5_files(self, file_pattern: str, key: str = 'Bunch') -> dict[str, np.ndarray]:
        """
        Loads and concatenates multiple .h5 files matching a pattern.
        Handles sequential parts (00, 01, ...) and transposes if necessary.
        Returns a dictionary of concatenated arrays.
        """
        # Find all matching files in the simulation folder
        pattern_path = os.path.join(self.path, file_pattern)
        files = sorted(glob(pattern_path))
        if not files:
            return {}
        data_dict = {}
        first_file = True
        for f_path in files:
            try:
                with self._h5.File(f_path, 'r') as fid:
                    # Determine if we need to look inside a group (e.g., 'Bunch' or 'Slices')
                    if key in fid:
                        group = fid[key]
                    else:
                        # Fallback: assume root level keys are the data
                        group = fid
                    current_data = {}
                    for k in group.keys():
                        val = np.array(group[k])
                        # Handle scalar vs array
                        if val.shape == ():
                            val = val.item()
                        current_data[k] = val
                    if first_file:
                        data_dict = {k: [v] for k, v in current_data.items()}
                        first_file = False
                    else:
                        for k, v in current_data.items():
                            if k in data_dict:
                                # Concatenate along the first axis (turns)
                                # Handle case where data might be 1D or 2D
                                if isinstance(data_dict[k][0], np.ndarray):
                                    data_dict[k].append(v)
                                else:
                                    # If it's a scalar, we can't concatenate, skip or overwrite?
                                    # Usually scalars are constant, so we ignore subsequent ones
                                    pass
            except Exception as e:
                print(f"Warning: Failed to load {f_path}: {e}")
                continue
        # Concatenate lists into final arrays
        final_data = {}
        for k, list_of_arrays in data_dict.items():
            if len(list_of_arrays) == 1:
                final_data[k] = list_of_arrays[0]
            else:
                # Stack along axis 0 (turns)
                try:
                    final_data[k] = np.concatenate(list_of_arrays, axis=0)
                except ValueError:
                    # Fallback if shapes don't match perfectly (rare)
                    final_data[k] = np.array(list_of_arrays) # Keep as object array if mismatch
        return final_data

    @property
    def data(self) -> dict[str, np.ndarray]:
        """
        Lazy-loaded property containing all merged simulation data.
        """
        if not hasattr(self, "_data_cache"):
            self._data_cache = {}
            # Load Bunch Evolution
            # Pattern matches: bunch_evolution.h5 OR bunch_evolution_00.h5, bunch_evolution_01.h5
            bunch_files = self._load_h5_files(InstabilityModel.BUNCH_EVOLUTION_PATTERN, key='Bunch')
            if bunch_files:
                self._data_cache.update(bunch_files)
                self.bunch_data_file = list(bunch_files.keys())[0] # Store reference to first file
            # Load Slice Evolution
            slice_files = self._load_h5_files(InstabilityModel.SLICE_EVOLUTION_PATTERN, key='Slices')
            if slice_files:
                self._data_cache.update(slice_files)
                self.slice_data_file = list(slice_files.keys())[0]
            # If no data found, raise error
            if not self._data_cache:
                raise IOError(f"No .h5 data files found in {self.path}")
        return self._data_cache

    def gen_n_turns(self):
        """Total number of turns in the simulation."""
        if 'mean_x' in self.data:
            self.n_turns = len(self.data['mean_x'])
        else:
            self.n_turns = 0

    def gen_mean_x(self):
        """Extract mean_x from data."""
        if 'mean_x' in self.data:
            self.mean_x = self.data['mean_x']
        else:
            self.mean_x = np.array([])

    def gen_mean_y(self):
        """Extract mean_y from data."""
        if 'mean_y' in self.data:
            self.mean_y = self.data['mean_y']
        else:
            self.mean_y = np.array([])

    def gen_epsn_x(self):
        """Extract normalized emittance X."""
        if 'epsn_x' in self.data:
            self.epsn_x = self.data['epsn_x']
        else:
            self.epsn_x = np.array([])

    def gen_epsn_y(self):
        """Extract normalized emittance Y."""
        if 'epsn_y' in self.data:
            self.epsn_y = self.data['epsn_y']
        else:
            self.epsn_y = np.array([])

    def gen_sigma_z(self):
        """Extract bunch length."""
        if 'sigma_z' in self.data:
            self.sigma_z = self.data['sigma_z']
        else:
            self.sigma_z = np.array([])

    def gen_macroparticlenumber(self):
        """Extract macroparticle count."""
        if 'macroparticlenumber' in self.data:
            self.macroparticlenumber = self.data['macroparticlenumber']
        else:
            self.macroparticlenumber = np.array([])

    def gen_growth_rate_centroid(self):
        """
        Calculates the growth rate of the centroid oscillation.
        Fits log(|mean_x|) to a line for the rising portion.
        """
        if not hasattr(self, 'mean_x') or len(self.mean_x) < 10:
            self.growth_rate_centroid = np.nan
            return
        x = self.mean_x
        turns = np.arange(len(x))
        # Filter out zeros/negatives for log
        mask = np.abs(x) > 1e-12
        if np.sum(mask) < 10:
            self.growth_rate_centroid = np.nan
            return
        # Fit log(|x|) = a*t + b -> growth rate = a
        try:
            coeffs = np.polyfit(turns[mask], np.log(np.abs(x[mask])), 1)
            self.growth_rate_centroid = coeffs[0]
        except Exception:
            self.growth_rate_centroid = np.nan

    def gen_dominant_mode_idx(self):
        """
        Finds the index of the dominant frequency in the centroid FFT.
        """
        if not hasattr(self, 'mean_x') or len(self.mean_x) < 2:
            self.dominant_mode_idx = 0
            return
        fft_x = np.fft.rfft(self.mean_x)
        power = np.abs(fft_x[1:])**2 # Exclude DC
        if len(power) == 0:
            self.dominant_mode_idx = 0
            return
        idx = np.argmax(power) + 1
        self.dominant_mode_idx = idx

    def gen_tune_centroid(self):
        """
        Estimates the tune from the dominant frequency.
        Tune = freq * N_turns (normalized to 0.5 Nyquist)
        Actually: freq in rfftfreq is cycles per turn.
        """
        if not hasattr(self, 'mean_x') or len(self.mean_x) < 2:
            self.tune_centroid = 0.0
            return

        freqs = np.fft.rfftfreq(len(self.mean_x))
        idx = self.dominant_mode_idx
        if idx < len(freqs):
            self.tune_centroid = freqs[idx]
        else:
            self.tune_centroid = 0.0

    def gen_growth_rate_mode(self):
        """
        Estimates growth rate of the dominant mode.
        Simplified: Uses the centroid growth rate as a proxy if mode isolation is complex.
        For a more advanced version, one would isolate the mode via inverse FFT and fit.
        """
        # For now, proxy with centroid rate. 
        # To improve: Filter FFT to dominant bin, inverse FFT, then fit envelope.
        self.growth_rate_mode = self.growth_rate_centroid

    def gen_blowup_turn_first(self):
        """
        Finds the first turn where emittance exceeds 1.2x initial value.
        """
        if not hasattr(self, 'epsn_x') or len(self.epsn_x) == 0:
            self.blowup_turn_first = np.nan
            return
        initial = self.epsn_x[0]
        threshold = initial * 1.2
        mask = self.epsn_x > threshold
        if np.any(mask):
            self.blowup_turn_first = float(np.argmax(mask))
        else:
            self.blowup_turn_first = np.nan

    def gen_max_emittance_ratio(self):
        """Max ratio of emittance to initial."""
        if not hasattr(self, 'epsn_x') or len(self.epsn_x) == 0:
            self.max_emittance_ratio = 1.0
            return
        initial = self.epsn_x[0]
        if initial == 0:
            self.max_emittance_ratio = np.nan
        else:
            self.max_emittance_ratio = float(np.max(self.epsn_x) / initial)

    def run_analysis(self):
        """
        Convenience method to trigger all generators.
        Since properties are lazy-loaded via _sync_get, accessing them triggers calculation.
        """
        _ = self.n_turns
        _ = self.growth_rate_centroid
        _ = self.dominant_mode_idx
        _ = self.tune_centroid
        _ = self.blowup_turn_first
        _ = self.max_emittance_ratio
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
            if i == -1: raise KeyError("Bounds were provided for non-existent variable {}".format(v))
            self.bounds_lower[i - 1] = b[0]
            self.bounds_upper[i - 1] = b[1]

    def initial(self, initial_info: dict):
        """Specify the initial guess for each variable: value"""
        for v, g in initial_info.items():
            i = self.variables.index(v)
            if i == -1: raise KeyError("An inital value was provided for non-existent variable {}".format(v))
            self.initial_guess[i - 1] = g

    def fix(self, fix_info: dict):
        """Specify fixed values for each variable: value. They will not be optimized"""
        for v, f in fix_info.items():
            i = self.variables.index(v)
            if i == -1: raise KeyError("An fixed value was provided for non-existent variable {}".format(v))
            self.fixed[i - 1] = True
            self.fixed_values[i - 1] = f

    def limit_estimate(self, xdata: np.ndarray, ydata: np.ndarray) -> float:
        """Estimate the limit of f(x) using the x and y data"""
        if len(ydata) == 0:
            return 0
        elif len(ydata) <= 5:
            return ydata[-1]
        smooth_diff = smooth(np.diff(ydata), max(ydata.size // 10, 5))
        if np.mean(smooth_diff[-max(ydata.size // 10, 5):]) >= ydata[-1]*1e-2:
            # We have likely not completed the buildup within the simulation window
            return  (3 + (np.argmax(smooth_diff) / smooth_diff.size))*np.max(ydata)
        else:
            # Return an estimate based on the average slope in the last window
            return min(np.mean(ydata[-5:-1]) + max(np.mean(smooth_diff[-max(ydata.size // 10, 5):]), 0), np.max(ydata))

    def _mget(self, attr: str, model: ECModel, default: bool | float | int | np.number | np.ndarray = False):
        return getattr(model, self.name+"_"+attr) if hasattr(model, "_"+self.name+"_"+attr) or hasattr(model, self.name+"_"+attr) else default
    
    def _mset(self, attr: str, value, model: ECModel):
        model.set_sync(self.name+"_"+attr)
        model._sync_set(self.name+"_"+attr, value)

    def _evaluate(self, code, names, f_mask, f_vals, x, *free_params):
        """
        Internal bridge that maps Scipy's (x, *free_params) call 
        to the full set of variables required by the expression.
        """
        eval_context = {"np": np, "scipy": scipy, names[0]: x}
        # Reconstruct the full parameter list from free and fixed values
        free_ptr = 0
        for i, name in enumerate(names[1:]):
            if f_mask[i]:
                eval_context[name] = f_vals[i]
            else:
                eval_context[name] = free_params[free_ptr]
                free_ptr += 1
        # Use a restricted eval on the pre-compiled code object
        return eval(code, eval_context)

    def _make_target(self):
        fixed_mask = self.fixed.copy()
        fixed_vals = self.fixed_values.copy()
        return functools_partial(
            self._evaluate, 
            self._compiled_fn, 
            self.variables, 
            fixed_mask, 
            fixed_vals
        )

    @property
    def full_names(self):
        """
        The full (database entry) names of each variable
        """
        return np.array([(self.name+"_"+v, self.name+"_"+v+"_err") for v in self.variables[1:]], dtype=str)

    def select_data(self, model: ECModel) -> tuple[np.ndarray, np.ndarray]:
        """
        Picks data from the model to fit to.
        """
        return self.selector.select(model)

    def scale_factor(self, model: ECModel, xdata: np.ndarray, ydata: np.ndarray) -> tuple[float | int | np.number, float | int | np.number]:
        """
        Returns the X and Y scale factors for the provided data. Override.
        """
        self._mset("scale_x", 1, model)
        self._mset("scale_y", 1, model)
        return (1, 1)

    def fit(self, model: ECModel, refit: bool = False) -> np.ndarray | None:
        """
        Apply the fit to the ECModel.
        First checks for an existing fit, and skips fitting unless "refit" is True.
        If the fit fails, the exception is stored as _furman_no_photo_fitting_error.
        """
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
                # Store the parameter values
                f = 0 # Counter for free variables (index into result)
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
                if hasattr(model, "_data"): delattr(model, "_data")
                if hasattr(model, "_smooth"): delattr(model, "_smooth")
                if hasattr(model, "_smooth_diff"): delattr(model, "_smooth_diff")
            except (RuntimeError, ValueError) as e:
                self._mset("success", False, model)
                setattr(model, "_"+self.name+"_fitting_error", str(e))
                return None
            self._reset_state()
        return np.array([(self._mget(v, model), self._mget(v+"_err", model)) for v in self.variables[1:]]).T

    def fit_function(self, model: ECModel, params: np.ndarray | None = None) -> Callable[..., float | np.number | np.ndarray]:
        """
        Returns a callable function of one variable, corresponding to the fit function.
        Can optionally override the parameters.
        """
        if params is None:
            params = self.fit(model)
            if params is None: raise RuntimeError("Cannot return a function for a failed fit!")
            params = params[0]
        scale_x = self._mget("scale_x", model, default=1)
        scale_y = self._mget("scale_y", model, default=1)
        if scale_x == False or scale_y == False or scale_y == 0:
            scale_x, scale_y = self.scale_factor(model, *self.select_data(model))
        target = self._make_target()
        if params is None: raise RuntimeError("Cannot return a function for a failed fit!")
        fn = lambda x: target(scale_x * x, *params[~self.fixed]) / scale_y
        setattr(fn, "variables", self.variables)
        setattr(fn, "parameters", params)
        setattr(fn, "model", self.name)
        return fn

    def scrub(self, model: ECModel):
        """
        Remove all parameters of the fit from this model.
        """
        for v in self.variables[1:]:
            if hasattr(model, "_"+self.name+"_"+v):
                delattr(model, self.name+"_"+v)
            if hasattr(model, "_"+self.name+"_"+v+"_err"):
                delattr(model, self.name+"_"+v+"_err")
            if hasattr(model, "_"+self.name+"_success"):
                delattr(model, self.name+"_success")


class FurmanNoPhotoFit(Fit):
    """
    "FURMAN" FULL MODEL (B) - NO PHOTOEMISSION
    x  -> scaled time
    yc -> Critical electron cloud density
    b  -> Furman beta parameter (multipacting)
    y0 -> [fixed] initial value = initial e- concentration
    """

    FURMAN_NO_PHOTO = """(y0 * yc * np.exp(beta * yc * x)) / (yc + y0 * (np.exp(beta * yc * x) - 1))"""

    def __init__(self, selector: None | DataSelector = None):
        super().__init__(
            "furman_np",
            FurmanNoPhotoFit.FURMAN_NO_PHOTO,
            ("x", "yc", "beta", "y0"),
            BunchAverageSelector() if selector is None else selector
        )

    def scale_factor(self, model: ECModel, xdata: np.ndarray, ydata: np.ndarray) -> tuple[float | int | np.number, float | int | np.number]:
        scale_x = np.reciprocal(model.b_spac)
        scale_y = np.reciprocal(model.mean_intensity)
        self._mset("scale_x", scale_x, model)
        self._mset("scale_y", scale_y, model)
        return (scale_x, scale_y)
    
    def fit(self, model: ECModel, refit: bool = False) -> np.ndarray | None:
        xdata, ydata = self.select_data(model)
        _, scale_y = self.scale_factor(model, xdata, ydata)
        self.fix({"y0": ydata[0] * scale_y})
        if model.buildup:
            mslope = np.argmin(np.square(ydata - ((ydata[-1] - ydata[0])/2)))
            bounds = {
                "yc": (ydata[0] * scale_y, self.limit_estimate(xdata, ydata)*scale_y),
                "beta": (0, 3*model.del_max)#max(5, 4*np.max(np.diff(ydata))/((ydata[-1]**2)*scale_y)))
            }
            self.initial({
                "yc": self.limit_estimate(xdata, ydata)*scale_y,
                "beta": min(1, abs(4*np.mean(np.diff(ydata[mslope-2:mslope+2]))/((ydata[-1]**2)*scale_y)))
            })
        else:
            bounds = {
                "yc": (0, np.inf),
                "beta": (-np.inf, 0)
            }
            self.initial({
                "yc": 2*ydata[0]*scale_y,
                "beta": -0.1
            })
        self.bound(bounds)
        return super().fit(model, refit)

class FurmanNPMCFit(FurmanNoPhotoFit):
    """
    "FURMAN" NO PHOTOEMISSION MODEL WITH MAGNETIC CORRECTION
    x  -> scaled time
    yc -> Critical electron cloud density
    b0 -> Furman beta parameter (multipacting)
    b1 -> Magnetic correction
    y0 -> [fixed] initial value = initial e- concentration

    Solves the modified ODE dy/dx = (0) + (b0 + b1*x) * (yc - y(x)) * y(x)
    """

    FURMAN_NO_PHOTO_MC = "(y0 * yc * np.exp(yc*x*(b0 + 0.5*b1*x))) / (yc + y0 * (np.exp(yc*x*(b0 + 0.5*b1*x)) - 1))"

    def __init__(self, db: SimDB, selector: None | DataSelector = None):
        self.db = db
        Fit.__init__(self,
            "furman_npmc",
            FurmanNPMCFit.FURMAN_NO_PHOTO_MC,
            ("x", "yc", "b0", "b1", "y0"),
            BunchAverageSelector() if selector is None else selector
        )
    
    def fit(self, model: ECModel, refit: bool = False) -> np.ndarray | None:
        np_result = FurmanNoPhotoFit(selector=self.selector).fit(model, refit=refit)
        if np_result is None:
            self._mset("success", False, model)
            setattr(model, "_"+self.name+"_fitting_error", "The underlying NoPhoto fit failed.")
            return None
        if list(model.B_multip) == [0]:
            self.fix({
                "y0": model.furman_np_y0,
                "yc": model.furman_np_yc,
                "b1": 0,
            })
            self.bound({"b0": (-2*abs(model.furman_np_beta), 2*abs(model.furman_np_beta))})
            self.initial({"b0": model.furman_np_beta})
        else:
            self.fix({
                    "y0": model.furman_np_y0,
                    "yc": model.furman_np_yc,
                })
            base_search = self.db.where(doc_id=model.doc_id)[0]
            base_model = self.db.closest(base_search, photoemission=False, furman_np_success=True, B_multip=[0])
            if len(base_model) != 0:
                base_model = ECModel(model.db, base_model[0])
                if base_model.furman_np_beta > 0:
                    base_beta = base_model.furman_np_beta
                else:
                    base_beta = (model.furman_np_beta / 2)
                self.initial({
                    "b0": base_beta,
                    "b1": 0
                })
                self.bound({
                    "b0": (0, 2*base_beta),
                    "b1": (-base_beta, base_beta),
                })
            else:
                self.initial({
                    "b0": 0.1,
                    "b1": 0
                })
                self.bound({
                    "b0": (0, np.inf),
                    "b1": (-np.inf, np.inf),
                })
        return Fit.fit(self, model, refit)

class FurmanPhotoFit(Fit):
    """
    "FURMAN" FULL MODEL (B) - NO PHOTOEMISSION
    x  -> scaled time
    yc -> Critical electron cloud density
    a  -> Furman alpha parameter
    b  -> Furman beta parameter (multipacting)
    y0 -> [fixed] initial value = initial e- concentration
    """

    FURMAN_PHOTO = """0.5*(yc + np.sqrt(yc**2 + (4*alpha/beta))*np.tanh((x/2)*np.sqrt(beta*(4*alpha + (yc**2)*beta)) - np.arctanh((yc - 2*y0)*np.sqrt(beta / (4*alpha + (yc**2)*beta)))))"""

    def __init__(self, selector: None | DataSelector = None):
        super().__init__(
            "furman_p",
            FurmanPhotoFit.FURMAN_PHOTO,
            ("x", "yc", "alpha", "beta", "y0"),
            BunchAverageSelector() if selector is None else selector
        )

    def scale_factor(self, model: ECModel, xdata: np.ndarray, ydata: np.ndarray) -> tuple[float | int | np.number, float | int | np.number]:
        scale_x = np.reciprocal(model.b_spac)
        scale_y = np.reciprocal(model.mean_intensity)
        self._mset("scale_x", scale_x, model)
        self._mset("scale_y", scale_y, model)
        return (scale_x, scale_y)
    
    def fit(self, model: ECModel, refit: bool = False) -> np.ndarray | None:
        xdata, ydata = self.select_data(model)
        _, scale_y = self.scale_factor(model, xdata, ydata)
        self.fix({"y0": ydata[0] * scale_y})
        if model.buildup:
            np_result = FurmanNoPhotoFit(selector=self.selector).fit(model, refit=refit)
            if np_result is None:
                self._mset("success", False, model)
                setattr(model, "_"+self.name+"_fitting_error", "The underlying NoPhoto fit failed.")
                return None
            self.bound({
                "yc": (ydata[0] * scale_y, max(self.limit_estimate(xdata, ydata)*scale_y, np_result[0,0])),
                "alpha": (0, model.k_pe_st*scipy.constants.c*model.b_spac),
                "beta": (0, max(np_result[0,1], 1))
            })
            self.initial({
                "yc": self.limit_estimate(xdata, ydata)*scale_y,
                "alpha": model.k_pe_st*scipy.constants.c*model.b_spac,
                "beta": min(1, np_result[0,1])
            })
            return super().fit(model, refit)
        else:
            self._mset("success", False, model)
            setattr(model, "_"+self.name+"_fitting_error", "Cannot fit photoemission model for non-buildup simulation.")
            return None
        

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