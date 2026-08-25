import os
from shutil import copy2, rmtree
from itertools import product
import numpy as np
from scipy.io import loadmat
from typing import Any, Iterable, Callable

from tinydb import JSONStorage, TinyDB, Query
from tinydb.storages import MemoryStorage, JSONStorage
from tinydb.table import Document
from tinydb.middlewares import CachingMiddleware

"""
EcA - Electron Cloud Analysis v1.1
@author Adam Furman
@email adam.furman@cern.ch

For use with PyECLOUD.
"""

# Utility Funcitons

def unique_with_tolerance(arr, tol):
    arr = np.sort(arr)
    # Find indices where the difference between consecutive elements exceeds tolerance
    split_indices = np.where(np.diff(arr) > tol)[0] + 1
    # Split the sorted array at those indices and take the first element of each group
    return np.array([group[0] for group in np.split(arr, split_indices)])

def realpath(path: str) -> str:
    return os.path.realpath(os.path.expanduser(path))

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

# Buildup databases

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
        wrote_values = 0
        for filename in DBFolder.FILENAMES.values():
            if filename != DBFolder.FILENAMES["data"]:
                this_file = [prop for prop in values.keys() if prop in self.property_map and self.property_map[prop] == filename]
                wrote_values += len(this_file)
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
        if len(values) > wrote_values:
            raise ValueError("Properties were set that were not found in any template file!")
        for resource in self.resources:
            if resource not in skip_resource:
                rsrc, rdest = realpath(os.path.join(self.path, resource)), realpath(os.path.join(path, resource))
                if rsrc != rdest:
                    copy2(rsrc, rdest)
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

    @staticmethod
    def tweak(path: str, **changes):
        """
        Tweak an existing simulation's configuration files, applying changes.
        """
        if not os.path.exists(path):
            raise ValueError("The provided simulation path does not exist.")
        template = TemplateSim(path)
        template.spawn(path, **changes)

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

    def insert(self, *documents: dict | Document):
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
                return [self.db.get(doc_id=int(i)) for i in search["doc_id"].match if not isinstance(i, tuple)]
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
                context = {"np": np, "ECModel": ECModel, "db": self}
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

    def ensure(self, attr: str, generator: Callable[ECModel, Any], **search) -> list[int]:
        """
        Ensure that each entry (from the query) has the "attr" property, calling the generator if needed.
        See the SynchedEntry ensure method.
        Returns the list of doc_ids where the generator was called.
        """
        without_attr = self.where(**{"_"+attr: WhereNot(Has(attr)), **search})
        for wa in without_attr:
            model = ECModel(self, wa)
            model.ensure(attr, lambda: generator(model))
        return [wa.doc_id for wa in without_attr]

    def extract(self, attrs: Iterable[str] | None, **search) -> SimDB:
        """
        Create a sub-database, extracting the properties "attrs" from documents matching the query.
        if attrs is None, include all attributes
        Ignores missing properties.
        """
        if isinstance(attrs, Iterable):
            attrs = tuple(attrs)
        sdb = TinyDB(storage=MemoryStorage)
        for result in self.where(**search):
            if attrs is None:
                sdb.insert(result)
            else:
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
        attrX = list(attrX) if type(attrX) != str else [attrX]
        attrY = list(attrY) if type(attrY) != str else [attrY]
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
    """
    def __init__(self, *items, str_match=True):
        self.match: list[tuple | Any] = [tuple(i) if isinstance(i, list) or isinstance(i, np.ndarray) else i for i in items]
        if str_match:
            self.match_str = {str(item) for item in items}
    
    def query(self, value, *_) -> bool:
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

class Has(WhereIn):
    """
    Matches if the database entry has the specified attribute (of any value)
    """
    def __init__(self, *attrs):
        self.match = attrs

    def query(self, *data):
        result = data[-1]
        for attr in self.match:
            if attr not in result:
                return False
        return True

class WhereNot(WhereIn):
    """
    Inverts the query.
    """
    def __init__(self, where_in: WhereIn):
        self.where_in = where_in

    def query(self, value, *result) -> bool:
        return not self.where_in.query(value, *result)


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
        if isinstance(result, list): result = result[0]
        super().__init__(db if isinstance(db, TinyDB) else db.db, result.doc_id, ECModel.PROPERTIES)
        self.doc_id: int = result.doc_id
        # Populate this object with information from the database entry
        for k, v in result.items():
            self.set_sync(k)
            if isinstance(v, list) and len(v) > 0 and not type(v[0]) == str:
                setattr(self, "_"+k, np.array(v, dtype=type(v[0])))
            else:
                setattr(self, "_"+k, v)

    @property
    def data(self) -> dict[str, np.ndarray]:
        if not hasattr(self, "_data"):
            self._data = loadmat(os.path.join(self.path, DBFolder.FILENAMES["data"]))
        return self._data

    @property
    def time(self) -> np.ndarray:
        return self.data["t"].ravel()

    @property
    def N_electrons(self) -> np.ndarray:
        return self.data["Nel_timep"].ravel()

    @property
    def central_density(self) -> np.ndarray:
        return self.data["cen_density"].ravel()

    @property
    def N_sec(self) -> np.ndarray:
        return self.data["Nel_emit_time"].ravel()

    @property
    def N_col(self) -> np.ndarray:
        return self.data["Nel_imp_time"].ravel()

    @property
    def intensity(self) -> np.ndarray:
        return self.data["lam_t_array"].ravel()

    @property
    def smooth(self) -> np.ndarray:
        if not hasattr(self, "_smooth"):
            conv_window = int(2*self.bunch_step)
            self._smooth = smooth(self.N_electrons, conv_window)
        return self._smooth

    @property
    def smooth_diff(self) -> np.ndarray:
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

    def time_to_index(self, t: np.ndarray | np.number | float) -> np.ndarray | np.integer | int:
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
            self.buildup: bool = bool(np.mean(self.smooth_diff[:self.cutoff:self.bunch_step]) > 0)
        except:
            self.buildup: bool = False

    def gen_bunch_step(self):
        """
        The number of indexes corresponding to b_spac
        """
        self.bunch_step: int = int(self.time_to_index(self.b_spac))

    def gen_bunch_times(self):
        """
        The times corresponding to maximum beam intensity.
        """
        self.bunch_times: np.ndarray = self.t_offs + (self.b_spac * np.argwhere(self.filling_pattern_file == 1).ravel())

    def gen_cutoff(self):
        """
        The index into the data arrays corresponding to the last bunch passage
        """
        self.cutoff: int = min(min(self.time.size, self.N_electrons.size)-1, int(self.time_to_index(self.bunch_times[-1])))

    def gen_half_bunch(self):
        """
        The true half-width of the simulated Gaussian bunch.
        """
        if self.time.size == 0 or self.time[-1] <= self.t_offs:
            self.half_bunch: float = self.t_offs
        else:
            self.half_bunch: float = self.t_offs - self.time[np.argwhere(self.intensity[:self.time_to_index(self.t_offs)] > 0).ravel()[0]]

    def gen_magnet(self):
        """
        The type of magnetic section.
        """
        if len(self.B_multip) == 1 and self.B_multip[0] == 0:
            self.magnet: int = 0
            return
        self.magnet: int = 2 * len(self.B_multip)

    def gen_mean_intensity(self):
        """
        The mean beam intensity.
        """
        start = np.argwhere(self.intensity[:self.time_to_index(self.t_offs)] > 0).ravel()[0]
        self.mean_intensity: np.floating = np.mean(self.intensity[start:start+self.time_to_index(self.b_spac)])

    def gen_Ne_0(self):
        """
        The initial amount of electrons - note that we do not use N_electrons[0] - instead, we take the amount at the point
        before the start of the first bunch passage.
        """
        self.Ne_0: float = float(self.N_electrons[np.argwhere(self.intensity[:self.time_to_index(self.t_offs)] > 0).ravel()[0]])
    
    def gen_perimeter(self):
        """
        The length of the perimeter of the beam chamber.
        """
        chamber_path = os.path.join(self.path, self.filename_chm)
        if not os.path.exists(chamber_path):
            raise IOError("There is not a chamber definition at: {}".format(chamber_path))
        chamber_data = loadmat(chamber_path)
        Vx, Vy = chamber_data["Vx"].ravel(), chamber_data["Vy"].ravel()
        Vx, Vy = Vx - np.mean(Vx), Vy - np.mean(Vy)
        angle = np.atan2(Vy, Vx)
        angle[angle < 0] += 2*np.pi
        order = np.argsort(angle)
        Vx, Vy = Vx[order], Vy[order]
        Vx, Vy = np.append(Vx, Vx[0]), np.append(Vy, Vy[0])
        self.perimeter: float = np.sum(
            np.sqrt(np.diff(Vx)**2 + np.diff(Vy)**2)
        )
        self.area: float = 0.5 * np.abs(
            np.sum(Vx[:-1] * Vy[1:] - Vx[1:] * Vy[:-1])
        )

    def gen_train_times(self):
        """
        Detects train starts by comparing total simulation time to pattern duration
        and identifying periodic repetitions in the filling pattern.
        """
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
        # Identify which bunches (by index) are filled
        filled_slots = np.where(self.filling_pattern_file == 1)[0]
        if len(filled_slots) == 0:
            self.train_times = np.array([])
            return
        # The first train starts at the first filled slot
        first_train_start = self.t_offs + (filled_slots[0] * self.b_spac)
        # Calculate starts for subsequent trains based on pattern duration
        num_repetitions = int(np.ceil(sim_duration / pattern_duration))
        self.train_times: np.ndarray = np.array([
            first_train_start + (i * pattern_duration) 
            for i in range(num_repetitions)
        ])