"""
EcA - Electron Cloud Analysis v1
@author Adam Furman
@email adam.furman@cern.ch

For use with PyECLOUD.
"""
from typing import Callable, Iterable

import numpy as np
import scipy
from tinydb import JSONStorage, TinyDB, Query
from tinydb.storages import MemoryStorage, JSONStorage
from tinydb.table import Document
from tinydb.middlewares import CachingMiddleware

import os

FILENAMES = {
    "data": "Pyecltest.mat",
    "sey": "secondary_emission_parameters.input",
    "machine": "machine_parameters.input",
    "simulation": "simulation_parameters.input",
    "beam": "beam.beam"
}

class DBFolder(object):
    """
    A source folder for building a database.
    Recursively searches the path for folders containing all FILENAMES
    corresponding to a finished simulation.
    -> shared_properties are meta-attributes that all simulations found
        within this DBFolder will inherit.
    """
    def __init__(self, path: str, **shared_properties):
        self.path = path
        self.shared_properties = shared_properties
        self._scan()

    def _scan(self):
        self.sim_folders = set()
        for r, _, files in os.walk(self.path):
            if sum([(1 if f in files else 0) for f in FILENAMES.values()]) == len(FILENAMES):
                self.sim_folders.add(r)
        self.sim_folders = list(sorted(self.sim_folders))

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
                self._add_single(*entry)
            if verbose: print("Done.      ")

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
        if isinstance(value, np.bool):
            # The numpy bool type is also not supported
            return bool(value)
        return value

    def _add_single(self, sim_folder: str, db_folder: DBFolder) -> bool:
        """
        Add a single simulation to the database.
        Returns True if it was added, and False
        if simulation with identical parameters already exists.
        """
        parameters = {**db_folder.shared_properties}
        # Read all input parameter files
        for name, path in FILENAMES.items():
            if name != "data":
                parameters.update(**SimDB.parse_input_file(os.path.join(sim_folder, path)))
        # Check if a simulation with these parameters exists
        search = self.db.search(Query().fragment(parameters))
        if len(search) == 0:
            # No existing entry with these parameters
            parameters["processed"] = False # Mark data file as un-processed
            parameters["path"] = sim_folder
            self.db.insert(parameters)
            return True
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
        if "query" in search and isinstance(search["query"], Query):
            return self.db.search(search["query"])
        non_dynamic = {}
        dynamic = {}
        for k, v in search.items():
            if k.startswith("_") and type(v) == str:
                comp = compile(v, "query_expr", "eval")
                context = {"np": np, "scipy": scipy, "ECModel": ECModel, "db": self}
                dynamic[k] = lambda r: eval(comp, context, {rk: rv for rk, rv in r.items()})
            elif isinstance(v, Callable):
                dynamic[k] = v
            elif isinstance(v, np.ndarray):
                non_dynamic[k] = v.tolist()
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


class SynchedEntry(object):
    """
    A SynchedEntry provides a way to automatically save data back to a TinyDB
    as it is changed by a Python object.
    Starting with an initial db entry (identified by a doc_id), Python properties
    can be marked to sync, meaning that changes to them will automatically be
    propagated to the database.
    """
    def __init__(self, db: TinyDB, doc_id: int, synch: Iterable[str] = []):
        self.db = db
        self.id = doc_id
        self.attached = True
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
            self.db.update({k: SimDB.prepare(v) for k, v in self._changes.items() if not (getattr(self, "_"+k))}, doc_ids=[self.id])

    def _sync_get(self, attr: str):
        """Retrieve the property."""
        return getattr(self, "_"+attr)

    def _sync_set(self, attr: str, value):
        """Set the property, and update its value in the database."""
        if hasattr(self, "_"+attr):
            if isinstance(value, np.ndarray):
                if getattr(self, "_"+attr).shape == value.shape and np.all(getattr(self, "_"+attr) == value):
                    return
            elif getattr(self, "_"+attr) == value:
                return
        setattr(self, "_"+attr, value)
        if self.attached:
            self.db.update({attr: SimDB.prepare(value)}, doc_ids=[self.id])
        else:
            self._changes[attr] = value

    def _sync_delete(self, attr: str):
        """Delete the property and its corresponding value in the database."""
        delattr(self, "_"+attr)
        if self.attached:
            result = self.db.get(doc_id=self.id)
            if not result is None:
                self.db.update({k: v for k, v in result.items() if k != attr}, doc_ids=[self.id])
        else:
            del self._changes[attr]

    def set_sync(self, attr: str):
        """Mark a property to be synced, attaching its getter, setter, and deleter."""
        if not hasattr(type(self), attr):
            setattr(type(self), attr, property(
                lambda s: s._sync_get(attr),
                lambda s, v: s._sync_set(attr, v),
                lambda s: s._sync_delete(attr),
                "DB Synched property {}".format(attr)
                ))

    def ensure(self, attr: str, gen: Callable) -> bool:
        """Ensure that the attribute exists and is synced, calling gen() to make it if it does not, returning True if generated."""
        self.set_sync(attr)
        if not hasattr(self, "_"+attr):
            self._sync_set(attr, gen())
            return True
        return False

def smooth(array: np.ndarray, window: int | np.integer) -> np.ndarray:
    """Smooths a NumPy array using convolution, preserving the initial value."""
    smoothed = np.zeros_like(array)
    if window > array.size / 2: window = array.size // 10
    for i in range(window):
        smoothed[i] = np.mean(array[:i+1])*(1 - (i/window)) + np.mean(array[i+1:i+1+window])*(i/window)
    smoothed[window:-window+1] = np.convolve(array[window:], np.ones(window) / window, "valid")
    for i in range(len(smoothed)-window+1, len(smoothed)):
        smoothed[i] = np.mean(array[i-window//2:])
    return smoothed

class ECModel(SynchedEntry):
    """
    An ECModel corresponds to the analysis of one PyECloud data file.
    """
    PROPERTIES = {
        "buildup",          # True if there is multipacting buildup of EC
        "bunch_step",       # The number of indexes corresponding to b_spac
        "bunch_times",      # Times of each bunch maximum (t_off + n * b_spac)
        "cutoff",           # Index into data arrays corresponding to the last bunch passage
        "half_bunch",       # True (cut-off) half-width of bunch
        "magnet",           # The type of magnetic section (0: drift, 2: dipole, 4: quadrupole, 6: sextupole)
        "Ne_0",             # Initial number of electrons
    }

    def __init__(self, db: TinyDB, doc: int | Document):
        result = doc if isinstance(doc, Document) else db.get(doc_id=doc)
        if result is None: raise KeyError("The provided doc_id={} does not exist in the database!".format(doc))
        # Populate this object with information from the database entry
        for k, v in result.items():
            if isinstance(v, list) and len(v) > 0 and not type(v[0]) == str:
                setattr(self, "_"+k, np.array(v, dtype=type(v[0])))
            else:
                setattr(self, "_"+k, v)
            self.set_sync(k)
        super().__init__(db, result.doc_id, ECModel.PROPERTIES)

    @property
    def data(self):
        if not hasattr("self", "_data"):
            self._data = scipy.io.loadmat(os.path.join(self.path, FILENAMES["data"]))
        return self._data

    @property
    def time(self):
        return self.data["t"].ravel()

    @property
    def N_electrons(self):
        return self.data["Nel_timep"].ravel()

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
        self.bunch_times = self.t_offs + (self.b_spac * np.arange(np.argwhere(self.filling_pattern_file == 1).ravel()[-1]))

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

    def gen_Ne_0(self):
        """
        The initial amount of electrons - note that we do not use N_electrons[0] - instead, we take the amount at the point
        before the start of the first bunch passage.
        """
        self.Ne_0 = self.N_electrons[np.argwhere(self.intensity[:self.time_to_index(self.t_offs)] > 0).ravel()[0]]

class Fit(object):
    """
    A Fit is the base class for applying a curve fit to an ECModel.
    """
    def __init__(self, name: str, function: str, variables: Iterable[str]):
        self.name = name
        self.function = function
        self.variables = tuple(variables)
        self._reset_state()

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

    def _mget(self, attr: str, model: ECModel, default: bool | float | int | np.number | np.ndarray = False):
        return getattr(model, self.name+"_"+attr) if hasattr(model, "_"+self.name+"_"+attr) else default
    
    def _mset(self, attr: str, value, model: ECModel):
        model.set_sync(self.name+"_"+attr)
        model._sync_set(self.name+"_"+attr, value)

    def _make_target(self):
        context = {self.variables[i+1]: self.fixed_values[i] for i in range(len(self.initial_guess)) if self.fixed[i]}
        exec("target = lambda {}, {} : ({})".format(
                self.variables[0],
                ", ".join(v for i, v in enumerate(self.variables[1:]) if not self.fixed[i]),
                self.function),
            globals={"np": np, "scipy": scipy, **context}, locals=context)
        target = context["target"]
        return target

    @property
    def full_names(self):
        """
        The full (database entry) names of each variable
        """
        return np.array([(self.name+"_"+v, self.name+"_"+v+"_err") for v in self.variables[1:]], dtype=str)

    def select_data(self, model: ECModel) -> tuple[np.ndarray, np.ndarray]:
        """
        Picks data from the model to fit to. Override.
        """
        return (model.time[:model.cutoff], model.N_electrons[:model.cutoff])

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
        if (not self._mget("success", model)) or refit:
            try:
                xdata, ydata = self.select_data(model)
                scale_x, scale_y = self.scale_factor(model, xdata, ydata)
                result, covariance = scipy.optimize.curve_fit(
                    self._make_target(),
                    scale_x * xdata, scale_y * ydata,
                    p0=self.initial_guess[~self.fixed],
                    bounds=(self.bounds_lower[~self.fixed], self.bounds_upper[~self.fixed]),
                    loss="soft_l1",
                    maxfev=1e4,
                    nan_policy="omit"
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
            except (RuntimeError, ValueError) as e:
                self._mset("success", False, model)
                setattr(model, "_"+self.name+"_fitting_error", e)
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

class BeforeBunchFit(Fit):
    """
    Selects the points immediately before each bunch passage as the fitting points.
    """

    def select_data(self, model: ECModel) -> tuple[np.ndarray, np.ndarray]:
        pre_bp = np.array(model.time_to_index(model.bunch_times - model.half_bunch))
        valid = pre_bp < model.time.size
        return (model.time[pre_bp[valid]], model.N_electrons[pre_bp[valid]])

class FurmanNoPhotoFit(BeforeBunchFit):
    """
    "FURMAN" FULL MODEL (B) - NO PHOTOEMISSION
    x  -> scaled time
    yc -> Critical electron cloud density
    b  -> Furman beta parameter (multipacting)
    y0 -> [fixed] initial value = initial e- concentration
    """

    FURMAN_NO_PHOTO = """(y0 * yc * np.exp(beta * yc * x)) / (yc + y0 * (np.exp(beta * yc * x) - 1))"""

    def __init__(self):
        super().__init__(
            "furman_np",
            FurmanNoPhotoFit.FURMAN_NO_PHOTO,
            ("x", "yc", "beta", "y0")
        )

    def scale_factor(self, model: ECModel, xdata: np.ndarray, ydata: np.ndarray) -> tuple[float | int | np.number, float | int | np.number]:
        # Ensure that we have a mean intensity (will only regenerate if missing)
        model.ensure(
            "mean_intensity", 
            lambda : np.mean(model.intensity[model.time_to_index(model.t_offs):model.time_to_index(model.t_offs+model.b_spac)])
        )
        scale_x = np.reciprocal(model.b_spac)
        scale_y = np.reciprocal(model.mean_intensity)
        self._mset("scale_x", scale_x, model)
        self._mset("scale_y", scale_y, model)
        return (scale_x, scale_y)
    
    def fit(self, model: ECModel, refit=False) -> np.ndarray | None:
        xdata, ydata = self.select_data(model)
        _, scale_y = self.scale_factor(model, xdata, ydata)
        self.fix({"y0": model.Ne_0 * scale_y})
        mslope = np.argmin(np.square(ydata - (ydata[-1]/2)))
        if model.buildup:
            bounds = {
                "yc": (model.Ne_0 * scale_y, 2*model.N_electrons[model.cutoff]*scale_y),
                "beta": (0, 4*np.max(np.diff(ydata))/((ydata[-1]**2)*scale_y))
            }
            self.initial({
                "yc": model.N_electrons[model.cutoff]*scale_y,
                "beta": 4*np.mean(np.diff(ydata[mslope-2:mslope+2]))/((ydata[-1]**2)*scale_y)
            })
        else:
            bounds = {
                "yc": (0, np.inf),
                "beta": (-np.inf, 0)
            }
            self.initial({
                "yc": 2*model.Ne_0*scale_y,
                "beta": -1
            })
        self.bound(bounds)
        return super().fit(model, refit)

class FurmanPhotoFit(BeforeBunchFit):
    """
    "FURMAN" FULL MODEL (B) - NO PHOTOEMISSION
    x  -> scaled time
    yc -> Critical electron cloud density
    a  -> Furman alpha parameter
    b  -> Furman beta parameter (multipacting)
    y0 -> [fixed] initial value = initial e- concentration
    """

    FURMAN_PHOTO = """0.5*(yc + np.sqrt(yc**2 + (4*alpha/beta))*np.tanh((x/2)*np.sqrt(beta*(4*alpha + (yc**2)*beta)) - np.arctanh((yc - 2*y0)*np.sqrt(beta / (4*alpha + (yc**2)*beta)))))"""

    def __init__(self):
        super().__init__(
            "furman_p",
            FurmanPhotoFit.FURMAN_PHOTO,
            ("x", "yc", "alpha", "beta", "y0")
        )

    def scale_factor(self, model: ECModel, xdata: np.ndarray, ydata: np.ndarray) -> tuple[float | int | np.number, float | int | np.number]:
        # Ensure that we have a mean intensity (will only regenerate if missing)
        model.ensure(
            "mean_intensity", 
            lambda : np.mean(model.intensity[model.time_to_index(model.t_offs):model.time_to_index(model.t_offs+model.b_spac)])
        )
        scale_x = np.reciprocal(model.b_spac)
        scale_y = np.reciprocal(model.mean_intensity)
        self._mset("scale_x", scale_x, model)
        self._mset("scale_y", scale_y, model)
        return (scale_x, scale_y)
    
    def fit(self, model: ECModel, refit=False) -> np.ndarray | None:
        xdata, ydata = self.select_data(model)
        _, scale_y = self.scale_factor(model, xdata, ydata)
        self.fix({"y0": model.Ne_0 * scale_y})
        mslope = np.argmin(np.square(ydata - (ydata[-1]/2)))
        if model.buildup:
            bounds = {
                "yc": (model.Ne_0 * scale_y, 2*model.N_electrons[model.cutoff]*scale_y),
                "alpha": (0, 5*model.k_pe_st),
                "beta": (0, 4*np.max(np.diff(ydata))/((ydata[-1]**2)*scale_y))
            }
            self.initial({
                "yc": model.N_electrons[model.cutoff]*scale_y,
                "alpha": model.k_pe_st,
                "beta": 4*np.mean(np.diff(ydata[mslope-2:mslope+2]))/((ydata[-1]**2)*scale_y)
            })
        else:
            bounds = {
                "yc": (0, np.inf),
                "alpha": (0, 5*model.k_pe_st),
                "beta": (-np.inf, 0)
            }
            self.initial({
                "yc": 2*model.Ne_0*scale_y,
                "alpha": model.k_pe_st,
                "beta": -1
            })
        self.bound(bounds)
        return super().fit(model, refit)

def fit_all_where(db: SimDB, fit: Fit, refit: bool = False, verbose: bool = True, **search) -> list[ECModel]:
    """
    For the database db, apply the Fit to each result of the search (using where()),
    returning an ECModel instance for each result.
    """
    models = [ECModel(db.db, result) for result in db.where(**search)]
    for i, model in enumerate(models):
        fit.fit(model, refit=refit)
        if verbose: print("{:<3}/{} {}".format(i, len(models), "S" if fit._mget("success", model) else "F"), end="\r")
    if verbose: print("Done.        ")
    if isinstance(db.db.storage, CachingMiddleware):
        db.db.storage.flush()
    return models

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
            code = "print(\"Interactive Python prompt, db is loaded\")"
            v = {"db": SimDB(argv[1]), "ECModel": ECModel, "np": np, "scipy": scipy}
            while code != "exit()":
                if "=" in code:
                    exec(code, v)
                else:
                    result = eval(code, v)
                    if not result is None: print("-> {}".format(result))
                code = input ("> ")
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