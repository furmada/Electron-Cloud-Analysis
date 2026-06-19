from eca import DBFolder, SynchedEntry

import os
import numpy as np
import scipy


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