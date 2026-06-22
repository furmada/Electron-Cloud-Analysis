"""
Instability Model and Database Components
Handles PyHEADTAIL instability simulation data with dynamic property computation.
"""

from eca import DBFolder, SynchedEntry
from glob import glob
import os
import numpy as np
import scipy.fft
import scipy.signal

from tinydb import TinyDB
from tinydb.table import Document
from eca import SimDB


class InstabilityDBFolder(DBFolder):
    """Database source folder for instability simulations."""
    CONFIG_FOLDER = "pyecloud_config"
    FILENAMES: dict[str, str] = {
        "sey": "secondary_emission_parameters.input",
        "machine": "machine_parameters.input",
        "simulation": "simulation_parameters.input"
    }
    INSTAB_CONF_FILE = "Simulation_parameters.py"

    def __init__(self, path: str, **shared_properties):
        super().__init__(path, **shared_properties)

    def _scan(self):
        """Scan for instability simulation folders."""
        self.sim_folders = set()
        self.run_folders = set()
        for r, folders, files in os.walk(self.path):
            if self.CONFIG_FOLDER not in folders:
                continue
                
            count = sum(1 for f in self.FILENAMES.values() 
                       if os.path.exists(os.path.join(r, self.CONFIG_FOLDER, f)))
            
            if count == len(self.FILENAMES):
                if any(f.endswith(".h5") for f in files):
                    self.sim_folders.add(r)
                else:
                    self.run_folders.add(r)
        
        self.sim_folders = sorted(self.sim_folders)
        self.run_folders = sorted(self.run_folders)

    def config_files_for(self, path: str) -> dict:
        """Return configuration file paths for a simulation folder."""
        base = {k: os.path.join(path, self.CONFIG_FOLDER, f) 
                for k, f in self.FILENAMES.items()}
        base["instability"] = os.path.join(path, self.INSTAB_CONF_FILE)
        return base


class InstabilityModel(SynchedEntry):
    """Represents a single PyHEADTAIL instability simulation."""
    
    PROPERTIES: set[str] = {
        "n_turns", "n_slices", "growth_rate_centroid", "growth_rate_mode",
        "dominant_mode_idx", "tune_centroid", "blowup_turn_first",
        "max_emittance_ratio", "growth_rate_r_squared", "instability_threshold",
        "dominant_mode_growth_rate"
    }

    BUNCH_EVOLUTION_PATTERN = "bunch_evolution*.h5"
    SLICE_EVOLUTION_PATTERN = "slice_evolution*.h5"
    _h5 = None

    def __init__(self, db: TinyDB | SimDB, doc: int | Document):
        if self._h5 is None:
            import h5py
            InstabilityModel._h5 = h5py
        
        # Get document from db
        result = doc if isinstance(doc, Document) else (db if isinstance(db, TinyDB) else db.db).get(doc_id=doc)
        if result is None:
            raise KeyError(f"Document with doc_id={doc} not found in database")
        if isinstance(result, list): result = result[0]

        super().__init__(db if isinstance(db, TinyDB) else db.db, result.doc_id, self.PROPERTIES)
        self.doc_id = result.doc_id
        
        # Initialize cached properties
        for k, v in result.items():
            self.set_sync(k)
            if isinstance(v, list) and v and not isinstance(v[0], str):
                setattr(self, "_" + k, np.array(v, dtype=type(v[0])))
            else:
                setattr(self, "_" + k, v)
        
        self._bunch_data_cache: dict[str, np.ndarray] = {}
        self._slice_data_cache: dict[str, np.ndarray] = {}
        self._bunch_data_loaded = False
        self._slice_data_loaded = False

    def _load_h5_files(self, file_pattern: str, key: str = 'Bunch') -> dict[str, np.ndarray]:
        """Load HDF5 files matching the pattern."""
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
        
        # Combine arrays
        final_data = {}
        for k, list_of_arrays in data_dict.items():
            if len(list_of_arrays) == 1:
                final_data[k] = list_of_arrays[0]
            else:
                try:
                    final_data[k] = np.concatenate(list_of_arrays, axis=0)
                except (ValueError, np.AxisError):
                    final_data[k] = list_of_arrays[0]
        
        return final_data

    @property
    def bunch_data(self) -> dict[str, np.ndarray]:
        """Lazy-load bunch evolution data."""
        if not self._bunch_data_loaded:
            self._bunch_data_cache = self._load_h5_files(self.BUNCH_EVOLUTION_PATTERN, key='Bunch')
            self._bunch_data_loaded = True
        return self._bunch_data_cache

    @property
    def slice_data(self) -> dict[str, np.ndarray]:
        """Lazy-load slice evolution data."""
        if not self._slice_data_loaded:
            self._slice_data_cache = self._load_h5_files(self.SLICE_EVOLUTION_PATTERN, key='Slices')
            self._slice_data_loaded = True
        return self._slice_data_cache

    @property
    def charge_weighted_rms_x(self) -> np.ndarray:
        """Compute charge-weighted RMS in X."""
        if hasattr(self, '_charge_weighted_rms_x_cache'):
            return self._charge_weighted_rms_x_cache
        
        if 'mean_x' not in self.slice_data or 'n_macroparticles_per_slice' not in self.slice_data:
            self._charge_weighted_rms_x_cache = np.array([])
            return self._charge_weighted_rms_x_cache
        
        mean_x = self.slice_data['mean_x']
        weights = self.slice_data['n_macroparticles_per_slice']
        
        if mean_x.ndim != 2 or weights.ndim != 2 or mean_x.shape != weights.shape:
            self._charge_weighted_rms_x_cache = np.array([])
            return self._charge_weighted_rms_x_cache
        
        w_sum = np.sum(weights, axis=0) + 1e-12
        weighted_mean = np.sum(mean_x * weights, axis=0) / w_sum
        self._charge_weighted_rms_x_cache = np.sqrt(
            np.sum((mean_x - weighted_mean) ** 2 * weights, axis=0) / w_sum
        )
        return self._charge_weighted_rms_x_cache

    @property
    def intrabunch_activity(self) -> np.ndarray:
        """Compute smoothed intra-bunch activity."""
        if hasattr(self, '_intrabunch_activity_cache'):
            return self._intrabunch_activity_cache
        
        rms = self.charge_weighted_rms_x
        if len(rms) == 0:
            self._intrabunch_activity_cache = np.array([])
        elif len(rms) <= 21:
            self._intrabunch_activity_cache = rms
        else:
            self._intrabunch_activity_cache = scipy.signal.savgol_filter(rms, 21, 3)
        
        return self._intrabunch_activity_cache

    # Data accessors
    @property
    def mean_x(self) -> np.ndarray:
        """Get X centroid (bunch or slice-averaged)."""
        if 'mean_x' in self.bunch_data:
            return self.bunch_data['mean_x']
        if 'mean_x' in self.slice_data:
            return np.mean(self.slice_data['mean_x'], axis=0)
        return np.array([])

    @property
    def mean_y(self) -> np.ndarray:
        """Get Y centroid (bunch or slice-averaged)."""
        if 'mean_y' in self.bunch_data:
            return self.bunch_data['mean_y']
        if 'mean_y' in self.slice_data:
            return np.mean(self.slice_data['mean_y'], axis=0)
        return np.array([])

    @property
    def epsn_x(self) -> np.ndarray:
        """Get normalized X emittance (bunch or slice-averaged)."""
        if 'epsn_x' in self.bunch_data:
            return self.bunch_data['epsn_x']
        if 'epsn_x' in self.slice_data:
            return np.mean(self.slice_data['epsn_x'], axis=0)
        return np.array([])

    @property
    def epsn_y(self) -> np.ndarray:
        """Get normalized Y emittance (bunch or slice-averaged)."""
        if 'epsn_y' in self.bunch_data:
            return self.bunch_data['epsn_y']
        if 'epsn_y' in self.slice_data:
            return np.mean(self.slice_data['epsn_y'], axis=0)
        return np.array([])

    @property
    def sigma_x(self) -> np.ndarray:
        """Get X beam size (bunch or slice-averaged)."""
        if 'sigma_x' in self.bunch_data:
            return self.bunch_data['sigma_x']
        if 'sigma_x' in self.slice_data:
            return np.mean(self.slice_data['sigma_x'], axis=0)
        return np.array([])

    @property
    def sigma_y(self) -> np.ndarray:
        """Get Y beam size (bunch or slice-averaged)."""
        if 'sigma_y' in self.bunch_data:
            return self.bunch_data['sigma_y']
        if 'sigma_y' in self.slice_data:
            return np.mean(self.slice_data['sigma_y'], axis=0)
        return np.array([])

    @property
    def slice_mean_x(self) -> np.ndarray:
        """Get slice-level X centroid (2D: slices × turns)."""
        return self.slice_data.get('mean_x', np.array([]))

    @property
    def slice_mean_y(self) -> np.ndarray:
        """Get slice-level Y centroid (2D: slices × turns)."""
        return self.slice_data.get('mean_y', np.array([]))

    @property
    def slice_n_macroparticles(self) -> np.ndarray:
        """Get slice-level macroparticle counts."""
        return self.slice_data.get('n_macroparticles_per_slice', np.array([]))

    @property
    def slice_fft_power(self) -> np.ndarray:
        """Compute PSD matrix from slice data. Shape: (Freq_Bins, Windows/Turns)"""
        if hasattr(self, '_slice_fft_cache'):
            return self._slice_fft_cache
        
        if len(self.slice_mean_x) == 0 or self.slice_mean_x.ndim != 2:
            self._slice_fft_cache = np.array([])
            return self._slice_fft_cache
        
        data = self.slice_mean_x.T  # Shape: (Turns, Slices)
        n_turns, n_slices = data.shape
        window_size = 64
        hop_size = 10
        
        if n_turns < window_size:
            # Short signals: average across slices
            fft_result = scipy.fft.rfft(data, axis=0)
            power = np.mean(np.abs(fft_result) ** 2, axis=1)[:, np.newaxis]
            self._slice_fft_cache = power
            return self._slice_fft_cache
        
        # Sliding window FFT
        n_freqs = window_size // 2 + 1
        n_output_turns = (n_turns - window_size) // hop_size + 1
        power_matrix = np.zeros((n_freqs, n_output_turns))
        
        window = scipy.signal.windows.hann(window_size)
        for i in range(n_output_turns):
            start = i * hop_size
            end = start + window_size
            segment = data[start:end, :] * window[:, np.newaxis]
            fft_seg = np.array(scipy.fft.rfft(segment, axis=0))
            power_matrix[:, i] = np.mean(np.abs(fft_seg) ** 2, axis=1)
        
        self._slice_fft_cache = power_matrix
        return self._slice_fft_cache

    @property
    def dominant_mode_freq(self) -> float:
        """Get frequency of dominant mode."""
        power = self.slice_fft_power
        if len(power) == 0:
            return 0.0
        freqs = scipy.fft.rfftfreq(64)
        idx = self.dominant_mode_idx
        return float(freqs[idx]) if idx < len(freqs) else 0.0

    def _sync_get(self, attr: str):
        """Dynamic attribute resolver with generator fallback."""
        if not hasattr(self, "_" + attr):
            gen_method = getattr(self, "gen_" + attr, None)
            if gen_method:
                gen_method()
        return super()._sync_get(attr)

    # ============== DB Synced Parameter Generators ==============

    def gen_n_turns(self):
        """Generate number of turns."""
        self.n_turns = len(self.mean_x) if len(self.mean_x) > 0 else 0

    def gen_n_slices(self):
        """Generate number of slices."""
        self.n_slices = self.slice_mean_x.shape[0] if self.slice_mean_x.ndim == 2 else 1

    def gen_growth_rate_centroid(self, fit_window: int | None = None):
        """Compute growth rate from X centroid via log-fit."""
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
            log_x = np.log(np.abs(x[mask]))
            coeffs = np.polyfit(turns[mask], log_x, 1)
            fitted = np.polyval(coeffs, turns[mask])
            ss_res = np.sum((log_x - fitted) ** 2)
            ss_tot = np.sum((log_x - np.mean(log_x)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            self.growth_rate_centroid = coeffs[0]
            self.growth_rate_r_squared = max(0.0, float(r_squared))
        except Exception:
            self.growth_rate_centroid, self.growth_rate_r_squared = np.nan, 0.0

    def gen_dominant_mode_idx(self):
        """Find dominant mode from FFT power matrix."""
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
        growth_rates = np.full(growth_phase.shape[0], -np.inf)
        
        for i in range(growth_phase.shape[0]):
            valid = growth_phase[i, :] > 1e-12
            if np.sum(valid) >= 5:
                try:
                    coeffs = np.polyfit(turns[valid], np.log(growth_phase[i, valid]), 1)
                    growth_rates[i] = coeffs[0]
                except Exception:
                    pass
        
        # Find mode with highest growth rate (skip index 0)
        if len(growth_rates) > 1:
            max_idx = np.argmax(growth_rates[1:]) + 1
            self.dominant_mode_idx = max_idx if growth_rates[max_idx] > 0 else 0
        else:
            self.dominant_mode_idx = 0

    def gen_tune_centroid(self):
        """Compute tune from FFT of centroid."""
        x = self.mean_x
        if len(x) < 2:
            self.tune_centroid = 0.0
            return
        
        fft_x = scipy.fft.rfft(x)
        power = np.abs(fft_x[1:]) ** 2
        idx = np.argmax(power) + 1 if len(power) > 0 else 0
        freqs = scipy.fft.rfftfreq(len(x))
        self.tune_centroid = freqs[idx] if idx < len(freqs) else 0.0

    def gen_growth_rate_mode(self):
        """Set mode growth rate (currently same as centroid)."""
        self.growth_rate_mode = self.growth_rate_centroid

    def gen_blowup_turn_first(self):
        """Find first turn where emittance exceeds 20% threshold."""
        epsn = self.epsn_x
        if len(epsn) == 0 or epsn[0] == 0:
            self.blowup_turn_first = np.nan
            return
        
        mask = epsn > (epsn[0] * 1.2)
        self.blowup_turn_first = float(np.argmax(mask)) if np.any(mask) else np.nan

    def gen_max_emittance_ratio(self):
        """Compute max/initial emittance ratio."""
        epsn = self.epsn_x
        if len(epsn) == 0 or epsn[0] == 0:
            self.max_emittance_ratio = np.nan
        else:
            self.max_emittance_ratio = float(np.max(epsn) / epsn[0])

    def gen_growth_rate_r_squared(self):
        """Ensure R² is computed (depends on centroid fit)."""
        if not hasattr(self, '_growth_rate_r_squared'):
            self.gen_growth_rate_centroid()

    def gen_instability_threshold(self):
        """Determine if growth rate exceeds instability threshold."""
        self.instability_threshold = bool(self.growth_rate_centroid > 0.001)

    def gen_dominant_mode_growth_rate(self):
        """Compute growth rate of dominant mode."""
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
        
        try:
            coeffs = np.polyfit(turns[valid], np.log(mode_signal[valid]), 1)
            self.dominant_mode_growth_rate = float(coeffs[0])
        except Exception:
            self.dominant_mode_growth_rate = np.nan

    def run_analysis(self):
        """Run all analysis generators and save to DB."""
        for prop in self.PROPERTIES:
            getattr(self, "gen_" + prop)()
        return self