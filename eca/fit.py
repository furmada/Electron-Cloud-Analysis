from eca import ECModel, SimDB

import numpy as np
from scipy.optimize import curve_fit
from scipy.constants import c
from functools import partial as functools_partial
from typing import Any, Callable, Iterable
from tinydb.middlewares import CachingMiddleware


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
    def select(self, model: ECModel) -> tuple[np.ndarray, np.ndarray]:
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
        
        # Compute derivatives globally for terminal tail checks
        dydx = np.diff(ydata) / np.where(np.diff(xdata) == 0, 1e-12, np.diff(xdata))
        tail_growth = np.mean(dydx[-5:])
        
        # Curve has naturally flattened or turned downward -> Already saturated
        if tail_growth <= 0.02 * np.max(dydx) or tail_growth <= 0:
            return max_y

        # Isolate the Active Buildup Zone
        # Define active zone where density has risen above a baseline floor
        active_mask = np.log(ydata) > 0.02 * np.log(max_y)
        active_idx = np.where(active_mask)[0]
        if len(active_idx) < 5: 
            return max_y * 1.5  # Not enough active points to extract curvature reliably

        # Slice data down to the active horizon
        y_active, dydx_active = ydata[active_idx], dydx[active_idx[:-1]]
        y_mid_active = 0.5 * (y_active[:-1] + y_active[1:])
        n_act = len(y_active)

        # Use a polynomial to estimate limit
        poly_estimate = max_y
        try:
            # Fit: dydx = p0*y^2 + p1*y + p2
            p = np.polyfit(y_mid_active, dydx_active, 2)
            # For a stable saturating curve, the y^2 coefficient must be negative
            if p[0] < 0:
                roots = np.roots(p)
                # We want the real root that represents forward saturation (yc > max_y)
                valid_roots = roots[np.isreal(roots) & (roots > max_y)]
                if len(valid_roots) > 0:
                    poly_estimate = float(np.clip(np.min(valid_roots), max_y, max_y * 5.0))
        except (np.linalg.LinAlgError, ValueError):
            pass
        # Sample 3 evenly spaced points strictly *within* the active window
        concavity_estimate = max_y
        k = (n_act - 1) // 2
        y1 = y_active[n_act - 1 - 2 * k]
        y2 = y_active[n_act - 1 - k]
        y3 = y_active[n_act - 1]
        
        denom = y1 * y3 - y2**2
        if denom < 0:  # Confirms curve is concave/decelerating toward a ceiling
            yc_alg = (2 * y1 * y2 * y3 - y2**2 * (y1 + y3)) / denom
            if yc_alg > max_y:
                concavity_estimate = float(min(yc_alg, max_y * 5.0)) 
            
        return (poly_estimate + concavity_estimate) / 2

    def _mget(self, attr: str, model: ECModel, default: Any = False):
        return getattr(model, self.name+"_"+attr) if hasattr(model, "_"+self.name+"_"+attr) or hasattr(model, self.name+"_"+attr) else default
    
    def _mset(self, attr: str, value, model: ECModel):
        model.set_sync(self.name+"_"+attr)
        model._sync_set(self.name+"_"+attr, value)

    def _evaluate(self, code, names, f_mask, f_vals, x, *free_params):
        eval_context = {"np": np, names[0]: x}
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
                result, covariance = curve_fit(
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
            p_fit = self.fit(model)
            if p_fit is None:
                raise RuntimeError("Cannot return a function for a failed fit!")
            else:
                params = p_fit[0]
        scale_x = self._mget("scale_x", model, default=1)
        scale_y = self._mget("scale_y", model, default=1)
        if not scale_x or not scale_y:
            scale_x, scale_y = self.scale_factor(model, *self.select_data(model))
        target = self._make_target()
        fn = lambda x: target(scale_x * x, *(params[~self.fixed] if params is not None else [])) / scale_y
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
            m0 = max(float(np.mean(dYdX[:max(1, len(dYdX)//20)])), 1e-12)
            
            L = self.limit_estimate(xdata, ydata) * scale_y
            yc_init = max(np_result[0, 0], L)
            alpha_init = max(min(model.k_pe_st * c * model.b_spac * 0.1, m0), 1e-8)
            beta_init = max(4.0 * (np.max(dYdX) - alpha_init) / (yc_init**2), 1e-8)
            delta_p0 = max(0.5 * (-yc_init + np.sqrt(yc_init**2 + 4.0 * alpha_init / beta_init)), 1e-8)

            def stable_delta_target(x, yc, delta, beta):
                D = yc + 2.0 * delta
                E = np.exp(-beta * D * x)
                denom = (Y[0] + delta) + (yc + delta - Y[0]) * E
                return ((yc + delta) * (Y[0] + delta) + (-delta) * (yc + delta - Y[0]) * E) / np.where(denom == 0, 1e-12, denom)

            result, covariance = curve_fit(
                stable_delta_target, X, Y, p0=[yc_init, delta_p0, beta_init],
                bounds=([Y[0], 1e-8, 0.0], [yc_init * 1.1, max(L, 10.0), max(beta_init * 10.0, 10.0)]),
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
    PHOTO_GAMMA = """(yc/2) + gamma*(( ((2*gamma - yc + 2*y0)/(2*gamma + yc - 2*y0))*np.exp(2*beta*gamma*x) - 1) / ( ((2*gamma - yc + 2*y0)/(2*gamma + yc - 2*y0))*np.exp(2*beta*gamma*x) + 1 ))"""

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
        if model.buildup:
            L = self.limit_estimate(xdata, ydata) * scale_y
            bounds = {
                "yc": (ydata[0] * scale_y, 1.1*L),
                "beta": (0, 4 * model.del_max if getattr(model, "k_pe_st", 0) < 1e-6 else 10 * model.del_max),
                "gamma": (0, L)
            }
            self.initial({
                "yc": max(ydata[0]*scale_y, L),
                "beta": model.del_max,
                "gamma": 0.25*L
            })
        else:
            ymax = np.max(ydata) * scale_y
            bounds = {
                "yc": (0, 10),
                "beta": (-1e3, 0),
                "gamma": (0, max(ymax, 10))
            }
            self.initial({
                "yc": ymax,
                "beta": -1,
                "gamma": ymax
            })
        self.bound(bounds)
        return super().fit(model, refit, xdata, ydata)


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