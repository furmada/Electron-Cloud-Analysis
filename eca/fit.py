import numpy as np
from scipy.optimize import curve_fit, minimize
from scipy.constants import c
from functools import partial as functools_partial
from typing import Any, Callable, Iterable
from tinydb.middlewares import CachingMiddleware

from eca import ECModel, SimDB


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
        starts = np.clip(model.time_to_index(model.bunch_times[valid] - model.half_bunch), 0, y_src.size - 1)
        
        max_step = model.bunch_step
        if len(starts) > 1:
            distances = np.diff(starts)
            steps = np.append(np.minimum(max_step, distances), max_step)
        else:
            steps = np.array([max_step])
            
        ends = np.clip(starts + steps, 0, y_src.size)

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
        self._compiled_fn = compile(self.function, f"<{self.name}_fitfn>", "eval")
        self._reset_state()

    def _reset_state(self):
        n_params = len(self.variables) - 1
        self.initial_guess = np.ones(shape=(n_params,))
        self.bounds_lower = np.full(shape=(n_params,), fill_value=-np.inf)
        self.bounds_upper = np.full(shape=(n_params,), fill_value=np.inf)
        self.fixed = np.zeros(shape=(n_params,), dtype=bool)
        self.fixed_values = np.zeros(shape=(n_params,))

    def bound(self, bound_info: dict):
        for v, b in bound_info.items():
            if v in self.variables[1:]:
                i = self.variables[1:].index(v)
                self.bounds_lower[i], self.bounds_upper[i] = b[0], b[1]

    def initial(self, initial_info: dict):
        for v, g in initial_info.items():
            if v in self.variables[1:]:
                i = self.variables[1:].index(v)
                self.initial_guess[i] = g

    def fix(self, fix_info: dict):
        for v, f in fix_info.items():
            if v in self.variables[1:]:
                i = self.variables[1:].index(v)
                self.fixed[i], self.fixed_values[i] = True, f

    def limit_estimate(self, xdata: np.ndarray, ydata: np.ndarray) -> float:
        """Estimates or constrains the mathematical saturation limit yc."""
        if (n := len(ydata)) == 0: return 0.0
        max_y = float(np.max(ydata))
        if n <= 5 or max_y == 0: return max_y
        
        dydx = np.diff(ydata) / np.where(np.diff(xdata) == 0, 1e-12, np.diff(xdata))
        tail_growth = np.mean(dydx[-5:])
        
        if tail_growth <= 0.02 * np.max(dydx) or tail_growth <= 0:
            return max_y

        active_mask = np.log(ydata) > 0.02 * np.log(max_y)
        active_idx = np.where(active_mask)[0]
        if len(active_idx) < 5: 
            return max_y * 1.5

        y_active, dydx_active = ydata[active_idx], dydx[active_idx[:-1]]
        y_mid_active = 0.5 * (y_active[:-1] + y_active[1:])
        n_act = len(y_active)

        poly_estimate = max_y
        try:
            p = np.polyfit(y_mid_active, dydx_active, 2)
            if p[0] < 0:
                roots = np.roots(p)
                valid_roots = roots[np.isreal(roots) & (roots > max_y)]
                if len(valid_roots) > 0:
                    poly_estimate = float(np.clip(np.min(valid_roots), max_y, max_y * 5.0))
        except (np.linalg.LinAlgError, ValueError):
            pass

        concavity_estimate = max_y
        k = (n_act - 1) // 2
        y1 = y_active[n_act - 1 - 2 * k]
        y2 = y_active[n_act - 1 - k]
        y3 = y_active[n_act - 1]
        
        denom = y1 * y3 - y2**2
        if denom < 0:
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
        self._mset("scale_x", 1.0, model)
        self._mset("scale_y", 1.0, model)
        return (1.0, 1.0)

    def _preprocess_data(self, model: ECModel, xdata: np.ndarray, ydata: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Hook method for subclasses to crop or filter raw input data prior to scaling."""
        return xdata, ydata

    def _configure_fit(self, model: ECModel, X: np.ndarray, Y: np.ndarray):
        """Hook method for subclasses to configure bounds, initial guesses, and fixed parameters."""
        pass

    def _run_fit(self, X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Default solver execution using scipy curve_fit."""
        free_mask = ~self.fixed
        p0 = self.initial_guess[free_mask]
        bounds = (self.bounds_lower[free_mask], self.bounds_upper[free_mask])

        result, covariance = curve_fit(
            self._make_target(), X, Y,
            p0=p0,
            bounds=bounds,
            loss="arctan", maxfev=int(1e5), nan_policy="omit", ftol=1e-10, xtol=1e-10, x_scale="jac"
        )
        perr = np.sqrt(np.diag(covariance))
        return result, perr

    def fit(self, model: ECModel, refit: bool = False, xdata: np.ndarray | None = None, ydata: np.ndarray | None = None) -> np.ndarray | None:
        """Template lifecycle method for running fits."""
        if not refit and self._mget("success", model):
            return np.array([(self._mget(v, model), self._mget(v+"_err", model)) for v in self.variables[1:]]).T

        try:
            self._reset_state()
            if xdata is None or ydata is None:
                xdata, ydata = self.select_data(model)

            if len(xdata) == 0 or len(ydata) == 0:
                raise ValueError("Insufficient or empty data selected for fitting.")

            xdata, ydata = self._preprocess_data(model, xdata, ydata)
            scale_x, scale_y = self.scale_factor(model, xdata, ydata)
            X, Y = xdata * scale_x, ydata * scale_y

            self._configure_fit(model, X, Y)
            free_results, free_errors = self._run_fit(X, Y)

            f = 0
            for i, v in enumerate(self.variables[1:]):
                if self.fixed[i]:
                    self._mset(v, self.fixed_values[i], model)
                    self._mset(v+"_err", 0.0, model)
                else:
                    self._mset(v, free_results[f], model)
                    self._mset(v+"_err", free_errors[f], model)
                    f += 1

            self._mset("success", True, model)
            self._mset("train", getattr(self.selector, "use_train", -1), model)

            for attr in ["_data", "_smooth", "_smooth_diff"]:
                if hasattr(model, attr): delattr(model, attr)

        except (RuntimeError, ValueError) as e:
            self._mset("success", False, model)
            setattr(model, f"_{self.name}_fitting_error", str(e))
            return None

        return np.array([(self._mget(v, model), self._mget(v+"_err", model)) for v in self.variables[1:]]).T

    def fit_function(self, model: ECModel, params: np.ndarray | None = None) -> Callable[..., np.ndarray]:
        if params is None:
            p_fit = self.fit(model)
            if p_fit is None:
                raise RuntimeError("Cannot return a function for a failed fit!")
            else:
                params = p_fit[0]
        scale_x = self._mget("scale_x", model, default=1.0)
        scale_y = self._mget("scale_y", model, default=1.0)
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

    def _preprocess_data(self, model: ECModel, xdata: np.ndarray, ydata: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not model.buildup:
            if (max_idx := np.argmax(ydata)) > 0:
                xdata, ydata = xdata[max_idx:], ydata[max_idx:]
            if (drop := ydata[0] - (np.mean(ydata[-5:]) if len(ydata) >= 5 else ydata[-1])) > 0:
                mask = ydata >= (ydata[-1] + 0.15 * drop)
                xdata, ydata = xdata[mask], ydata[mask]
            xdata = xdata - xdata[0]
        return xdata, ydata

    def _configure_fit(self, model: ECModel, X: np.ndarray, Y: np.ndarray):
        if model.buildup:
            self.function = self.FURMAN_BUILDUP
            self._compiled_fn = compile(self.function, f"<{self.name}_fitfn>", "eval")
            self.fix({"y0": Y[0]})
            
            lim_est_scaled = self.limit_estimate(X, Y)
            max_y_scaled = float(np.max(Y))
            
            self.bound({
                "yc": (Y[0], max(lim_est_scaled, max_y_scaled * 1.001)),
                "beta": (1e-6, 3 * model.del_max if getattr(model, "k_pe_st", 0) < 1e-6 else 10 * model.del_max)
            })
            mslope = np.argmin(np.square(Y - ((Y[-1] - Y[0]) / 2)))
            dYdX = np.diff(Y) / np.where(np.diff(X) == 0, 1e-12, np.diff(X))
            mean_slope = float(np.mean(dYdX[max(0, mslope-2):min(len(dYdX), mslope+2)])) if len(dYdX) > 0 else 0.1
            
            self.initial({
                "yc": lim_est_scaled,
                "beta": min(1.0, abs(4 * mean_slope / (Y[-1]**2)))
            })
        else:
            self.function = self.FURMAN_DECAY
            self._compiled_fn = compile(self.function, f"<{self.name}_fitfn>", "eval")
            self.fix({"y0": Y[0]})
            
            dX = X[1] - X[0] if len(X) > 1 else 1.0
            gamma_est = (np.log(Y[1]) - np.log(Y[0])) / dX if len(X) > 1 and Y[1] > 0 and Y[0] > 0 else -0.1
            if gamma_est >= 0: gamma_est = -0.1
            
            self.bound({"yc": (-np.inf, -1e-12), "beta": (1e-6, np.inf)})
            self.initial({"yc": gamma_est, "beta": 1.0})


class FurmanNPMCFit(FurmanBaseFit):
    """No Photoemission variant corrected for background magnetic multi-pole boundaries."""
    FURMAN_NO_PHOTO_MC = "(y0 * yc * np.exp(yc*x*(b0 + 0.5*b1*x))) / (yc + y0 * (np.exp(yc*x*(b0 + 0.5*b1*x)) - 1))"

    def __init__(self, db: SimDB, selector: None | DataSelector = None):
        self.db = db
        super().__init__("furman_npmc", self.FURMAN_NO_PHOTO_MC, ("x", "yc", "b0", "b1", "y0"), BunchAverageSelector() if selector is None else selector)

    def _configure_fit(self, model: ECModel, X: np.ndarray, Y: np.ndarray):
        if FurmanNoPhotoFit(selector=self.selector).fit(model, refit=False) is None:
            raise ValueError("The underlying NoPhoto fit failed.")

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


class FurmanPhotoFit(FurmanBaseFit):
    """
    Full Model optimized via stable PhotoGamma reparameterization to accommodate 
    active photoemission without exponential solver instability. Fully supports
    refit chaining and cached parameter reuse.
    """
    FURMAN_PHOTO_READABLE = """0.5*(yc + np.sqrt(yc**2 + (4*alpha/beta))*np.tanh((x/2)*np.sqrt(beta*(4*alpha + (yc**2)*beta)) - np.arctanh((yc - 2*y0)*np.sqrt(beta / (4*alpha + (yc**2)*beta)))))"""

    def __init__(self, selector: None | DataSelector = None):
        super().__init__(
            "furman_p", 
            self.FURMAN_PHOTO_READABLE, 
            ("x", "yc", "alpha", "beta", "y0"), 
            BunchAverageSelector() if selector is None else selector
        )

    def fit(self, model: ECModel, refit: bool = False, xdata: np.ndarray | None = None, ydata: np.ndarray | None = None) -> np.ndarray | None:
        """Capture execution flags to propagate refit behavior down to chained models."""
        self._current_refit = refit
        self._current_xdata = xdata
        self._current_ydata = ydata
        return super().fit(model, refit=refit, xdata=xdata, ydata=ydata)

    def _configure_fit(self, model: ECModel, X: np.ndarray, Y: np.ndarray):
        if not getattr(model, "buildup", True):
            raise ValueError("Cannot fit photoemission model for non-buildup simulation.")

        self.fix({"y0": Y[0]})
        self._current_model = model
        self._zero_photo_fallback = getattr(model, "k_pe_st", 0.0) <= 0

    def _run_fit(self, X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Handle zero-photoemission fallback with refit flag propagation
        if getattr(self, "_zero_photo_fallback", False):
            np_result = FurmanNoPhotoFit(selector=self.selector).fit(
                self._current_model, 
                refit=self._current_refit
            )
            if np_result is None:
                raise ValueError("Zero-photoemission NoPhoto fallback failed.")
            return np.array([np_result[0, 0], 0.0, np_result[0, 1]]), np.array([np_result[1, 0], 0.0, np_result[1, 1]])

        # Execute PhotoGammaFit through full fit() lifecycle
        pg_fit = PhotoGammaFit(selector=self.selector)
        pg_res = pg_fit.fit(
            self._current_model, 
            refit=self._current_refit, 
            xdata=self._current_xdata, 
            ydata=self._current_ydata
        )
        
        if pg_res is None:
            raise ValueError("Underlying PhotoGamma fit failed.")

        # Extract values (row 0) and errors (row 1)
        ys_val, beta_val, gamma_val = pg_res[0]
        ys_err, beta_err, gamma_err = pg_res[1]

        # Parameter transformation
        yc_val = 2.0 * ys_val - gamma_val
        alpha_val = beta_val * ys_val * (gamma_val - ys_val)

        # Error propagation (Jacobian mapping)
        yc_err = np.sqrt(4.0 * (ys_err**2) + (gamma_err**2))
        
        dalpha_dys = beta_val * (gamma_val - 2.0 * ys_val)
        dalpha_dbeta = ys_val * (gamma_val - ys_val)
        dalpha_dgamma = beta_val * ys_val
        
        alpha_err = np.sqrt(
            (dalpha_dys * ys_err)**2 + 
            (dalpha_dbeta * beta_err)**2 + 
            (dalpha_dgamma * gamma_err)**2
        )

        return np.array([yc_val, alpha_val, beta_val]), np.array([yc_err, alpha_err, beta_err])

class PhotoGammaFit(FurmanBaseFit):
    """
    Conservative PhotoGamma fit with safeguards against explosive exponential growth
    and denominator singularities. Uses multi-target minimization over both the 
    function and analytical derivative.
    """
    # Retained for compatibility with the Fit base class signature compilation.
    PHOTO_GAMMA = "ys + (gamma * (y0 - ys)) / (ys - y0 + np.exp(beta * gamma * x) * (gamma - ys + y0))"
    
    def __init__(self, selector=None, deriv_weight=0.15, conservative_exponential=True):
        super().__init__(
            "pgfit",
            self.PHOTO_GAMMA,
            ("x", "ys", "beta", "gamma", "y0"),
            BunchAverageSelector() if selector is None else selector
        )
        self.deriv_weight = deriv_weight
        self.conservative_exponential = conservative_exponential

    def _configure_fit(self, model: "ECModel", X: np.ndarray, Y: np.ndarray):
        self.fix({"y0": Y[0]})
        y0 = Y[0]
        
        max_x = max(float(np.max(X)), 1.0)
        is_growing = getattr(model, "buildup", True) and (Y[-1] >= Y[0])

        if is_growing:
            lim_est = self.limit_estimate(X, Y)
            max_y = float(np.max(Y)) if len(Y) > 0 else 1.0
            
            ys_init = max(lim_est, max_y * 1.05)
            ys_lower = y0
            ys_upper = max(max_y, lim_est)
            
            gamma_lower = max(1e-6, ys_init - y0 + 1e-4)
            gamma_init = max(gamma_lower * 1.1, max_y)
            gamma_upper = max(max_y * 50.0, 100.0)
            
            # Constrain beta upper limit dynamically to prevent vanishing gradients
            beta_upper = max(1.0, 50.0 / max_x)
            beta_init = min(0.1, beta_upper * 0.5)
        else:
            min_y = float(np.min(Y))
            
            ys_init = max(0.0, min_y)
            ys_lower = 0.0
            ys_upper = y0
            
            gamma_lower = 1e-6
            gamma_init = max(y0, 1.0)
            gamma_upper = max(y0 * 50.0, 100.0)
            
            beta_upper = max(1.0, 50.0 / max_x)
            beta_init = min(0.1, beta_upper * 0.5)

        self.bound({
            "ys": (ys_lower, ys_upper),
            "beta": (1e-6, beta_upper),
            "gamma": (gamma_lower, gamma_upper)
        })
        
        self.initial({
            "ys": ys_init,
            "beta": beta_init,
            "gamma": gamma_init
        })

    def _run_fit(self, X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if len(X) < 5:
            raise ValueError("Insufficient data points for PhotoGamma fitting.")

        X_mid = 0.5 * (X[:-1] + X[1:])
        dX = np.diff(X)
        dX = np.where(dX == 0, 1e-12, dX)
        dYdX = np.diff(Y) / dX

        def compute_vals(x_arr, ys, beta, gamma, y0):
            arg = np.clip(beta * gamma * x_arr, -700.0, 700.0)
            E = np.exp(arg)
            denom = (ys - y0) + E * (gamma - ys + y0)
            denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
            return ys + (gamma * (y0 - ys)) / denom

        def compute_deriv(x_arr, ys, beta, gamma, y0):
            arg = np.clip(beta * gamma * x_arr, -700.0, 700.0)
            E = np.exp(arg)
            A = y0 - ys
            B = gamma - ys + y0
            denom = (ys - y0) + E * B
            denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
            return (beta * (gamma**2) * A * B * E) / np.square(denom)

        def get_full_params(p_free):
            params = np.empty(4)
            free_ptr = 0
            for i in range(4):
                if self.fixed[i]:
                    params[i] = self.fixed_values[i]
                else:
                    params[i] = p_free[free_ptr]
                    free_ptr += 1
            return params

        def objective(p_free):
            params = get_full_params(p_free)
            y_pred = compute_vals(X, *params)
            d_pred = compute_deriv(X_mid, *params)

            if not (np.all(np.isfinite(y_pred)) and np.all(np.isfinite(d_pred))):
                return 1e12

            res_y = y_pred - Y
            res_d = d_pred - dYdX
            loss = np.sum(np.square(res_y)) + self.deriv_weight * np.sum(np.square(res_d))
            
            return loss if np.isfinite(loss) else 1e12

        # ---------------------------------------------------------
        # NEW: Explicit Singularity Constraint for SLSQP
        # Ensures: gamma - ys + y0 >= 1e-5
        # ---------------------------------------------------------
        def singularity_constraint(p_free):
            ys_val, _, gamma_val, y0_val = get_full_params(p_free)
            return (gamma_val - ys_val + y0_val) - 1e-5
            
        constraints = [{'type': 'ineq', 'fun': singularity_constraint}]

        free_mask = ~self.fixed
        p0 = self.initial_guess[free_mask]
        bounds = list(zip(self.bounds_lower[free_mask], self.bounds_upper[free_mask]))

        # Switched method to SLSQP to utilize the constraints argument
        opt_res = minimize(
            objective, p0, bounds=bounds, method="SLSQP", constraints=constraints,
            options={"maxiter": 10000, "ftol": 1e-12}
        )

        if not opt_res.success:
            raise RuntimeError(f"PhotoGamma SLSQP optimization failed: {opt_res.message}")

        opt_params = opt_res.x

        # Calculate standard uncertainties via Jacobian matrix estimation
        eps = 1e-6
        n_obs = len(Y) + len(dYdX)
        n_param = len(opt_params)
        J = np.zeros((n_obs, n_param))

        def compute_res(p_free):
            p_full = get_full_params(p_free)
            y_pred = compute_vals(X, *p_full)
            d_pred = compute_deriv(X_mid, *p_full)
            return np.concatenate([y_pred - Y, np.sqrt(self.deriv_weight) * (d_pred - dYdX)])

        r0 = compute_res(opt_params)

        for i in range(n_param):
            p_step = opt_params.copy()
            step = max(abs(opt_params[i]) * eps, eps)
            p_step[i] += step
            r_step = compute_res(p_step)
            J[:, i] = (r_step - r0) / step

        jtj = J.T @ J
        dof = max(1, n_obs - n_param)
        s_sq = np.sum(np.square(r0)) / dof
        
        try:
            cov = np.linalg.inv(jtj) * s_sq
            perr = np.sqrt(np.maximum(0.0, np.diag(cov)))
        except np.linalg.LinAlgError:
            perr = np.zeros_like(opt_params)

        return opt_params, perr


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