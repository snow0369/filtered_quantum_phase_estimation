from typing import Tuple

import numpy as np
from scipy.optimize import minimize_scalar


def max_amplitude_trig(
    c: np.ndarray,
    time_step: float = np.pi,
    oversample: int = 64,
    top_k: int = 12,
    window_scale: float = 1.5,
    xatol: float = 1e-12,
    maxiter: int = 2000,
    polish_full: bool = True,
) -> Tuple[float, float]:
    r"""
    Maximize |f(x)| on x ∈ [-1,1], where
        f(x) = Σ_k c_k * exp(-i * k * time_step * x),
    with c ordered as [c_-n, ..., c_0, ..., c_n] (len = 2n+1).

    Strategy: dense scan → pick top peaks → per-peak bounded Brent refine.
    """

    c = np.asarray(c, dtype=np.complex128)
    n = (len(c) - 1) // 2
    if 2 * n + 1 != len(c):
        raise ValueError("c must have odd length = 2n+1, ordered [c_-n ... c_n].")

    k = np.arange(-n, n + 1)

    # ---- helpers -----------------------------------------------------------
    def f_scalar(x: float) -> complex:
        return (c * np.exp(-1j * time_step * k * x)).sum()

    def phi(x: float) -> float:   # objective to MAXIMIZE
        return np.abs(f_scalar(x))

    # ---- 1) dense scan -----------------------------------------------------
    m = int(max(8, oversample) * (2 * n + 1))       # ≥ 8 samples per Fourier node
    x_grid = np.linspace(-1.0, 1.0, m, endpoint=False)
    f_vals = (c[:, None] * np.exp(-1j * time_step * np.outer(k, x_grid))).sum(axis=0)
    abs_vals = np.abs(f_vals)

    # Include endpoints explicitly
    end_x  = np.array([-1.0, 1.0])
    end_phi = np.array([phi(-1.0), phi( 1.0)])

    # ---- 2) choose top-K local maxima on grid -----------------------------
    prev = np.roll(abs_vals, 1)
    nxt  = np.roll(abs_vals, -1)
    is_peak = (abs_vals >= prev) & (abs_vals > nxt)
    peak_idx = np.flatnonzero(is_peak)

    if peak_idx.size == 0:
        # Fall back to best grid point
        j0 = int(abs_vals.argmax())
        x0 = float(x_grid[j0])
        dx = (x_grid[1] - x_grid[0])
        a = max(-1.0, x0 - window_scale * dx)
        b = min( 1.0, x0 + window_scale * dx)
        res = minimize_scalar(lambda x: -phi(x), method="bounded", bounds=(a, b),
                              options={"xatol": xatol, "maxiter": maxiter})
        x_best = float(res.x if res.success else x0)
        y_best = float((-res.fun) if res.success else phi(x_best))
    else:
        order = np.argsort(abs_vals[peak_idx])[::-1]
        cand_idx = peak_idx[order[:top_k]]

        dx = (x_grid[1] - x_grid[0])
        best_x, best_y = None, -np.inf

        for j in cand_idx:
            x0 = float(x_grid[j])
            a = max(-1.0, x0 - window_scale * dx)
            b = min( 1.0, x0 + window_scale * dx)

            res = minimize_scalar(lambda x: -phi(x), method="bounded", bounds=(a, b),
                                  options={"xatol": xatol, "maxiter": maxiter})
            if res.success:
                y_loc = float(-res.fun)
                x_loc = float(res.x)
            else:
                # fallback to center of window if optimizer failed
                x_loc = 0.5 * (a + b)
                y_loc = phi(x_loc)

            if y_loc > best_y:
                best_y, best_x = y_loc, x_loc

        # compare endpoints too
        if end_phi.max() > best_y:
            best_y = float(end_phi.max())
            best_x = float(end_x[end_phi.argmax()])

        x_best, y_best = best_x, best_y

    # ---- 3) optional full-interval polish ---------------------------------
    if polish_full:
        res = minimize_scalar(lambda x: -phi(x), method="bounded",
                              bounds=(-1.0 + 1e-12, 1.0 - 1e-12),
                              options={"xatol": xatol, "maxiter": maxiter})
        if res.success and -res.fun > y_best:
            x_best, y_best = float(res.x), float(-res.fun)

    return float(x_best), float(y_best)


def apply_filter_to_state_trig(degenerate_block,
                               c,
                               time_step):
    eigvals = np.array([eigval for _, eigval, _ in degenerate_block])
    ov2 = np.array([ov2 for _, _, ov2 in degenerate_block])
    filtered_state_ov2 = (np.abs(filter_func_eval_trig(eigvals, c, time_step)) ** 2) * ov2
    norm = np.sum(filtered_state_ov2)
    filtered_state_ov2 /= norm
    ret_degenblock = [(idxs, eigval, ov2) for (idxs, eigval, _), ov2 in zip(degenerate_block, filtered_state_ov2)]
    return ret_degenblock, norm


def filter_func_eval_trig(x, c, time_step):
    c = np.asarray(c, dtype=np.complex128)
    if c.ndim != 1 or len(c) % 2 == 0:
        raise ValueError("`c` must be a 1-D array of odd length (2n+1).")

    n      = len(c) // 2
    k      = np.arange(-n, n + 1)                   # [-n … n]
    x_arr  = np.asarray(x, dtype=float)

    # --- broadcasting: shape  (*x_shape,  2n+1)
    phase  = np.exp(1j * time_step * np.multiply.outer(x_arr, k))

    # --- contract the last axis with coefficients
    #     tensordot keeps the leading shape of x_arr intact
    f_vals = np.tensordot(phase, c, axes=([-1], [0]))

    # tensordot returns a 0-D array for scalar input; convert to Python scalar
    return f_vals.item() if f_vals.ndim == 0 else f_vals


def effective_gaussian_params_trig(width, period, epsilon):
    n_fourier = int(np.ceil(2 * period * np.log(epsilon ** -1) / (np.pi * width)))
    width_rescaled = width / np.sqrt(2 * np.log(epsilon ** -1))
    return n_fourier * 2 + 1, width_rescaled
