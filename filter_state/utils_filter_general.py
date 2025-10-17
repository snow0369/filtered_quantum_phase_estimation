from typing import Callable, Tuple

import numpy as np
from numpy.polynomial.chebyshev import chebval
from ofex.classical_algorithms.filter_functions import gaussian_function_fourier, gaussian_function_cheby
from scipy.optimize import minimize_scalar

from filter_state.utils_filter_cheby import max_amplitude_cheby, apply_filter_to_state_cheby, filter_func_eval_cheby, \
    effective_gaussian_params_cheby
from filter_state.utils_filter_trig import max_amplitude_trig, apply_filter_to_state_trig, filter_func_eval_trig, \
    effective_gaussian_params_trig


def max_amplitude(basis_type, c, time_step=np.pi, oversample=64, use_scipy=True):
    if basis_type == "trig":
        max_x, max_amp = max_amplitude_trig(c, time_step, oversample, use_scipy)
    elif basis_type == "cheby":
        max_x, max_amp = max_amplitude_cheby(c, oversample, use_scipy)
    else:
        raise ValueError
    return max_x, max_amp


def apply_filter_to_state(basis_type, degenerate_block, c, time_step=np.pi, precalc_cheby=None):
    if basis_type == "trig":
        return apply_filter_to_state_trig(degenerate_block, c, time_step)
    elif basis_type == "cheby":
        return apply_filter_to_state_cheby(degenerate_block, c, precalc_cheby)
    else:
        raise ValueError


def filter_func_eval(basis_type, x, c, time_step=np.pi, precalc_cheby=None):
    if basis_type == "trig":
        return filter_func_eval_trig(x, c, time_step)
    elif basis_type == "cheby":
        return filter_func_eval_cheby(x, c, precalc_cheby)
    else:
        raise ValueError


def effective_gaussian_params(basis_type, width, epsilon, period):
    if basis_type == "trig":
        return effective_gaussian_params_trig(width, period, epsilon)
    elif basis_type == "cheby":
        return effective_gaussian_params_cheby(width, epsilon)
    else:
        raise ValueError


def gaussian_function(basis_type, n_basis, width, center, period=2.0, peak_height=1.0):
    if n_basis % 2 == 0:
        raise ValueError
    if basis_type == "trig":
        n_fourier = (n_basis - 1) // 2
        return gaussian_function_fourier(n_fourier, width, center, period, peak_height)[:2]
    elif basis_type == "cheby":
        n_poly = n_basis - 1
        return gaussian_function_cheby(n_poly, width, center, period, peak_height)
    else:
        raise ValueError


def _series_fourier(coeff: np.ndarray, freqs: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """Return f(x) = Σ c_k e^{i f_k x} for vector x."""
    c = coeff[:, None]  # (K,1)
    f = freqs[:, None]  # (K,1)
    return lambda x: (c * np.exp(1j * f * x)).sum(axis=0)


def _series_cheby(coeff: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """Return f(x) = Σ c_k T_k(x) for vector x in [-1,1] (or rescaled domain)."""
    c = np.asarray(coeff)
    return lambda x: chebval(x, c)


def gaussian_width_fit_general(basis_type: str, c: np.ndarray, period: float = 2.0, center: float = 0.0,
                               peak_height: float = 1.0, oversample: int = 128,
                               width_bounds: Tuple[float, float] = (1e-3, 2.0),
                               tol: float = 1e-8) -> Tuple[float, np.ndarray]:
    r"""
    Find Gaussian width w so that the Gaussian *series* best matches the *main lobe*
    of the target series f_T.

    Target series:
      - Fourier:     f_T(x) = Σ c_k^T · e^{i (2π/period) k x}, k=-n..n
      - Chebyshev:   f_T(x) = Σ c_k^T · T_k(x), k=0..N

    Gaussian builder callables (you provide):
      - make_fourier_gaussian(n, w, center=..., period=..., peak_height=...) -> (expr, coeff, freqs)
      - make_chebyshev_gaussian(n_cheb, w, **cheby_kwargs) -> coeff_cheb

    Returns
    -------
    w_opt : float
    coeff_G : ndarray
        Gaussian series coefficients at w_opt (Fourier or Chebyshev).
    """
    c = np.asarray(c)

    if basis_type not in {"trig", "cheby"}:
        raise ValueError("basis must be 'trig' or 'cheby'.")

    # ---------- build target evaluator & sampling grid ----------
    if basis_type == "trig":
        # infer k and freqs from c_target length
        n = (len(c) - 1) // 2
        if 2 * n + 1 != len(c):
            raise ValueError("Fourier c_target must have length 2n+1 (k = -n..n).")
        k = np.arange(-n, n + 1)
        freqs = k * (2 * np.pi / period)
        f_T = _series_fourier(c, freqs)

        # sample one period centred at `center`
        m_grid = oversample * (2 * n + 1)
        x_grid = np.linspace(center - period / 2, center + period / 2, m_grid, endpoint=False)

    else:  # chebyshev
        # c_target is [c_0, c_1, ..., c_N]
        N = len(c) - 1
        f_T = _series_cheby(c)

        # sample given domain
        a, b = center - period / 2, center + period / 2
        m_grid = oversample * (N + 1)
        x_grid = np.linspace(a, b, m_grid, endpoint=False)

    # ---------- detect main lobe of target ----------
    amp = np.abs(f_T(x_grid))
    fmax = amp.max()
    mask_main = amp >= 0.5 * fmax
    x_main = x_grid[mask_main]
    amp_T = amp[mask_main].real  # magnitude (real, nonnegative)

    # ---------- objective: RMS error on main lobe ----------
    def error_width(w: float) -> float:
        if basis_type == "trig":
            # builder expected to return (expr, coeff, freqs)
            _, c_G, _ = gaussian_function_fourier(
                n, w, center, period, peak_height)
            f_G = _series_fourier(c_G, freqs)
        else:
            _, c_G = gaussian_function_cheby(
                N * 2, w, center, period, peak_height)
            f_G = _series_cheby(c_G)

        err = np.abs(f_G(x_main)) - amp_T
        return float(np.sqrt(np.mean(err ** 2)))  # RMS

    # ---------- search ----------
    res = minimize_scalar(error_width, bounds=width_bounds, method='bounded', options={'xatol': tol})
    w_opt = float(res.x)

    # ---------- rebuild at optimal width ----------
    if basis_type == "trig":
        _, coeff_G, _ = gaussian_function_fourier(n, w_opt, center, period, peak_height)
    else:
        _, coeff_G = gaussian_function_cheby(N, w_opt, center, period, peak_height)

    return w_opt, coeff_G
