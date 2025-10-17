from typing import Tuple, Union

import numpy as np
from numpy.polynomial.chebyshev import Chebyshev
from scipy.optimize import fminbound


def calc_cheby(values: Union[float, np.ndarray], n_cheby: int) -> np.ndarray:
    """
    Return Chebyshev T_k(x) values up to degree n_cheby-1 for each x in `values`.

    - If `values` is a scalar, returns shape (n_cheby,).
    - If `values` is a 1D array of length N, returns shape (N, n_cheby).

    Recurrence: T_0(x)=1, T_1(x)=x, T_{n}(x)=2x T_{n-1}(x) - T_{n-2}(x)
    """
    if n_cheby < 0:
        raise ValueError("n_cheby must be >= 0")

    x = np.asarray(values)
    if x.ndim > 1:
        raise ValueError("values must be a scalar or 1D array")

    scalar_input = (x.ndim == 0)
    if scalar_input:
        x = x.reshape(1)  # treat as length-1 for unified logic

    # Promote dtype to float (or keep complex) to avoid integer arithmetic
    dtype = np.result_type(x.dtype, np.float64)
    N = x.shape[0]
    out = np.empty((N, n_cheby), dtype=dtype) if n_cheby > 0 else np.empty((N, 0), dtype=dtype)

    if n_cheby == 0:
        return out[0] if scalar_input else out

    # T_0
    out[:, 0] = 1.0

    if n_cheby == 1:
        return out[0] if scalar_input else out

    # T_1
    out[:, 1] = x.astype(dtype, copy=False)

    # T_n
    for n in range(2, n_cheby):
        out[:, n] = 2.0 * x * out[:, n - 1] - out[:, n - 2]

    return out[0] if scalar_input else out


def max_amplitude_cheby(c, oversample=64, use_scipy=True) -> Tuple[float, float]:
    """
    Locate the global maximum of |Σ c_k T_k(x)| on x ∈ [-1, 1].

    Parameters
    ----------
    c : array_like
        Chebyshev coefficients [c0, c1, ..., cn].
    oversample : int or None, optional
        Extra uniform sample points:
          - None or 0  → no uniform grid (default behaviour).
          - k > 0      → evaluate on `k * (n_coeffs)` equally spaced points,
                         and merge those into candidate set.  This is *cheap*
                         (O(k*n)) and helps when roots of f' are numerically
                         missed.
    use_scipy : bool, default True
        If SciPy is available and True, refine the maximum by minimising
        -|f(x)| with `scipy.optimize.fminbound`.

    Returns
    -------
    x_max : float
        Location where |f(x)| is maximal.
    max_abs : float
        The maximum absolute value |f(x_max)|.
    """
    xtol = 1e-10

    f = Chebyshev(c, domain=[-1, 1])

    # ---- 1) endpoints & stationary points -------------------------------
    df = f.deriv()
    crit = np.asarray(
        [z.real for z in df.roots() if np.isreal(z) and -1.0 <= z.real <= 1.0],
        dtype=float,
    )

    candidates = np.concatenate(([-1.0, 1.0], crit))

    # ---- 2) optional uniform oversampling -------------------------------
    if oversample:
        m = int(max(1, oversample)) * len(c)
        grid = np.linspace(-1.0, 1.0, m, endpoint=True)
        candidates = np.unique(np.concatenate((candidates, grid)))

    vals = f(candidates)
    idx = np.argmax(np.abs(vals))
    best_x, best_val = float(candidates[idx]), float(np.abs(vals[idx]))

    # ---- 3) optional SciPy refinement -----------------------------------
    if use_scipy:
        try:
            x_opt = fminbound(lambda x: -abs(f(x)), -1.0, 1.0, xtol=xtol, disp=0)
            val_opt = abs(f(x_opt))
            if val_opt > best_val + 1e-12 * max(1.0, best_val):
                best_x, best_val = float(x_opt), float(val_opt)
        except Exception as err:  # keep the analytic result on failure
            print("[max_amplitude_cheby] SciPy refinement failed:", err)

    return best_x, best_val


def apply_filter_to_state_cheby(degenerate_block,
                                c,
                                precalc_cheby=None):
    """
    Returns:
      ret_degenblock: list[(indices, eigval, new_ov2)]
      norm: float = sum(new_ov2)
    """
    N = len(degenerate_block)
    if N == 0:
        return [], 0.0

    eigvals = np.array([eig for _, eig, _ in degenerate_block], dtype=float)
    ov2 = np.array([w for _, _, w in degenerate_block], dtype=float)

    # If a precomputed Chebyshev table is provided, ensure it matches (N, n_cheby) or (n_cheby,) for scalar.
    if precalc_cheby is not None:
        T = np.asarray(precalc_cheby)
        n = len(c)
        if eigvals.ndim == 1:  # vector case
            if T.ndim != 2 or T.shape[0] != N or T.shape[1] < n:
                # Mismatch -> recompute
                precalc_cheby = None
            else:
                precalc_cheby = T[:, :n]
        else:  # scalar case (not used here because N>=1), keep as-is
            precalc_cheby = T[..., :n]

    fvals = filter_func_eval_cheby(eigvals, c, precalc_cheby)  # shape (N,)
    w = (fvals.conj() * fvals).real  # |f|^2 without sqrt
    filtered_ov2 = ov2 * w
    norm = float(filtered_ov2.sum())
    filtered_ov2 /= norm
    ret_degenblock = [
        (idxs, eigval, float(new_w))
        for (idxs, eigval, _), new_w in zip(degenerate_block, filtered_ov2)
    ]
    return ret_degenblock, norm


def filter_func_eval_cheby(x, c, precalc_cheby=None):
    """
    Evaluate f(x) = sum_k coeff[k] * T_k(x).

    - x: float or 1-D array (length N)
    - coeff: 1-D array of length n_cheby
    - precalc_cheby: optional precomputed T matrix
        * if x is scalar: shape (n_cheby,)
        * if x is 1-D:    shape (N, n_cheby)
    Returns:
        float if x is scalar, or shape (N,) if x is 1-D.
    """
    c = np.asarray(c).reshape(-1)
    n = c.size

    x_arr = np.asarray(x)
    is_scalar = (x_arr.ndim == 0)

    if precalc_cheby is None:
        T = calc_cheby(x_arr, n)  # (n,) for scalar, (N, n) for array
    else:
        T = np.asarray(precalc_cheby)[..., :n]

    if is_scalar:
        # Accept (n,) or (1,n) and make it 1-D
        T = T.reshape(-1)
        y = T @ c  # numpy 0-d scalar
        return y.item()  # Python scalar (float/complex)
    else:
        # Ensure (N, n) @ (n,) -> (N,)
        if T.ndim == 1:
            T = T.reshape(1, -1)
        return T @ c


def effective_gaussian_params_cheby(width, epsilon):
    n_poly = 2 * (int(np.ceil(np.sqrt(2 * np.log(4 / epsilon) * max(np.e ** 2 * np.log(1 / epsilon) / (2 * width ** 2),
                                                                   np.log(2 / epsilon))))) // 2)
    n_basis = n_poly + 1
    width_rescaled = width / np.sqrt(2 * np.log(epsilon ** -1))
    return n_basis, width_rescaled
