from numbers import Number
from typing import List, Tuple

import numpy as np

from chemistry_data.chem_tools import spectrum_analysis
from ofex.linalg.sparse_tools import state_dot
from ofex.operators import normalize_by_lcu_norm
from ofex.state.state_tools import normalize


def initialize_reference_state(pham, hf, mol_name, transform, tag, initial_overlap, n_qubits, mol_param, print_progress,
                               **_):
    pham_normalized, norm = normalize_by_lcu_norm(pham, level=1)

    # Spectral analysis
    eigval_overlap_pair, eig_energies = spectrum_analysis(pham, hf, mol_name, transform, ref_name="HF", tag=tag,
                                                          print_progress=print_progress, save=True, load=True,
                                                          mol_param=mol_param)
    for idx, (eig_idx, eigval, overlap, eigvec) in enumerate(eigval_overlap_pair):
        eigval_overlap_pair[idx] = (eig_idx, eigval / norm, overlap, eigvec)

    # Evenly allocated state
    if isinstance(initial_overlap, Number):
        ref_state = np.zeros(2 ** n_qubits, dtype=np.complex128)
        n_effective_states = len(eigval_overlap_pair)
        other_overlap = np.sqrt((1 - abs(initial_overlap) ** 2) / (n_effective_states - 1))
        for idx, (eig_idx, eigval, overlap, eigvec) in enumerate(eigval_overlap_pair):
            if eig_idx == 0:
                ref_state += eigvec * initial_overlap
            else:
                ref_state += eigvec * other_overlap
            # eigval_overlap_pair[idx] = (eig_idx, eigval, 1/np.sqrt(n_effective_states) ,eigvec)
    elif initial_overlap == "HF":
        ref_state = hf
    elif initial_overlap == "even":
        ref_state = np.zeros(2 ** n_qubits, dtype=np.complex128)
        for eig_idx, eigval, overlap, eigvec in eigval_overlap_pair:
            ref_state += eigvec
    else:
        raise ValueError(f"Invalid initial_overlap: {initial_overlap}")
    ref_state = normalize(ref_state)
    for idx, (eig_idx, eigval, overlap, eigvec) in enumerate(eigval_overlap_pair):
        eigval_overlap_pair[idx] = (eig_idx, eigval, state_dot(ref_state, eigvec), eigvec)

    return pham_normalized, norm, ref_state, eigval_overlap_pair


def collect_degeneracy(eigval_overlap_pair: List[Tuple[int, complex, complex, np.ndarray]],
                       atol_degeneracy: float = 1e-8,
                       norm: float = 1.0) -> List[Tuple[List[int], float, float]]:
    """
    Merge consecutive eigenpairs whose eigenvalues differ by <= `atol_degeneracy`
    (absolute tolerance, rtol=0).  Return a list of

        ([eig_indices_in_block], representative_eigval, sum_overlap²)

    where `representative_eigval` is the arithmetic mean of the block.  All
    overlaps are converted to probabilities |⟨ref|ψ⟩|² before accumulation.
    """

    if not eigval_overlap_pair:  # -- guard for empty input
        return []

    # sort by eigenvalue ---------------------------
    eigval_overlap_pair = sorted(eigval_overlap_pair, key=lambda t: t[1].real)

    # seed the first block -------------------------
    current_idx_list: List[int] = [eigval_overlap_pair[0][0]]
    current_eigvals: List[float] = [eigval_overlap_pair[0][1]]
    current_overlap2: float = abs(eigval_overlap_pair[0][2]) ** 2

    blocks: List[Tuple[List[int], float, float]] = []

    # walk through the rest ------------------------
    for eigidx, eigval, overlap, _ in eigval_overlap_pair[1:]:
        overlap_sq = abs(overlap) ** 2

        if np.isclose(eigval, current_eigvals[-1],
                      atol=atol_degeneracy, rtol=0.0):
            # same degenerate block
            current_idx_list.append(eigidx)
            current_eigvals.append(eigval)
            current_overlap2 += overlap_sq
        else:
            # flush previous block
            mean_eigval = float(np.mean(current_eigvals)) / norm
            blocks.append((current_idx_list, mean_eigval, current_overlap2))

            # start new block
            current_idx_list = [eigidx]
            current_eigvals = [eigval]
            current_overlap2 = overlap_sq

    # flush final block ----------------------------
    mean_eigval = float(np.mean(current_eigvals)) / norm
    blocks.append((current_idx_list, mean_eigval, current_overlap2))

    return blocks


def gen_eig_truncated(H, S, thresh=1e-8, *, relative=True, return_transform=False):
    """
    Solve  H c = E S c  after truncating the tiny eigenvalues of S.

    Parameters
    ----------
    H : (N,N) array_like
        Hermitian (or real-symmetric) Hamiltonian.
    S : (N,N) array_like
        Hermitian positive-semidefinite overlap matrix.
    thresh : float, default 1e-8
        Cut-off for the eigenvalues of S.
        If `relative=True` it is interpreted as *relative to* max(s_i).
        Otherwise it is an absolute cut-off.
    relative : bool, default True
        Whether `thresh` is relative.
    return_transform : bool, default False
        If True also return T = U_keep diag(s_keep^−1/2).

    Returns
    -------
    evals : (M,) ndarray
        Eigenvalues  E  (M ≤ N after truncation).
    evecs : (N,M) ndarray
        Eigenvectors  c  in the original basis (columns).
    T     : (N,M) ndarray, optional
        Transformation that maps reduced vectors  z  to  c = T z.
    """
    # --- convert to complex ndarray (NumPy's eigh prefers float/complex) ----
    H = np.asarray(H, dtype=np.complex128)
    S = np.asarray(S, dtype=np.complex128)

    # --- 1. spectral decomposition of S ------------------------------------
    s_vals, U = np.linalg.eigh(S)  # S = U diag(s) U†  ;  s_vals sorted ascending
    if relative:
        thresh = thresh * s_vals.max()

    keep = s_vals >= thresh
    if not np.any(keep):
        raise ValueError("All eigenvalues of S are below the threshold.")

    s_keep = s_vals[keep]  # kept eigenvalues   (shape (M,))
    U_keep = U[:, keep]  # kept eigenvectors  (N × M)

    # --- 2. build the orthonormalising transform  T = U Σ^{-1/2} ------------
    T = U_keep / np.sqrt(s_keep)[None, :]  # broadcast division, shape (N × M)

    # --- 3. transform H and solve the reduced Hermitian problem ------------
    H_bar = T.conj().T @ H @ T  # M × M  Hermitian
    evals, Z = np.linalg.eigh(H_bar)  # standard eigen-problem

    # --- 4. back-transform eigenvectors ------------------------------------
    evecs = T @ Z  # columns are full-space eigenvectors

    if return_transform:
        return evals, evecs, T
    return evals, evecs


