import os
from copy import deepcopy

import numpy as np
from matplotlib import pyplot as plt
from openfermion import LinearQubitOperator
from tqdm import tqdm

from chemistry_data.chem_tools import real_time_propagator, real_time_propagator_path, spectrum_analysis, \
    prepare_hamiltonian_refstates
from chemistry_data.example_model import hubbard_examples
from filter_state.utils_quantum import gen_eig_truncated, collect_degeneracy
from ofex.linalg.sparse_tools import state_dot, apply_operator
from ofex.operators import normalize_by_lcu_norm
from ofex.state.state_tools import to_dense
from ofex_algorithms.qksd.qksd_utils import toeplitz_arr_to_mat


def generate_smatrix(n_basis, n_trotter, time_step, ref_state,
                     mol_name, transform, pham, n_qubits, mol_param,
                     conj=False,
                     tag=None, n_digits=6, save=True, load=True, print_progress=False,
                     make_h_matrix=False,
                     **_):
    prop = real_time_propagator(mol_name, transform, pham, time_step, n_qubits, n_trotter,
                                tag, n_digits, save, load, print_progress, mol_param)
    prop_state = deepcopy(ref_state)

    prop_path = real_time_propagator_path(mol_name, transform, time_step, n_trotter, tag, 6, mol_param)
    if make_h_matrix:
        s_mat, h_mat = load_s_matrix(n_basis, prop_path, conj, make_h_matrix=True)
        if s_mat is not None and h_mat is not None:
            return s_mat, h_mat
    else:
        s_mat = load_s_matrix(n_basis, prop_path, conj, make_h_matrix=False)
        if s_mat is not None:
            return s_mat

    s_arr = np.zeros(n_basis, dtype=np.complex128)
    h_arr = np.zeros(n_basis, dtype=np.complex128) if make_h_matrix else None

    for i in tqdm(range(n_basis)):
        if conj:
            s_arr[i] = state_dot(ref_state, prop_state)
        else:
            s_arr[i] = state_dot(ref_state, prop_state)
        if make_h_matrix:
            h_prop_state = apply_operator(pham, prop_state)
            if conj:
                h_arr[i] = state_dot(ref_state, h_prop_state)
            else:
                h_arr[i] = state_dot(ref_state, h_prop_state)
        if i != n_basis - 1:
            prop_state = apply_operator(prop, prop_state)

    s_mat = toeplitz_arr_to_mat(s_arr.conj() if conj else s_arr)
    if make_h_matrix:
        h_mat = toeplitz_arr_to_mat(h_arr.conj() if conj else h_arr)
        if save:
            save_matrices(prop_path, s_arr, h_arr)
        return s_mat, h_mat
    else:
        if save:
            save_matrices(prop_path, s_arr, None)
        return s_mat


def load_s_matrix(n_basis, prop_path,
                  conj, make_h_matrix):
    path_matrix = prop_path.split('/')[-1]
    path_matrix = '.'.join(path_matrix.split('.')[:-1])
    dir_matrix = "./data/qksd_matrices/" + '/'.join(prop_path.split('/')[-3:-1])
    if not os.path.exists(dir_matrix):
        os.makedirs(dir_matrix)
    path_s_matrix = os.path.join(dir_matrix, path_matrix + '_S.npy')
    path_h_matrix = os.path.join(dir_matrix, path_matrix + '_H.npy')

    # Load S matrix
    if not os.path.exists(path_s_matrix):
        return None if not make_h_matrix else (None, None)
    s_arr = np.array(np.load(path_s_matrix))

    if len(s_arr) < n_basis:
        return None if not make_h_matrix else (None, None)
    s_mat = toeplitz_arr_to_mat(s_arr[:n_basis].conj() if conj else s_arr[:n_basis])

    if not make_h_matrix:
        return s_mat

    # Load H matrix
    if not os.path.isfile(path_h_matrix):
        return None, None
    h_arr = np.array(np.load(path_h_matrix))

    if len(h_arr) < n_basis:
        return None, None
    h_mat = toeplitz_arr_to_mat(h_arr[:n_basis].conj() if conj else h_arr[:n_basis])
    return s_mat, h_mat


def save_matrices(prop_path, s_arr, h_arr):
    path_matrix = prop_path.split('/')[-1]
    path_matrix = '.'.join(path_matrix.split('.')[:-1])
    dir_matrix = "./data/qksd_matrices/" + '/'.join(prop_path.split('/')[-3:-1])
    if not os.path.exists(dir_matrix):
        os.makedirs(dir_matrix)
    path_s_matrix = os.path.join(dir_matrix, path_matrix + '_S.npy')
    path_h_matrix = os.path.join(dir_matrix, path_matrix + '_H.npy')

    if os.path.exists(path_s_matrix):
        loaded_s_arr = np.array(np.load(path_s_matrix))
        if len(loaded_s_arr) >= len(s_arr):
            return
    np.save(path_s_matrix, s_arr)

    if h_arr is not None:
        if os.path.exists(path_h_matrix):
            loaded_h_arr = np.array(np.load(path_h_matrix))
            if len(loaded_h_arr) >= len(h_arr):
                return
        np.save(path_h_matrix, h_arr)


def generate_smatrix_cheby(n_basis, ref_state,
                           mol_name, transform, pham, n_qubits, mol_param,
                           tag="", save=True, load=True, **_):
    if mol_param is None:
        mol_param = ""
    elif isinstance(mol_param, dict):
        mol_param = "".join([f"{k}={v}" for k, v in mol_param.items()])
    else:
        mol_param = str(mol_param)
    path_matrix = "./data/qksd_matrices/" + '/'.join([mol_name, transform])
    if not os.path.exists(path_matrix):
        os.makedirs(path_matrix)
    path_matrix = path_matrix + '/' + '_'.join([tag, mol_param]) + '.npy'

    a, b = None, None
    mlen = 2 * n_basis - 1
    if os.path.exists(path_matrix) and load:
        ab = np.load(path_matrix)
        if ab.shape[1] >= mlen:
            a = ab[0, : mlen]
            b = ab[1, : mlen]
            save = False
            print(f"Cheby matrix loaded from {path_matrix}")

    if a is None:
        print(f"Generating Cheby matrix (mlen={mlen}))")
        a = np.zeros(mlen, dtype=np.complex128)
        b = np.zeros(mlen, dtype=np.complex128)
        pham = LinearQubitOperator(pham, n_qubits=n_qubits)

        # w_0 = H|psi>, w_1 = H^2|psi>
        w_nm2 = to_dense(apply_operator(pham, ref_state))  # w_0
        b[0] = state_dot(ref_state, w_nm2)  # b_0 = <psi|H|psi>
        a[0] = 1.0
        a[1] = b[0]

        w_nm1 = apply_operator(pham, w_nm2)  # w_1 = H^2|psi>
        b[1] = state_dot(ref_state, w_nm1)
        # We can now get a[2] = 2*b[1] - a[0] in the loop.
        for n in tqdm(range(2, mlen)):
            # Recurrence for a_n (needs b_{n-1}, a_{n-2})
            a[n] = 2 * b[n - 1] - a[n - 2]

            # Update vectors for next step of b
            # w_n = 2 H w_{n-1} - w_{n-2}
            w_n = 2 * apply_operator(pham, w_nm1) - w_nm2
            b[n] = state_dot(ref_state, w_n)

            # slide window
            w_nm2, w_nm1 = w_nm1, w_n

    if save:
        ab = np.zeros((2, mlen), dtype=np.complex128)
        ab[0, :] = a
        ab[1, :] = b
        np.save(path_matrix, ab)

    # Build S, H matrices
    idx_sum = np.add.outer(np.arange(n_basis), np.arange(n_basis))
    idx_sub = np.abs(np.subtract.outer(np.arange(n_basis), np.arange(n_basis)))
    s_mat = (a[idx_sum] + a[idx_sub]) / 2
    h_mat = (b[idx_sum] + b[idx_sub]) / 2

    return s_mat, h_mat


def example_script(pham, hf, mol_name, transform, tag, n_qubits, mol_param, **kwargs):
    max_basis = 60
    thresh = 1e-12
    print_progress = True

    pham_normalized, norm = normalize_by_lcu_norm(pham, level=1)
    eigval_overlap_pair, eigen_energies = spectrum_analysis(pham, hf, mol_name, transform,
                                                            ref_name="HF", tag=tag,
                                                            save=True, load=True, print_progress=print_progress,
                                                            mol_param=mol_param)
    degenerate_block = collect_degeneracy(eigval_overlap_pair, atol_degeneracy=1e-8, norm=norm)
    gnd_energy = degenerate_block[0][1].real
    first_energy = degenerate_block[1][1].real
    spectral_gap = first_energy - gnd_energy


    s_full, h_full = generate_smatrix_cheby(max_basis, hf, mol_name, transform, pham_normalized, n_qubits,
                                            mol_param, **kwargs)

    delta_e0, delta_e1 = np.zeros(max_basis + 1), np.zeros(max_basis + 1)
    basis_list = np.arange(2, max_basis + 1)
    for n_basis in basis_list:
        s_part, h_part = s_full[:n_basis, :n_basis], h_full[:n_basis, :n_basis]
        val, vec = gen_eig_truncated(h_part, s_part, thresh)
        qksd_gnd_energy = val[0]
        qksd_1st_energy = val[1]
        for idx_basis in range(1, n_basis):
            if not np.isclose(qksd_gnd_energy, val[idx_basis]):
                qksd_1st_energy = val[idx_basis]
                break
        delta_e0[n_basis] = abs(qksd_gnd_energy - gnd_energy) / spectral_gap
        delta_e1[n_basis] = abs(qksd_1st_energy - first_energy) / spectral_gap
        print(f"\tn_basis = {n_basis} \tΔE0 = {delta_e0[n_basis]:.3e}\tΔE1 = {delta_e1[n_basis]:.3e}")

    plt.title(mol_name)
    plt.plot(basis_list, delta_e0[2:], label="Delta E0")
    plt.plot(basis_list, delta_e1[2:], label="Delta E1")
    plt.legend()
    plt.yscale('log')
    plt.show()


if __name__ == "__main__":
    for name, model in hubbard_examples.items():
        print(f"{name}")
        data = prepare_hamiltonian_refstates(**model)
        example_script(**data)
