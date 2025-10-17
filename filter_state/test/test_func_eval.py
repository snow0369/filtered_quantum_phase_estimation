import time

import numpy as np
from matplotlib import pyplot as plt

from chemistry_data.chem_tools import spectrum_analysis, prepare_hamiltonian_refstates
from chemistry_data.example_model import hubbard_examples
from filter_state.utils_filter_cheby import effective_gaussian_params_cheby, filter_func_eval_cheby, calc_cheby, \
    apply_filter_to_state_cheby
from filter_state.utils_quantum import collect_degeneracy
from filter_state.utils_filter_trig import filter_func_eval_trig, apply_filter_to_state_trig, \
    effective_gaussian_params_trig
from filter_state.utils_krylov import generate_smatrix, generate_smatrix_cheby
from ofex.classical_algorithms.filter_functions import gaussian_function_fourier
from ofex.classical_algorithms.filter_functions.gaussian import gaussian_function_cheby
from ofex.operators import normalize_by_lcu_norm


def script(pham, hf, mol_name, transform, tag, n_qubits, mol_param, print_progress=True,
           **kwargs):
    filter_period = 2
    time_step = 2.0 * np.pi / filter_period
    filter_epsilon = 1e-2
    n_trotter = None

    # ===== Normalize Hamiltonian and obtain spectrum ===== #
    pham_normalized, norm = normalize_by_lcu_norm(pham, level=1)
    eigval_overlap_pair, eigen_energies = spectrum_analysis(pham, hf, mol_name, transform,
                                                            ref_name="HF", tag=tag,
                                                            save=True, load=True, print_progress=print_progress,
                                                            mol_param=mol_param)
    degenerate_block_normalized = collect_degeneracy(eigval_overlap_pair, atol_degeneracy=1e-8, norm=norm)

    gnd_energy = degenerate_block_normalized[0][1].real
    first_energy = degenerate_block_normalized[1][1].real
    ov2 = degenerate_block_normalized[0][2]

    center, width = gnd_energy, first_energy - gnd_energy

    test_grid = np.linspace(-1, 1, 1000)

    fig, (ax1, ax2) = plt.subplots(2, 1)

    # ===== Trigonometric filter function ==== #
    n_basis, rescaled_width = effective_gaussian_params_trig(width, filter_period, filter_epsilon)

    t = time.time()
    print(f"Generating trig S matrix (size = {n_basis}) ... ", end="")
    s_mat = generate_smatrix(n_basis, n_trotter, time_step, hf,
                             mol_name, transform, pham_normalized, n_qubits, mol_param,
                             conj=True, print_progress=print_progress,
                             tag="normalized_1_" + tag)
    print(f"done ({time.time() - t} sec)")

    n_fourier = (n_basis - 1) // 2
    filter_func, filter_coeff, _ = gaussian_function_fourier(n_fourier, rescaled_width, center, filter_period)

    # 1. f(test_grid) -> will be plotted later
    t = time.time()
    print(f"trig test: f(x_grid) (size = {len(test_grid)}) ... ", end="")
    test_y_grid_trig = filter_func_eval_trig(test_grid, filter_coeff, time_step)
    print(f"done ({time.time() - t})")

    ax1.plot(test_grid, np.abs(test_y_grid_trig) ** 2, label="trig test")
    ax1.set_yscale('log')
    ax1.legend()

    # 2. apply filter to the state
    t = time.time()
    print(f"trig test: apply state (size = {len(degenerate_block_normalized)}) ... ", end="")
    f_degenblock_1, norm_1 = apply_filter_to_state_trig(degenerate_block_normalized, filter_coeff, time_step)
    print(f"done ({time.time() - t} sec)")

    # 3. norm calculation by smat and f(E0) -> compare with 2
    t = time.time()
    print(f"trig test: norm and f(E0) with smat (size = {s_mat.shape[0]}) ... ", end="")
    norm_2 = (filter_coeff.T.conj() @ s_mat @ filter_coeff).real
    fe0 = filter_func_eval_trig(gnd_energy, filter_coeff, time_step)
    ov2_filtered = ov2 * abs(fe0) ** 2 / norm_2
    print(f"done ({time.time() - t} sec)")

    assert np.isclose(norm_1, norm_2), f"norm_1 = {norm_1}, norm_2 = {norm_2}"
    assert np.isclose(f_degenblock_1[0][2], ov2_filtered), \
        f"ov2_filtered_1 = {f_degenblock_1[0][2]}, ov2_filtered_2 = {ov2_filtered}"
    print(f"norm = {norm_1}, ov2 = {ov2_filtered}")

    # ===== Chebyshev filter function ===== #
    n_basis, rescaled_width = effective_gaussian_params_cheby(width, filter_epsilon)

    t = time.time()
    print(f"Generating cheby S matrix (size = {n_basis}) ... ", end="")
    s_mat, _ = generate_smatrix_cheby(n_basis, hf, mol_name, transform, pham_normalized, n_qubits, mol_param,
                                      tag="normalized_1_" + tag)
    print(f"done ({time.time() - t} sec)")

    n_poly = n_basis - 1
    filter_func, filter_coeff = gaussian_function_cheby(n_poly, rescaled_width, center, filter_period)

    # 0. Precalculate cheby(eig)
    t = time.time()
    print(f"cheby precalc: (size = {len(degenerate_block_normalized)}) ... ", end="")
    eigvals = np.array([eig.real for _, eig, _ in degenerate_block_normalized], dtype=float)
    precalc = calc_cheby(eigvals, n_basis)
    print(f"done ({time.time() - t} sec)")

    # 1. f(test_grid) -> will be plotted later
    t = time.time()
    print(f"cheby test: f(x_grid) (size = {len(test_grid)}) ... ", end="")
    test_y_grid_cheby = filter_func_eval_cheby(test_grid, filter_coeff)
    print(f"done ({time.time() - t} sec)")

    ax2.plot(test_grid, np.abs(test_y_grid_cheby) ** 2, label="cheby test")
    ax2.set_yscale('log')
    ax2.legend()
    plt.show()

    # 2. apply filter to the state
    t = time.time()
    print(f"cheby test: apply state (size = {len(degenerate_block_normalized)}) ... ", end="")
    f_degenblock_1, norm_1 = apply_filter_to_state_cheby(degenerate_block_normalized, filter_coeff, precalc)
    print(f"done ({time.time() - t} sec)")

    # 3. norm calculation by smat and f(E0) -> compare with 2
    t = time.time()
    print(f"cheby test: norm and f(E0) with smat (size = {s_mat.shape[0]}) ... ", end="")
    norm_2 = (filter_coeff.T.conj() @ s_mat @ filter_coeff).real
    fe0 = filter_func_eval_cheby(gnd_energy, filter_coeff, precalc[0, :])
    ov2_filtered = ov2 * abs(fe0) ** 2 / norm_2
    print(f"done ({time.time() - t} sec)")

    assert np.isclose(norm_1, norm_2), f"norm_1 = {norm_1}, norm_2 = {norm_2}"
    assert np.isclose(f_degenblock_1[0][2], ov2_filtered), \
        f"ov2_filtered_1 = {f_degenblock_1[0][2]}, ov2_filtered_2 = {ov2_filtered}"
    print(f"norm = {norm_1}, ov2 = {ov2_filtered}")


if __name__ == "__main__":
    name = 'hubbard-7'
    model = hubbard_examples[name]

    data = prepare_hamiltonian_refstates(**model)
    script(**data)
