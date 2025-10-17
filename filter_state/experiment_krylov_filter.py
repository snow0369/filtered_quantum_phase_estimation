import os
import pickle
import time

import numpy as np
from ofex.linalg.sparse_tools import expectation
from ofex.operators import normalize_by_lcu_norm

from chemistry_data.chem_tools import spectrum_analysis, prepare_hamiltonian_refstates
from chemistry_data.example_model import hubbard_examples
from filter_state.utils_filter_cheby import calc_cheby
from filter_state.utils_filter_general import gaussian_width_fit_general, max_amplitude, filter_func_eval, \
    gaussian_function, apply_filter_to_state
from filter_state.utils_krylov import generate_smatrix, generate_smatrix_cheby
from filter_state.utils_quantum import collect_degeneracy, gen_eig_truncated

run_setting = {
    "hubbard-6": {
        "eigthresh": 1e-14,
        "max_n_basis": 30,
        "lambda_list": np.logspace(-10, -1, 1001),
    },
    "hubbard-(2,3)": {
        "eigthresh": 1e-14,
        "max_n_basis": 30,
        "lambda_list": np.logspace(-10, -1, 1001),
    },
    "hubbard-7": {
        "eigthresh": 1e-14,
        "max_n_basis": 60,
        "lambda_list": np.logspace(-10, -1, 1001),
    }
}

def krylov_filter_property(filter_basis_type, c, time_step, precalc,
                           s_part, gnd_energy_normalized, gamma_0_sq):
    _, max_amp = max_amplitude(filter_basis_type, c, time_step)
    c /= max_amp

    normalizer = np.vdot(c, s_part.conj() @ c)
    assert np.isclose(normalizer.imag, 0.0), normalizer
    normalizer = normalizer.real

    p_f = normalizer

    if p_f > 1.0 and np.isclose(p_f, 1.0):
        p_f = 1.0
    assert 0 <= p_f <= 1.0, p_f

    f_e0 = complex(filter_func_eval(filter_basis_type, gnd_energy_normalized, c, time_step,
                                    precalc[0, :] if precalc is not None else None))
    gamma_f0_sq = gamma_0_sq * np.abs(f_e0) ** 2 / normalizer

    if gamma_f0_sq > 1.0 and np.isclose(gamma_f0_sq, 1.0, atol=1e-4):
        gamma_f0_sq = 1.0
    assert 0 <= gamma_f0_sq <= 1.0, gamma_f0_sq

    return gamma_f0_sq, p_f


def relative_fpqe_cost(succ_prob, gamma_f02, n_basis, epsilon_normalized, gamma_02):
    qpe_cost = 1 / (epsilon_normalized * gamma_02)
    mf = 1 / (gamma_f02 * succ_prob)
    depth = 2 * n_basis
    fqpe_cost = mf * depth + 1 / (epsilon_normalized * gamma_f02)
    return fqpe_cost / qpe_cost


def script(filter_basis_type, pham, hf, mol_name, transform, tag, n_qubits, mol_param,
           eigthresh, max_n_basis, lambda_list,
           print_progress=True, **kwargs):
    filter_period = 2.0
    epsilon_to_gap = 1e-4
    n_trotter = None  # None for exact propagation
    time_step = 2.0 * np.pi / filter_period

    # Normalize Hamiltonian and obtain spectrum
    pham_normalized, norm = normalize_by_lcu_norm(pham, level=1)
    eigval_overlap_pair, eigen_energies = spectrum_analysis(pham, hf, mol_name, transform,
                                                            ref_name="HF", tag=tag,
                                                            save=True, load=True, print_progress=print_progress,
                                                            mol_param=mol_param)
    degenerate_block = collect_degeneracy(eigval_overlap_pair, atol_degeneracy=1e-8, norm=norm)

    # Print energy values
    hf_energy = expectation(pham_normalized, hf, sparse=True).real
    gnd_energy = degenerate_block[0][1].real
    first_energy = degenerate_block[1][1].real
    spectral_gap = first_energy - gnd_energy
    print(f"GND Energy (normalized): {gnd_energy})")
    print(f"Spect Gap  (normalized): {spectral_gap})")
    print(f"HF Energy  (normalized): {hf_energy})")

    gamma_0_sq = degenerate_block[0][2]
    gamma_1_sq = degenerate_block[1][2]
    print(f"|γ_0|^2: {gamma_0_sq}")
    print(f"|γ_1|^2: {gamma_1_sq}")

    epsilon_normalized = epsilon_to_gap * spectral_gap

    eigvals = np.array([eig for _, eig, _ in degenerate_block])

    # Set maximum basis
    n_basis_list = np.arange(1, max_n_basis + 2, 2)  # only odd basis (trig: -N ~ N, cheby: 0~2N)
    num_n_basis = len(n_basis_list)
    max_n_basis = int(n_basis_list[-1])
    assert max_n_basis % 2 == 1

    sparse_lambda_factor_list = [0.1, 1.0, 10]

    # Generate Krylov matrices
    t = time.time()
    print(f"Generating S matrix ({max_n_basis} x {max_n_basis}) ... ")
    if filter_basis_type == "trig":
        s_mat, h_mat = generate_smatrix(max_n_basis, n_trotter, time_step, hf,
                                        mol_name, transform, pham_normalized, n_qubits, mol_param,
                                        print_progress=print_progress, tag="normalized_1_" + tag,
                                        make_h_matrix=True)
        precalc = None
    elif filter_basis_type == "cheby":
        s_mat, h_mat = generate_smatrix_cheby(max_n_basis, hf, mol_name, transform, pham_normalized, n_qubits, mol_param,
                                              tag="normalized_1_"+tag)
        precalc = calc_cheby(eigvals, max_n_basis + 1)
    else:
        raise ValueError("filter_basis_type must be 'trig' or 'cheby'")
    print(f"done ({time.time() - t} sec)")


    # 1. Krylov analysis from basis = 1 to max
    # Show the convergence of |γ_f|^2 and E_0, E_1 and behavior of p_f and fqpe cost.
    print(" === KSD Analysis with the basis number ===")
    filter_overlap, succ_prob, delta_e0, delta_e1, fqpe_cost = (np.zeros(num_n_basis),
                                                                np.zeros(num_n_basis),
                                                                np.zeros(num_n_basis),
                                                                np.zeros(num_n_basis - 1),
                                                                np.zeros(num_n_basis))
    # n_basis = 1; same as hf energy, No E_1.
    filter_overlap[0], succ_prob[0], delta_e0[0], fqpe_cost[0] = \
        gamma_0_sq, 1.0, abs(hf_energy - gnd_energy), 1.0
    print(f"\tn_basis = {1} \t|γf0|^2 = {gamma_0_sq:.3e}\tp_f = {1.0}\tΔE0 = {delta_e0[0]:.3e}")

    ksd_gnd_energy, c = hf_energy, None
    for idx_n in range(1, num_n_basis):
        n_basis = n_basis_list[idx_n]
        s_part, h_part = s_mat[:n_basis, :n_basis], h_mat[:n_basis, :n_basis]
        val, vec = gen_eig_truncated(h_part, s_part, eigthresh)

        # KSD eigenenergies
        ksd_gnd_energy = val[0]
        ksd_1st_energy = val[1]
        # Avoid degeneracy
        for n in range(1, n_basis):
            if not np.isclose(ksd_gnd_energy, val[n]):
                ksd_1st_energy = val[n]
                break

        # KSD eigen filter function
        c = vec[:, 0].conj()
        gamma_f0_sq, p_f = krylov_filter_property(filter_basis_type, c, time_step, precalc,
                                                  s_part, gnd_energy, gamma_0_sq)

        # save calculated values
        filter_overlap[idx_n] = gamma_f0_sq
        succ_prob[idx_n] = p_f
        delta_e0[idx_n] = np.abs(ksd_gnd_energy - gnd_energy)
        delta_e1[idx_n - 1] = np.abs(ksd_1st_energy - first_energy)
        fqpe_cost[idx_n] = relative_fpqe_cost(p_f, gamma_f0_sq, n_basis, epsilon_normalized, gamma_0_sq)

        print(f"\tn_basis = {n_basis} \t|γf0|^2 = {gamma_f0_sq:.3e}\tp_f = {p_f:.3e}\t"
              f"ΔE0 = {delta_e0[idx_n]:.3e}\tΔE1 = {delta_e1[idx_n - 1]:.3e}\tC_FQPE/C_QPE={fqpe_cost[idx_n]:.3e}")

    c_final = c
    normal_ksd_gnd_energy = ksd_gnd_energy

    # 2. Apply the modified KSD
    # Show the trade-off between p_f and |γ_f|^2 in terms of the lagrangian factor.
    print(" === Modified KSD analysis with the lagrangian coefficient === ")
    n_lambda = len(lambda_list)
    tot_modified_overlap, tot_modified_succprob, tot_modified_deltae0, tot_modified_deltae1, tot_modified_fqpe_cost = \
        (np.zeros((num_n_basis, n_lambda)), np.zeros((num_n_basis, n_lambda)),
         np.zeros((num_n_basis, n_lambda)), np.zeros((num_n_basis - 1, n_lambda)),
         np.zeros((num_n_basis, n_lambda)))
    (best_modified_overlap, best_modified_succprob, best_modified_deltae0,
     best_modified_deltae1, best_modified_fqpe_cost) = \
        (np.zeros(num_n_basis), np.zeros(num_n_basis), np.zeros(num_n_basis),
         np.zeros(num_n_basis - 1), np.zeros(num_n_basis))
    min_cost_lmda_idx = np.zeros(num_n_basis, dtype=int)

    n_sparse_lambda = len(sparse_lambda_factor_list)
    tot_sparse_modified_overlap, tot_sparse_modified_succprob, \
    tot_sparse_modified_deltae0, tot_sparse_modified_deltae1, \
    tot_sparse_modified_fqpe_cost =\
        (np.zeros((n_sparse_lambda, num_n_basis)), np.zeros((n_sparse_lambda, num_n_basis)),
         np.zeros((n_sparse_lambda, num_n_basis)), np.zeros((n_sparse_lambda, num_n_basis - 1)),
         np.zeros((n_sparse_lambda, num_n_basis)))


    modified_c_vec = list()

    tot_modified_overlap[0, :], tot_modified_succprob[0, :], tot_modified_deltae0[0, :], tot_modified_fqpe_cost[0, :] = \
        gamma_0_sq, 1.0, abs(hf_energy - gnd_energy), 1.0
    best_modified_overlap[0], best_modified_succprob[0], best_modified_deltae0[0], best_modified_fqpe_cost[0] = \
        gamma_0_sq, 1.0, abs(hf_energy - gnd_energy), 1.0
    best_modified_c_vec = None
    tot_sparse_modified_overlap[:, 0], tot_sparse_modified_succprob[:, 0], tot_sparse_modified_deltae0[:, 0],\
    tot_sparse_modified_fqpe_cost[:, 0] = gamma_0_sq, 1.0, abs(hf_energy - gnd_energy), 1.0

    for idx_n in range(1, num_n_basis):
        n_basis = n_basis_list[idx_n]
        s_part, h_part = s_mat[:n_basis, :n_basis], h_mat[:n_basis, :n_basis]
        # ----- dense optimization of lambda -----
        for idx_lmda, lmda in enumerate(lambda_list):
            h_shift = h_part + np.eye(h_part.shape[0], dtype=complex) * lmda
            val, vec = gen_eig_truncated(h_shift, s_part, eigthresh)

            # modified KSD eigen filter function
            c0, c1 = vec[:, 0].conj(), vec[:, 1].conj()
            gamma_f0_sq, p_f = krylov_filter_property(filter_basis_type,
                                                      c0, time_step, precalc, s_part, gnd_energy, gamma_0_sq)

            # modified KSD eigenenergies
            ksd_gnd_energy = (vec[:, 0].T.conj() @ h_part @ vec[:, 0]) / (vec[:, 0].T.conj() @ s_part @ vec[:, 0])
            ksd_1st_energy = (vec[:, 1].T.conj() @ h_part @ vec[:, 1]) / (vec[:, 1].T.conj() @ s_part @ vec[:, 1])
            # Avoid degeneracy
            for n in range(1, h_part.shape[0]):
                tmp_c = vec[:, n]
                tmp_ksd_1st_energy = (tmp_c.T.conj() @ h_part @ tmp_c) / (tmp_c.T.conj() @ s_part @ tmp_c)
                if not np.isclose(ksd_gnd_energy, tmp_ksd_1st_energy):
                    ksd_1st_energy = tmp_ksd_1st_energy
                    break

            # save calculated values
            tot_modified_overlap[idx_n, idx_lmda] = gamma_f0_sq
            tot_modified_succprob[idx_n, idx_lmda] = p_f
            tot_modified_deltae0[idx_n, idx_lmda] = np.abs(ksd_gnd_energy - gnd_energy)
            tot_modified_deltae1[idx_n - 1, idx_lmda] = np.abs(ksd_1st_energy - first_energy)
            tot_modified_fqpe_cost[idx_n, idx_lmda] = relative_fpqe_cost(p_f, gamma_f0_sq, max_n_basis,
                                                                         epsilon_normalized, gamma_0_sq)

            if n_basis == max_n_basis:
                modified_c_vec.append(c0)

            if idx_lmda % 10 == 0:
                print(
                    f"\tlambda = {lmda:.3e} \t|γf0|^2 = {gamma_f0_sq:.3e}\tp_f = {p_f:.3e}\t"
                    f"ΔE0 = {tot_modified_deltae0[idx_n, idx_lmda]:.3e}\tΔE1 = {tot_modified_deltae1[idx_n - 1, idx_lmda]:.3e}\t"
                    f"C_FQPE/C_QPE={tot_modified_fqpe_cost[idx_n, idx_lmda]:.3e}")

        # find the best lambda parameter
        min_idx = np.argmin(tot_modified_fqpe_cost[idx_n, :])
        (best_modified_overlap[idx_n], best_modified_succprob[idx_n], best_modified_deltae0[idx_n],
         best_modified_deltae1[idx_n - 1], best_modified_fqpe_cost[idx_n]) = \
            (tot_modified_overlap[idx_n, min_idx], tot_modified_succprob[idx_n, min_idx],
             tot_modified_deltae0[idx_n, min_idx], tot_modified_deltae1[idx_n - 1, min_idx],
             tot_modified_fqpe_cost[idx_n, min_idx])
        min_cost_lmda_idx[idx_n] = min_idx

        if n_basis == max_n_basis:
            best_modified_c_vec = modified_c_vec[min_idx]

        # ----- sparse experiment -----
        for idx_lmda, lmda_factor in enumerate(sparse_lambda_factor_list):
            lmda = lmda_factor * epsilon_normalized * n_basis
            h_shift = h_part + np.eye(h_part.shape[0], dtype=complex) * lmda
            val, vec = gen_eig_truncated(h_shift, s_part, eigthresh)

            # modified KSD eigen filter function
            c0, c1 = vec[:, 0].conj(), vec[:, 1].conj()
            gamma_f0_sq, p_f = krylov_filter_property(filter_basis_type,
                                                      c0, time_step, precalc, s_part, gnd_energy, gamma_0_sq)

            # modified KSD eigenenergies
            ksd_gnd_energy = (vec[:, 0].T.conj() @ h_part @ vec[:, 0]) / (vec[:, 0].T.conj() @ s_part @ vec[:, 0])
            ksd_1st_energy = (vec[:, 1].T.conj() @ h_part @ vec[:, 1]) / (vec[:, 1].T.conj() @ s_part @ vec[:, 1])
            # Avoid degeneracy
            for n in range(1, h_part.shape[0]):
                tmp_c = vec[:, n]
                tmp_ksd_1st_energy = (tmp_c.T.conj() @ h_part @ tmp_c) / (tmp_c.T.conj() @ s_part @ tmp_c)
                if not np.isclose(ksd_gnd_energy, tmp_ksd_1st_energy):
                    ksd_1st_energy = tmp_ksd_1st_energy
                    break

            tot_sparse_modified_overlap[idx_lmda, idx_n] = gamma_f0_sq
            tot_sparse_modified_succprob[idx_lmda, idx_n] = p_f
            tot_sparse_modified_deltae0[idx_lmda, idx_n] = np.abs(ksd_gnd_energy - gnd_energy)
            tot_sparse_modified_deltae1[idx_lmda, idx_n - 1] = np.abs(ksd_1st_energy - first_energy)
            tot_sparse_modified_fqpe_cost[idx_lmda, idx_n] = relative_fpqe_cost(p_f, gamma_f0_sq, max_n_basis,
                                                                                epsilon_normalized, gamma_0_sq)

    min_cost_lambda = lambda_list[min_cost_lmda_idx]
    # 3. Compare to the Gaussian filter
    print(" === Find similar Gaussian filter === ")
    # 3-1. Determine the width of the original Krylov filter
    x_max, max_amp = max_amplitude(filter_basis_type, best_modified_c_vec, time_step)
    w_opt, _ = gaussian_width_fit_general(filter_basis_type, best_modified_c_vec, period=filter_period, center=x_max,
                                          width_bounds=(1e-4, 1e-1))
    print(f"Center: {x_max}, Width: {w_opt}")

    # 3-2. Generate gaussian filter and apply filters to the state
    _, gaussian_c_vec = gaussian_function(filter_basis_type,
                                          max_n_basis if filter_basis_type == "trig" else max_n_basis * 2 + 1,
                                          w_opt, normal_ksd_gnd_energy, filter_period)
    degen_block_krylov, pf_krylov = apply_filter_to_state(filter_basis_type, degenerate_block,
                                                          c_final, time_step, precalc)
    degen_block_modkry, pf_modkry = apply_filter_to_state(filter_basis_type, degenerate_block,
                                                          best_modified_c_vec, time_step, precalc)
    degen_block_gauss, pf_gauss = apply_filter_to_state(filter_basis_type, degenerate_block,
                                                        gaussian_c_vec, time_step, precalc)
    gamma0_krylov_2 = degen_block_krylov[0][2]
    gamma0_modkry_2 = degen_block_modkry[0][2]
    gamma0_gauss_2 = degen_block_gauss[0][2]
    cost_krylov = relative_fpqe_cost(pf_krylov, gamma0_krylov_2, max_n_basis, epsilon_normalized, gamma_0_sq)
    cost_modkry = relative_fpqe_cost(pf_modkry, gamma0_modkry_2, max_n_basis, epsilon_normalized, gamma_0_sq)
    cost_gauss = relative_fpqe_cost(pf_gauss, gamma0_gauss_2, max_n_basis, epsilon_normalized, gamma_0_sq)

    # 4. Save the calculated values into a file.
    plot_bundle = {
        # --- identification / mode ---
        "mol_name": mol_name,
        "tag": tag,
        "filter_basis_type": filter_basis_type,   # "trig" or "cheby"

        # --- basic scalars ---
        "time_step": time_step,
        "gnd_energy": gnd_energy,
        "first_energy": first_energy,
        "spectral_gap": spectral_gap,
        "epsilon_to_gap": epsilon_to_gap,
        "epsilon_normalized": epsilon_normalized,
        "max_n_basis": int(max_n_basis),

        # --- grids / lists ---
        "n_basis_list": np.asarray(n_basis_list),
        "lambda_list": np.asarray(lambda_list),

        # --- originals (analysis vs N) ---
        # x-axis in your plots used n_basis_list
        "originals": {
            "x": np.asarray(n_basis_list),
            "filter_overlap": np.asarray(filter_overlap),
            "succ_prob": np.asarray(succ_prob),
            "delta_e0": np.asarray(delta_e0),
            "delta_e1": np.asarray(delta_e1),
            "fqpe_cost": np.asarray(fqpe_cost),
            # divide by this in plots for normalized errors
            "spectral_gap_normalized": float(spectral_gap),
        },

        # --- modified (analysis vs lambda) ---
        "modified": {
            "x_lambda": np.asarray(lambda_list),
            "overlap": np.asarray(tot_modified_overlap[-1, :]),
            "succprob": np.asarray(tot_modified_succprob[-1, :]),
            "deltae0": np.asarray(tot_modified_deltae0[-1, :]),
            "deltae1": np.asarray(tot_modified_deltae1[-1, :]),
            "cost": np.asarray(tot_modified_fqpe_cost[-1, :]),
            "min_cost_idx": int(min_cost_lmda_idx[-1]),
            "spectral_gap_normalized": float(spectral_gap),
        },

        "modified_best": {
            "x": np.asarray(n_basis_list),
            "filter_overlap": np.asarray(best_modified_overlap),
            "succ_prob": np.array(best_modified_succprob),
            "delta_e0": np.array(best_modified_deltae0),
            "delta_e1": np.array(best_modified_deltae1),
            "fqpe_cost": np.array(best_modified_fqpe_cost),
            "spectral_gap_normalized": float(spectral_gap),
            "min_cost_lambda": min_cost_lambda,
        },

        "modified_sparse": {
            "x": np.asarray(n_basis_list),
            "epsilon_normalized": epsilon_normalized,
            "sparse_lambda": np.array(sparse_lambda_factor_list),
            "filter_overlap_by_lambda": np.asarray(tot_sparse_modified_overlap),
            "succ_prob_by_lambda": np.array(tot_sparse_modified_succprob),
            "delta_e0_by_lambda": np.array(tot_sparse_modified_deltae0),
            "delta_e1_by_lambda": np.array(tot_sparse_modified_deltae1),
            "fqpe_cost_by_lambda": np.array(tot_sparse_modified_fqpe_cost),
            "spectral_gap_normalized": float(spectral_gap),
        },

        # --- filters & state blocks (for function plots + histograms) ---
        "filters_blocks": {
            "c_final": np.asarray(c_final) if c_final is not None else None,
            "best_modified_c_vec": np.asarray(best_modified_c_vec),
            "gaussian_c_vec": np.asarray(gaussian_c_vec),
            "degenerate_block": degenerate_block,            # [(idx, E, |<E|phi_0>|^2), ...]
            "degen_block_krylov": degen_block_krylov,        # after applying filters
            "degen_block_modkry": degen_block_modkry,
            "degen_block_gauss": degen_block_gauss,
            "precalc": precalc,                              # for chebyshev path; None for trig
        },

        # --- application metrics (for text boxes in hist figs) ---
        "annotations": {
            "pf_krylov": float(pf_krylov),
            "gamma0_krylov_2": float(gamma0_krylov_2),
            "cost_krylov": float(cost_krylov),
            "pf_modkry": float(pf_modkry),
            "gamma0_modkry_2": float(gamma0_modkry_2),
            "cost_modkry": float(cost_modkry),
            "pf_gauss": float(pf_gauss),
            "gamma0_gauss_2": float(gamma0_gauss_2),
            "cost_gauss": float(cost_gauss),
        },

        # --- gaussian fit aux (optional but handy) ---
        "gaussian_fit": {
            "x_max": float(x_max),
            "w_opt": float(w_opt),
        },
    }

    out_dir = f'./data/krylov_analysis_{filter_basis_type}/'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{mol_name}_{transform}_{tag}_plot_inputs.pkl')

    with open(out_path, 'wb') as f:
        pickle.dump(plot_bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[saved plot bundle] {out_path}")
    # (optionally) return the path
    return out_path


if __name__ == "__main__":
    for name, model in hubbard_examples.items():
        data = prepare_hamiltonian_refstates(**model)
        data = data | run_setting[name]
        print(f"{name} Trig basis")
        script(filter_basis_type="trig", **data)
        print(f"{name} Cheby basis")
        script(filter_basis_type="cheby", **data)
