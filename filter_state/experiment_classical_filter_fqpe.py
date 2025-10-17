import json
import multiprocessing
import os
import pickle
import random
import threading
import time
import warnings
from datetime import datetime
from itertools import product

import numpy as np
from filelock import FileLock
from ofex.linalg.sparse_tools import expectation
from ofex.operators import normalize_by_lcu_norm
from scipy.special import lambertw
from tqdm import tqdm

from chemistry_data.chem_tools import prepare_hamiltonian_refstates, spectrum_analysis
from chemistry_data.example_model import hubbard_examples
from filter_state.utils_filter_cheby import calc_cheby
from filter_state.utils_filter_general import apply_filter_to_state, filter_func_eval, effective_gaussian_params, \
    gaussian_function
from filter_state.utils_krylov import generate_smatrix, generate_smatrix_cheby
from filter_state.utils_quantum import collect_degeneracy
from filter_state.utils_repr import round_to_2_sigfigs_scientific

warnings.filterwarnings("ignore")


# =================================== Tools for log and parallel works =================================== #
def init_worker(shared_dict,  # Manager.dict – tiny proxy object
                filter_period,
                filter_epsilon,
                filter_basis_type,
                s_mat,
                precalc,
                degenerate_block_normalized,
                debug,
                log_fname,
                log_flock_name):
    """
    Runs once in *each* worker.  Stores big read-only objects in globals so
    every task can use them without being re-pickled and shipped again.
    """
    global FILTER_PROP, PERIOD, EPS, BASIS, S_MAT, PRECALC, DEG, DEBUG, LOG, FLOCK
    FILTER_PROP = shared_dict
    PERIOD = filter_period
    EPS = filter_epsilon
    BASIS = filter_basis_type
    S_MAT = s_mat
    PRECALC = precalc
    DEG = degenerate_block_normalized
    DEBUG = debug
    LOG = log_fname
    FLOCK = FileLock(log_flock_name, timeout=30)


def tail_thread(log_fname, total, pbar, stop_event):
    with open(log_fname, "r") as f:
        f.seek(0, os.SEEK_END)
        while not stop_event.is_set():
            line = f.readline()
            if line:  # got one (maybe several) new line(s)
                pbar.update(line.count("\n"))
            else:  # nothing new – wait a bit
                time.sleep(0.1)


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def log_line(center, width, gnd_energy_normalized, spectral_gap_normalized, result):
    msg = dict()
    msg["ts"] = timestamp()
    msg["pid"] = os.getpid()
    msg["center"] = f"{(center - gnd_energy_normalized) / spectral_gap_normalized:.4e} Δ + E0"
    msg["width"] = f"{width / spectral_gap_normalized:.4e} Δ"
    msg["result"] = result
    return json.dumps(msg, separators=(', ', ':')) + '\n'


# ======================================== Gaussian filter Helpers ======================================== #
def optimal_gaussian_width(delta_0, delta_1, overlap, epsilon):
    # delta_i = |E_i - \tilde{E}_0|
    r = delta_1 / delta_0
    opt_width = delta_1 / np.sqrt(lambertw(np.sqrt(2) * (1 - np.abs(overlap) ** 2) * (r ** 2 - 1)
                                           * np.exp(r ** 2) * delta_1 / epsilon)
                                  - r ** 2)
    assert np.isclose(opt_width.imag, 0.0), opt_width
    return opt_width.real


# ============================================= Main Script ============================================= #
def script_cost_analysis(filter_basis_type, pham, hf, mol_name, transform, tag, n_qubits, mol_param,
                         epsilon_to_gap, print_progress=True, **kwargs):
    # ===== Parameter Setting ===== #
    # Can be frequently changed
    max_center_deviation_factor = 3.0
    min_filter_width_factor = 1 / 3
    max_filter_width_factor = 5.0
    n_filter_center = 121
    n_filter_width = 121

    # Rarely changed
    epsilon_simulation = None  # If none, it is set to epsilon_normalized
    filter_period = 2.0
    debug = False
    max_smat_dim = 14001  # should be odd, adjust depending on your time/memory budget
    num_workers = 8

    n_trotter = None  # None for exact propagation, only for trigonometric filters

    # ===== Normalize Hamiltonian and obtain spectrum ===== #
    pham_normalized, norm = normalize_by_lcu_norm(pham, level=1)
    eigval_overlap_pair, eigen_energies = spectrum_analysis(pham, hf, mol_name, transform,
                                                            ref_name="HF", tag=tag,
                                                            save=True, load=True, print_progress=print_progress,
                                                            mol_param=mol_param)
    degenerate_block_normalized = collect_degeneracy(eigval_overlap_pair, atol_degeneracy=1e-8, norm=norm)
    eigvals = np.array([eig for _, eig, _ in degenerate_block_normalized])

    # Print energy values
    hf_energy_normalized = expectation(pham_normalized, hf, sparse=True).real
    gnd_energy_normalized = degenerate_block_normalized[0][1].real
    first_energy_normalized = degenerate_block_normalized[1][1].real
    spectral_gap_normalized = first_energy_normalized - gnd_energy_normalized
    print(f"GND Energy (normalized): {gnd_energy_normalized}")
    print(f"Spect Gap  (normalized): {spectral_gap_normalized}")
    print(f"HF Energy  (normalized): {hf_energy_normalized}")

    gamma_0_sq = degenerate_block_normalized[0][2]
    gamma_1_sq = degenerate_block_normalized[1][2]
    print(f"|γ_0|^2: {gamma_0_sq}")
    print(f"|γ_1|^2: {gamma_1_sq}")

    epsilon_to_gap = round_to_2_sigfigs_scientific(epsilon_to_gap)
    epsilon_normalized = epsilon_to_gap * spectral_gap_normalized
    print(f"epsilon to gap      : {epsilon_to_gap}")
    print(f"epsilon (normalized): {epsilon_normalized}")

    if epsilon_simulation is None:
        epsilon_simulation = epsilon_normalized

    # ===== Filter Properties ===== #
    filter_center = np.linspace(gnd_energy_normalized - spectral_gap_normalized * max_center_deviation_factor,
                                gnd_energy_normalized + spectral_gap_normalized * max_center_deviation_factor,
                                n_filter_center)
    filter_width = np.linspace(spectral_gap_normalized * min_filter_width_factor,
                               spectral_gap_normalized * max_filter_width_factor,
                               n_filter_width)

    print(f"filter center min/max: {min(filter_center)}/{max(filter_center)}")
    print(f"filter width  min/max: {min(filter_width)}/{max(filter_width)}")

    min_filter_width = np.min(filter_width)

    # Set filter fluctuation
    if filter_basis_type == "trig":
        c = 5 * epsilon_normalized / (4 * np.pi * spectral_gap_normalized)
        filter_epsilon = np.sqrt(c * lambertw(1 / c))
    elif filter_basis_type == "cheby":
        c = 5 * np.e * epsilon_normalized / (32 * spectral_gap_normalized * np.log(1 / epsilon_simulation))
        filter_epsilon = np.sqrt(c * lambertw(16 / c))
    else:
        raise ValueError
    assert np.isclose(filter_epsilon.imag, 0.0)
    assert filter_epsilon.real > 0.0
    filter_epsilon = filter_epsilon.real
    print(f"filter_epsilon: {filter_epsilon}")

    # Get the max number of basis.
    assert min_filter_width > 0.0
    max_n_basis, _ = effective_gaussian_params(filter_basis_type, min_filter_width, filter_epsilon, filter_period)

    print(f"min_filter_width: {min_filter_width}")
    print(f"max_n_basis: {max_n_basis}")
    # num_workers = max(1, min((2 ** 30) // ((2 * max_n_basis + 1) ** 2), 8))
    # print(f"num workers = {num_workers}")

    # For efficient computation of filter properties, generate S matrix
    t = time.time()
    time_step = 2.0 * np.pi / filter_period
    n_dim_smat = min(max_n_basis, max_smat_dim)
    print(f"Generating S matrix ({n_dim_smat} x {n_dim_smat})... ", end="")
    if filter_basis_type == "trig":
        s_mat = generate_smatrix(n_dim_smat, n_trotter, time_step, hf,
                                 mol_name, transform, pham_normalized, n_qubits, mol_param,
                                 conj=True,
                                 print_progress=print_progress, tag="normalized_1_" + tag)
        precalc = None
    else:
        s_mat, _ = generate_smatrix_cheby(n_dim_smat, hf,
                                          mol_name, transform, pham_normalized, n_qubits, mol_param,
                                          tag="normalized_1_" + tag)
        precalc = calc_cheby(eigvals, max_n_basis)
    print(f"done ({time.time() - t} sec)")

    # File names
    dir_name = f"./data/fstate_width_center_{filter_basis_type}/"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    properties_fname = \
        os.path.join(dir_name,
                     f"{mol_name}_{transform}_{epsilon_to_gap:.2e}_filter_properties.pkl" if tag is None else \
                         f"{mol_name}_{transform}_{epsilon_to_gap:.2e}_{tag}_filter_properties.pkl")

    # Load previously worked data
    filter_property_list = dict()
    if os.path.exists(properties_fname):
        with open(properties_fname, 'rb') as f:
            _gnd_energy_normalized, _spectral_gap_normalized, _epsilon_to_gap, \
                _gamma_0_sq, _filter_type, _filter_center, _filter_width, \
                filter_property_list = pickle.load(f)
        assert np.isclose(_gnd_energy_normalized, gnd_energy_normalized)
        assert np.isclose(_spectral_gap_normalized, spectral_gap_normalized)
        assert np.isclose(_epsilon_to_gap, epsilon_to_gap)
        assert np.isclose(_gamma_0_sq, gamma_0_sq)
        assert _filter_type == ["gaussian_function_fourier"]

    # Log file
    log_dir_name = f"./log/fstate_width_center_fixed_{filter_basis_type}/"
    if not os.path.exists(log_dir_name):
        os.makedirs(log_dir_name)
    log_fname = \
        os.path.join(log_dir_name,
                     f"{mol_name}_{transform}_{epsilon_to_gap:.2e}_filter_properties.log" if tag is None else \
                         f"{mol_name}_{transform}_{epsilon_to_gap:.2e}_{tag}_filter_properties.log")
    open(log_fname, "w").close()   # truncate old log
    log_flock_name = log_fname + ".lock"

    # Prepare parallel works.
    work_list = list(product(filter_center, filter_width))
    random.shuffle(work_list)
    manager = multiprocessing.Manager()
    filter_property_list = manager.dict(filter_property_list.items())
    chunksize = max(1, len(work_list) // (num_workers * 4))

    stop_evt = threading.Event()
    pbar = tqdm(total=len(work_list))
    tail_thr = threading.Thread(target=tail_thread,
                                args=(log_fname, len(work_list), pbar, stop_evt),
                                daemon=True)
    tail_thr.start()

    with multiprocessing.Pool(
            processes=num_workers,
            initializer=init_worker,
            initargs=(filter_property_list, filter_period, filter_epsilon,
                      filter_basis_type, s_mat, precalc, degenerate_block_normalized,
                      debug, log_fname, log_flock_name)
    ) as pool:
        async_res = pool.map_async(_parallel_work, work_list, chunksize=chunksize)
        async_res.get()

    stop_evt.set()
    tail_thr.join()
    pbar.close()

    with open(properties_fname, "wb") as f:
        pickle.dump((gnd_energy_normalized, spectral_gap_normalized, epsilon_to_gap,
                     gamma_0_sq, ["gaussian_function_fourier"],
                     filter_center.tolist(), filter_width.tolist(),
                     dict(filter_property_list)), f)


def _parallel_work(args):
    def q(x, ndigits=7):
        return round(x, ndigits)

    filter_center, filter_width = args

    filter_property_list, filter_period, filter_epsilon, filter_basis_type, \
        s_mat, precalc, degenerate_block, debug, log_fname = \
        FILTER_PROP, PERIOD, EPS, BASIS, S_MAT, PRECALC, DEG, DEBUG, LOG

    gnd_energy = degenerate_block[0][1]
    first_energy = degenerate_block[1][1]
    spectral_gap_normalized = first_energy - gnd_energy

    filter_center, filter_width = q(filter_center), q(filter_width)
    time_step = 2.0 * np.pi / filter_period

    if ("gaussian_function_fourier", filter_center, filter_width) in filter_property_list:
        with FLOCK:
            with open(log_fname, "a") as f:
                f.write(log_line(filter_center, filter_width, gnd_energy, spectral_gap_normalized,
                                 "skipped_computation"))
                f.flush()
        return
    assert filter_width > 0

    n_basis, rescaled_width = effective_gaussian_params(filter_basis_type, filter_width, filter_epsilon, filter_period)
    _, coeff_gauss = gaussian_function(filter_basis_type, n_basis, rescaled_width, filter_center, filter_period)

    if s_mat.shape[0] >= n_basis:
        s_mat = s_mat[:n_basis, :n_basis]
    else:
        s_mat = None

    gauss_properties = filter_properties(coeff_gauss, time_step, n_basis, filter_center, filter_width,
                                         filter_basis_type, s_mat, precalc,
                                         degenerate_block, log_fname)
    filter_property_list[("gaussian_function_fourier", filter_center, filter_width)] = gauss_properties


def filter_properties(coeff: np.ndarray,
                      time_step: np.ndarray,
                      n_basis: int,
                      filter_center: float,
                      filter_width: float,
                      filter_basis_type: str,
                      s_mat: np.ndarray,
                      precalc,
                      degenerate_block,
                      log_fname):
    gnd_energy = degenerate_block[0][1]
    first_energy = degenerate_block[1][1]
    spectral_gap_normalized = first_energy - gnd_energy
    initial_overlap_squared = degenerate_block[0][2]

    depth = n_basis * 2 if filter_basis_type == 'trig' else n_basis

    # 1. Calculate f_e0
    f_e0 = filter_func_eval(filter_basis_type, gnd_energy, coeff, time_step, \
                            precalc[0, :] if precalc is not None else None)
    process_result = ""
    try:
        if s_mat is None:
            raise Exception("safe:no_smat")

        # 2. Calculate prob with s matrix
        succ_prob = (coeff.T.conj() @ s_mat @ coeff).real

        if succ_prob > 1.0 and np.isclose(succ_prob, 1.0):
            succ_prob = 1.0
            process_result = "normal"
        elif (succ_prob > 1.0 and not np.isclose(succ_prob, 1.0)) or \
                (succ_prob < 0.0 and not np.isclose(succ_prob, 0.0)):
            raise Exception("safe:pf_outbound")
        elif succ_prob < 0.0 and np.isclose(succ_prob, 0.0) and np.isclose(abs(f_e0), 0.0):
            raise Exception("safe:small_f0")  # Numerically unstable overlap
        elif succ_prob < 0.0:
            process_result = "normal:pf_negative"
            succ_prob = 1e-13
        else:
            process_result = "normal"

        # 3. Calculate filtered overlap
        overlap_squared = initial_overlap_squared * abs(f_e0) ** 2 / succ_prob

    except Exception as exc:
        ret_degen_block, succ_prob = apply_filter_to_state(filter_basis_type, degenerate_block, coeff,
                                                           time_step, precalc)
        process_result = str(exc)
        if succ_prob > 1.0 and np.isclose(succ_prob, 1.0):
            succ_prob = 1.0
            process_result = process_result + "_pf_renormalized"
        assert 0 <= succ_prob <= 1.0

        gamma_f_ei_sq = np.array([gamma_f_sq for _, _, gamma_f_sq in ret_degen_block])
        overlap_squared = 1 / (1 + np.sum(gamma_f_ei_sq[1:]) / gamma_f_ei_sq[0])

    # 4. Log
    with FLOCK:
        with open(log_fname, "a") as f:
            f.write(log_line(filter_center, filter_width, gnd_energy, spectral_gap_normalized,
                             process_result))
            f.flush()
    return succ_prob, f_e0, overlap_squared, depth


if __name__ == '__main__':
    for name, model in hubbard_examples.items():
        for epsilon in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
            data = prepare_hamiltonian_refstates(**model)
            print(f"{name}, epsilon = {epsilon}, cheby filters")
            script_cost_analysis(filter_basis_type='cheby', epsilon_to_gap=epsilon, **data)
            print(f"{name}, epsilon = {epsilon}, trig filters")
            script_cost_analysis(filter_basis_type='trig', epsilon_to_gap=epsilon, **data)
