import os
import pickle
import re
from numbers import Number
from time import time
from typing import Tuple, List, Optional, Union, Dict

import numpy as np
from openfermion import MolecularData, FermionOperator, get_fermion_operator, QubitOperator

from ofex.hamiltonian import PolyacenePPP, FermionicHubbard
from ofex.hamiltonian.electronic_structures import CustomElectronicStructure
from ofex.linalg.sparse_tools import diagonalization, state_dot
from ofex.propagator import exact_rte, trotter_rte_by_si_ref
from ofex.state.chem_ref_state import hf_ground, cisd_ground
from ofex.state.state_tools import get_num_qubits
from ofex.transforms import fermion_to_qubit_state, fermion_to_qubit_operator
from ofex.utils.chem import molecule_example, run_driver

DIR_HAMILTONIAN = '../chemistry_data/hamiltonian/'
DIR_REFSTATE = "../chemistry_data/reference_state/"
DIR_SPECTRUM = "../chemistry_data/spectrum"
DIR_PROPAGATOR = "../chemistry_data/propagator"

DEFAULT_DRIVER = "pyscf"
CHEMICAL_ACCURACY = 0.00159362


def rec_path(dir_list: Tuple[str, ...], fname, tag):
    d: str = os.path.join(*dir_list)
    if not os.path.isdir(d):
        os.makedirs(d)
    if tag is None:
        return os.path.join(d, fname)
    else:
        fname = '.'.join(fname.split('.')[:-1]) + f"_{tag}." + fname.split('.')[-1]
        return os.path.join(d, fname)


def path_fermion_hamiltonian(mol_name, tag=None):
    return rec_path((DIR_HAMILTONIAN, mol_name), "fermion.pkl", tag)


def path_pauli_hamiltonian(mol_name, transform, tag=None):
    return rec_path((DIR_HAMILTONIAN, mol_name), f"{transform}.pkl", tag)


def path_cisd_state(mol_name, transform, tag=None):
    return rec_path((DIR_REFSTATE, mol_name), f"cisd_{transform}.pkl", tag)


def path_spectrum_analysis(mol_name, ref_name, transform, tag=None):
    return rec_path((DIR_SPECTRUM, mol_name), f"{ref_name}_{transform}.pkl", tag)


def path_propagator(mol_name, transform, time_step_str, n_trotter, tag=None):
    if n_trotter == 0 or n_trotter is None:
        return rec_path((DIR_PROPAGATOR, mol_name, transform), f"Δt={time_step_str}_exactrte.npy", tag)
    else:
        return rec_path((DIR_PROPAGATOR, mol_name, transform), f"Δt={time_step_str}_trotter={n_trotter}.npy", tag)


def spectrum_analysis(pham, ref, mol_name, transform,
                      ref_name="HF", tag=None, atol=1e-8,
                      save=True,
                      load=True,
                      print_progress=False,
                      mol_param=None) \
        -> Tuple[List[Tuple[int, complex, complex, np.ndarray]], np.ndarray]:
    if mol_param is not None:
        if not isinstance(mol_param, dict):
            tag = str(mol_param) if tag is None else tag+"_"+str(mol_param)
        else:
            dict_string = "".join([f"{k}={v}" for k, v in mol_param.items()])
            tag = dict_string if tag is None else tag+"_"+dict_string
    spectrum_path = path_spectrum_analysis(mol_name, ref_name, transform, tag)

    if os.path.isfile(spectrum_path) and load:
        with open(spectrum_path, 'rb') as f:
            loaded_data = pickle.load(f)
            if len(loaded_data) == 2 and isinstance(loaded_data[0], list) and isinstance(loaded_data[1], list):
                eigval_overlap_pair, true_eig_energies = loaded_data
                for idx, (idx_eig, eigval, overlap, eigvec) in enumerate(eigval_overlap_pair):
                    eigval_overlap_pair[idx] = (idx_eig, eigval, overlap, np.array(eigvec))
                if print_progress:
                    print(f" === Spectrum Analysis Loaded from {spectrum_path} ===")
                return eigval_overlap_pair, np.array(true_eig_energies)
            else:
                save=True
    if print_progress:
        print(" === Spectrum Analysis Started ... ", end="")
    t = time()
    n_qubits = get_num_qubits(ref)
    w, v = diagonalization(pham, n_qubits, sparse_eig=False)
    idx_order = np.argsort(w)
    true_eig_energies = w[idx_order]
    eigval_overlap_pair = list()
    eigval_overlap_pair_save = list()
    for idx_eig in idx_order:
        eigval, eigvec = w[idx_eig], v[:, idx_eig]
        overlap = state_dot(ref, eigvec)
        if abs(overlap) ** 2 > atol:
            eigval_overlap_pair.append((idx_eig, eigval, overlap, eigvec))
            eigval_overlap_pair_save.append((idx_eig, eigval, overlap, eigvec.tolist()))
    if save:
        with open(spectrum_path, 'wb') as f:
            pickle.dump((eigval_overlap_pair_save, list(true_eig_energies)), f)
    if print_progress:
        print(f" Done ({time() - t} sec) ===")

    return eigval_overlap_pair, true_eig_energies


def real_time_propagator_path(mol_name, transform, time_step, n_trotter,
                              tag=None, n_digits_t=6, mol_param=None):
    if mol_param is not None:
        if not isinstance(mol_param, dict):
            tag = str(mol_param) if tag is None else tag+"_"+str(mol_param)
        else:
            dict_string = "".join([f"{k}={v}" for k, v in mol_param.items()])
            tag = dict_string if tag is None else tag+"_"+dict_string
    if time_step <= 0:
        raise ValueError
    time_step = round(time_step, n_digits_t - int(np.floor(np.log10(abs(time_step)))) - 1)
    time_step_str = ("{" + f":.{n_digits_t - 1}e" + "}").format(time_step)
    prop_path = path_propagator(mol_name, transform, time_step_str, n_trotter, tag)
    return prop_path

def real_time_propagator(mol_name, transform, pham, time_step, n_qubits, n_trotter,
                         tag=None, n_digits_t=6,
                         save=True,
                         load=True,
                         print_progress=False,
                         mol_param=None):
    prop_path = real_time_propagator_path(mol_name, transform, time_step, n_trotter, tag, n_digits_t, mol_param)
    if os.path.isfile(prop_path) and load:
        prop = np.load(prop_path)
        if print_progress:
            print(f" === Propagator Loaded from {prop_path} ===")
    else:
        if print_progress:
            print(f" === Propagator Evaluation Started ... ", end="")
        t = time()
        if n_trotter == 0 or n_trotter is None:
            prop = exact_rte(pham, time_step, exact_sparse=True)
        else:
            prop = trotter_rte_by_si_ref(pham, time_step, n_qubits, n_trotter, exact_sparse=False)
        if save:
            np.save(prop_path, prop)
            if print_progress:
                print(f" Done({time() - t} sec) and saved to {prop_path} ===")
        elif print_progress:
            print(f" Done({time() - t} sec) ===")
    return prop


def prepare_hamiltonian_refstates(mol_name, transform,
                                  tag=None,
                                  save=True,
                                  load=True,
                                  print_progress=False,
                                  mol_param:Optional[Union[Number, List[Number], Dict[str, Number]]] = None,):
    if mol_param is not None:
        if not isinstance(mol_param, dict):
            tag = str(mol_param) if tag is None else tag+"_"+str(mol_param)
        else:
            dict_string = "".join([f"{k}={v}" for k, v in mol_param.items()])
            tag = dict_string if tag is None else tag+"_"+dict_string
    if print_progress:
        print(" === Start Preparing Hamiltonian ===")

    t = time()
    match_hubbard_1d = re.match(r'hubbard-(\d+)', mol_name)
    match_hubbard_2d = re.match(r'hubbard-\((\d+),(\d+)\)', mol_name)
    if match_hubbard_1d:
        n_hubbard = int(match_hubbard_1d.group(1))
        if mol_param is None:
            mol_param = dict()
        if 'n_electrons' not in mol_param:
            mol_param['n_electrons'] = n_hubbard
        if 'run_rhf' not in mol_param:
            mol_param['run_rhf'] = True
        if 'n' in mol_param:
            raise ValueError
        mol = FermionicHubbard(n_hubbard, **mol_param)
    elif match_hubbard_2d:
        n_x, n_y = int(match_hubbard_2d.group(1)), int(match_hubbard_2d.group(2))
        if mol_param is None:
            mol_param = dict()
        if 'n_electrons' not in mol_param:
            mol_param['n_electrons'] = n_x * n_y
        if 'run_rhf' not in mol_param:
            mol_param['run_rhf'] = True
        if 'n' in mol_param:
            raise ValueError
        mol = FermionicHubbard(n_x, n_y, **mol_param)

    elif mol_name not in ["Benzene_PPP"]:
        mol = molecule_example(mol_name, param=mol_param)
        mol.load()
        mol = run_driver(mol, run_cisd=True, run_fci=True, driver=DEFAULT_DRIVER)
    elif mol_name == "Benzene_PPP":
        mol = PolyacenePPP(n_ring=1)
    else:
        raise ValueError
    if print_progress:
        print(f" === MolecularData Done ({time() - t} sec) ===")

    # keyword arguments required in ofex.transforms.fermion_to_qubit_operator
    if transform == "bravyi_kitaev":
        f2q_kwargs = {"n_qubits": mol.n_qubits}
    elif transform == "symmetry_conserving_bravyi_kitaev":
        f2q_kwargs = {"active_fermions": mol.n_electrons,
                      "active_orbitals": mol.n_qubits}
    elif transform == "jordan_wigner":
        f2q_kwargs = dict()
    else:
        raise NotImplementedError

    # HF Reference State
    if isinstance(mol, MolecularData):
        hf_fermion = hf_ground(mol)
    elif isinstance(mol, CustomElectronicStructure):
        hf_fermion = mol.hf_state()
    else:
        raise TypeError
    hf = fermion_to_qubit_state(hf_fermion, transform, **f2q_kwargs)

    # PySCF results may be different for every run. Thus, the objects need to be pickled.
    # Prepare Fermion Hamiltonian
    fham_path = path_fermion_hamiltonian(mol_name, tag)
    if os.path.isfile(fham_path) and load:
        with open(fham_path, 'rb') as f:
            fham_t, f_const = pickle.load(f)
        fham = FermionOperator()
        fham.terms = fham_t
        if print_progress:
            print(f" === Fermion Hamiltonian Loaded from {fham_path} ===")
    else:
        fham = mol.get_molecular_hamiltonian()
        if not isinstance(fham, FermionOperator):
            fham = get_fermion_operator(fham)
        f_const = fham.constant
        fham = fham - f_const
        if save:
            with open(fham_path, 'wb') as f:
                pickle.dump((fham.terms, f_const), f)
            if print_progress:
                print(f" === Fermion Hamiltonian Evaluated and Saved to {fham_path} ===")
        elif print_progress:
            print(f" === Fermion Hamiltonian Evaluated ===")

    # Prepare Pauli Hamiltonian
    pham_path = path_pauli_hamiltonian(mol_name, transform, tag)
    if os.path.isfile(pham_path) and load:
        with open(pham_path, 'rb') as f:
            pham_t, p_const = pickle.load(f)
        pham = QubitOperator()
        pham.terms = pham_t
        if print_progress:
            print(f" === Qubit Hamiltonian Loaded from {pham_path} ===")
    else:
        pham = fermion_to_qubit_operator(fham, transform, **f2q_kwargs)
        p_const = pham.constant
        pham = pham - p_const
        if save:
            with open(pham_path, 'wb') as f:
                pickle.dump((pham.terms, p_const), f)
            if print_progress:
                print(f" === Qubit Hamiltonian Evaluated and Saved to {pham_path} ===")
        elif print_progress:
            print(f" === Qubit Hamiltonian Evaluated ===")

    if print_progress:
        print(f" === Hamiltonian Preparation Done ({time() - t}) ===")

    # Prepare CISD State
    cisd_state_path = path_cisd_state(mol_name, transform, tag)
    if os.path.isfile(cisd_state_path) and load:
        with open(cisd_state_path, 'rb') as f:
            cisd_state = pickle.load(f)
        if print_progress:
            print(f" === CISD state Loaded from {cisd_state_path} ===")
    elif isinstance(mol, MolecularData):
        cisd_state = cisd_ground(mol)
        cisd_state = fermion_to_qubit_state(cisd_state, transform, **f2q_kwargs)
        if save:
            with open(cisd_state_path, 'wb') as f:
                pickle.dump(cisd_state, f)
            if print_progress:
                print(f" === CISD state Evaluated and Saved to {cisd_state_path} ===")
        elif print_progress:
            print(f" === CISD state Evaluated ===")
    elif isinstance(mol, CustomElectronicStructure):
        cisd_state = None
    else:
        raise TypeError

    n_qubits = get_num_qubits(hf)

    return {"pham": pham, "fham": fham, "p_const": p_const, "f_const": f_const,
            "hf": hf, "hf_fermion": hf_fermion, "cisd_state": cisd_state,
            "mol_name": mol_name, "mol_param":mol_param, "mol": mol, "transform": transform, "f2q_kwargs": f2q_kwargs,
            "tag": tag, "n_qubits": n_qubits}
