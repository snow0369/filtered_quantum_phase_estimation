hubbard_examples = {
    "hubbard-6": {"mol_name": "hubbard-6",
                  "transform": "symmetry_conserving_bravyi_kitaev",
                  "mol_param": {"tunneling": 1.0 / 10, "coulomb": 1.0, 'run_rhf': False, "n_electrons": 4}
                  },
    "hubbard-(2,3)": {"mol_name": "hubbard-(2,3)",
                      "transform": "symmetry_conserving_bravyi_kitaev",
                      "mol_param": {"tunneling": 1.0 / 10, "coulomb": 1.0, 'run_rhf': False, "n_electrons": 3}
                      },
    "hubbard-7": {"mol_name": "hubbard-7",
                  "transform": "symmetry_conserving_bravyi_kitaev",
                  "mol_param": {"tunneling": 1.0 / 10, "coulomb": 1.0, 'run_rhf': False, "n_electrons": 4}
                  }
}
