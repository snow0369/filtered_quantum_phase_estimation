# Filtered Quantum Phase Estimation — Code & Data

Raw data, reproduction scripts, and figures for the manuscript (arXiv:2510.04294).  
Main code and data live in `filter_state/`.

## Repository layout
```
filtered_quantum_phase_estimation/
├─ filter_state/
│  ├─ experiment_*.py        # Run experiments; generates data into filter_state/data
│  ├─ plot*.py               # Generate all paper figures from data
│  ├─ data/                  # (created by experiments) raw & processed outputs
│  ├─ data_produced/         # Pre-generated results (use to skip long runs)
│  └─ figures/               # Resulting figures saved here by plot scripts
├─ chemistry_data/           # Molecule/problem inputs used by experiments
├─ requirements.txt          # List of Python dependencies
├─ LICENSE                   # MIT
└─ README.md
```

## Requirements

- Python 3.x  
- [OFEX (OpenFermion EXpansion)](https://github.com/snow0369/ofex) — **Note:** this package is *not* available on PyPI and must be installed directly from GitHub (see below).  
- Typical scientific Python stack (`numpy`, `scipy`, `matplotlib`, etc.), listed in `requirements.txt`.  

### Quick OFEX install (from source)
```bash
git clone https://github.com/snow0369/ofex.git
cd ofex
pip install -r requirements.txt
pip install -e .
```
(If you use `psi4`, follow the `ofex` README instructions and set `PSI4PATH` accordingly.)

### Install dependencies for this repository
```bash
# from the top-level directory
pip install -r requirements.txt
```

## Quick start

1. **Clone this repo**
```bash
git clone https://github.com/snow0369/filtered_quantum_phase_estimation.git
cd filtered_quantum_phase_estimation
```

2. **(Optional) Create a clean environment**
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Install OFEX** (must be done separately)
```bash
git clone https://github.com/snow0369/ofex.git
cd ofex
pip install -r requirements.txt
pip install -e .
```

5. **Run experiments** (this will populate `filter_state/data/`)
```bash
cd ../filtered_quantum_phase_estimation/filter_state
python experiment_<name>.py
# e.g.
# python experiment_gaussian_filter.py
```

> ⏱️ **Skip long runs:** copy the previously generated results from `data_produced/` to `data/`:
```bash
# from inside filter_state/
cp -r data_produced/* data/
```
(Then you can go straight to plotting.)

6. **Generate paper figures**
```bash
# still in filter_state/
python plot_<name>.py
# Output figures appear in filter_state/figures/
```

## Reproducibility notes

- All figures in the paper can be recreated by running the `plot*.py` scripts; they read from `filter_state/data/`.  
- If you haven’t run the experiments, use the ready-made outputs in `filter_state/data_produced/` by copying them into `data/`.  
- Problem instances and chemistry inputs (when applicable) are under `chemistry_data/`.

## Data availability

Generated data (and pre-generated outputs) are included in this repository under `filter_state/data` and `filter_state/data_produced`.  
Figures created by `plot*.py` are saved into `filter_state/figures/`.

## Citation

If you use this code or data, please cite the manuscript (arXiv:2510.04294) and this repository.

```
@misc{lee2025filteredqpe,
  title   = {Filtered Quantum Phase Estimation},
  author  = {Gwonhak Lee and collaborators},
  year    = {2025},
  eprint  = {2510.04294},
  archivePrefix = {arXiv}
}
```

## License
MIT. See `LICENSE`.
