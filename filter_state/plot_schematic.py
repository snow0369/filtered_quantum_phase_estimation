import os

import numpy as np
import sympy
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter

from chemistry_data.chem_tools import prepare_hamiltonian_refstates
from filter_state.utils_filter_general import apply_filter_to_state
from filter_state.utils_plot import plot_state_histogram, apply_mpl_style
from filter_state.utils_quantum import initialize_reference_state, collect_degeneracy
from ofex.classical_algorithms.filter_functions import gaussian_function_fourier
from ofex.classical_algorithms.funcapprox import plot_functions

if __name__ == "__main__":
    apply_mpl_style()

    # Preprocessing
    save_dir = "./figures/schematic_overview/"
    os.makedirs(save_dir, exist_ok=True)  # ensure directory exists

    mol_name = "H4"
    data = prepare_hamiltonian_refstates(mol_name=mol_name, transform="symmetry_conserving_bravyi_kitaev",
                                         print_progress=True)
    pham, norm, ref_state, eigval_overlap_pair = initialize_reference_state(**data, print_progress=True,
                                                                            initial_overlap='even')
    transform, n_qubits, = data['transform'], data['n_qubits']

    # Initial State
    fig_state, ax_state = plt.subplots(1, 1, figsize=(5, 5), dpi=300)
    degenerated = collect_degeneracy(eigval_overlap_pair)
    plot_state_histogram(fig_state, ax_state, degenerated, min_gap=0.05, normalized=True, annotate_eigidx=None)
    ax_state.set_xlim(-1.0, 1.0)
    ax_state.set_ylabel(r"$|\braket{E_i|\phi_0}|^2$")

    # --- Initial panel annotation (bigger) ---
    """
    ann0 = r"$\ket{\phi_0}=\Sigma_{i=0}^{d-1}\gamma_i\ket{E_i}$"
    ax_state.text(
        0.98, 0.98, ann0,
        transform=ax_state.transAxes,
        ha="right", va="top",
        fontsize=plt.rcParams.get("legend.fontsize", 10) * 1.3,
        zorder=100
    )
    """
    # Save fig to save_dir
    fig_state.savefig(os.path.join(save_dir, f"{mol_name}_initial_state_hist.png"),
                      dpi=300, bbox_inches="tight")

    # Filter
    target_width = 0.03
    target_center = -0.5
    period = 2.0

    n_fourier = 6
    filter_func, filter_coeff, _ = gaussian_function_fourier(
        n_fourier, target_width, target_center, period
    )
    fig_filter, ax_filter = plt.subplots(1, 1, figsize=(5, 5), dpi=300)
    plot_functions(
        {"filter": filter_func},
        sympy.symbols("x"),
        plot_options={"filter": {"linewidth": 2.0}},
        x_points=np.linspace(-1.0, 1.0, 1000),
        axes=ax_filter,
        plot=False
    )

    # --- Filter panel: remove legend & draw axes through origin with ticks/labels on them ---
    leg = ax_filter.get_legend()
    if leg is not None:
        leg.remove()

    # Ensure 0 is inside view
    xmin, xmax = ax_filter.get_xlim()
    ymin, ymax = ax_filter.get_ylim()
    ax_filter.set_xlim(min(xmin, 0), max(xmax, 0))
    ax_filter.set_ylim(min(ymin, 0), max(ymax, 0))

    # Move spines to the origin so tick labels sit at the crossing
    for side in ("right", "top"):
        ax_filter.spines[side].set_visible(False)
    ax_filter.spines["left"].set_position(("data", 0))
    ax_filter.spines["bottom"].set_position(("data", 0))
    ax_filter.spines["left"].set_zorder(10)
    ax_filter.spines["bottom"].set_zorder(10)

    # Ticks/labels snug to spines at origin
    ax_filter.xaxis.set_ticks_position("bottom")
    ax_filter.yaxis.set_ticks_position("left")
    ax_filter.tick_params(axis="both", which="both", direction="out", pad=2)

    # Clear normal labels (since we'll annotate instead)
    ax_filter.set_xlabel("")
    ax_filter.set_ylabel("")

    # Current axis limits (your plot is ~ x∈[-1,1], y∈[-0.2,1])
    xmin, xmax = ax_filter.get_xlim()
    ymin, ymax = ax_filter.get_ylim()

    # Annotate x-axis label at (xmax, 0)
    ax_filter.annotate(
        r"$E$", xy=(xmax-0.05, 0), xycoords="data",
        xytext=(4, 2), textcoords="offset points",  # small nudge outwards
        ha="left", va="bottom", clip_on=False, zorder=20,
        fontsize=plt.rcParams.get("legend.fontsize", 10)
    )

    # Annotate y-axis label at (0, ymax)
    ax_filter.annotate(
        r"$f(E)$", xy=(0, ymax-0.05), xycoords="data",
        xytext=(2, 4), textcoords="offset points",
        ha="left", va="bottom", clip_on=False, zorder=20,
        fontsize=plt.rcParams.get("legend.fontsize", 10)
    )

    ax_filter.xaxis.set_major_formatter(
        FuncFormatter(lambda x, pos: "" if np.isclose(x, 0.0, atol=1e-12) else f"{x:g}")
    )
    ax_filter.yaxis.set_major_formatter(
        FuncFormatter(lambda y, pos: "" if np.isclose(y, 0.0, atol=1e-12) else f"{y:g}")
    )

    # Save fig
    fig_filter.savefig(
        os.path.join(save_dir, f"{mol_name}_filter_function.png"),
        dpi=300, bbox_inches="tight"
    )

    # Apply filter
    filtered_eigval_overlap_pair, _ = apply_filter_to_state("trig", degenerated, filter_coeff,
                                                            time_step=2 * np.pi / period)

    # Plot filtered state
    fig_final, ax_final = plt.subplots(1, 1, figsize=(5, 5), dpi=300)
    ax_final.set_xlim(-1.0, 1.0)
    plot_state_histogram(fig_final, ax_final, filtered_eigval_overlap_pair,
                         min_gap=0.05, normalized=True, annotate_eigidx=None)
    ax_final.set_ylabel(r"$|\braket{E_i|\phi_f}|^2$")
    # ---- Annotation for the final panel (top-right) ----
    """
    annf = r"\begin{equation*}\ket{\phi_f}:=\frac{f(\hat{H})\ket{\phi_0}}{\|f(\hat{H})\ket{\phi_0}\|}\end{equation*}"
    ax_final.text(
        0.98, 0.98, annf,
        transform=ax_final.transAxes,
        ha="right", va="top",
        fontsize=plt.rcParams.get("legend.fontsize", 10) * 1.3,
        zorder=100
    )
    """
    # Save fig
    fig_final.savefig(os.path.join(save_dir, f"{mol_name}_filtered_state_hist.png"),
                      dpi=300, bbox_inches="tight")
