from copy import deepcopy
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from chemistry_data.chem_tools import prepare_hamiltonian_refstates
from chemistry_data.example_model import hubbard_examples
from filter_state.utils_plot import apply_mpl_style, FigureConfig, OriginalsAnalysis, load_plot_bundle, save_figs, \
    to_latex_sci

omit_expensive_part = True
omit_cost = 4.0

def plot_original_and_modified_analysis(
        *, fig_cfg: FigureConfig, originals: OriginalsAnalysis, modified_sparse: Dict[float, OriginalsAnalysis],
):
    apply_mpl_style()

    # Local settings
    figsize = (12.56, 6.75)

    """Create the two 'analysis' figures (original vs N, modified vs Λ)."""

    # --- Original analysis (3 rows) ---
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize, dpi=fig_cfg.dpi, sharex="col")
    (ax_overlap, ax_cost), (ax_succprob, ax_deltae) = axes
    linestyles = {"ori": '--', "0.1": '-', "1": '-', "10": '-'}
    colors = {"ori": "k", "0.1": "tab:orange", "1": "tab:green", "10": "tab:red"}
    alphas = {"ori": 1.0, "0.1": 0.6, "1": 1.0, "10": 0.6}
    linewidths = {"ori": 1.5, "0.1": 1.5, "1": 2.0, "10": 1.5}
    e0_marker = '|'
    e1_marker = '.'

    lmda_ftr = sorted(list(modified_sparse.keys()))
    lmda_result = [modified_sparse[k] for k in lmda_ftr]
    lmda_ftr = ['%g' % k for k in lmda_ftr]
    for fig_name, fig_data in zip(["ori"] + lmda_ftr, [originals] + lmda_result):
        # Plots
        ax_overlap.plot(fig_data.x - 1, fig_data.filter_overlap,
                        linestyle=linestyles[fig_name], color=colors[fig_name],
                        alpha=alphas[fig_name], linewidth=linewidths[fig_name],
                        label=r"$|\gamma_{f0}|^2$" if fig_name == "ori" else None)
        ax_deltae.plot(fig_data.x - 1, fig_data.delta_e1 / fig_data.spectral_gap_normalized,
                       linestyle=linestyles[fig_name], marker=e1_marker, color=colors[fig_name],
                       # color='tab:purple',
                       alpha=alphas[fig_name], linewidth=linewidths[fig_name],
                       label=r"$|E^{\mathrm{(QKSD)}}_1-E_1|/\Delta E_0$" if fig_name == "ori" else None)
        ax_deltae.plot(fig_data.x - 1, fig_data.delta_e0 / fig_data.spectral_gap_normalized,
                       linestyle=linestyles[fig_name], marker=e0_marker, color=colors[fig_name],
                       alpha=alphas[fig_name], linewidth=linewidths[fig_name],
                       label=r"$|E^{\mathrm{(QKSD)}}_0-E_0|/\Delta E_0$" if fig_name == "ori" else None)
        ax_succprob.plot(fig_data.x - 1, fig_data.succ_prob,
                         linestyle=linestyles[fig_name], color=colors[fig_name],
                         alpha=alphas[fig_name], linewidth=linewidths[fig_name],
                         label=r"$p_f$" if fig_name == "ori" else None)
        ax_cost.plot(fig_data.x - 1, fig_data.fqpe_cost,
                     linestyle=linestyles[fig_name], color=colors[fig_name],
                     alpha=alphas[fig_name], linewidth=linewidths[fig_name],
                     label=r"$C_{\mathrm{FQPE}}/C_{\mathrm{QPE}}$" if fig_name == "ori" else None)
        ax_cost.axhline(1.0, linestyle='-.', color='k')

        if omit_expensive_part and fig_name == "ori":
            import matplotlib as mpl
            large_costs = fig_data.fqpe_cost[fig_data.fqpe_cost > omit_cost].mean()
            large_costs = to_latex_sci(large_costs, precision=2, with_dollar=False, negative_space=True)
            first_omit = np.argwhere(fig_data.fqpe_cost > omit_cost)[0, 0] + 4
            ax_cost.text(x=first_omit, y=omit_cost*0.5, s=rf"$\gtrsim"+large_costs+"$", ha="left", va="center",
                         fontsize=mpl.rcParams["axes.labelsize"])

    # Legends
    labels = {"ori": r"$\Lambda\!=\!0$ (Standard Krylov)",
              "0.1": r"$\Lambda\!=\!0.1D_{\mathrm{sp}}/D_{\mathrm{QPE}}\!=\!0.1 N \epsilon$",
              "1": r"$\bm{\Lambda\!=\!D_{\mathrm{sp}}/D_{\mathrm{QPE}}\!=\!N \epsilon}$",
              "10": r"$\Lambda\!=\!10D_{\mathrm{sp}}/D_{\mathrm{QPE}}\!=\!10 N \epsilon$", }
    fake_lines = [
        Line2D([0], [0], color=colors[fig_name], linestyle=linestyles[fig_name], alpha=alphas[fig_name],
               linewidth=linewidths[fig_name], label=labels[fig_name])
        for fig_name in ["ori"] + lmda_ftr
    ]
    ax_overlap.legend(fake_lines, [h.get_label() for h in fake_lines], loc="lower right")

    fake_lines = [
        Line2D([0], [0], color="k", marker=e1_marker, label=r"$|\tilde{E}_1-E_1|/\Delta E_0$"),
        Line2D([0], [0], color="k", marker=e0_marker, label=r"$|\tilde{E}_0-E_0|/\Delta E_0$"),
    ]
    ax_deltae.legend(fake_lines, [h.get_label() for h in fake_lines], loc="lower left")

    # labels/scales
    ax_overlap.set_ylabel(r"$|\gamma_{f0}|^2$")
    ax_succprob.set_ylabel(r"$p_f$")
    ax_cost.set_ylabel(r"$C_{\mathrm{FQPE}}/C_{\mathrm{QPE}}$")
    ax_deltae.set_ylabel(r"KSD Energy Error")

    ax_succprob.set_xlabel(r"$N$")
    ax_deltae.set_xlabel(r"$N$")
    for a in (ax_overlap, ax_succprob, ax_cost, ax_deltae):
        a.set_yscale("log")

    ax_cost.set_ylim(ax_cost.get_ylim()[0], omit_cost)

    fig_names = ["new_analysis"]
    figures = {"new_analysis": fig}

    assert set(fig_names) == set(figures.keys())
    fig.tight_layout(w_pad=3.2, h_pad=0.6, pad=0.4)

    return fig_names, figures


def merge_images(paths, out_path="merged.png"):
    """
    Merge 6 images into a 2x3 grid.
    :param paths: list of 6 image file paths
    :param out_path: output file path
    """
    from PIL import Image
    if len(paths) != 6:
        raise ValueError("Exactly 6 image paths are required")

    # Open images
    imgs = [Image.open(p) for p in paths]

    # Optionally resize to the same size
    w, h = imgs[0].size
    imgs = [im.resize((w, h)) for im in imgs]

    # Create blank canvas
    merged = Image.new("RGB", (3 * w, 2 * h))

    # Paste images row by row
    for idx, im in enumerate(imgs):
        x = (idx % 3) * w
        y = (idx // 3) * h
        merged.paste(im, (x, y))

    merged.save(out_path)
    print(f"Saved merged image to {out_path}")


if __name__ == "__main__":
    import os, shutil
    default_figureconfig = FigureConfig()
    merged_fig_base_dir = "./figures/krylov_analysis"
    for name, model in hubbard_examples.items():
        data = prepare_hamiltonian_refstates(**model)
        for filter_basis_type in ["trig", "cheby"]:
            plt_name = f"{name}_{filter_basis_type}"
            print(plt_name)
            fig_base_dir = f"./figures/krylov_analysis/{plt_name}"

            plt_config = load_plot_bundle(filter_basis_type, **data)
            print(plt_config["extra_information"])
            fig_names, figures = plot_original_and_modified_analysis(fig_cfg=default_figureconfig,
                                                                     originals=plt_config["originals"],
                                                                     modified_sparse=plt_config["modified_sparse"])
            save_figs(fig_base_dir, merged_fig_base_dir, plt_name, fig_names, figures,
                      merged_shape=None)
            plt.close()
    img_path_list = [
        "./figures/krylov_analysis/hubbard-6_trig/0_hubbard-6_trig_new_analysis.png",
        "./figures/krylov_analysis/hubbard-(2,3)_trig/0_hubbard-(2,3)_trig_new_analysis.png",
        "./figures/krylov_analysis/hubbard-7_trig/0_hubbard-7_trig_new_analysis.png",
        "./figures/krylov_analysis/hubbard-6_cheby/0_hubbard-6_cheby_new_analysis.png",
        "./figures/krylov_analysis/hubbard-(2,3)_cheby/0_hubbard-(2,3)_cheby_new_analysis.png",
        "./figures/krylov_analysis/hubbard-7_cheby/0_hubbard-7_cheby_new_analysis.png",
    ]
    merge_images(img_path_list, out_path="./figures/krylov_analysis/merged.png")
    out_dir = "./figures/krylov_analysis/final_imgs/"
    os.makedirs(out_dir, exist_ok=True)
    for src in img_path_list:
        dst = os.path.join(out_dir, os.path.basename(src))
        shutil.copy(src, dst)
