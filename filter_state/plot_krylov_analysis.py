from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

from chemistry_data.chem_tools import prepare_hamiltonian_refstates
from chemistry_data.example_model import hubbard_examples
from filter_state.utils_plot import apply_mpl_style, FigureConfig, OriginalsAnalysis, ModifiedAnalysis, \
    load_plot_bundle, save_figs, ModifiedBestAnalysis


def plot_original_and_modified_analysis(
    *, fig_cfg: FigureConfig, originals: OriginalsAnalysis, modified_best: ModifiedBestAnalysis,
        modified: ModifiedAnalysis
):
    apply_mpl_style()

    # Local settings
    figsize = (8, 6.75)
    # figsize_long = (16, 6.75)
    figsize_noleg = (6.28, 6.75)
    # figsize_long_noleg = (12.56, 6.75)
    tl_rect = [0, 0, 0.8, 1]
    leg1_pos = (1.40, 1.0)
    leg2_pos = (1.425, 1.0)

    """Create the two 'analysis' figures (original vs N, modified vs Λ)."""

    # --- Original analysis (3 rows) ---
    fig_ori, axes_ori = plt.subplots(nrows=3, ncols=1, figsize=figsize, dpi=fig_cfg.dpi, sharex='col')
    fig_mdb, axes_mdb = plt.subplots(nrows=3, ncols=1, figsize=figsize, dpi=fig_cfg.dpi, sharex='col')

    for fig_name, fig, axes, fig_data in zip(["ori", "mdb"], [fig_ori, fig_mdb],
                                             [axes_ori, axes_mdb], [originals, modified_best]):
        ax_overlap, ax_cost, ax_deltae = axes
        ax_succprob = ax_overlap.twinx()

        # Plots
        line_overlap, = ax_overlap.plot(fig_data.x - 1, fig_data.filter_overlap, '-', color='r',
                                        label=r"$|\gamma_{f0}|^2$")
        line_deltae0, = ax_deltae.plot(fig_data.x - 1, fig_data.delta_e0 / fig_data.spectral_gap_normalized,
                                       '-', color='tab:orange',
                                       label=r"$|E^{\mathrm{(QKSD)}}_0-E_0|/\Delta E_0$")
        line_deltae1, = ax_deltae.plot(fig_data.x - 1, fig_data.delta_e1 / fig_data.spectral_gap_normalized,
                                       '-', color='tab:purple',
                                       label=r"$|E^{\mathrm{(QKSD)}}_1-E_1|/\Delta E_0$")
        line_succprob, = ax_succprob.plot(fig_data.x - 1, fig_data.succ_prob, '-', color='b', label=r"$p_f$")
        line_cost, = ax_cost.plot(fig_data.x - 1, fig_data.fqpe_cost, '-', color='g',
                                  label=r"$C_{\mathrm{FQPE}}/C_{\mathrm{QPE}}$")
        ax_cost.axhline(1.0, linestyle='-.', color='k')

        # labels/scales
        ax_overlap.set_ylabel(r"$|\gamma_{f0}|^2$" if fig_name == "ori" else r"$|\gamma_{f0;\Lambda^{\star}}|^2$",
                              color="r")
        ax_succprob.set_ylabel(r"$p_f$" if fig_name == "ori" else r"$p_{f;\Lambda^{\star}}$", color="b")
        ax_cost.set_ylabel(r"$C_{\mathrm{FQPE}}/C_{\mathrm{QPE}}$")
        ax_deltae.set_ylabel(r"QKSD Energy Error")
        ax_deltae.set_xlabel(r"$N$")
        for a in (ax_overlap, ax_succprob, ax_cost, ax_deltae):
            a.set_yscale("log")

        # legend (pack on delta-e axis)
        ax0_plot_list = [line_overlap, line_succprob, line_deltae0, line_deltae1, line_cost]
        leg1 = ax_deltae.legend(ax0_plot_list, [l.get_label() for l in ax0_plot_list],
                                loc="upper right", bbox_to_anchor=leg1_pos,
                                bbox_transform=ax_deltae.transAxes, borderaxespad=0)
        leg1.set_in_layout(False)
        fig.tight_layout(rect=tl_rect)

    # --- Modified analysis (3 rows) ---
    fig_mod, axes = plt.subplots(nrows=3, ncols=1, figsize=figsize, dpi=fig_cfg.dpi, sharex='col')
    ax_m_overlap, ax_m_cost, ax_m_deltae = axes
    ax_m_succprob = ax_m_overlap.twinx()

    # Plots
    line_m_overlap, = ax_m_overlap.plot(modified.x_lambda, modified.overlap, '-', color='r',
                                        label=r"$|\gamma_{f0}|^2$")
    line_m_deltae0, = ax_m_deltae.plot(modified.x_lambda, modified.deltae0 / modified.spectral_gap_normalized,
                                       '-', color='tab:orange',
                                       label=r"$|E^{(\mathrm{QKSD})}_0-E_0|/\Delta E_0$")
    line_m_deltae1, = ax_m_deltae.plot(modified.x_lambda, modified.deltae1 / modified.spectral_gap_normalized,
                                       '-', color='tab:purple',
                                       label=r"$|E^{(\mathrm{QKSD})}_1-E_1|/\Delta E_0$")
    line_m_succprob, = ax_m_succprob.plot(modified.x_lambda, modified.succprob, '-', color='b', label=r"$p_{f}$")
    line_m_cost, = ax_m_cost.plot(modified.x_lambda, modified.cost, '-', color='g',
                                  label=r"$C_{\mathrm{FQPE}}/C_{\mathrm{QPE}}$")
    ax_m_deltae.axvline(modified.x_lambda[modified.min_cost_idx], linestyle='-.', color='k')
    ax_m_cost.axvline(modified.x_lambda[modified.min_cost_idx], linestyle='-.', color='k')
    if np.max(modified.cost) > 0.1:
        ax_m_cost.axhline(1.0, linestyle='-.', color='k')

    # labels/scales
    ax_m_overlap.set_ylabel(r"$|\gamma_{f0;\Lambda}|^2$", color="r")
    ax_m_succprob.set_ylabel(r"$p_{f;\Lambda}$", color="b")
    ax_m_cost.set_ylabel(r"$C_{\mathrm{FQPE}}/C_{\mathrm{QPE}}$")
    ax_m_deltae.set_ylabel(r"QKSD Energy Error")
    ax_m_deltae.set_xlabel(rf"$\Lambda ~ (N={max(originals.x)-1})$")
    for a in (ax_m_overlap, ax_m_succprob, ax_m_cost, ax_m_deltae):
        a.set_yscale("log")
    for a in (ax_m_overlap, ax_m_succprob, ax_m_cost, ax_m_deltae):
        a.set_xscale("log")

    # legend (pack on delta-e axis)
    ax2_plot_list = [line_m_overlap, line_m_succprob, line_m_deltae0, line_m_deltae1, line_m_cost]
    leg2 = ax_m_deltae.legend(ax2_plot_list, [l.get_label() for l in ax2_plot_list],
                              loc="upper right", bbox_to_anchor=leg2_pos,
                              bbox_transform=ax_m_deltae.transAxes, borderaxespad=0)
    leg2.set_in_layout(False)
    fig_mod.tight_layout(rect=tl_rect)

    fig_names = ["ori_analysis", "ori_analysis_NOLEG",
                 "modbest_analysis", "modbest_analysis_NOLEG",
                 "mod_analysis", "mod_analysis_NOLEG"]
    figures = {"ori_analysis": fig_ori, "modbest_analysis": fig_mdb, "mod_analysis": fig_mod,
               "ori_analysis_NOLEG": deepcopy(fig_ori),
               "modbest_analysis_NOLEG": deepcopy(fig_mdb),
               "mod_analysis_NOLEG": deepcopy(fig_mod)}

    for fig_name in fig_names:
        if "NOLEG" not in fig_name:
            continue
        fig = figures[fig_name]
        for ax in fig.axes:
            if ax.get_legend() is not None:
                ax.get_legend().remove()
        #if fig_name == "mod_analysis_NOLEG":
        #    fig.set_size_inches(*figsize_long_noleg)
        #else:
        fig.set_size_inches(*figsize_noleg)
        fig.tight_layout()

    assert set(fig_names) == set(figures.keys())

    return fig_names, figures


if __name__ == "__main__":
    default_figureconfig = FigureConfig()
    merged_fig_base_dir = "./figures/krylov_analysis"
    for name, model in hubbard_examples.items():
        data = prepare_hamiltonian_refstates(**model)
        for filter_basis_type in ["trig", "cheby"]:
            plt_name = f"{name}_{filter_basis_type}"
            print(plt_name)
            fig_base_dir = f"./figures/krylov_analysis/{plt_name}"

            plt_config = load_plot_bundle(filter_basis_type, **data)
            fig_names, figures = plot_original_and_modified_analysis(fig_cfg=default_figureconfig,
                                                                     originals=plt_config["originals"],
                                                                     modified_best=plt_config["modified_best"],
                                                                     modified=plt_config["modified"], )
            save_figs(fig_base_dir, merged_fig_base_dir, plt_name, fig_names, figures,
                      merged_shape=(3, 1))
            plt.close()
