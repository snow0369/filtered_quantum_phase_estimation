import os
from copy import deepcopy
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

from chemistry_data.chem_tools import prepare_hamiltonian_refstates
from chemistry_data.example_model import hubbard_examples
from filter_state.utils_filter_general import filter_func_eval
from filter_state.utils_plot import to_latex_sci, plot_state_histogram, apply_mpl_style, FigureConfig, InsetsConfig, \
    FiltersAndBlocks, YXLimits, Annotations, settings_to_configs, load_plot_bundle, save_figs, merge_with_arrow

log_label = True


def set_log_ticks(ax, step=2, numticks=None, threshold=None):
    ax.set_yscale('log', base=10)

    lo, hi = ax.get_ylim()
    lo = max(lo, np.finfo(float).tiny)

    # exponents that are guaranteed INSIDE the limits:
    kmin = int(np.ceil(np.log10(lo)))     # first k with 10^k >= lo
    kmax = int(np.floor(np.log10(hi)))    # last  k with 10^k <= hi

    if kmin > kmax:
        return  # nothing sensible to show

    # align to multiples of 'step'
    start = int(np.ceil(kmin / step)) * step
    end   = int(np.floor(kmax / step)) * step
    exps  = np.arange(start, end + 1, step, dtype=int)

    # optional thinning to at most numticks (preserve the one nearest 0)
    if numticks is not None and numticks > 0 and len(exps) > numticks:
        stride = int(np.ceil(len(exps) / numticks))
        anchor_idx = int(np.argmin(np.abs(exps)))  # closest to 0 (or 0 itself)
        offset = anchor_idx % stride
        exps = exps[offset::stride]

    ticks = 10.0 ** exps
    ax.yaxis.set_major_locator(mticker.FixedLocator(ticks))
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    # label as -log10(y); if threshold is set, hide labels for y > threshold
    if threshold is None:
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda y, pos: f"{-int(round(np.log10(y)))}")
        )
    else:
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(
                lambda y, pos, thr=threshold:
                    "" if (y > thr) else f"{-int(round(np.log10(y)))}"
            )
        )

def _inset_axes_kwargs(i: int, insets_cfg: InsetsConfig, fig_cfg: FigureConfig):
    return insets_cfg.per_plot_inset_axes_kwargs[i] or fig_cfg.default_inset_axes_kwargs


def plot_filter_functions_and_histograms(
        *,
        fig_cfg: FigureConfig,
        insets_cfg: InsetsConfig,
        filt: FiltersAndBlocks,
        limits: YXLimits,
        ann: Annotations,
):
    apply_mpl_style(larger_fonts=True)
    # Local settings
    figsize = (8, 4.5)
    figsize_noleg = (8, 4.5)
    ins_label_size = int(mpl.rcParams["xtick.labelsize"] * 0.7)
    ins_box_alpha = 0.6

    """Create figures for: func+hist (orig/mod/gauss) and applied histograms (orig/mod/gauss)."""

    # ----- inset x-range around E0
    ins_x_min = filt.gnd_energy + insets_cfg.ins_x_min_factor * filt.spectral_gap
    ins_x_max = filt.gnd_energy + insets_cfg.ins_x_max_factor * filt.spectral_gap
    ins_xrange = np.linspace(ins_x_min, ins_x_max, 1000)

    # ----- figures
    fig_ori_funchist, ax_histogram = plt.subplots(1, 1, figsize=figsize, dpi=fig_cfg.dpi)
    ax_krylov_filter = ax_histogram.twinx()

    fig_mod_funchist, ax_mod_histogram = plt.subplots(1, 1, figsize=figsize, dpi=fig_cfg.dpi)
    ax_mod_krylov_filter = ax_mod_histogram.twinx()

    fig_gaussian_funchist, ax_gaussian_histogram = plt.subplots(1, 1, figsize=figsize, dpi=fig_cfg.dpi)
    ax_gaussian_krylov_filter = ax_gaussian_histogram.twinx()

    fig_appl_ori_hist, ax_appl_ori_hist = plt.subplots(1, 1, figsize=figsize, dpi=fig_cfg.dpi)
    fig_appl_mod_hist, ax_appl_mod_hist = plt.subplots(1, 1, figsize=figsize, dpi=fig_cfg.dpi)
    fig_appl_gau_hist, ax_appl_gau_hist = plt.subplots(1, 1, figsize=figsize, dpi=fig_cfg.dpi)

    # ----- function curves + base histograms
    def plot_func_and_hist(ax_func, ax_hist, c_vec, label):
        line_func, = ax_func.plot(
            filt.x_range,
            np.abs(filter_func_eval(filt.basis_type, filt.x_range, c_vec, filt.time_step)) ** 2,
            label=label,
        )
        min_gap = fig_cfg.main_hist_bar_width * (filt.x_range.max() - filt.x_range.min()) / 2
        _, (kernel_plot, bar_gnd, bar_ext) = plot_state_histogram(ax_func.figure, ax_hist, filt.degenerate_block,
                                                                  min_gap=min_gap,
                                                                  normalized=True,
                                                                  plot_hist=True,
                                                                  plot_kernel=True,
                                                                  kernel_bw=min_gap * fig_cfg.kernel_bw_factor,
                                                                  label_gnd=r"$|\gamma_0|^2$",
                                                                  label_ext=r"$|\gamma_{i>0}|^2$",
                                                                  annotate_eigidx=None,
                                                                  alpha_hist=fig_cfg.histogram_alpha)
        return line_func, kernel_plot, bar_gnd, bar_ext

    line_func, kernel_plot, bar_gnd, bar_ext = plot_func_and_hist(ax_krylov_filter, ax_histogram, filt.c_final,
                                                                  r"$|f(E)|^2$")
    mod_line_func, mod_kernel_plot, mod_bar_gnd, mod_bar_ext = plot_func_and_hist(
        ax_mod_krylov_filter, ax_mod_histogram, filt.best_modified_c_vec, r"$|f_{\Lambda}(E)|^2$"
    )
    gau_line_func, gau_kernel_plot, gau_bar_gnd, gau_bar_ext = plot_func_and_hist(
        ax_gaussian_krylov_filter, ax_gaussian_histogram, filt.gaussian_c_vec, r"$|g(E)|^2$"
    )

    # ----- applied histograms (after filtering)
    def plot_applied_hist(flt_name, fig, ax, block, lg_gnd, lg_ext):
        min_gap = fig_cfg.main_hist_bar_width * (filt.x_range.max() - filt.x_range.min()) / 2
        _, (kernel_plot, bar_fg, bar_fe) = plot_state_histogram(
            fig, ax, block,
            min_gap=min_gap,
            normalized=True,
            plot_hist=True,
            plot_kernel=True,
            kernel_bw=min_gap * fig_cfg.applied_kernel_bw_factor[flt_name],
            label_gnd=lg_gnd, label_ext=lg_ext,
            annotate_eigidx=None,
            alpha_hist=fig_cfg.histogram_alpha)
        return bar_fg, bar_fe

    bar_filt_ori_gnd, bar_filt_ori_ext = plot_applied_hist(
        "ori", fig_appl_ori_hist, ax_appl_ori_hist, filt.degen_block_krylov,
        r"$|\gamma_{f0}|^2$", r"$|\gamma_{fi>0}|^2$"
    )
    bar_filt_mod_gnd, bar_filt_mod_ext = plot_applied_hist(
        "mod", fig_appl_mod_hist, ax_appl_mod_hist, filt.degen_block_modkry,
        r"$|\gamma_{f\Lambda;0}|^2$", r"$|\gamma_{f\Lambda;i>0}|^2$"
    )
    bar_filt_gau_gnd, bar_filt_gau_ext = plot_applied_hist(
        "gau", fig_appl_gau_hist, ax_appl_gau_hist, filt.degen_block_gauss,
        r"$|\gamma_{g0}|^2$", r"$|\gamma_{gi>0}|^2$"
    )

    # ----- insets (6 panels)
    func_axes = [ax_krylov_filter, ax_mod_krylov_filter, ax_gaussian_krylov_filter, None, None, None]
    hist_axes = [ax_histogram, ax_mod_histogram, ax_gaussian_histogram, ax_appl_ori_hist, ax_appl_mod_hist,
                 ax_appl_gau_hist]
    figs = [fig_ori_funchist, fig_mod_funchist, fig_gaussian_funchist, fig_appl_ori_hist, fig_appl_mod_hist,
            fig_appl_gau_hist]
    hist_blocks = [
        filt.degenerate_block, filt.degenerate_block, filt.degenerate_block,
        filt.degen_block_krylov, filt.degen_block_modkry, filt.degen_block_gauss
    ]

    def _inset_func_values(i: int) -> Optional[np.ndarray]:
        if i == 0:
            return np.abs(filter_func_eval(filt.basis_type, ins_xrange, filt.c_final, filt.time_step)) ** 2
        if i == 1:
            return np.abs(filter_func_eval(filt.basis_type, ins_xrange, filt.best_modified_c_vec, filt.time_step)) ** 2
        if i == 2:
            return np.abs(filter_func_eval(filt.basis_type, ins_xrange, filt.gaussian_c_vec, filt.time_step)) ** 2
        return None

    for i, (fig, axf, axh) in enumerate(zip(figs, func_axes, hist_axes)):
        if axf is not None:
            axins_hist = inset_axes(axf, bbox_transform=axf.transAxes, **_inset_axes_kwargs(i, insets_cfg, fig_cfg))
            axins_func = axins_hist.twinx()
        else:
            axins_hist = inset_axes(axh, bbox_transform=axh.transAxes, **_inset_axes_kwargs(i, insets_cfg, fig_cfg))
            axins_func = None

        ins_funcval = _inset_func_values(i)
        if ins_funcval is not None:
            axins_func.plot(ins_xrange, ins_funcval)

        plot_state_histogram(fig, axins_hist, hist_blocks[i],
                             min_gap=fig_cfg.inset_hist_bar_width * (ins_x_max - ins_x_min), normalized=True,
                             annotate_eigidx=[0, 1], annotate_offset=0.0,
                             alpha_hist=insets_cfg.histogram_alpha)

        # limits
        axins_hist.set_xlim(ins_x_min, ins_x_max)
        if axins_func is not None:
            axins_func.set_xlim(ins_x_min, ins_x_max)

        # y-lims
        # ins_ov2 = [ov2 for _, e, ov2 in hist_blocks[i] if ins_x_min <= e <= ins_x_max]
        # ins_ymin_hist, ins_ymax_hist = min(ins_ov2), max(ins_ov2)
        ymin, ymax = insets_cfg.hist_y_minmax[i]
        # ymin = max(ins_ymin_hist * insets_cfg.factor_ins_y_min, insets_cfg.hist_y_minmax[i][0])
        # ymax = min(ins_ymax_hist * insets_cfg.factor_ins_y_max, insets_cfg.hist_y_minmax[i][1])
        axins_hist.set_ylim(ymin, ymax)

        if axins_func is not None and ins_funcval is not None and insets_cfg.func_y_minmax[i] is not None:
            fymin = max(np.min(ins_funcval) * insets_cfg.factor_ins_y_min, insets_cfg.func_y_minmax[i][0])
            fymax = min(np.max(ins_funcval) * insets_cfg.factor_ins_y_max, insets_cfg.func_y_minmax[i][1])
            axins_func.set_ylim(fymin, fymax)

        # scales + cosmetics
        axins_hist.set_yscale("log")
        if axins_func is not None:
            axins_func.set_yscale("log")

        thr = insets_cfg.ins_skip_yticks[i]
        if log_label:
            set_log_ticks(axins_hist, step=2, threshold=thr)
            if axins_func is not None:
                set_log_ticks(axins_func, step=2, numticks=6, threshold=thr)

        axins_hist.set_ylabel("")
        axins_hist.set_xlabel("")
        axins_hist.tick_params(axis='x', labelleft=True, labelright=False, labelsize=ins_label_size)
        axins_hist.tick_params(axis='y', labelsize=ins_label_size)
        if axins_func is not None:
            axins_func.tick_params(axis='x', labelsize=ins_label_size)
            axins_func.tick_params(axis='y', labelsize=ins_label_size)

        # connector box
        #if axins_func is not None:
        #    _, con1, con2 = mark_inset(axf, axins_func, loc1=3, loc2=3, fc='none', ec='k', lw=1, linestyle='--')
        #else:
        rect, con1, con2 = mark_inset(axh, axins_hist, loc1=1, loc2=4, fc='none', ec='k', lw=1, linestyle='--')
        con1.loc1, con1.loc2 = 2, 1  # parent upper-right -> inset upper-left
        con2.loc1, con2.loc2 = 3, 4  # parent lower-right -> inset lower-left
        rect.set_alpha(ins_box_alpha)
        con1.set_alpha(ins_box_alpha)
        con2.set_alpha(ins_box_alpha)

    # ----- labels/scales/limits for big axes
    # y labels
    ax_krylov_filter.set_ylabel(r"$|f(E)|^2$")
    ax_histogram.set_ylabel(r"$|\braket{E_i|\phi_0}|^2$")
    ax_mod_krylov_filter.set_ylabel(r"$|f(E;\Lambda)|^2$")
    ax_mod_histogram.set_ylabel(r"$|\braket{E_i|\phi_0}|^2$")
    ax_gaussian_krylov_filter.set_ylabel(r"$|g(E)|^2$")
    ax_gaussian_histogram.set_ylabel(r"$|\braket{E_i|\phi_0}|^2$")
    ax_appl_ori_hist.set_ylabel(r"$|\braket{E_i|\phi_f}|^2$")
    ax_appl_mod_hist.set_ylabel(r"$|\braket{E_i|\phi_{f;\Lambda}}|^2$")
    ax_appl_gau_hist.set_ylabel(r"$|\braket{E_i|\phi_g}|^2$")

    # x labels
    for ax in (ax_histogram, ax_mod_histogram, ax_gaussian_histogram,
               ax_appl_ori_hist, ax_appl_mod_hist, ax_appl_gau_hist):
        ax.set_xlabel(r"$E$")

    # y-lims
    ax_krylov_filter.set_ylim(*limits.ori_func_y_minmax)
    ax_mod_krylov_filter.set_ylim(*limits.mod_func_y_minmax)
    ax_gaussian_krylov_filter.set_ylim(*limits.gau_func_y_minmax)
    ax_histogram.set_ylim(*limits.ori_hist_y_minmax)
    ax_mod_histogram.set_ylim(*limits.mod_hist_y_minmax)
    ax_gaussian_histogram.set_ylim(*limits.gau_hist_y_minmax)
    ax_appl_ori_hist.set_ylim(*limits.ori_hist_y_minmax)
    ax_appl_mod_hist.set_ylim(*limits.mod_hist_y_minmax)
    ax_appl_gau_hist.set_ylim(*limits.gau_hist_y_minmax)

    # scales
    for a in (ax_histogram, ax_mod_histogram, ax_gaussian_histogram,
              ax_appl_ori_hist, ax_appl_mod_hist, ax_appl_gau_hist):
        a.set_yscale("log", base=10)
        if log_label:
            if a.get_ylabel():
                a.set_ylabel(r"$-\log_{10}"+a.get_ylabel()[1:])
            set_log_ticks(a, step=2)

    for a in (ax_krylov_filter, ax_mod_krylov_filter, ax_gaussian_krylov_filter):
        a.set_yscale("log", base=10)
        if log_label:
            if a.get_ylabel():
                a.set_ylabel(r"$-\log_{10}"+a.get_ylabel()[1:])
            set_log_ticks(a, step=4)

    # x-lims shared
    xmin, xmax = float(filt.x_range.min()), float(filt.x_range.max())
    for a in (ax_krylov_filter, ax_mod_krylov_filter, ax_gaussian_krylov_filter,
              ax_histogram, ax_mod_histogram, ax_gaussian_histogram,
              ax_appl_ori_hist, ax_appl_mod_hist, ax_appl_gau_hist):
        a.set_xlim(xmin, xmax)

    # legends
    """
    ax1_plot_list = [line_func, bar_gnd, bar_ext, kernel_plot]
    ax2_plot_list = [mod_line_func, mod_bar_gnd, mod_bar_ext, mod_kernel_plot]
    ax3_plot_list = [gau_line_func, gau_bar_gnd, gau_bar_ext, gau_kernel_plot]
    ax_histogram.legend(ax1_plot_list, [l.get_label() for l in ax1_plot_list],
                        loc="upper right", bbox_to_anchor=(1.42, 1.0), borderaxespad=0)
    ax_mod_histogram.legend(ax2_plot_list, [l.get_label() for l in ax2_plot_list],
                            loc="upper right", bbox_to_anchor=(1.42, 1.0), borderaxespad=0)
    ax_gaussian_histogram.legend(ax3_plot_list, [l.get_label() for l in ax3_plot_list],
                                 loc="upper right", bbox_to_anchor=(1.42, 1.0), borderaxespad=0)
    """
    # z-order
    for axf, axh in ((ax_krylov_filter, ax_histogram),
                     (ax_mod_krylov_filter, ax_mod_histogram),
                     (ax_gaussian_krylov_filter, ax_gaussian_histogram)):
        axf.set_zorder(axh.zorder + 1)
        axf.patch.set_visible(False)

    # annotations (stats boxes)
    def annotate_str(pf, gamma, cost, func):
        return (r"$\begin{aligned}"
                + rf"p_{{{func}}}=&" + to_latex_sci(pf, precision=3, with_dollar=False) + r"\\"
                + rf"|\gamma_{{{func}0}}|^2=&" + to_latex_sci(gamma, precision=3, with_dollar=False) + r"\\"
                + r"C_{\mathrm{FQPE}}/C_{\mathrm{QPE}}=&" + to_latex_sci(cost, precision=3, with_dollar=False)
                + r"\end{aligned}$")

    bbox = dict(
        boxstyle="round,pad=0.3",  # rounded box
        facecolor="w",  # fill color (like legend)
        edgecolor="k",  # border color
        linewidth=0.8,
    )
    fig_appl_ori_hist.text(*ann.txt_xy, annotate_str(ann.pf_krylov, ann.gamma0_krylov_2, ann.cost_krylov, "f"),
                           ha="right", va="top", fontsize=mpl.rcParams["legend.fontsize"], bbox=bbox, zorder=100000)
    fig_appl_mod_hist.text(*ann.txt_xy, annotate_str(ann.pf_modkry, ann.gamma0_modkry_2, ann.cost_modkry, r"f;\Lambda"),
                           ha="right", va="top", fontsize=mpl.rcParams["legend.fontsize"], bbox=bbox, zorder=100000)
    fig_appl_gau_hist.text(*ann.txt_xy, annotate_str(ann.pf_gauss, ann.gamma0_gauss_2, ann.cost_gauss, "g"),
                           ha="right", va="top", fontsize=mpl.rcParams["legend.fontsize"], bbox=bbox, zorder=100000)

    """
    # special relabel of x tick "$E_0$"
    for fig in (fig_ori_funchist, fig_mod_funchist, fig_gaussian_funchist,
                fig_appl_ori_hist, fig_appl_mod_hist, fig_appl_gau_hist):
        fig.canvas.draw()
    target = r"$E_0$"
    for ax in (ax_histogram, ax_mod_histogram, ax_gaussian_histogram,
               ax_krylov_filter, ax_mod_krylov_filter, ax_gaussian_krylov_filter,
               ax_appl_ori_hist, ax_appl_mod_hist, ax_appl_gau_hist):
        for tick in ax.xaxis.get_major_ticks():
            if tick.label1.get_text() == target:
                tick.label1.set_visible(False)
                tick.label2.set_visible(False)
                ax.annotate(target, xy=(tick.get_loc(), 0),
                            xycoords=('data', 'axes fraction'),
                            xytext=(0, -16), textcoords='offset points',
                            ha='center', va='top', fontweight='bold')
    """
    # finalize
    for fig in (fig_ori_funchist, fig_mod_funchist, fig_gaussian_funchist,
                fig_appl_ori_hist, fig_appl_mod_hist, fig_appl_gau_hist):
        fig.tight_layout()

    fig_names = ["ori_funchist_NOLEG",
                 "ori_applstate_NOLEG",
                 "mod_funchist_NOLEG",
                 "mod_applstate_NOLEG",
                 "gau_funchist_NOLEG",
                 "gau_applstate_NOLEG", ]
    figures = {"ori_funchist_NOLEG": fig_ori_funchist,
               "mod_funchist_NOLEG": fig_mod_funchist,
               "gau_funchist_NOLEG": fig_gaussian_funchist,
               "ori_applstate_NOLEG": fig_appl_ori_hist,
               "mod_applstate_NOLEG": fig_appl_mod_hist,
               "gau_applstate_NOLEG": fig_appl_gau_hist}

    """
    fig_names = ["ori_funchist", "ori_funchist_NOLEG",
                 "ori_applstate", "ori_applstate_NOLEG",
                 "mod_funchist", "mod_funchist_NOLEG",
                 "mod_applstate", "mod_applstate_NOLEG",
                 "gau_funchist", "gau_funchist_NOLEG",
                 "gau_applstate", "gau_applstate_NOLEG", ]
    figures = {"ori_funchist": fig_ori_funchist,
               "mod_funchist": fig_mod_funchist,
               "gau_funchist": fig_gaussian_funchist,
               "ori_applstate": fig_appl_ori_hist,
               "mod_applstate": fig_appl_mod_hist,
               "gau_applstate": fig_appl_gau_hist}
    figure_noleg = dict()
    for fig_name, fig in figures.items():
        fig = deepcopy(fig)
        for ax in fig.axes:
            if ax.get_legend() is not None:
                ax.get_legend().remove()
        fig.set_size_inches(*figsize_noleg)
        fig.tight_layout()
        figure_noleg[fig_name + "_NOLEG"] = fig
    figures = figures | figure_noleg
    """
    assert set(fig_names) == set(figures.keys())

    return fig_names, figures


if __name__ == '__main__':
    run_test = False
    test_name, test_basis = "hubbard-7", "trig"

    for name, model in hubbard_examples.items():
        if run_test and name != test_name:
            continue
        default_figureconfig = FigureConfig()
        if name in ["hubbard-6", "hubbard-7"]:
            default_figureconfig.applied_kernel_bw_factor = {"ori": 40, "mod": 40, "gau": 20}
        data = prepare_hamiltonian_refstates(**model)
        for filter_basis_type in ["trig", "cheby"]:
            if run_test and filter_basis_type != test_basis:
                continue
            plt_name = f"{name}_{filter_basis_type}"
            print(plt_name)
            fig_base_dir = f"./figures/filter_func_and_state_{filter_basis_type}/{name}/"
            merged_fig_base_dir = f"./figures/filter_func_and_state_{filter_basis_type}/"

            plt_config = load_plot_bundle(filter_basis_type, **data)
            insets_cfg, limits = settings_to_configs(plt_config["settings"])
            fig_names, figures = plot_filter_functions_and_histograms(
                fig_cfg=default_figureconfig,
                insets_cfg=insets_cfg,
                filt=plt_config["filt"],
                limits=limits,
                ann=plt_config["ann"],
            )
            save_figs(fig_base_dir, merged_fig_base_dir, plt_name, fig_names, figures,
                      merged_shape=None)
            plt.close()

            for idx, filt_name in enumerate(["ori", "mod", "gau"]):
                left_img = os.path.join(fig_base_dir,
                                        f"{idx * 2}_{name}_{filter_basis_type}_{filt_name}_funchist_NOLEG.png")
                right_img = os.path.join(fig_base_dir,
                                         f"{idx * 2 + 1}_{name}_{filter_basis_type}_{filt_name}_applstate_NOLEG.png")
                out_path = os.path.join(merged_fig_base_dir,
                                        f"{name}_{filter_basis_type}_{idx }_{filt_name}_arrowmerged.png")
                merge_with_arrow(left_img, right_img, out_path)
