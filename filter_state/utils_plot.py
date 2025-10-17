import os
import pickle
from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Optional, List, Tuple, Union, Mapping, Any, Dict, Sequence

import matplotlib as mpl
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from scipy.stats import gaussian_kde
from PIL import Image, ImageDraw


def plot_filter_qpe_cost(ax,
                         data: np.ndarray,
                         filter_center_list: np.ndarray,
                         filter_width_list: np.ndarray,
                         ceil_log10: int,
                         lowerbound_log10: int,
                         gnd_energy,
                         spectral_gap,
                         epsilon
                         ):
    x_label = [f"{c:.2e}" for c in filter_center_list - gnd_energy]
    y_label = [f"{w:.2e}" for w in filter_width_list]

    ceil = 10 ** ceil_log10
    data = np.array(data)
    data[data > ceil] = ceil
    # min_idx_c, min_idx_w = np.unravel_index(np.argmin(data), data.shape)
    lowerbound = 10 ** lowerbound_log10
    if np.min(data) < lowerbound:
        raise ValueError(np.min(data))

    # Set colormap and plot
    middle = -np.log(lowerbound) / (np.log(ceil) - np.log(lowerbound))
    colors = [(0, "green"), (middle ** 2, "blue"), (middle, "white"), (1, "red")]
    custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)

    heatmap = sns.heatmap(
        data.T,
        annot=False,  # Annotate with values
        cmap=custom_cmap,
        norm=LogNorm(vmin=lowerbound, vmax=ceil),
        xticklabels=x_label,
        yticklabels=y_label,
        cbar=True,  # Show color bar
        ax=ax,
    )

    # Plot further information
    len_x, len_y = len(filter_center_list), len(filter_width_list)
    normalizer_center = lambda val: len_x * (val - np.min(filter_center_list)) / (
            np.max(filter_center_list) - np.min(filter_center_list))
    normalizer_width = lambda val: len_y * (val - np.min(filter_width_list)) / (
            np.max(filter_width_list) - np.min(filter_width_list))

    min_center, max_center = np.min(filter_center_list), np.max(filter_center_list)
    min_center_tick = int((min_center - gnd_energy) / spectral_gap)
    max_center_tick = int((max_center - gnd_energy) / spectral_gap)
    center_tick_idxs = np.concatenate((np.arange(min_center_tick, 0), np.arange(1, max_center_tick + 1)))
    center_tick_idxs_with_gnd = np.arange(min_center_tick, max_center_tick + 1)
    center_ticks = center_tick_idxs * spectral_gap + gnd_energy
    center_ticks_with_gnd = center_tick_idxs_with_gnd * spectral_gap + gnd_energy

    ax.hlines(normalizer_center(center_ticks), 0, len_x, colors="black", linestyles="dotted")
    ax.hlines(normalizer_center(gnd_energy), 0, len_x, colors="black", linestyles="dashed")

    min_width_tick, max_width_tick = (int(np.min(filter_width_list) / spectral_gap),
                                      int(np.max(filter_width_list) / spectral_gap))
    min_width_tick = max(min_width_tick, 1)
    width_tick_idxs = np.arange(min_width_tick, max_width_tick + 1)
    width_ticks = width_tick_idxs * spectral_gap

    ax.vlines(normalizer_width(width_ticks), 0, len_y, colors="black", linestyles="dotted")

    min_idx_c, min_idx_w = np.unravel_index(np.argmin(data), data.shape)
    ax.scatter(min_idx_w, min_idx_c, color="k", marker="x",
               label=r"$\mathrm{min}(C_\mathrm{FQPE}/C_{\mathrm{QPE}})=%.3f$" % data[min_idx_c, min_idx_w])

    # Set Color bar
    cbar = heatmap.collections[0].colorbar
    tick_positions = list(cbar.get_ticks())
    tick_labels = [label.get_text() for label in cbar.ax.get_yticklabels()]

    for i in range(len(tick_positions)):
        if np.isclose(tick_positions[i], ceil):
            tick_positions = tick_positions[:i] + [ceil] + tick_positions[i + 1:]
            tick_labels = tick_labels[:i] + [r"$>\mathdefault{10^{%d}}$" % ceil_log10] + tick_labels[i + 1:]
            break
        elif tick_positions[i] > ceil:
            tick_positions = tick_positions[:i] + [ceil] + tick_positions[i:]
            tick_labels = tick_labels[:i] + [r"$>\mathdefault{10^{%d}}$" % ceil_log10] + tick_labels[i:]
            break

    for i in range(len(tick_positions)):
        if np.isclose(tick_positions[i], lowerbound):
            tick_positions = tick_positions[:i] + [lowerbound] + tick_positions[i + 1:]
            tick_labels = tick_labels[:i] + [r"$\mathdefault{10^{%d}}$" % lowerbound_log10] + tick_labels[i + 1:]
            break
        elif tick_positions[i] > lowerbound:
            tick_positions = tick_positions[:i] + [lowerbound] + tick_positions[i:]
            tick_labels = tick_labels[:i] + [r"$\mathdefault{10^{%d}}$" % lowerbound_log10] + tick_labels[i:]
            break

    cbar.set_ticks(tick_positions[1:-1])
    cbar.set_ticklabels(tick_labels[1:-1])  # Format ticks

    # Adjust axis label
    # x_ticks, y_ticks = ax.get_xticks(), ax.get_yticks()
    # x_ticklabels, y_ticklabels = ax.get_xticklabels(), ax.get_yticklabels()
    # n_ticks = 11
    # ax.set_xticks(x_ticks[::len(x_ticks) // n_ticks])
    # ax.set_xticklabels(x_ticklabels[::len(x_ticklabels) // n_ticks],  rotation=45, ha='right', va='top')
    # ax.set_yticks(y_ticks[::len(y_ticks) // n_ticks])
    # ax.set_yticklabels(y_ticklabels[::len(y_ticklabels) // n_ticks])
    ax.set_xticks(normalizer_center(center_ticks_with_gnd))
    ax.set_xticklabels([r"$E_\mathrm{GND} + %d\Delta$" % i if i > 0 else
                        r"$E_\mathrm{GND} - %d\Delta$" % -i if i < 0 else r"$E_\mathrm{GND}$"
                        for i in center_tick_idxs_with_gnd])
    ax.set_yticks(normalizer_width(width_ticks))
    ax.set_yticklabels([fr"${i}\Delta$" for i in width_tick_idxs], rotation=0)

    # Axis label
    ax.set_xlabel(r"$Center$")
    ax.set_ylabel(r"$Width$")
    ax.invert_yaxis()
    ax.legend()

    return ax


def plot_properties(filter_center, filter_width, filter_property_list,
                    c_def, w_def,
                    gnd_energy, epsilon, axes):
    n_fliter_centers = len(filter_center)
    n_filter_widths = len(filter_width)
    filter_type = "gaussian_function_fourier"

    # Fix width
    succ_prob_list_fix_width = np.array([filter_property_list[(filter_type, c, w_def)][0] for c in filter_center])
    overlap_squared_list_fix_width = np.array([filter_property_list[(filter_type, c, w_def)][2] for c in filter_center])
    succ_prob_list_fix_center = np.array([filter_property_list[(filter_type, c_def, w)][0] for w in filter_width])
    overlap_squared_list_fix_center = np.array([filter_property_list[(filter_type, c_def, w)][2] for w in filter_width])
    depth_list_fix_width = np.array([filter_property_list[(filter_type, c, w_def)][3] for c in filter_center])
    depth_list_fix_center = np.array([filter_property_list[(filter_type, c_def, w)][3] for w in filter_width])

    filter_center_shifted = filter_center - gnd_energy

    for idx_axis, (x_axis, succ_prob_list, overlap_squared_list, depth_list) \
            in enumerate(zip([filter_center_shifted, filter_width],
                             [succ_prob_list_fix_width, succ_prob_list_fix_center],
                             [overlap_squared_list_fix_width, overlap_squared_list_fix_center],
                             [depth_list_fix_width, depth_list_fix_center])):
        axes[idx_axis, 0].plot(x_axis, succ_prob_list, label="prob")
        axes[idx_axis, 0].plot(x_axis, overlap_squared_list, label="overlap2")

        mf_list = 1 / (succ_prob_list * overlap_squared_list)
        qpe_depth = 1 / (epsilon * overlap_squared_list)
        overall_cost = mf_list * depth_list + qpe_depth

        axes[idx_axis, 1].plot(x_axis, mf_list, label="mf")
        axes[idx_axis, 1].plot(x_axis, qpe_depth, label="QPE depth")
        axes[idx_axis, 1].plot(x_axis, overall_cost, label="cost")
        axes[idx_axis, 1].set_yscale('log')

    axes[0, 0].set_xlabel("center")
    axes[0, 1].set_xlabel("center")

    axes[1, 0].set_xlabel("width")
    axes[1, 1].set_xlabel("width")

    # Add legends
    for ax in axes.flatten():
        ax.legend()

    return axes



def plot_state_histogram(fig, ax, eigval_overlap2_pair: List[Tuple[List[int], complex, complex]],
                         min_gap: Optional[float] = None, normalized=False,
                         plot_hist=True, alpha_hist=1.0,
                         plot_kernel=False, kernel_xgrid=None, kernel_bw="silverman", alpha_kernel=1.0,
                         label_gnd=None, label_ext=None, annotate_eigidx=(0,),
                         annotate_offset=0.03, **kwargs):
    if annotate_eigidx is None:
        annotate_eigidx = tuple()
    eigvals = np.array([eigval for _, eigval, _ in eigval_overlap2_pair])
    min_eigval, max_eigval = min(eigvals), max(eigvals)
    overlap_sqs = np.array([overlap_sq for _, _, overlap_sq in eigval_overlap2_pair])
    gap = np.min(np.diff(eigvals))
    if min_gap is not None:
        gap = max([gap, min_gap])

    if normalized:
        margin_low, margin_high = -1.0, 1.0
    else:
        margin_low, margin_high = min_eigval * 1.1 if min_eigval < 0 else min_eigval * 0.9, \
            max_eigval * 1.1 if max_eigval > 0 else max_eigval * 0.9

    if kernel_xgrid is None:
        kernel_xgrid = np.linspace(margin_low, margin_high, 1001)

    plt_object = list()
    if plot_hist:
        plt_gnd = ax.bar(eigvals[0], overlap_sqs[0], color="blue", width=gap, align="center", label=label_gnd,
                         alpha=alpha_hist, **kwargs)
        plt_ext = ax.bar(eigvals[1:], overlap_sqs[1:], color="red", width=gap, align="center", label=label_ext,
                         alpha=alpha_hist, **kwargs)
        plt_object += [plt_gnd, plt_ext]
    if plot_kernel:
        kde = gaussian_kde(eigvals, weights=overlap_sqs, bw_method=kernel_bw)
        kernel_ygrid = kde(kernel_xgrid)
        kernel_ygrid = kernel_ygrid/np.max(kernel_ygrid)
        kernel_plot, =ax.plot(kernel_xgrid, kernel_ygrid,
                              color="red", label=r"$\mathrm{Ker}(\{\gamma_i, E_i\})$",
                              alpha=alpha_kernel, **kwargs)
        plt_object.append(kernel_plot)

    ax.set_xlabel(r"$E_i$")
    ax.set_ylabel(r"$|\langle E_i | \psi \rangle|^2$")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(margin_low, margin_high)

    old_pos = ax.get_xticks()
    old_label = [t.get_text() for t in ax.get_xticklabels()]
    old_dict = dict(zip(old_pos, old_label))  # {tick → label}

    if annotate_eigidx == "all":
        annotate_eigidx = list(range(len(eigvals)))
    annotate_eigidx = list(annotate_eigidx)
    eig_pos = np.atleast_1d(eigvals[annotate_eigidx])
    eig_label = {v: rf"$E_{i}$" for i, v in zip(annotate_eigidx, eig_pos)}

    ticks = np.unique(np.concatenate((old_pos, eig_pos)))  # sorted union
    ax.set_xticks(ticks)

    labels = [eig_label.get(t,  # latex "E_i" if this is an eigen-value
                            old_dict.get(t, f"{t:g}"))  # else old numeric label
              for t in ticks]

    ax.set_xticklabels(labels, rotation=0, ha="center")

    if annotate_offset > 0.0:
        # draw once so that tick objects exist
        fig.canvas.draw()

        # find the Text object for our new tick and move it down a bit
        for txt in ax.get_xticklabels():
            if txt.get_text() in eig_label.values():
                x, y = txt.get_position()  # (data_coord, axis_fraction)
                txt.set_position((x, y - annotate_offset))  # 0.03 axis units lower
                txt.set_fontweight("bold")  # optional highlight

    return ax, plt_object


def plot_worst_cost(ax,
                    data: np.ndarray,
                    filter_center_list: np.ndarray,
                    filter_width_list: np.ndarray,
                    ceil_log10: int,
                    lowerbound_log10: int,
                    gnd_energy,
                    spectral_gap,
                    epsilon,
                    n_points=20):
    ceil = 10 ** ceil_log10
    data = np.array(data)
    # data[data > ceil] = ceil
    # min_idx_c, min_idx_w = np.unravel_index(np.argmin(data), data.shape)
    lowerbound = 10 ** lowerbound_log10
    if np.min(data) < lowerbound:
        raise ValueError


    prior_energy_0, prior_energy_1 = np.zeros_like(data), np.zeros_like(data)
    prior_energy_0[:] = filter_center_list[:, None]
    prior_energy_1[:] = filter_center_list[:, None]
    prior_energy_1[:] += filter_width_list

    first_energy = gnd_energy + spectral_gap
    prior_energy_err = np.linspace(np.min(np.abs(filter_center_list - gnd_energy)),
                                   min(np.max(np.abs(filter_center_list - gnd_energy)), spectral_gap * 1.01), n_points)
    worst_points = list()
    first_omit = False
    for idx_err, err in enumerate(prior_energy_err):
        energy_indices_0, energy_indices_1 = np.where(np.logical_and(np.abs(prior_energy_0 - gnd_energy) <= err,
                                                                     np.abs(prior_energy_1 - first_energy) <= err))
        cost_data = list()
        for idx0, idx1 in product(energy_indices_0, energy_indices_1):
            cost_data.append(data[idx0, idx1])
        if len(cost_data) == 0 and idx_err == 0:
            first_omit = True
            continue
        elif len(cost_data) == 0:
            worst_points.append(worst_points[-1])
        else:
            worst_points.append(np.max(cost_data))

    if first_omit:
        worst_points = [worst_points[0]] + worst_points

    ax.plot(prior_energy_err, worst_points)
    ax.hlines(1.0, np.min(prior_energy_err), np.max(prior_energy_err), colors="black", linestyles="dashed")
    ax.vlines(0.5 * spectral_gap, np.min(worst_points) / 10, np.max(worst_points) * 10, colors="black",
              linestyles="dashed")

    n_half_delta = int(2 * max(prior_energy_err) / spectral_gap)

    ax.set_xticks([d * spectral_gap / 2 for d in range(n_half_delta + 1)])
    ax.set_xticklabels([fr"${i / 2:.1f}\Delta$" for i in range(n_half_delta + 1)], rotation=0)

    ax.set_xlim(0, spectral_gap)
    ax.set_ylim(np.min(worst_points) / 2, np.max(worst_points) * 2)

    ax.set_yscale('log')

    ax.set_xlabel(r"$\epsilon'$")
    ax.set_ylabel(r"Worst $C_{\mathrm{FQPE}}/C_{\mathrm{QPE}}$")


def to_latex_sci(x, precision=1, with_dollar=True, negative_space=False):
    if precision == 0 :
        mant, exp = f"{x:.1e}".split("e")
        return rf"$10^{{{int(exp)}}}$" if with_dollar else rf"10^{{{int(mant)}}}"
    else:
        mant, exp = f"{x:.{precision}e}".split("e")
        if negative_space:
            return rf"${mant}\!\times\!10^{{{int(exp)}}}$" if with_dollar else rf"{mant}\!\times\! 10^{{{int(exp)}}}"
        else:
            return rf"${mant}\times 10^{{{int(exp)}}}$" if with_dollar else rf"{mant}\times 10^{{{int(exp)}}}"


def apply_mpl_style(larger_fonts=False):
    mpl.rcParams["text.usetex"] = True  # enable full LaTeX
    mpl.rcParams['axes.linewidth'] = 1.2
    # mpl.rcParams["font.size"] = 11  # global default for *all* text
    if larger_fonts:
        mpl.rcParams["axes.labelsize"] = 23  # x– and y–axis labels only
        mpl.rcParams["axes.titlesize"] = 23  # subplot titles
        mpl.rcParams["xtick.labelsize"] = 23
        mpl.rcParams["ytick.labelsize"] = 23
        mpl.rcParams["legend.fontsize"] = 21
    else:
        mpl.rcParams["axes.labelsize"] = 18  # x– and y–axis labels only
        mpl.rcParams["axes.titlesize"] = 21  # subplot titles
        mpl.rcParams["xtick.labelsize"] = 16
        mpl.rcParams["ytick.labelsize"] = 16
        mpl.rcParams["legend.fontsize"] = 16
    mpl.rcParams.update({
        "text.usetex": True,  # let TeX typeset strings
        "font.family": "serif",
        "text.latex.preamble": r"""
            \usepackage{lmodern}             % or newtxtext,newtxmath …
            \usepackage{amsmath,amssymb}
            \usepackage{braket}
            \usepackage{bm}
        """,
    })


def merge_with_arrow(
    left_path: str,
    right_path: str,
    out_path: str,
    *,
    match_height: str = "max",     # "max" | "min" | "pad"
    gap: int = 140,                # width of the arrow area (px)
    pad: int = -35,                # outer padding around everything (px)
    arrow_width: int = 40,         # shaft thickness (px)
    arrow_head_len: int = 70,      # head length (px)
    arrow_head_wid: int = 90,      # head width (px)
    arrow_color=(31, 119, 180, 255),    # RGBA
    bg=(255, 255, 255, 255),       # RGBA background (use (0,0,0,0) for transparent)
) -> str:
    """
    Merge two PNGs side-by-side with a right arrow between them.

    Returns:
        out_path
    """
    L = Image.open(left_path).convert("RGBA")
    R = Image.open(right_path).convert("RGBA")

    def resize_keep_h(img, target_h):
        if img.height == target_h: return img
        w = round(img.width * target_h / img.height)
        return img.resize((w, target_h), Image.LANCZOS)

    if match_height in ("max", "min"):
        target_h = max(L.height, R.height) if match_height == "max" else min(L.height, R.height)
        Lr, Rr = resize_keep_h(L, target_h), resize_keep_h(R, target_h)
    elif match_height == "pad":
        target_h = max(L.height, R.height)

        def pad_h(img):
            if img.height == target_h: return img
            canvas = Image.new("RGBA", (img.width, target_h), bg)
            y = (target_h - img.height) // 2
            canvas.paste(img, (0, y), img)
            return canvas

        Lr, Rr = pad_h(L), pad_h(R)
    else:
        raise ValueError("match_height must be 'max', 'min', or 'pad'")

    # Interpret negative pad as arrow overlap (don’t shrink the canvas)
    overlap = max(0, -int(pad))  # pixels arrow may intrude into each image
    pad = max(0, int(pad))  # real outer padding for the canvas

    # Canvas
    W = pad + Lr.width + gap + Rr.width + pad
    H = target_h + 2 * pad
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    canvas = Image.new("RGBA", (W, H), bg)

    # Paste positions
    y0 = pad
    x_left = pad
    x_right = pad + Lr.width + gap
    canvas.paste(Lr, (x_left, y0), Lr)
    canvas.paste(Rr, (x_right, y0), Rr)

    # Arrow
    draw = ImageDraw.Draw(canvas)
    y_mid = pad + target_h // 2

    lane_x1 = x_left + Lr.width  # left edge of the gap
    lane_x2 = x_right  # right edge of the gap

    # Start/end of shaft inside lane with small margins, then extend by overlap
    inner_margin = max(0, int(0.15 * gap))
    x1 = lane_x1 + inner_margin - overlap  # extend into left image
    x2 = lane_x2 - inner_margin + overlap  # extend into right image

    avail = max(1, x2 - x1)
    min_shaft = max(2, arrow_width)
    head_len = min(arrow_head_len, max(4, int(avail * 0.45)))
    head_wid = min(arrow_head_wid, max(4, int(target_h * 0.4)))

    base_x = x2 - head_len
    if base_x - x1 < min_shaft:  # ensure visible shaft
        head_len = max(4, avail - min_shaft)
        base_x = x2 - head_len

    # Shaft stops at the head base (no bleed)
    draw.line([(x1, y_mid), (base_x, y_mid)], fill=arrow_color, width=arrow_width)

    # Head
    head = [(x2, y_mid),
            (base_x, y_mid - head_wid // 2),
            (base_x, y_mid + head_wid // 2)]
    draw.polygon(head, fill=arrow_color)

    canvas.save(out_path)
    return out_path


@dataclass
class FigureConfig:
    dpi: int = 300
    histogram_alpha: float = 0.05
    main_hist_bar_width: float = 0.01
    inset_hist_bar_width: float = 0.03
    default_inset_axes_kwargs: Mapping[str, Any] = None
    kernel_bw_factor: float = 10
    applied_kernel_bw_factor: Dict[str, float] = None
    def __post_init__(self):
        if self.default_inset_axes_kwargs is None:
            self.default_inset_axes_kwargs = {"alpha": 1.0, "fc": "white", "zorder": 100}
        if self.applied_kernel_bw_factor is None:
            self.applied_kernel_bw_factor = {"ori": 20, "mod":20, "gau": 20}


@dataclass
class InsetsConfig:
    # x-window factors around E0 for the insets
    ins_x_min_factor: float
    ins_x_max_factor: float
    # y scaling factors for inset autoscale
    factor_ins_y_min: float
    factor_ins_y_max: float
    # formatter suppression thresholds per inset (len = 6)
    ins_skip_yticks: Sequence[Optional[float]]
    # y-limits for the small function plots (None for “hist-only” rows)
    func_y_minmax: Sequence[Optional[Tuple[float, float]]]  # len 6
    # y-limits for the small histogram plots
    hist_y_minmax: Sequence[Tuple[float, float]]            # len 6
    # individual inset-axes kwargs (fall back to FigureConfig.default_inset_axes_kwargs if None)
    per_plot_inset_axes_kwargs: Sequence[Optional[Mapping[str, Any]]]  # len 6
    histogram_alpha: float = 0.3


@dataclass
class OriginalsAnalysis:
    x: np.ndarray
    filter_overlap: np.ndarray
    delta_e0: np.ndarray
    delta_e1: np.ndarray
    succ_prob: np.ndarray
    fqpe_cost: np.ndarray
    spectral_gap_normalized: float


@dataclass
class ModifiedBestAnalysis:
    x: np.ndarray
    filter_overlap: np.ndarray
    delta_e0: np.ndarray
    delta_e1: np.ndarray
    succ_prob: np.ndarray
    fqpe_cost: np.ndarray
    spectral_gap_normalized: float
    min_cost_lambda: np.ndarray


@dataclass
class ModifiedAnalysis:
    x_lambda: np.ndarray
    overlap: np.ndarray
    deltae0: np.ndarray
    deltae1: np.ndarray
    succprob: np.ndarray
    cost: np.ndarray
    min_cost_idx: int
    spectral_gap_normalized: float
    max_n_basis: int


@dataclass
class FiltersAndBlocks:
    # function domain & coefficients
    basis_type: str
    x_range: np.ndarray
    time_step: float
    c_final: np.ndarray
    best_modified_c_vec: np.ndarray
    gaussian_c_vec: np.ndarray
    # histogram blocks
    degenerate_block: Sequence[Tuple[int, float, float]]
    degen_block_krylov: Sequence[Tuple[int, float, float]]
    degen_block_modkry: Sequence[Tuple[int, float, float]]
    degen_block_gauss: Sequence[Tuple[int, float, float]]
    # global energy data
    gnd_energy: float
    spectral_gap: float


@dataclass
class YXLimits:
    ori_func_y_minmax: Tuple[float, float]
    mod_func_y_minmax: Tuple[float, float]
    gau_func_y_minmax: Tuple[float, float]
    ori_hist_y_minmax: Tuple[float, float]
    mod_hist_y_minmax: Tuple[float, float]
    gau_hist_y_minmax: Tuple[float, float]


@dataclass
class Annotations:
    pf_krylov: float
    gamma0_krylov_2: float
    cost_krylov: float
    pf_modkry: float
    gamma0_modkry_2: float
    cost_modkry: float
    pf_gauss: float
    gamma0_gauss_2: float
    cost_gauss: float
    txt_xy: Tuple[float, float] = (0.98, 0.975)  # (x, y) in axes fractions


def _ensure_inset_copies(setting: dict) -> dict:
    """If mod/gau inset kwargs are None, copy ori’s (deep)."""
    if setting.get("mod_inset_axes_kwargs") is None:
        setting["mod_inset_axes_kwargs"] = deepcopy(setting["ori_inset_axes_kwargs"])
    if setting.get("gau_inset_axes_kwargs") is None:
        setting["gau_inset_axes_kwargs"] = deepcopy(setting["ori_inset_axes_kwargs"])
    if setting.get("appl_mod_inset_axes_kwargs") is None:
        setting["appl_mod_inset_axes_kwargs"] = deepcopy(setting["appl_ori_inset_axes_kwargs"])
    if setting.get("appl_gau_inset_axes_kwargs") is None:
        setting["appl_gau_inset_axes_kwargs"] = deepcopy(setting["appl_ori_inset_axes_kwargs"])
    return setting


def _deep_merge(base: dict, patch: dict) -> dict:
    out = deepcopy(base)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def generate_plot_setting() -> dict:
    """
    Return a dict:
        { 'hubbard-6'     : <settings-dict>,
          'hubbard-(2,3)' : <settings-dict>, ... }
    You can add as many molecules as you like.
    """
    pl = {
        ("hubbard-6", "trig"): deepcopy(default_plot_settings),

        ("hubbard-(2,3)", "trig"): _deep_merge(default_plot_settings, {
            "ins_x_max_factor": 3.0,
        }),

        ("hubbard-7", "trig"): deepcopy(default_plot_settings),

        ("hubbard-6", "cheby"): deepcopy(default_plot_settings),

        ("hubbard-(2,3)", "cheby"): _deep_merge(default_plot_settings, {
            "ins_x_max_factor": 3.0,
        }),

        ("hubbard-7", "cheby"): deepcopy(default_plot_settings),
    }

    # ensure every molecule has its own mod/gau inset dicts
    for name, st in pl.items():
        pl[name] = _ensure_inset_copies(st)

    # Fine adjustment of inset position
    # x0, y0 (lower-left), width, height
    pl[("hubbard-6", "trig")]["mod_inset_axes_kwargs"]["bbox_to_anchor"] = (0.070, -0.085, 0.85, 0.9)
    pl[("hubbard-6", "trig")]["gau_inset_axes_kwargs"]["bbox_to_anchor"] = (0.085, -0.085, 0.85, 0.9)
    pl[("hubbard-6", "cheby")]["ori_inset_axes_kwargs"]["bbox_to_anchor"] = (0.075, -0.05, 0.85, 0.9)
    pl[("hubbard-6", "cheby")]["mod_inset_axes_kwargs"]["bbox_to_anchor"] = (0.078, -0.085, 0.85, 0.9)
    pl[("hubbard-6", "cheby")]["gau_inset_axes_kwargs"]["bbox_to_anchor"] = (0.062, -0.085, 0.85, 0.9)

    pl[("hubbard-(2,3)", "trig")]["ori_inset_axes_kwargs"]["bbox_to_anchor"] = (0.075, -0.085, 0.85, 0.9)
    pl[("hubbard-(2,3)", "trig")]["mod_inset_axes_kwargs"]["bbox_to_anchor"] = (0.050, -0.085, 0.85, 0.9)
    pl[("hubbard-(2,3)", "trig")]["gau_inset_axes_kwargs"]["bbox_to_anchor"] = (0.080, -0.085, 0.85, 0.9)
    pl[("hubbard-(2,3)", "cheby")]["ori_inset_axes_kwargs"]["bbox_to_anchor"] = (0.065, -0.085, 0.85, 0.9)
    pl[("hubbard-(2,3)", "cheby")]["mod_inset_axes_kwargs"]["bbox_to_anchor"] = (0.085, -0.085, 0.85, 0.9)
    pl[("hubbard-(2,3)", "cheby")]["gau_inset_axes_kwargs"]["bbox_to_anchor"] = (0.055, -0.085, 0.85, 0.9)

    pl[("hubbard-7", "trig")]["x_range"] = np.linspace(-0.55, 0.8, 1000)
    pl[("hubbard-7", "trig")]["ori_inset_axes_kwargs"]["bbox_to_anchor"] = (0.075, -0.085, 0.85, 0.9)
    pl[("hubbard-7", "trig")]["mod_inset_axes_kwargs"]["bbox_to_anchor"] = (0.080, -0.085, 0.85, 0.9)
    pl[("hubbard-7", "trig")]["gau_inset_axes_kwargs"]["bbox_to_anchor"] = (0.075, -0.085, 0.85, 0.9)
    pl[("hubbard-7", "trig")]["gau_ins_skip_yticks"] = 0.1

    pl[("hubbard-7", "cheby")]["ori_inset_axes_kwargs"]["bbox_to_anchor"] = (0.075, -0.085, 0.85, 0.9)
    pl[("hubbard-7", "cheby")]["mod_inset_axes_kwargs"]["bbox_to_anchor"] = (0.075, -0.085, 0.85, 0.9)
    pl[("hubbard-7", "cheby")]["gau_inset_axes_kwargs"]["bbox_to_anchor"] = (0.075, -0.085, 0.85, 0.9)

    return pl


def settings_to_configs(setting: dict):
    """
    Convert one `setting` dict (exactly one molecule) to the two
    dataclass objects that the plotting layer wants.
    """
    func_minmax = [
        setting["ori_func_y_minmax"],
        setting["mod_func_y_minmax"],
        setting["gau_func_y_minmax"],
        None, None, None,
    ]
    hist_minmax = [
        setting["ori_hist_y_minmax"],
        setting["mod_hist_y_minmax"],
        setting["gau_hist_y_minmax"],
        setting["ori_hist_y_minmax"],
        setting["mod_hist_y_minmax"],
        setting["gau_hist_y_minmax"],
    ]
    inset_kwargs = [
        setting["ori_inset_axes_kwargs"],
        setting["mod_inset_axes_kwargs"] or setting["ori_inset_axes_kwargs"],
        setting["gau_inset_axes_kwargs"] or setting["ori_inset_axes_kwargs"],
        setting["appl_ori_inset_axes_kwargs"],
        setting["appl_mod_inset_axes_kwargs"] or setting["appl_ori_inset_axes_kwargs"],
        setting["appl_gau_inset_axes_kwargs"] or setting["appl_ori_inset_axes_kwargs"],
    ]
    ins_skip = [
        setting["ori_ins_skip_yticks"],
        setting["mod_ins_skip_yticks"],
        setting["gau_ins_skip_yticks"],
        setting["appl_ori_ins_skip_yticks"],
        setting["appl_mod_ins_skip_yticks"],
        setting["appl_gau_ins_skip_yticks"],
    ]

    insets_cfg = InsetsConfig(
        ins_x_min_factor=setting["ins_x_min_factor"],
        ins_x_max_factor=setting["ins_x_max_factor"],
        factor_ins_y_min=setting["ins_y_min_factor"],
        factor_ins_y_max=setting["ins_y_max_factor"],
        ins_skip_yticks=ins_skip,
        func_y_minmax=func_minmax,
        hist_y_minmax=hist_minmax,
        per_plot_inset_axes_kwargs=inset_kwargs,
    )

    limits = YXLimits(
        ori_func_y_minmax=setting["ori_func_y_minmax"],
        mod_func_y_minmax=setting["mod_func_y_minmax"],
        gau_func_y_minmax=setting["gau_func_y_minmax"],
        ori_hist_y_minmax=setting["ori_hist_y_minmax"],
        mod_hist_y_minmax=setting["mod_hist_y_minmax"],
        gau_hist_y_minmax=setting["gau_hist_y_minmax"],
    )
    return insets_cfg, limits


def _select_settings(mol_name: str, filter_basis_type: str):
    return generate_plot_setting()[(mol_name, filter_basis_type)]


def load_plot_bundle(filter_basis_type, mol_name, transform, tag, **_):
    pkl_path = f'./data/krylov_analysis_{filter_basis_type}/{mol_name}_{transform}_{tag}_plot_inputs.pkl'
    with open(pkl_path, "rb") as f:
        bundle = pickle.load(f)

    # ------------------- originals (vs N) ---------------------------------
    o, mb, ms = bundle["originals"], bundle["modified_best"], bundle["modified_sparse"]
    # delta_e1 in file has length len(x)-1, so left-pad with NaN

    originals = OriginalsAnalysis(
        x=o["x"],
        filter_overlap=o["filter_overlap"],
        delta_e0=o["delta_e0"],
        delta_e1=np.concatenate([[np.nan], o["delta_e1"]]),
        succ_prob=o["succ_prob"],
        fqpe_cost=o["fqpe_cost"],
        spectral_gap_normalized=o["spectral_gap_normalized"],
    )

    modified_best = ModifiedBestAnalysis(
        x=mb["x"],
        filter_overlap=mb["filter_overlap"],
        delta_e0=mb["delta_e0"],
        delta_e1=np.concatenate([[np.nan], mb["delta_e1"]]),
        succ_prob=mb["succ_prob"],
        fqpe_cost=mb["fqpe_cost"],
        spectral_gap_normalized=mb["spectral_gap_normalized"],
        min_cost_lambda=mb["min_cost_lambda"],
    )

    modified_sparse = {
        ftr: OriginalsAnalysis(
            x=ms["x"],
            filter_overlap=ms["filter_overlap_by_lambda"][idx_ftr],
            delta_e0=ms["delta_e0_by_lambda"][idx_ftr],
            delta_e1=np.concatenate([[np.nan], ms["delta_e1_by_lambda"][idx_ftr]]),
            succ_prob=ms["succ_prob_by_lambda"][idx_ftr],
            fqpe_cost=ms["fqpe_cost_by_lambda"][idx_ftr],
            spectral_gap_normalized=ms["spectral_gap_normalized"],
        )
        for idx_ftr, ftr in enumerate(ms["sparse_lambda"])
    }

    # ------------------- modified (vs λ) ----------------------------------
    m = bundle["modified"]
    modified = ModifiedAnalysis(
        x_lambda=m["x_lambda"],
        overlap=m["overlap"],
        deltae0=m["deltae0"],
        deltae1=m["deltae1"],
        succprob=m["succprob"],
        cost=m["cost"],
        min_cost_idx=m["min_cost_idx"],
        spectral_gap_normalized=m["spectral_gap_normalized"],
        max_n_basis=bundle["max_n_basis"],
    )

    # ------------------- filters & blocks ---------------------------------
    fb = bundle["filters_blocks"]
    settings = _select_settings(bundle["mol_name"], filter_basis_type)

    filt = FiltersAndBlocks(
        basis_type=filter_basis_type,
        x_range=np.asarray(settings["x_range"]),
        time_step=bundle["time_step"],
        c_final=fb["c_final"],
        best_modified_c_vec=fb["best_modified_c_vec"],
        gaussian_c_vec=fb["gaussian_c_vec"],
        degenerate_block=fb["degenerate_block"],
        degen_block_krylov=fb["degen_block_krylov"],
        degen_block_modkry=fb["degen_block_modkry"],
        degen_block_gauss=fb["degen_block_gauss"],
        gnd_energy=bundle["gnd_energy"],
        spectral_gap=bundle["spectral_gap"],
    )

    # ------------------- annotations --------------------------------------
    a = bundle["annotations"]
    ann = Annotations(
        pf_krylov=a["pf_krylov"], gamma0_krylov_2=a["gamma0_krylov_2"], cost_krylov=a["cost_krylov"],
        pf_modkry=a["pf_modkry"], gamma0_modkry_2=a["gamma0_modkry_2"], cost_modkry=a["cost_modkry"],
        pf_gauss=a["pf_gauss"], gamma0_gauss_2=a["gamma0_gauss_2"], cost_gauss=a["cost_gauss"],
    )

    extra_information = dict(
        epsilon_normalized=bundle["epsilon_normalized"]
    )

    return dict(
        originals=originals,
        modified_best=modified_best,
        modified=modified,
        modified_sparse=modified_sparse,
        filt=filt,
        ann=ann,
        basis_type=bundle["filter_basis_type"],
        mol_name=bundle["mol_name"],
        tag=bundle["tag"],
        settings=settings,
        extra_information=extra_information,
    )


def save_figs(base_dir, merged_based_dir, plt_name, fig_names, figs, merged_shape=None):
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    merged_based_dir = Path(merged_based_dir)
    merged_based_dir.mkdir(parents=True, exist_ok=True)

    image_path = list()
    for idx, fig_name in enumerate(fig_names):
        fig = figs[fig_name]
        fig_path = os.path.join(base_dir, f"{idx}_{plt_name}_{fig_name}.png")
        fig.savefig(fig_path)
        if "NOLEG" not in fig_name:
            image_path.append(fig_path)

    if merged_shape is not None:
        n_cols, n_rows = merged_shape
        if n_cols * n_rows < len(image_path):
            raise ValueError
        merged_img_path = os.path.join(merged_based_dir, f"{plt_name}_merged.png")
        total_cells = n_rows * n_cols

        # pad with None if not enough paths
        paths_padded = image_path + [None] * (total_cells - len(image_path))

        # build list-of-lists grid
        im_list = []
        for r in range(n_rows):
            row = []
            for c in range(n_cols):
                p = paths_padded[r * n_cols + c]
                if p is None:
                    row.append(None)
                else:
                    row.append(Image.open(p))
            im_list.append(row)

        row_widths = []
        for row in im_list:
            w_sum = 0
            for im in row:
                if im is not None:
                    w_sum += im.width
            row_widths.append(w_sum)
        tot_w = max(row_widths) if row_widths else 0

        col_heights = []
        for c in range(n_cols):
            h_sum = 0
            for r in range(n_rows):
                if c < len(im_list[r]):
                    im = im_list[r][c]
                    if im is not None:
                        h_sum += im.height
            col_heights.append(h_sum)
        tot_h = max(col_heights) if col_heights else 0

        merged_img = Image.new("RGBA", (tot_w, tot_h))

        pos_y = 0
        for idx_row, row in enumerate(im_list):
            pos_x = 0
            for im in row:
                if im is not None:
                    merged_img.paste(im, (pos_x, pos_y))
                    pos_x += im.width
            pos_y += max([im.height for im in row if im is not None])

        merged_img.save(merged_img_path)


default_plot_settings = {
    # ---------- main curves ----------
    "x_range": np.linspace(-0.65, 1.0, 1000),

    # This all should be normalized!
    "ori_func_y_minmax": (1e-16, 1.1),
    "ori_hist_y_minmax": (1e-8,  1.1),
    "mod_func_y_minmax": (1e-16, 1.1),
    "mod_hist_y_minmax": (1e-8,  1.1),
    "gau_func_y_minmax": (1e-16, 1.1),
    "gau_hist_y_minmax": (1e-8,  1.1),

    # ---------- global inset tuning ----------
    "ins_x_min_factor": -1.0,
    "ins_x_max_factor":  9.0,
    "ins_y_min_factor":  0.01,
    "ins_y_max_factor": 10.0,

    # (None  → don’t suppress any ticks;   numeric → hide labels > threshold)
    "ori_ins_skip_yticks": None,
    "mod_ins_skip_yticks": None,
    "gau_ins_skip_yticks": None,

    "appl_ori_ins_skip_yticks": None,
    "appl_mod_ins_skip_yticks": None,
    "appl_gau_ins_skip_yticks": None,

    # ---------- per-figure inset placement ----------
    "ori_inset_axes_kwargs": {
        "width": "40%", "height": "60%",
        "loc": "right", "bbox_to_anchor": (0.075, -0.05, 0.85, 0.9),  #x0, y0 (lower-left), width, height
        "axes_kwargs":  {"alpha": 1.0, "fc": "white", "zorder": 100},
    },
    # copy for other types (don’t alias! we’ll deep-copy later)
    # mod/gau start identical, vary per molecule
    "mod_inset_axes_kwargs": None,   # None → take default
    "gau_inset_axes_kwargs": None,
    # ---- For applied state histogram
    "appl_ori_inset_axes_kwargs": {
        "width": "40%", "height": "60%",
        "loc": "right", "bbox_to_anchor": (0.1, -0.085, 0.85, 0.9),
        "axes_kwargs":  {"alpha": 1.0, "fc": "white", "zorder": 100},
    },
    "appl_mod_inset_axes_kwargs": None,
    "appl_gau_inset_axes_kwargs": None,
}
