import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from fractions import Fraction
from typing import Sequence, Tuple

from filter_state.utils_plot import apply_mpl_style

TXTPNT_TO_INCH = 1 / 72


def plot_gaussian_grid_irregular(
        *,
        center_factors: Sequence[float],
        width_factors: Sequence[float],
        inset_frac: float = 0.78,
        figsize: Tuple[float, float] = (8, 5.2),
        dpi: int = 300,
):
    """Draw one mini Gaussian at every (c′, w′) pair on an irregular grid."""
    apply_mpl_style()
    import matplotlib as mpl
    mpl.rcParams["axes.labelsize"] = 23  # x– and y–axis labels only
    mpl.rcParams["axes.titlesize"] = 23  # subplot titles
    mpl.rcParams["xtick.labelsize"] = 23
    mpl.rcParams["ytick.labelsize"] = 23

    inset_label_fontsize = 19

    # ---------- helpers ----------------------------------------------------
    def _gauss(Ep, cp, wp, eps=1e-2):
        return np.exp(np.log(eps) * ((Ep - cp) / wp) ** 2)

    def _fmt_center(cp):
        if abs(cp) < 1e-10:
            return r"$E_0$"
        if abs(cp - round(cp)) < 1e-10:
            m = int(round(cp));
            s = "+" if m > 0 else "-"
            if abs(m) == 1:
                return rf"$E_0 {s} \Delta E_0$"
            else:
                return rf"$E_0 {s} {abs(m)}\Delta E_0$"
        frac = Fraction(cp).limit_denominator(10)
        n, d = frac.numerator, frac.denominator
        s = "+" if cp > 0 else "-"
        return rf"$E_0 {s} {abs(n) if n != 1 else ''}\Delta E_0/{d}$"

    def _fmt_width(wp):
        if abs(wp - round(wp)) < 1e-10:
            m = int(round(wp))
            return rf"${m}\Delta E_0$" if m != 1 else r"$\Delta E_0$"
        frac = Fraction(wp).limit_denominator(10)
        n, d = frac.numerator, frac.denominator
        return rf"${n if n != 1 else ''}\Delta E_0/{d}$"

    # ---------- prep -------------------------------------------------------
    center_factors = np.asarray(center_factors, float)
    width_factors = np.asarray(width_factors, float)

    # gaps used to size the insets
    def _gap(a):
        return np.min(np.diff(np.sort(a))) if len(a) > 1 else 1.0

    dx, dy = _gap(center_factors), _gap(width_factors)
    box_w, box_h = inset_frac * dx, inset_frac * dy  # inset size (data units)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_xlim(center_factors.min() - box_w * 0.60, center_factors.max() + box_w * 0.60)
    ax.set_ylim(width_factors.min() - box_h * 0.90, width_factors.max() + box_h * 0.60)
    # ax.set_aspect("equal")

    # ticks exactly at data positions
    ax.set_xticks(center_factors, [_fmt_center(c) for c in center_factors])
    #  rotation=45, ha="right")
    ax.set_yticks(width_factors, [_fmt_width(w) for w in width_factors])
    ax.tick_params(length=0)
    ax.set_xlabel(r"Center")
    ax.set_ylabel(r"Width", labelpad=-10)

    # ---------- draw every inset ------------------------------------------
    Ep = np.linspace(min(center_factors) - max(width_factors),
                     max(center_factors) + max(width_factors), 250)
    for cp in center_factors:
        for wp in width_factors:
            # bounding box in DATA coordinates
            bbox = (cp - box_w / 2, wp - box_h / 2, box_w, box_h)
            ax_in = inset_axes(
                ax,
                width="100%", height="100%",  # fill the bbox
                bbox_to_anchor=bbox,
                bbox_transform=ax.transData,  # <-- key line
                loc="lower left",
                borderpad=0,
            )
            ax_in.plot(Ep, _gauss(Ep, cp, wp), lw=0.8)
            ax_in.set_ylim(0, 1.05)
            ax_in.set_xticks([0, 1])
            e1_label = r"$E_1\cdots$"
            ax_in.set_xticklabels(["$E_0$", e1_label], fontsize=inset_label_fontsize)
            ax_in.set_yticks([])
            for txt in ax_in.get_xticklabels():
                if txt.get_text() == e1_label:
                    txt.set_transform(
                        txt.get_transform() + ScaledTranslation(15 * TXTPNT_TO_INCH, 0, fig.dpi_scale_trans)
                    )
                    break
            for s in ("right", "left", "top"): ax_in.spines[s].set_visible(False)

    fig.tight_layout()
    return fig, ax


# ------------------------------ demo ----------------------------------------
if __name__ == "__main__":
    centers = [-1, 0, 1]
    widths = [1 / 3, 1, 2, ]
    fig, ax = plot_gaussian_grid_irregular(center_factors=centers,
                                           width_factors=widths,
                                           inset_frac=0.78)
    fig_path = "./figures/filter_previewpannel.png"
    fig.savefig(fig_path)
