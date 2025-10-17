from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import seaborn as sns
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, LogNorm

from chemistry_data.chem_tools import prepare_hamiltonian_refstates
from chemistry_data.example_model import hubbard_examples
from filter_state.utils_plot import to_latex_sci, apply_mpl_style


# ---------- small utilities -------------------------------------------------

def q(x: float, ndigits: int = 7) -> float:
    """Round for use as dictionary keys/labels to avoid FP noise."""
    return round(x, ndigits)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Paths:
    data_dir: Path  # where *_filter_properties.pkl live
    fig_dir: Path  # base folder to save figures


@dataclass
class Inputs:
    filter_basis_type: str  # "trig" | "cheby"
    mol_name: str
    transform: str
    tag: Optional[str]
    epsilons_by_gap: Tuple[float, ...] = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
    ceil_log10: int = 2
    floor_log10: int = -4
    x_crop_px: int = 2020  # crop line for *_crop.png
    max_epsilon_prime: float = 0.7


# ---------- Prop finder ----------------------------------------------------------

Key = Tuple[str, float, float]  # (filter_type, center, width)
Value = Tuple[Any, ...]  # succ_prob, f_e0, overlap_sq, depth


def _nearest_prop(
        table: Dict[Key, Value],
        ftype: str,
        c_req: float,
        w_req: float,
) -> Tuple[Key, Value]:
    """Return (nearest_key, table[nearest_key]) for given request."""
    best_key: Optional[Key] = None
    best_dist = math.inf
    for k in table.keys():
        if k[0] != ftype:
            continue
        dist = abs(k[1] - c_req) + abs(k[2] - w_req)  # 1-norm
        if dist < best_dist:
            best_dist = dist
            best_key = k
    if best_key is None:
        raise KeyError(f"No entries with filter_type={ftype!r}")
    return best_key, table[best_key]


# ---------- I/O -------------------------------------------------------------

def _prop_file_name(paths: Paths, inp: Inputs, epsilon_by_gap: float) -> Path:
    base = f"{inp.mol_name}_{inp.transform}_{epsilon_by_gap:.2e}"
    if inp.tag:
        base += f"_{inp.tag}"
    return paths.data_dir / f"{base}_filter_properties.pkl"


def _load_filter_properties(path: Path):
    """
    Expected pickle structure:
        (gnd_energy_normalized, spectral_gap_normalized, _, gamma0_sq, _, filter_center, filter_width, filter_property_list)
    """
    import pickle
    with path.open("rb") as f:
        return pickle.load(f)


# ---------- math helpers ----------------------------------------------------

def _normalize_and_clip(cost: np.ndarray, floor_log10: int, ceil_log10: int) -> np.ndarray:
    out = cost.copy()
    ceil = 10 ** ceil_log10
    floor = 10 ** floor_log10
    out[out > ceil] = ceil
    if np.min(out) < floor:
        raise ValueError(f"cost contains values below 10^{floor_log10}: {np.min(out)}")
    return out


def _custom_cmap(floor_log10: int, ceil_log10: int) -> LinearSegmentedColormap:
    ceil = 10 ** ceil_log10
    flor = 10 ** floor_log10
    middle = -np.log(flor) / (np.log(ceil) - np.log(flor))
    # green → blue → white → red, with a sharper mid-to-low emphasis
    colors = [(0, "green"), (middle ** 2, "blue"), (middle, "white"), (1, "red")]
    return LinearSegmentedColormap.from_list("custom_cmap", colors)


# ---------- plotting small helpers -----------------------------------------

def _format_y_tick_label(
        wfactor: float,
        depth: int
) -> str:
    """Build the multi-line y tick label showing width in ΔE0 units and N estimate."""
    str_n_basis = to_latex_sci(depth, precision=2, with_dollar=False, negative_space=True)

    if abs(wfactor - round(wfactor)) < 1e-5:
        wfactor = int(round(wfactor))
        assert wfactor >= 0
        coeff = (f"{wfactor}\\Delta E_0" if wfactor > 1 else "\\Delta E_0")
    else:
        fr = Fraction(wfactor).limit_denominator(10)
        num, den = fr.numerator, fr.denominator
        coeff = f"\\Delta E_0 / {den}" if num == 1 else f"{num}/{den} \\Delta E_0"

    label = (
            r"\begin{align*}"
            + rf"{coeff}& \\[-0.4em] "
            + r"(" + str_n_basis + r")&"
            + r"\end{align*}"
    )
    return label


def _tweak_colorbar_ticks(cbar, floor_log10: int, ceil_log10: int) -> None:
    ceil = 10 ** ceil_log10
    flor = 10 ** floor_log10

    ticks = list(cbar.get_ticks())
    labels = [t.get_text() for t in cbar.ax.get_yticklabels()]

    def _inject(value: float, text: str):
        for i, t in enumerate(ticks):
            if np.isclose(t, value):
                ticks[i] = value
                labels[i] = text
                return
            if t > value:
                ticks.insert(i, value)
                labels.insert(i, text)
                return
        ticks.append(value)
        labels.append(text)

    _inject(ceil, rf"$>\mathdefault{{10^{ceil_log10}}}$")
    _inject(flor, rf"$\mathdefault{{10^{floor_log10}}}$")

    # Trim first & last (often out of bd)
    if len(ticks) > 2:
        ticks = ticks[1:-1]
        labels = labels[1:-1]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(labels)


# ---------- main per-epsilon heatmap routine -------------------------------

def _plot_heatmap_for_epsilon(
        ax: Axes,
        cost_now: np.ndarray,
        filter_center_list: np.ndarray,
        filter_width_list: np.ndarray,
        depth_array: np.ndarray,
        gnd_energy: float,
        spectral_gap: float,
        floor_log10: int,
        ceil_log10: int,
) -> Tuple[int, int]:
    """
    Draw the heatmap and return the (ix, iy) of the min cost.
    """
    cmap = _custom_cmap(floor_log10, ceil_log10)
    ceil = 10 ** ceil_log10
    flor = 10 ** floor_log10

    # Build tick labels shown on the axes (strings)
    x_tick_label = [f"{c:.2e}" for c in (filter_center_list - gnd_energy)]
    y_tick_label = [f"{w:.2e}" for w in filter_width_list]

    hm = sns.heatmap(
        cost_now.T,
        annot=False,
        cmap=cmap,
        norm=LogNorm(vmin=flor, vmax=ceil),
        xticklabels=x_tick_label,
        yticklabels=y_tick_label,
        cbar=True,
        ax=ax,
    )

    # normalizers map energy→index-space for grid overlays
    len_x, len_y = len(filter_center_list), len(filter_width_list)
    min_x, max_x = float(np.min(filter_center_list)), float(np.max(filter_center_list))
    min_y, max_y = float(np.min(filter_width_list)), float(np.max(filter_width_list))
    nx = lambda val: len_x * (val - min_x) / (max_x - min_x)
    ny = lambda val: len_y * (val - min_y) / (max_y - min_y)

    # grid ticks (X, energies at E0 + m Δ)
    min_x_tick = int((min_x - gnd_energy) / spectral_gap)
    max_x_tick = int((max_x - gnd_energy) / spectral_gap)
    x_tick_fctr = np.concatenate((np.arange(min_x_tick, 0), np.arange(1, max_x_tick + 1)))
    x_tick_fctr_with_gnd = np.arange(min_x_tick, max_x_tick + 1)
    x_ticks_without_gnd = x_tick_fctr * spectral_gap + gnd_energy
    x_ticks_with_gnd = x_tick_fctr_with_gnd * spectral_gap + gnd_energy

    # Y ticks (width multiples of Δ)
    mw = np.min(filter_width_list) / spectral_gap
    min_y_tick, max_y_tick = max(int(mw), 1), int(np.max(filter_width_list) / spectral_gap)
    y_tick_fctr = np.arange(min_y_tick, max_y_tick + 1)
    if int(mw) == 0:
        mw_frac = float(Fraction(mw).limit_denominator(10))
        y_tick_fctr = np.concatenate(([mw_frac], y_tick_fctr))
    y_ticks = y_tick_fctr * spectral_gap

    # grid lines
    grid_alpha = 0.5
    grid_linewidth = 0.8
    ax.vlines(nx(x_ticks_without_gnd), 0, len_y, colors="black", linestyles="--", alpha=grid_alpha,
              linewidth=grid_linewidth)
    ax.vlines(nx(gnd_energy), 0, len_y, colors="black", linestyles="-.", alpha=grid_alpha, linewidth=grid_linewidth)
    ax.hlines(ny(y_ticks), 0, len_x, colors="black", linestyles="--", alpha=grid_alpha, linewidth=grid_linewidth)

    # min point
    min_ix, min_iy = np.unravel_index(np.argmin(cost_now), cost_now.shape)
    ax.scatter(min_ix, min_iy, color="yellow", marker="x",
               label=r"$\mathrm{min}(C_\mathrm{FQPE}/C_{\mathrm{QPE}})="
                     + to_latex_sci(cost_now[min_ix, min_iy], precision=3, with_dollar=False) + "$")

    # colorbar tweaks
    cbar = hm.collections[0].colorbar
    _tweak_colorbar_ticks(cbar, floor_log10, ceil_log10)

    # Calculate the depth_list
    depth_list: list[int] = []
    for w_abs in y_ticks:  # iterate over absolute widths
        exact = np.where(np.isclose(filter_width_list, w_abs, atol=1e-12, rtol=0))[0]
        idx = int(exact[0]) if exact.size else int(np.argmin(np.abs(filter_width_list - w_abs)))
        depth_list.append(int(depth_array[idx]))

    # axis ticks/labels
    ax.set_xticks(nx(x_ticks_with_gnd)[1:])
    ax.set_xticklabels([
        r"$0$" if i == 0 else
        r"$\Delta E_0$" if i == 1 else
        r"$- \Delta E_0$" if i == -1 else
        rf"${i}\Delta E_0$" if i > 0 else
        rf"$- {-i}\Delta E_0$"
        #    r"$E_\mathrm{0}$" if i == 0 else
        #    r"$E_\mathrm{0} + \Delta E_0$" if i == 1 else
        #    r"$E_\mathrm{0} - \Delta E_0$" if i == -1 else
        #    rf"$E_\mathrm{{0}} + {i}\Delta E_0$" if i > 0 else
        #    rf"$E_\mathrm{{0}} - {-i}\Delta E_0$"
        for i in x_tick_fctr_with_gnd[1:]
    ])
    ax.set_yticks(ny(y_ticks))
    ax.set_yticklabels(
        [_format_y_tick_label(wf, depth) for wf, depth in zip(y_tick_fctr, depth_list)],
        rotation=0
    )
    ax.tick_params(axis="x", rotation=0)

    # ax.set_xlabel(r"Center bias with respect to $E_0$")
    # ax.set_ylabel(r"Width")
    ax.invert_yaxis()
    ax.legend(loc="lower left",
              handlelength=1.0,
              handletextpad=-0.1,
              borderpad=0.2)

    return int(min_ix), int(min_iy)


# ---------- worst-case curve (ε′ axis) -------------------------------------

def _collect_worst_points(
        cost_now: np.ndarray,
        filter_center_list: np.ndarray,
        filter_width_list: np.ndarray,
        gnd_energy: float,
        spectral_gap: float,
        max_epsilon_prime: float,
        n_points: int = 20,
) -> Tuple[np.ndarray, List[float]]:
    # --- validate & shape ---
    center = np.asarray(filter_center_list, float).squeeze()
    width = np.asarray(filter_width_list, float).squeeze()
    if center.ndim != 1 or width.ndim != 1:
        raise ValueError("filter_center_list and filter_width_list must be 1-D.")
    Nc, Nw = center.size, width.size
    if cost_now.shape != (Nc, Nw):
        raise ValueError(f"cost_now shape {cost_now.shape} != ({Nc}, {Nw}).")

    # --- grids via broadcasting (no copies) ---
    center_col = center[:, None]  # (Nc, 1)
    width_row = width[None, :]  # (1, Nw)
    prior_energy_0 = np.broadcast_to(center_col, (Nc, Nw))  # E0'
    prior_energy_1 = center_col + width_row  # E1' = E0' + width

    first_energy = gnd_energy + spectral_gap

    # probe eps' from smallest center error up to max range
    eps_grid = np.linspace(
        float(np.min(np.abs(center - gnd_energy))),
        float(spectral_gap * (max_epsilon_prime + 0.01)),
        n_points
    )

    worst_points: List[Optional[float]] = []
    last_val: Optional[float] = None
    first_hit_index: Optional[int] = None

    for idx_err, err in enumerate(eps_grid):
        mask = (
                (np.abs(prior_energy_0 - gnd_energy) <= err) &
                (np.abs(prior_energy_1 - first_energy) <= err)
        )
        vals = cost_now[mask]
        if vals.size == 0:
            # no candidates at this radius
            # - if we’ve already seen a value, repeat it (plateau)
            # - else, append placeholder; we’ll backfill later
            worst_points.append(last_val if last_val is not None else None)
            continue

        # we have candidates; take worst (max)
        v = float(vals.max())
        last_val = v
        worst_points.append(v)
        if first_hit_index is None:
            first_hit_index = idx_err

    # post-process leading None values
    if first_hit_index is None:
        # never found any hits at any radius
        raise ValueError("No matching grid points found for any epsilon'. Check grids and thresholds.")
    # backfill all leading None with the first actual value
    first_val = worst_points[first_hit_index]
    for i in range(first_hit_index):
        worst_points[i] = first_val

    # at this point, any remaining None would be from logic errors; guard anyway
    worst_points = [float(v) if v is not None else float(first_val) for v in worst_points]

    # return ε' normalized by ΔE0
    return eps_grid / spectral_gap, worst_points


# ---------- top-level driver (refactor of `script`) ------------------------

def run(
        filter_basis_type: str,
        mol_name: str,
        transform: str,
        tag: Optional[str],
        *,
        paths: Optional[Paths] = None,
        inputs: Optional[Inputs] = None,
) -> None:
    """
    Refactored entry-point. Matches original behavior, cleaner structure.
    """
    apply_mpl_style()

    inputs = inputs or Inputs(
        filter_basis_type=filter_basis_type,
        mol_name=mol_name,
        transform=transform,
        tag=tag,
    )
    paths = paths or Paths(
        data_dir=Path(f"data/fstate_width_center_{filter_basis_type}"),
        fig_dir=Path(f"figures/filtered_qpe_cost_{filter_basis_type}") / mol_name
    )
    _ensure_dir(paths.data_dir)
    _ensure_dir(paths.fig_dir)

    print(mol_name, filter_basis_type)
    image_paths: List[Path] = []

    # worst-case plot
    fig_wc, ax_wc = plt.subplots(figsize=(8, 4.5), dpi=300)
    wc_y_min, wc_y_max = math.inf, 0.0

    filter_center_list: Optional[np.ndarray] = None
    filter_width_list: Optional[np.ndarray] = None

    failed_load = False

    for epsilon_by_gap in inputs.epsilons_by_gap:
        print(f"epsilon_normalized: {epsilon_by_gap}")
        pkl_path = _prop_file_name(paths, inputs, epsilon_by_gap)
        try:
            (gnd_E, gap, _unused1, gamma0_sq, _unused2,
             filter_center, filter_width, filt_prop_list) = _load_filter_properties(pkl_path)
        except FileNotFoundError as e:
            failed_load = True
            print(e)
            continue

        # unify center/width arrays across epsilons
        if filter_center_list is None:
            filter_center_list = np.asarray(filter_center, float)
            filter_width_list = np.asarray(filter_width, float)
        else:
            _c = np.asarray(filter_center, float)
            _w = np.asarray(filter_width, float)
            if not (np.allclose(filter_center_list, _c) and np.allclose(filter_width_list, _w)):
                raise ValueError("filter_center/width grids differ across epsilon values.")

        # compute normalized epsilon for QPE
        epsilon_norm = float(epsilon_by_gap * gap)

        # base QPE cost
        qpe_cost = 1.0 / (epsilon_norm * gamma0_sq)

        # fill cost matrix (center × width) and extract the depth list
        C = np.zeros((len(filter_center_list), len(filter_width_list)), float)
        depth_array = np.zeros(len(filter_width_list), int)
        filter_type = "gaussian_function_fourier"
        for (ix, c_val), (iy, w_val) in product(enumerate(filter_center_list), enumerate(filter_width_list)):
            c_rd, w_rd = q(c_val), q(w_val)
            try:
                succ_prob, f_e0, overlap_sq, depth = filt_prop_list[(filter_type, c_rd, w_rd)]
            except KeyError:
                (k_found, (succ_prob, f_e0, overlap_sq, depth)) = _nearest_prop(
                    filt_prop_list, filter_type, c_rd, w_rd
                )
                diff_center = abs(k_found[1] - c_rd)
                diff_width = abs(k_found[2] - w_rd)
                print(
                    f"[WARN] Key ({c_rd:.3e}, {w_rd:.3e}) not found; "
                    f"using nearest ({k_found[1]:.3e}, {k_found[2]:.3e}) "
                    f"(Δcenter={diff_center / gap:.3e}ΔE0 , Δwidth={diff_width / gap:.3e}ΔE0)"
                )

            assert 0 <= abs(f_e0) <= 1.0, (f_e0, ix, c_val, iy, w_val)
            assert 0 <= succ_prob <= 1.0, (succ_prob, ix, c_val, iy, w_val)
            mf = 1.0 / (overlap_sq * succ_prob)
            fqpe_cost = mf * depth + 1.0 / (epsilon_norm * overlap_sq)
            C[ix, iy] = fqpe_cost / qpe_cost

            depth = int(depth)
            assert depth > 0
            assert depth == depth_array[iy] or depth_array[iy] == 0
            depth_array[iy] = depth

        # clip/normalize for plotting
        C_plot = _normalize_and_clip(C, inputs.floor_log10, inputs.ceil_log10)

        # --- heatmap figure
        fig_hm, ax_hm = plt.subplots(figsize=(8, 4.5), dpi=300)
        min_ix, min_iy = _plot_heatmap_for_epsilon(
            ax=ax_hm,
            cost_now=C_plot,
            filter_center_list=filter_center_list,
            filter_width_list=filter_width_list,
            depth_array=depth_array,
            gnd_energy=gnd_E,
            spectral_gap=gap,
            floor_log10=inputs.floor_log10,
            ceil_log10=inputs.ceil_log10,
        )
        print(f"min cost = {C_plot[min_ix, min_iy]:.3e}")
        print(f"min center = {(filter_center_list[min_ix] - gnd_E) / gap:.3e} Δ + E0")
        print(f"min width  = {filter_width_list[min_iy] / gap:.3e} Δ")

        # ax_hm.set_title(rf"{mol_name} — $\epsilon={epsilon_by_gap:.3e}\Delta E_0$")
        fig_hm.tight_layout()

        # save + crop
        out = paths.fig_dir / f"{mol_name}_ε={epsilon_by_gap:.3e}.png"
        fig_hm.savefig(out)
        image_paths.append(out)

        crop_out = paths.fig_dir / f"{mol_name}_ε={epsilon_by_gap:.3e}_crop.png"
        img = Image.open(out)
        w, h = img.size
        x_line = inputs.x_crop_px
        if not (0 <= x_line < w):
            raise ValueError(f"x coordinate {x_line} is outside image width 0…{w - 1}")
        cropped = img.crop((0, 0, x_line, h))
        cropped.save(crop_out)

        # --- worst-case accumulation
        eps_prime, worst_pts = _collect_worst_points(
            C_plot, filter_center_list, filter_width_list, gnd_E, gap, inputs.max_epsilon_prime
        )
        ax_wc.plot(
            eps_prime, worst_pts,
            label=rf"$\epsilon\!=\!{to_latex_sci(epsilon_by_gap, precision=2, with_dollar=False, negative_space=True)}\Delta E_0$"
        )
        wc_y_min, wc_y_max = min(wc_y_min, np.min(worst_pts)), max(wc_y_max, np.max(worst_pts))

        plt.close(fig_hm)

    if failed_load:
        print("There are some files failed to load. Skip the worst case plot.")
    else:
        # finalize worst-case figure
        ax_wc.axhline(1.0, color="black", linestyle="dashed", linewidth=0.5)
        ax_wc.set_xlim(0, inputs.max_epsilon_prime)
        ax_wc.set_ylim(wc_y_min / 2, wc_y_max * 2)
        ax_wc.set_yscale("log")
        ax_wc.set_xlabel(r"$\epsilon'$")
        ax_wc.set_ylabel(r"Worst $C_{\mathrm{FQPE}}/C_{\mathrm{QPE}}$")
        handles, labels = ax_wc.get_legend_handles_labels()
        ax_wc.legend(handles[::-1], labels[::-1])
        fig_wc.tight_layout()

        worst_out = paths.fig_dir / f"{inputs.mol_name}_worst_cost.png"
        fig_wc.savefig(worst_out)
        plt.close(fig_wc)


if __name__ == "__main__":
    for name, model in hubbard_examples.items():
        data = prepare_hamiltonian_refstates(**model)
        for basis_type in ["cheby", "trig"]:
            run(basis_type, name, data["transform"], data["tag"])
