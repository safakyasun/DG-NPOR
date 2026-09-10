#!/usr/bin/env python3
"""Create a publication quality comparison of inputs and the final K=3 state.

The upper row shows every pair among three physics motivated reconstructed
observables.  The lower row shows every pair among the three exact normalized
orbital coordinates supplied to the frozen structured PSD measurement.  Class
conditional 68 and 95 percent density contours make overlap and tails readable
without a perspective dependent three dimensional view.  No generic
dimensionality reduction is used.

This script is read only with respect to the fitted method.  It requires that
the locked test has already been scored and never refits the encoder, POR
operator, derivative gate, or PSD measurement.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

CLASS_STYLE = {
    0: {"name": "Light jets", "color": "#0072B2"},
    1: {"name": r"$b$ jets", "color": "#D55E00"},
}

PHYSICAL_INPUT_NAMES = (
    "log1p_max_abs_d0_significance",
    "log1p_max_abs_z0_sin_theta_significance",
    "number_of_reconstructed_tracks",
)

PHYSICAL_INPUT_LABELS = (
    r"$D_{d_0}$",
    r"$D_{z_0}$",
    r"$N_{\mathrm{trk}}$",
)

PAIR_INDICES = ((0, 1), (0, 2), (1, 2))
CONTOUR_MASSES = (0.68, 0.95)
DENSITY_GRID_BINS = 90
PHYSICAL_CONTINUOUS_SMOOTHING = 2.0
PHYSICAL_DISCRETE_SMOOTHING = 3.2
ORBITAL_SMOOTHING = 2.4


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Plot the reconstructed input distribution and the exact final "
            "three dimensional DG-NPOR orbital state."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "output_dir",
        help="Frozen self configuring result directory containing the joblib bundle.",
    )
    parser.add_argument("--h5-file", required=True, help="Official ATLAS JetSet HDF5 file.")
    parser.add_argument(
        "--figure-dir",
        default=None,
        help="Destination; default OUTPUT_DIR/paper_representation_figure.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=12000,
        help="Maximum class balanced number of locked test jets displayed.",
    )
    parser.add_argument("--read-batch-size", type=int, default=1024)
    parser.add_argument("--query-batch-size", type=int, default=1024)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args(argv)


def _resolve(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_scored_bundle(output_dir):
    status_path = output_dir / "test_scoring_status.json"
    bundle_path = output_dir / "trained_self_configuring_dg_npor_v9.joblib"
    if not status_path.exists():
        raise FileNotFoundError(status_path)
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if not bool(status.get("scored", False)):
        raise RuntimeError(
            "The locked test has not been scored. Run the official locked test "
            "scorer before creating a test distribution figure."
        )
    if not bundle_path.exists():
        raise FileNotFoundError(bundle_path)
    bundle = joblib.load(bundle_path)
    if not bool(bundle.get("method_resolved_before_test", False)):
        raise RuntimeError("The frozen self configuring method is unresolved.")
    return bundle


def _verify_frozen_run(bundle, h5_file):
    from por_hep.atlas_jetset import verify_official_file
    from por_hep.structured_tracks import load_atlas_jetset_index
    from run_atlas_jetset_por import workspace_source_sha256

    integrity = verify_official_file(h5_file, tier=bundle["dataset_tier"])
    if h5_file.name != bundle["official_file_name"]:
        raise RuntimeError("JetSet filename differs from the frozen run.")
    if int(integrity["size_bytes"]) != int(bundle["official_file_size"]):
        raise RuntimeError("JetSet size differs from the frozen run.")
    if integrity["adler32"] != bundle["official_file_adler32"]:
        raise RuntimeError("JetSet Adler 32 differs from the frozen run.")
    if workspace_source_sha256() != bundle["source_tree_sha256"]:
        raise RuntimeError("Workspace source SHA256 differs from the frozen run.")

    dataset = load_atlas_jetset_index(
        h5_file,
        sample=int(bundle["sample"]),
        random_state=int(bundle["random_state"]),
        label_definition=bundle["label_definition"],
        protocol=bundle["analysis_protocol"],
        max_source_events=int(bundle["max_source_events"]),
    )
    if not np.array_equal(dataset.row_index, bundle["sampled_source_rows"]):
        raise RuntimeError("Sampled source rows differ from the frozen run.")
    if not np.array_equal(dataset.group_id, bundle["sampled_event_numbers"]):
        raise RuntimeError("Sampled event numbers differ from the frozen run.")
    return dataset


def _balanced_subset(indices, labels, maximum, random_state):
    indices = np.asarray(indices, dtype=int)
    labels = np.asarray(labels, dtype=int)
    if maximum < 2:
        raise ValueError("max-points must be at least two.")
    rng = np.random.default_rng(int(random_state))
    per_class = max(1, int(maximum) // 2)
    selected = []
    for label in (0, 1):
        candidates = indices[labels == label]
        if len(candidates) == 0:
            raise RuntimeError("The locked test does not contain both classes.")
        take = min(per_class, len(candidates))
        selected.append(rng.choice(candidates, size=take, replace=False))
    result = np.concatenate(selected)
    return result[rng.permutation(len(result))]


def _masked_max(values, mask):
    safe = np.where(np.asarray(mask, dtype=bool), values, -np.inf)
    result = np.max(safe, axis=1)
    result[~np.isfinite(result)] = 0.0
    return result


def _physical_input_coordinates(tracks, mask, jet_context, field_names=None):
    """Return three reconstructed observables without fitting a projection.

    H5StructuredTrackSource has already applied signed log1p to both impact
    parameter significances.  Their absolute values are therefore
    log(1 + |significance|), which keeps long detector tails readable while
    remaining exactly aligned with the transformed quantities seen by the
    encoder.  Track multiplicity is the untruncated valid track count stored in
    the fourth jet context coordinate.
    """
    if field_names is None:
        from por_hep.atlas_jetset import RECONSTRUCTED_TRACK_INPUT_FIELDS

        field_names = RECONSTRUCTED_TRACK_INPUT_FIELDS
    names = tuple(field_names)
    d0_index = names.index("lifetimeSignedD0Significance")
    z0_index = names.index("lifetimeSignedZ0SinThetaSignificance")
    d0 = _masked_max(np.abs(tracks[:, :, d0_index]), mask)
    z0 = _masked_max(np.abs(tracks[:, :, z0_index]), mask)
    tracks_total = np.asarray(jet_context[:, 3], dtype=float)
    coordinates = np.column_stack([d0, z0, tracks_total]).astype(np.float32)
    if not np.isfinite(coordinates).all():
        raise RuntimeError("Physical input coordinates contain nonfinite values.")
    return coordinates


def _transform_model_in_batches(model, adapter, geometry, batch_size):
    states = []
    size = max(1, int(batch_size))
    for start in range(0, len(geometry), size):
        model_input = adapter.transform(geometry[start : start + size])
        states.append(model.transform(model_input))
    return np.vstack(states)


def _paper_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 9.5,
            "legend.fontsize": 8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.linewidth": 0.75,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "pdf.compression": 9,
        }
    )


def _class_and_contour_handles(labels):
    from matplotlib.lines import Line2D

    labels = np.asarray(labels, dtype=int)
    class_handles = [
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker="o",
            markersize=5,
            markerfacecolor=CLASS_STYLE[label]["color"],
            markeredgewidth=0,
            label=(
                f'{CLASS_STYLE[label]["name"]}  '
                f'(n={int(np.sum(labels == label)):,})'
            ),
        )
        for label in (0, 1)
    ]
    contour_handles = [
        Line2D([0], [0], color="0.25", linewidth=1.35, linestyle="-", label="68% HDR"),
        Line2D([0], [0], color="0.25", linewidth=0.85, linestyle="--", label="95% HDR"),
    ]
    return class_handles + contour_handles


def _class_handles(labels):
    return _class_and_contour_handles(labels)[:2]


def _robust_limits(coordinates, bounded=False):
    limits = []
    for column in np.asarray(coordinates, dtype=float).T:
        if bounded:
            low = float(np.min(column))
            high = float(np.max(column))
        else:
            low, high = np.quantile(column, [0.005, 0.995])
            low = max(0.0, float(low))
            high = float(high)
        span = high - low
        if not np.isfinite(span) or span <= 0:
            span = max(abs(low), 1.0)
        pad = 0.035 * span
        if bounded:
            limits.append((max(-1.02, low - pad), min(1.02, high + pad)))
        else:
            limits.append((max(0.0, low - pad), high + pad))
    return limits


def _density_grid(
    x,
    y,
    x_limit,
    y_limit,
    bins=DENSITY_GRID_BINS,
    smoothing=(PHYSICAL_CONTINUOUS_SMOOTHING, PHYSICAL_CONTINUOUS_SMOOTHING),
):
    from scipy.ndimage import gaussian_filter

    density, x_edges, y_edges = np.histogram2d(
        x,
        y,
        bins=int(bins),
        range=[x_limit, y_limit],
    )
    density = gaussian_filter(density.astype(float), sigma=smoothing, mode="nearest")
    x_centres = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centres = 0.5 * (y_edges[:-1] + y_edges[1:])
    return x_centres, y_centres, density


def _density_threshold(density, enclosed_mass):
    values = np.sort(np.asarray(density, dtype=float).ravel())[::-1]
    total = float(np.sum(values))
    if total <= 0 or not np.isfinite(total):
        return None
    cumulative = np.cumsum(values) / total
    index = min(int(np.searchsorted(cumulative, float(enclosed_mass))), len(values) - 1)
    threshold = float(values[index])
    return threshold if threshold > 0 and np.isfinite(threshold) else None


def _pairwise_panel(
    axis,
    coordinates,
    labels,
    pair,
    limits,
    axis_labels,
    seed,
    discrete_indices=(),
    continuous_smoothing=PHYSICAL_CONTINUOUS_SMOOTHING,
    discrete_smoothing=PHYSICAL_DISCRETE_SMOOTHING,
):
    from matplotlib.ticker import MaxNLocator

    x_index, y_index = pair
    x = np.asarray(coordinates[:, x_index], dtype=float)
    y = np.asarray(coordinates[:, y_index], dtype=float)
    labels = np.asarray(labels, dtype=int)
    order = np.random.default_rng(int(seed)).permutation(len(labels))
    colors = np.asarray(
        [CLASS_STYLE[int(label)]["color"] for label in labels], dtype=object
    )
    axis.scatter(
        x[order],
        y[order],
        c=colors[order],
        s=2.2,
        alpha=0.055,
        linewidths=0,
        rasterized=True,
    )

    for label in (0, 1):
        keep = labels == label
        smoothing = (
            discrete_smoothing
            if x_index in discrete_indices
            else continuous_smoothing,
            discrete_smoothing
            if y_index in discrete_indices
            else continuous_smoothing,
        )
        x_centres, y_centres, density = _density_grid(
            x[keep],
            y[keep],
            limits[x_index],
            limits[y_index],
            smoothing=smoothing,
        )
        thresholds = {
            mass: _density_threshold(density, mass) for mass in CONTOUR_MASSES
        }
        for mass, linestyle, linewidth in ((0.95, "--", 0.85), (0.68, "-", 1.35)):
            threshold = thresholds[mass]
            if threshold is None or threshold >= float(np.max(density)):
                continue
            axis.contour(
                x_centres,
                y_centres,
                density.T,
                levels=[threshold],
                colors=[CLASS_STYLE[label]["color"]],
                linestyles=[linestyle],
                linewidths=[linewidth],
                zorder=4,
            )

    axis.set_xlim(*limits[x_index])
    axis.set_ylim(*limits[y_index])
    axis.set_xlabel(axis_labels[x_index])
    axis.set_ylabel(axis_labels[y_index])
    axis.xaxis.set_major_locator(MaxNLocator(4))
    axis.yaxis.set_major_locator(MaxNLocator(4))
    axis.grid(True, color="#E7E7E7", linewidth=0.5)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.set_box_aspect(1.0)


def _make_figure(physical_input, orbital_state, labels, orbital_numbers):
    _paper_style()
    figure = plt.figure(figsize=(7.2, 5.35))
    grid = figure.add_gridspec(
        5,
        3,
        height_ratios=(0.13, 0.12, 1.0, 0.15, 1.0),
        hspace=0.42,
        wspace=0.38,
        left=0.085,
        right=0.985,
        bottom=0.09,
        top=0.985,
    )

    legend_axis = figure.add_subplot(grid[0, :])
    legend_axis.axis("off")
    legend_axis.legend(
        handles=_class_and_contour_handles(labels),
        loc="center",
        ncol=4,
        frameon=False,
        handlelength=2.0,
        columnspacing=1.6,
        borderaxespad=0.0,
    )

    physical_title = figure.add_subplot(grid[1, :])
    physical_title.axis("off")
    physical_title.text(
        0.0,
        0.48,
        "(a) Reconstructed observables",
        ha="left",
        va="center",
        fontsize=9.5,
    )

    physical_limits = _robust_limits(physical_input, bounded=False)
    for column, pair in enumerate(PAIR_INDICES):
        axis = figure.add_subplot(grid[2, column])
        _pairwise_panel(
            axis,
            physical_input,
            labels,
            pair,
            physical_limits,
            PHYSICAL_INPUT_LABELS,
            seed=2026082900 + column,
            discrete_indices=(2,),
        )

    title_bottom = figure.add_subplot(grid[3, :])
    title_bottom.axis("off")
    title_bottom.text(
        0.0,
        0.36,
        rf"(b) Normalized DG-NPOR state  ($K_{{\mathrm{{DG}}}}=3$)",
        ha="left",
        va="center",
        fontsize=9.5,
    )

    orbital_labels = tuple(rf"$\phi_{{{number}}}$" for number in orbital_numbers)
    orbital_limits = _robust_limits(orbital_state, bounded=True)
    for column, pair in enumerate(PAIR_INDICES):
        axis = figure.add_subplot(grid[4, column])
        _pairwise_panel(
            axis,
            orbital_state,
            labels,
            pair,
            orbital_limits,
            orbital_labels,
            seed=2026082910 + column,
            continuous_smoothing=ORBITAL_SMOOTHING,
        )
    return figure, physical_limits, orbital_limits


def _style_3d_axis(axis, limits, axis_labels, box_aspect, elev=22, azim=-55):
    from matplotlib.ticker import MaxNLocator

    axis.set_xlim(*limits[0])
    axis.set_ylim(*limits[1])
    axis.set_zlim(*limits[2])
    # Native mplot3d label placement is view- and backend-dependent.  Place the
    # compact symbols in axes coordinates so PDF and PNG exports agree.
    axis.set_xlabel("")
    axis.set_ylabel("")
    axis.set_zlabel("")
    axis.text2D(
        0.82, -0.055, axis_labels[0], transform=axis.transAxes,
        ha="center", va="top", fontsize=9,
    )
    axis.text2D(
        0.22, -0.035, axis_labels[1], transform=axis.transAxes,
        ha="center", va="top", fontsize=9,
    )
    axis.text2D(
        1.085, 0.52, axis_labels[2], transform=axis.transAxes,
        ha="left", va="center", rotation=90, fontsize=9,
    )
    axis.xaxis.set_major_locator(MaxNLocator(4))
    axis.yaxis.set_major_locator(MaxNLocator(4))
    axis.zaxis.set_major_locator(MaxNLocator(4))
    axis.tick_params(axis="both", which="major", pad=0)
    axis.view_init(elev=float(elev), azim=float(azim))
    axis.set_proj_type("persp")
    axis.set_box_aspect(box_aspect)
    for pane in (axis.xaxis.pane, axis.yaxis.pane, axis.zaxis.pane):
        pane.set_facecolor((0.975, 0.975, 0.975, 1.0))
        pane.set_edgecolor((0.86, 0.86, 0.86, 1.0))
    for coordinate_axis in (axis.xaxis, axis.yaxis, axis.zaxis):
        coordinate_axis._axinfo["grid"].update(
            {"color": (0.82, 0.82, 0.82, 1.0), "linewidth": 0.55}
        )


def _scatter_3d(axis, coordinates, labels, seed):
    labels = np.asarray(labels, dtype=int)
    order = np.random.default_rng(int(seed)).permutation(len(labels))
    colors = np.asarray(
        [CLASS_STYLE[int(label)]["color"] for label in labels], dtype=object
    )
    axis.scatter(
        coordinates[order, 0],
        coordinates[order, 1],
        coordinates[order, 2],
        c=colors[order],
        s=1.7,
        alpha=0.11,
        linewidths=0,
        depthshade=False,
        rasterized=True,
    )


def _make_3d_figure(physical_input, orbital_state, labels, orbital_numbers):
    _paper_style()
    figure = plt.figure(figsize=(7.2, 3.75))
    physical_axis = figure.add_subplot(1, 2, 1, projection="3d")
    orbital_axis = figure.add_subplot(1, 2, 2, projection="3d")
    figure.subplots_adjust(
        left=0.025, right=0.975, bottom=0.105, top=0.84, wspace=0.06
    )
    figure.legend(
        handles=_class_handles(labels),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=2,
        frameon=False,
        columnspacing=2.2,
        handletextpad=0.7,
    )

    physical_limits = _robust_limits(physical_input, bounded=False)
    _scatter_3d(physical_axis, physical_input, labels, seed=2026083001)
    _style_3d_axis(
        physical_axis,
        physical_limits,
        PHYSICAL_INPUT_LABELS,
        box_aspect=(1.0, 1.0, 0.82),
        elev=18,
        azim=35,
    )
    physical_axis.set_title(
        "(a) Reconstructed observables", loc="left", pad=5, fontsize=9.5
    )

    orbital_labels = tuple(rf"$\phi_{{{number}}}$" for number in orbital_numbers)
    orbital_limits = _robust_limits(orbital_state, bounded=True)
    _scatter_3d(orbital_axis, orbital_state, labels, seed=2026083002)
    _style_3d_axis(
        orbital_axis,
        orbital_limits,
        orbital_labels,
        box_aspect=(1.0, 1.0, 1.0),
        elev=18,
        azim=35,
    )
    orbital_axis.set_title(
        rf"(b) Normalized DG-NPOR state  ($K_{{\mathrm{{DG}}}}=3$)",
        loc="left",
        pad=5,
        fontsize=9.5,
    )
    return figure, physical_limits, orbital_limits


def _write_caption(path, count, orbital_numbers):
    orbitals = ", ".join(rf"phi_{number}" for number in orbital_numbers)
    caption = (
        "Class conditional representations of a balanced subset of "
        f"{count:,} locked test jets. Panel (a) shows three reconstructed input "
        "observables motivated by displaced track structure. We define "
        "D_q = log[1 + max_i |S_{q,i}|] for q in {d0, z0}, where i indexes "
        "tracks and S denotes the corresponding signed impact parameter "
        "significance, together with reconstructed track "
        "multiplicity. For legibility, the physical axes span the central 99 "
        "percent of the displayed sample; all selected jets remain in the "
        "archived coordinate file. Panel (b) shows the exact normalized three dimensional "
        "state supplied to the frozen PSD measurement through all three pairwise "
        "orbital planes, formed from the derivative selected orbitals "
        f"{orbitals}. Solid and dashed curves delineate the 68 and 95 percent "
        "class conditional highest density regions (HDRs), respectively. No "
        "PCA, generic embedding, "
        "additional projection, or model refitting is used."
    )
    path.write_text(caption + "\n", encoding="utf-8")


def _write_3d_caption(path, count, orbital_numbers):
    orbitals = ", ".join(rf"phi_{number}" for number in orbital_numbers)
    caption = (
        "Fixed-view three dimensional perspective projections of a balanced subset of "
        f"{count:,} locked test jets. Panel (a) shows the reconstructed "
        "observables D_q = log[1 + max_i |S_{q,i}|] for q in {d0, z0} and "
        "track multiplicity N_trk. Panel (b) shows the exact unit normalized "
        f"DG-NPOR state formed from the selected orbitals {orbitals}. The "
        "fixed camera orientation is used for qualitative visualization of the "
        "manifold; apparent overlap depends on the viewing direction. No PCA, generic embedding, "
        "or model refitting is used."
    )
    path.write_text(caption + "\n", encoding="utf-8")


def main(argv=None):
    args = parse_args(argv)
    from por_hep.structured_tracks import H5StructuredTrackSource

    output_dir = _resolve(args.output_dir)
    h5_file = _resolve(args.h5_file)
    figure_dir = (
        _resolve(args.figure_dir)
        if args.figure_dir is not None
        else output_dir / "paper_representation_figure"
    )
    if not h5_file.exists():
        raise FileNotFoundError(h5_file)
    figure_dir.mkdir(parents=True, exist_ok=True)

    bundle = _load_scored_bundle(output_dir)
    dataset = _verify_frozen_run(bundle, h5_file)
    model = bundle["neural_por_model"]
    encoder = bundle["structured_track_encoder"]
    adapter = bundle["input_feature_adapter"]

    selected_k = int(model.selected_k_)
    if selected_k != 3:
        raise RuntimeError(
            f"This paper figure requires K_DG=3, but the frozen model has K_DG={selected_k}."
        )
    orbital_numbers = [int(value) + 1 for value in model.selected_orbital_indices_]
    if len(orbital_numbers) != 3:
        raise RuntimeError("The frozen model does not contain exactly three selected orbitals.")

    locked_test = np.asarray(bundle["independent_test_indices"], dtype=int)
    locked_labels = dataset.y[locked_test]
    plot_indices = _balanced_subset(
        locked_test, locked_labels, args.max_points, args.random_state
    )
    plot_labels = dataset.y[plot_indices]

    with H5StructuredTrackSource(
        h5_file,
        dataset.row_index,
        max_tracks=int(bundle["max_tracks"]),
    ) as source:
        physical_input_parts = []
        geometry_parts = []
        read_size = max(1, int(args.read_batch_size))
        for start in range(0, len(plot_indices), read_size):
            tracks, mask, jet_context = source.read(
                plot_indices[start : start + read_size]
            )
            physical_input_parts.append(
                _physical_input_coordinates(tracks, mask, jet_context)
            )
            geometry, _ = encoder.transform_arrays(tracks, mask, jet_context)
            geometry_parts.append(geometry)

    physical_input = np.vstack(physical_input_parts)
    encoded_geometry = np.vstack(geometry_parts)
    orbital_state = _transform_model_in_batches(
        model,
        adapter,
        encoded_geometry,
        args.query_batch_size,
    )
    orbital_state = np.asarray(orbital_state, dtype=float)
    if orbital_state.shape != (len(plot_indices), 3):
        raise RuntimeError(
            f"Unexpected final state shape {orbital_state.shape}; expected "
            f"({len(plot_indices)}, 3)."
        )
    if not np.isfinite(orbital_state).all():
        raise RuntimeError("The final orbital state contains nonfinite values.")
    norms = np.linalg.norm(orbital_state, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-6, rtol=1e-6):
        raise RuntimeError("The selected orbital state is not unit normalized.")

    figure, physical_limits, orbital_limits = _make_figure(
        physical_input,
        orbital_state,
        plot_labels,
        orbital_numbers,
    )
    pairwise_pdf_path = figure_dir / "dg_npor_input_vs_k3_orbitals.pdf"
    pairwise_png_path = figure_dir / "dg_npor_input_vs_k3_orbitals.png"
    figure.savefig(pairwise_pdf_path, bbox_inches="tight", pad_inches=0.08)
    figure.savefig(
        pairwise_png_path,
        dpi=int(args.dpi),
        bbox_inches="tight",
        pad_inches=0.08,
    )
    plt.close(figure)

    figure_3d, _, _ = _make_3d_figure(
        physical_input,
        orbital_state,
        plot_labels,
        orbital_numbers,
    )
    three_d_pdf_path = figure_dir / "dg_npor_input_vs_k3_orbitals_3d.pdf"
    three_d_png_path = figure_dir / "dg_npor_input_vs_k3_orbitals_3d.png"
    figure_3d.savefig(three_d_pdf_path, bbox_inches="tight", pad_inches=0.04)
    figure_3d.savefig(
        three_d_png_path,
        dpi=int(args.dpi),
        bbox_inches="tight",
        pad_inches=0.04,
    )
    plt.close(figure_3d)

    np.savez_compressed(
        figure_dir / "dg_npor_input_vs_k3_coordinates.npz",
        source_row=dataset.row_index[plot_indices],
        event_number=dataset.group_id[plot_indices],
        y_true_light0_b1=plot_labels,
        physical_input_coordinates=physical_input,
        physical_input_names=np.asarray(PHYSICAL_INPUT_NAMES),
        normalized_orbital_state=orbital_state,
        selected_orbitals_one_based=np.asarray(orbital_numbers, dtype=int),
    )
    manifest = {
        "figure": "dg_npor_input_vs_k3_orbitals",
        "figure_files": {
            "pairwise_pdf": pairwise_pdf_path.name,
            "pairwise_png": pairwise_png_path.name,
            "fixed_view_3d_pdf": three_d_pdf_path.name,
            "fixed_view_3d_png": three_d_png_path.name,
        },
        "evaluation_role": "already_scored_locked_test",
        "balanced_visualization_subset": True,
        "number_of_displayed_jets": int(len(plot_indices)),
        "class_counts_light0_b1": {
            "0": int(np.sum(plot_labels == 0)),
            "1": int(np.sum(plot_labels == 1)),
        },
        "input_panel_observables": list(PHYSICAL_INPUT_NAMES),
        "input_panel_projection": "none; three reconstructed physical observables",
        "display_style": "all pairwise planes with class conditional density contours",
        "contour_enclosed_masses": list(CONTOUR_MASSES),
        "density_estimator": "Gaussian-smoothed two-dimensional histogram",
        "density_grid_bins_per_axis": int(DENSITY_GRID_BINS),
        "physical_continuous_grid_smoothing_sigma": float(
            PHYSICAL_CONTINUOUS_SMOOTHING
        ),
        "physical_discrete_grid_smoothing_sigma": float(
            PHYSICAL_DISCRETE_SMOOTHING
        ),
        "orbital_grid_smoothing_sigma": float(ORBITAL_SMOOTHING),
        "physical_axis_limits": [list(values) for values in physical_limits],
        "orbital_axis_limits": [list(values) for values in orbital_limits],
        "physical_axis_limit_quantiles": [0.005, 0.995],
        "PCA_used": False,
        "generic_embedding_used_for_figure": False,
        "encoder_geometry_dimension": int(encoded_geometry.shape[1]),
        "final_orbital_dimension": int(orbital_state.shape[1]),
        "selected_orbitals_one_based": orbital_numbers,
        "final_panel_projection": "none; exact normalized selected orbital state",
        "orbital_norm_min": float(np.min(norms)),
        "orbital_norm_max": float(np.max(norms)),
        "random_state_for_display_subset": int(args.random_state),
        "model_refit": False,
        "encoder_refit": False,
        "operator_refit": False,
        "derivative_gate_refit": False,
        "PSD_measurement_refit": False,
    }
    (figure_dir / "dg_npor_input_vs_k3_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    _write_caption(
        figure_dir / "dg_npor_input_vs_k3_caption.txt",
        len(plot_indices),
        orbital_numbers,
    )
    _write_3d_caption(
        figure_dir / "dg_npor_input_vs_k3_3d_caption.txt",
        len(plot_indices),
        orbital_numbers,
    )

    print("Paper representation figure completed")
    print(f"Displayed jets: {len(plot_indices):,}")
    print(f"Selected orbitals, one based: {orbital_numbers}")
    print(f"Pairwise PDF: {pairwise_pdf_path.resolve()}")
    print(f"Pairwise PNG: {pairwise_png_path.resolve()}")
    print(f"Fixed-view 3D PDF: {three_d_pdf_path.resolve()}")
    print(f"Fixed-view 3D PNG: {three_d_png_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
