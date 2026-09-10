"""Publication figures for the compact DG NPOR result section."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import roc_points
from .orbital_analysis import FEATURE_GROUPS, ORBITAL_COLUMNS
from .paper_inputs import PARTICLENET_METHOD, POR_METHOD


COLORS = {
    POR_METHOD: "#0072B2",
    PARTICLENET_METHOD: "#D55E00",
    "ATLAS GN2v01 audit": "#009E73",
    "ATLAS DL1dv01 audit": "#7A5195",
}


def configure_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.labelsize": 9.0,
            "axes.titlesize": 9.5,
            "legend.fontsize": 7.8,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.8,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )


def _save(fig, base_path):
    base = Path(base_path)
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".pdf"))
    fig.savefig(base.with_suffix(".png"), dpi=320)
    plt.close(fig)


def _thin(frame, maximum=2400):
    if len(frame) <= maximum:
        return frame
    index = np.unique(np.linspace(0, len(frame) - 1, maximum).astype(int))
    return frame.iloc[index]


def save_performance_dimension_figure(y_true, scores, main_table, output_base):
    """Two panel paper figure: rejection curve and decision dimension."""
    configure_style()
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.15, 3.05),
        gridspec_kw={"width_ratios": (1.55, 1.0)},
        constrained_layout=True,
    )
    curve_axis, dimension_axis = axes
    finite_values = []
    table = main_table.set_index("method")
    for method in (POR_METHOD, PARTICLENET_METHOD):
        frame = _thin(roc_points(y_true, scores[method]))
        selected = frame[
            frame["b_efficiency"].between(0.50, 0.90)
            & np.isfinite(frame["light_rejection"])
            & (frame["light_rejection"] > 0)
        ]
        finite_values.extend(selected["light_rejection"].tolist())
        curve_axis.plot(
            selected["b_efficiency"],
            selected["light_rejection"],
            color=COLORS[method],
            label=f"{method}, AUC {table.loc[method, 'auc']:.4f}",
        )
        for efficiency, column in ((0.70, "r_light_70"), (0.77, "r_light_77")):
            value = float(table.loc[method, column])
            if np.isfinite(value):
                curve_axis.scatter(
                    [efficiency],
                    [value],
                    s=24,
                    color=COLORS[method],
                    edgecolor="white",
                    linewidth=0.6,
                    zorder=4,
                )
    curve_axis.axvline(0.70, color="#666666", ls=":", lw=1.0)
    curve_axis.axvline(0.77, color="#999999", ls=":", lw=1.0)
    curve_axis.set_yscale("log")
    curve_axis.set_xlim(0.50, 0.90)
    if finite_values:
        lower = max(1.0, float(np.nanpercentile(finite_values, 1)) * 0.8)
        upper = max(10.0, float(np.nanpercentile(finite_values, 99.5)) * 1.35)
        curve_axis.set_ylim(lower, upper)
    curve_axis.set_xlabel(r"Bottom jet efficiency $\epsilon_b$")
    curve_axis.set_ylabel(r"Light jet rejection $1/\epsilon_{\mathrm{light}}$")
    curve_axis.grid(True, which="both", color="#dddddd", lw=0.6)
    curve_axis.legend(loc="best", frameon=False)
    curve_axis.set_title("a  Locked test discrimination", loc="left", fontweight="bold")

    methods = [PARTICLENET_METHOD, POR_METHOD]
    dimensions = [int(table.loc[method, "decision_dimension"]) for method in methods]
    positions = np.arange(len(methods))
    bars = dimension_axis.barh(
        positions,
        dimensions,
        color=[COLORS[method] for method in methods],
        height=0.52,
    )
    dimension_axis.set_xscale("log")
    dimension_axis.set_xlim(1.0, max(dimensions) * 2.2)
    dimension_axis.set_yticks(positions, ["ParticleNet", "DG NPOR"])
    dimension_axis.set_xlabel("State dimension available to output layer")
    dimension_axis.grid(True, axis="x", which="both", color="#dddddd", lw=0.6)
    dimension_axis.set_axisbelow(True)
    for bar, dimension in zip(bars, dimensions):
        dimension_axis.text(
            dimension * 1.12,
            bar.get_y() + bar.get_height() / 2,
            str(dimension),
            va="center",
            ha="left",
            fontweight="bold",
        )
    factor = dimensions[0] / dimensions[1]
    dimension_axis.text(
        0.97,
        0.94,
        f"Dimension ratio {factor:.1f}",
        transform=dimension_axis.transAxes,
        ha="right",
        va="top",
        color=COLORS[POR_METHOD],
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5},
    )
    dimension_axis.set_title("b  Explicit decision state", loc="left", fontweight="bold")
    _save(fig, output_base)


def save_all_score_roc(y_true, main_scores, audit_scores, metric_table, output_base):
    """Supplementary same jet ROC with audit scores visually distinguished."""
    configure_style()
    fig, axis = plt.subplots(figsize=(5.2, 3.8), constrained_layout=True)
    auc = metric_table.set_index("method")["roc_auc"].to_dict()
    combined = {**main_scores, **audit_scores}
    for method, score in combined.items():
        frame = _thin(roc_points(y_true, score))
        is_audit = method in audit_scores
        axis.plot(
            frame["light_efficiency"],
            frame["b_efficiency"],
            color=COLORS[method],
            ls="--" if is_audit else "-",
            alpha=0.75 if is_audit else 1.0,
            label=f"{method}, AUC {auc[method]:.4f}",
        )
    axis.set_xscale("log")
    axis.set_xlim(1e-5, 1.0)
    axis.set_ylim(0.0, 1.01)
    axis.set_xlabel(r"Light jet efficiency $\epsilon_{\mathrm{light}}$")
    axis.set_ylabel(r"Bottom jet efficiency $\epsilon_b$")
    axis.grid(True, which="both", color="#dddddd", lw=0.6)
    axis.legend(frameon=False, loc="lower right")
    axis.set_title("Common ATLAS JetSet locked test")
    _save(fig, output_base)


def save_orbital_content_figure(analysis, output_base):
    """Class distributions and grouped reconstructed feature associations."""
    configure_style()
    merged = analysis["merged"]
    representatives = analysis["group_representatives"]
    orbital_columns = analysis.get("orbital_columns", list(ORBITAL_COLUMNS))
    count = len(orbital_columns)
    ncols = min(3, count)
    nrows = (count + ncols - 1) // ncols
    fig = plt.figure(figsize=(7.15, 2.2 * nrows + max(2.0, 0.25 * count)), constrained_layout=True)
    grid = fig.add_gridspec(nrows + 1, ncols, height_ratios=[1.05] * nrows + [max(0.95, 0.12 * count)])
    for index, orbital in enumerate(orbital_columns):
        axis = fig.add_subplot(grid[index // ncols, index % ncols])
        values = merged[orbital].to_numpy(float)
        lower, upper = np.quantile(values, [0.005, 0.995])
        if upper <= lower:
            lower, upper = lower - 0.5, upper + 0.5
        bins = np.linspace(lower, upper, 45)
        for label, name, color in (
            (0, "Light", "#666666"),
            (1, "Bottom", COLORS[POR_METHOD]),
        ):
            selected = merged.loc[merged["y_true_light0_b1"] == label, orbital]
            axis.hist(
                selected,
                bins=bins,
                density=True,
                histtype="step",
                color=color,
                lw=1.6,
                label=name,
            )
        axis.set_xlabel(rf"Selected coordinate $z_{{{index + 1}}}$")
        if index == 0:
            axis.set_ylabel("Density")
            axis.legend(frameon=False)
        axis.grid(True, color="#eeeeee", lw=0.5)
        axis.set_title(chr(ord("a") + index), loc="left", fontweight="bold")

    heat_axis = fig.add_subplot(grid[nrows, :])
    groups = list(FEATURE_GROUPS)
    matrix = np.zeros((count, len(groups)), dtype=float)
    labels = np.empty(matrix.shape, dtype=object)
    for row, orbital in enumerate(orbital_columns):
        for column, group in enumerate(groups):
            selected = representatives[
                (representatives["orbital"] == orbital)
                & (representatives["feature_group"] == group)
            ].iloc[0]
            matrix[row, column] = selected["spearman_rho"]
            labels[row, column] = f"{selected['spearman_rho']:+.2f}"
    image = heat_axis.imshow(matrix, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")
    heat_axis.set_xticks(np.arange(len(groups)), groups)
    heat_axis.set_yticks(np.arange(count), [rf"$z_{{{i + 1}}}$" for i in range(count)])
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            heat_axis.text(
                column,
                row,
                labels[row, column],
                ha="center",
                va="center",
                color="white" if abs(matrix[row, column]) > 0.5 else "black",
                fontsize=8,
            )
    colorbar = fig.colorbar(image, ax=heat_axis, pad=0.02, fraction=0.035)
    colorbar.set_label(r"Representative Spearman $\rho$")
    heat_axis.set_title(
        "Strongest association within each reconstructed feature group",
        loc="left",
        fontweight="bold",
    )
    _save(fig, output_base)


def save_particle_training_figure(history_path, output_base):
    path = Path(history_path)
    if not path.exists():
        return False
    history = pd.read_csv(path)
    required = {
        "epoch",
        "train_batch_loss",
        "validation_auc",
        "validation_log_loss",
    }
    if not required.issubset(history.columns):
        return False
    configure_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.25), constrained_layout=True)
    specifications = (
        ("train_batch_loss", "Training batch loss"),
        ("validation_auc", "Validation AUC"),
        ("validation_log_loss", "Validation log loss"),
    )
    for axis, (column, label) in zip(axes, specifications):
        axis.plot(history["epoch"], history[column], marker="o", ms=2.8, color=COLORS[PARTICLENET_METHOD])
        axis.set_xlabel("Epoch")
        axis.set_ylabel(label)
        axis.grid(True, color="#eeeeee", lw=0.5)
    _save(fig, output_base)
    return True
