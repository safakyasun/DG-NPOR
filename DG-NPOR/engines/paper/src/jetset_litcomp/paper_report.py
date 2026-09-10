"""LaTeX tables, macros, and result prose generated from locked test numbers."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from .paper_inputs import PARTICLENET_METHOD, POR_METHOD


def latex_escape(value):
    text = str(value)
    replacements = (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
    )
    for source, target in replacements:
        text = text.replace(source, target)
    return text


def _interval(value, lower, upper, digits=3):
    return f"{value:.{digits}f} [{lower:.{digits}f}, {upper:.{digits}f}]"


def _finite(value, digits=1):
    return r"$\infty$" if not np.isfinite(value) else f"{value:.{digits}f}"


def _rejection_interval(row, digits=1):
    center = _finite(row["light_rejection"], digits)
    lower = _finite(row["light_rejection_lower_68"], digits)
    upper = _finite(row["light_rejection_upper_68"], digits)
    return f"{center} [{lower}, {upper}]"


def make_main_table(method_metrics, working_points, dimensions):
    selected = working_points[
        working_points["method"].isin([POR_METHOD, PARTICLENET_METHOD])
        & working_points["target_b_efficiency"].isin([0.70, 0.77])
    ].copy()
    pivot = selected.pivot(index="method", columns="target_b_efficiency")
    rows = []
    metric_index = method_metrics.set_index("method")
    for method in (POR_METHOD, PARTICLENET_METHOD):
        metric = metric_index.loc[method]
        wp70 = selected[(selected["method"] == method) & np.isclose(selected["target_b_efficiency"], 0.70)].iloc[0]
        wp77 = selected[(selected["method"] == method) & np.isclose(selected["target_b_efficiency"], 0.77)].iloc[0]
        rows.append(
            {
                "method": method,
                "decision_dimension": int(dimensions[method]),
                "auc": float(metric["roc_auc"]),
                "auc_lower_68": float(metric["auc_lower_68"]),
                "auc_upper_68": float(metric["auc_upper_68"]),
                "r_light_70": float(wp70["light_rejection"]),
                "r_light_70_lower_68": float(wp70["light_rejection_lower_68"]),
                "r_light_70_upper_68": float(wp70["light_rejection_upper_68"]),
                "r_light_77": float(wp77["light_rejection"]),
                "r_light_77_lower_68": float(wp77["light_rejection_lower_68"]),
                "r_light_77_upper_68": float(wp77["light_rejection_upper_68"]),
                "log_loss": float(metric["log_loss"]),
                "average_precision": float(metric["average_precision"]),
            }
        )
    return pd.DataFrame(rows)


def main_table_latex(frame):
    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Method & Decision dim. & AUC & $R_{\mathrm{light}}$ at $70\%$ $\epsilon_b$ & $R_{\mathrm{light}}$ at $77\%$ $\epsilon_b$ \\",
        r"\midrule",
    ]
    for row in frame.to_dict(orient="records"):
        values = (
            latex_escape(row["method"]),
            str(int(row["decision_dimension"])),
            _interval(row["auc"], row["auc_lower_68"], row["auc_upper_68"], 4),
            _rejection_interval(
                {
                    "light_rejection": row["r_light_70"],
                    "light_rejection_lower_68": row["r_light_70_lower_68"],
                    "light_rejection_upper_68": row["r_light_70_upper_68"],
                },
                1,
            ),
            _rejection_interval(
                {
                    "light_rejection": row["r_light_77"],
                    "light_rejection_lower_68": row["r_light_77_lower_68"],
                    "light_rejection_upper_68": row["r_light_77_upper_68"],
                },
                1,
            ),
        )
        lines.append(" & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def make_audit_table(method_metrics, working_points, audit_methods):
    metric_index = method_metrics.set_index("method")
    rows = []
    for method in audit_methods:
        metric = metric_index.loc[method]
        wp = working_points[
            (working_points["method"] == method)
            & np.isclose(working_points["target_b_efficiency"], 0.70)
        ].iloc[0]
        rows.append(
            {
                "method": method,
                "comparison_status": "same jet audit only",
                "auc": float(metric["roc_auc"]),
                "auc_lower_68": float(metric["auc_lower_68"]),
                "auc_upper_68": float(metric["auc_upper_68"]),
                "light_rejection": float(wp["light_rejection"]),
                "light_rejection_lower_68": float(wp["light_rejection_lower_68"]),
                "light_rejection_upper_68": float(wp["light_rejection_upper_68"]),
            }
        )
    return pd.DataFrame(rows)


def audit_table_latex(frame):
    lines = [
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Audit score & AUC & $R_{\mathrm{light}}$ at $70\%$ $\epsilon_b$ \\",
        r"\midrule",
    ]
    for row in frame.to_dict(orient="records"):
        values = (
            latex_escape(row["method"]),
            _interval(row["auc"], row["auc_lower_68"], row["auc_upper_68"], 4),
            _rejection_interval(row, 1),
        )
        lines.append(" & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def paper_macros(main_table, locked_rows, locked_events):
    rows = main_table.set_index("method")
    por = rows.loc[POR_METHOD]
    particle = rows.loc[PARTICLENET_METHOD]
    compression = float(particle["decision_dimension"] / por["decision_dimension"])
    values = {
        "LockedTestJets": f"{int(locked_rows):,}",
        "LockedTestEvents": f"{int(locked_events):,}",
        "PorDecisionDimension": str(int(por["decision_dimension"])),
        "ParticleNetDecisionDimension": str(int(particle["decision_dimension"])),
        "DecisionCompressionFactor": f"{compression:.1f}",
        "PorAUC": f"{por['auc']:.4f}",
        "ParticleNetAUC": f"{particle['auc']:.4f}",
        "PorRLightSeventy": _finite(por["r_light_70"], 1).replace("$", ""),
        "ParticleNetRLightSeventy": _finite(particle["r_light_70"], 1).replace("$", ""),
        "PorRLightSeventySeven": _finite(por["r_light_77"], 1).replace("$", ""),
        "ParticleNetRLightSeventySeven": _finite(particle["r_light_77"], 1).replace("$", ""),
    }
    return "\n".join(
        rf"\newcommand{{\{name}}}{{{value}}}" for name, value in values.items()
    ) + "\n"


def results_text(main_table, paired_differences, locked_rows, locked_events):
    rows = main_table.set_index("method")
    por = rows.loc[POR_METHOD]
    particle = rows.loc[PARTICLENET_METHOD]
    compression = particle["decision_dimension"] / por["decision_dimension"]
    dimension = int(por["decision_dimension"])
    delta_row = paired_differences[
        paired_differences["method"] == PARTICLENET_METHOD
    ].iloc[0]
    delta = float(delta_row["auc_difference_method_minus_reference"])
    if delta >= 0:
        comparison = (
            f"ParticleNet gives the larger AUC by {delta:.4f}, while DG NPOR "
            f"restricts the final decision to {dimension} explicit coordinates."
        )
    else:
        comparison = (
            f"DG NPOR gives the larger AUC by {abs(delta):.4f} while using only "
            f"{dimension} explicit decision coordinates."
        )
    return (
        "\\paragraph{Locked test performance.} "
        f"The event disjoint locked test contains {int(locked_rows):,} jets from "
        f"{int(locked_events):,} events. DG NPOR obtains an AUC of "
        f"{por['auc']:.4f} with a 68\\% event bootstrap interval of "
        f"[{por['auc_lower_68']:.4f}, {por['auc_upper_68']:.4f}], and a light jet "
        f"rejection of {_finite(por['r_light_70'], 1)} at 70\\% bottom jet efficiency. "
        f"The matched input ParticleNet reference obtains an AUC of "
        f"{particle['auc']:.4f} and a light jet rejection of "
        f"{_finite(particle['r_light_70'], 1)} at the same efficiency. {comparison} "
        f"Its 256 dimensional final hidden representation is {compression:.1f} times "
        f"larger than the {dimension} dimensional DG NPOR readout. The reported dimension "
        "comparison concerns the states directly available to the respective output "
        "layers and does not equate parameter count with representation dimension.\n"
    )


def write_text(path, content):
    Path(path).write_text(content, encoding="utf-8")
