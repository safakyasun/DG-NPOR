#!/usr/bin/env python3
"""Draw a detailed publication schematic of the self-configuring DG-NPOR model."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


INK = "#17212B"
MUTED = "#566471"
LIGHT_TEXT = "#6F7B85"
RULE = "#CAD2D9"
PANEL = "#FAFBFC"
BLUE = "#356FA8"
BLUE_LIGHT = "#EAF2FA"
PURPLE = "#6256A0"
PURPLE_LIGHT = "#F0EEFA"
ORANGE = "#C15D20"
ORANGE_LIGHT = "#FFF1E8"
TEAL = "#17776F"
TEAL_LIGHT = "#E8F6F4"
GOLD = "#D99A20"
SUPERVISION = "#A64B32"


def _rounded(ax, x, y, w, h, *, face=PANEL, edge=RULE, radius=0.010,
             linewidth=0.8, zorder=1):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.004,rounding_size={radius}",
        facecolor=face, edgecolor=edge, linewidth=linewidth, zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def _arrow(ax, start, end, *, color=INK, width=0.95, style="-",
           connection="arc3,rad=0", mutation=7.5, zorder=6):
    ax.add_patch(
        FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=mutation,
            linewidth=width, linestyle=style, color=color,
            connectionstyle=connection, shrinkA=1.2, shrinkB=1.2,
            zorder=zorder,
        )
    )


def _step_box(ax, x, y, w, h, number, title, accent, *, face=PANEL):
    _rounded(ax, x, y, w, h, face=face, edge=RULE, radius=0.009, linewidth=0.78)
    ax.plot([x, x + w], [y + h, y + h], color=accent, linewidth=2.15,
            solid_capstyle="butt", zorder=3)
    ax.add_patch(Circle((x + 0.016, y + h - 0.025), 0.0115,
                        facecolor=accent, edgecolor="none", zorder=4))
    ax.text(x + 0.016, y + h - 0.025, str(number), ha="center", va="center",
            fontsize=5.4, color="white", weight="bold", zorder=5)
    ax.text(x + 0.033, y + h - 0.025, title, ha="left", va="center",
            fontsize=6.6, color=INK, weight="bold", zorder=5)
    ax.plot([x + 0.011, x + w - 0.011], [y + h - 0.050, y + h - 0.050],
            color=RULE, linewidth=0.55, zorder=3)


def _row_header(ax, y, title, color, line_start):
    ax.text(0.025, y, title, ha="left", va="center", fontsize=7.2,
            color=color, weight="bold")
    ax.plot([line_start, 0.975], [y, y], color=color, linewidth=1.15,
            alpha=0.70, solid_capstyle="butt")


def _label_chip(ax, x, y, text, *, face, edge, color=INK, width=0.054):
    _rounded(ax, x, y, width, 0.026, face=face, edge=edge,
             radius=0.006, linewidth=0.65, zorder=4)
    ax.text(x + width / 2, y + 0.013, text, ha="center", va="center",
            fontsize=5.1, color=color, zorder=5)


def _matrix(ax, x, y, w, h, rows=5, cols=5, color=BLUE):
    for r in range(rows):
        for c in range(cols):
            value = 0.16 + 0.68 * ((3 * r + 2 * c) % 7) / 6.0
            ax.add_patch(Rectangle(
                (x + c * w / cols, y + (rows - 1 - r) * h / rows),
                0.92 * w / cols, 0.88 * h / rows,
                facecolor=mpl.colors.to_rgba(color, value), edgecolor="white",
                linewidth=0.25, zorder=3,
            ))


def _input_block(ax, x, y, w, h):
    _matrix(ax, x + 0.014, y + 0.078, 0.070, 0.085, rows=6, cols=5)
    ax.text(x + 0.045, y + 0.067, r"$X_{\rm trk}:N_{\rm trk}\times19$",
            ha="center", va="top", fontsize=5.0, color=BLUE)
    for i in range(6):
        ax.add_patch(Rectangle((x + 0.096, y + 0.082 + i * 0.012), 0.010, 0.008,
                               facecolor=INK if i < 4 else "white",
                               edgecolor=INK, linewidth=0.45, zorder=3))
    ax.text(x + 0.101, y + 0.067, "mask", ha="center", va="top",
            fontsize=5.0, color=MUTED)
    context_colors = ["#BFD9F0", "#CDE5D4", "#F4D9B9", "#E7C7D7"]
    for i, color in enumerate(context_colors):
        ax.add_patch(Circle((x + 0.135 + 0.016 * i, y + 0.126), 0.0068,
                            facecolor=color, edgecolor=INK, linewidth=0.45, zorder=3))
    ax.text(x + 0.159, y + 0.105, r"$j=[\log p_T,\eta,|\eta|,N_{\rm trk}]$",
            ha="center", va="top", fontsize=4.9, color=MUTED)
    ax.text(x + 0.014, y + 0.048,
            r"$N_{\rm trk}\leq40$; rank tracks by $|S_{d_0}|$",
            ha="left", va="center", fontsize=5.0, color=INK)
    ax.text(x + 0.014, y + 0.026,
            "19 features: impact - angle - momentum - uncertainty - hits",
            ha="left", va="center", fontsize=4.45, color=LIGHT_TEXT)
    ax.text(x + 0.014, y + 0.009,
            "physics transforms + robust scaling",
            ha="left", va="center", fontsize=4.55, color=LIGHT_TEXT)


def _relation_block(ax, x, y, w, h):
    ax.text(x + 0.012, y + h - 0.067,
            r"relations from $(\Delta\eta,\Delta\phi,S_{d_0},S_{z_0})$",
            ha="left", va="center", fontsize=5.1, color=MUTED)
    lanes = [
        (r"$H$", "mandatory residual", BLUE, "1"),
        (r"$A^sH$", "local kNN paths", BLUE, r"$g_s$"),
        (r"$GH$", "dense global broadcast", PURPLE, r"$g_G$"),
        (r"$H-GH$", "global residual", PURPLE, r"$g_R$"),
    ]
    y0 = y + 0.138
    for i, (symbol, label, color, gate) in enumerate(lanes):
        yy = y0 - i * 0.029
        _rounded(ax, x + 0.013, yy - 0.012, 0.044, 0.024,
                 face=mpl.colors.to_rgba(color, 0.10), edge=color,
                 radius=0.004, linewidth=0.55, zorder=3)
        ax.text(x + 0.035, yy, symbol, ha="center", va="center",
                fontsize=5.5, color=color)
        ax.text(x + 0.066, yy, label, ha="left", va="center",
                fontsize=4.65, color=INK)
        ax.plot([x + 0.159, x + 0.183], [yy, yy], color=color, linewidth=0.8)
        _rounded(ax, x + 0.184, yy - 0.010, 0.033, 0.020,
                 face="white", edge=color, radius=0.004, linewidth=0.55, zorder=3)
        ax.text(x + 0.2005, yy, gate, ha="center", va="center",
                fontsize=4.9, color=color)
    ax.text(x + 0.013, y + 0.015,
            r"hard-concrete gates + $\ell_0$; concatenate active branches",
            ha="left", va="center", fontsize=4.65, color=LIGHT_TEXT)


def _encoder_block(ax, x, y, w, h):
    for i in range(5):
        yy = y + 0.143 - i * 0.019
        ax.add_patch(Circle((x + 0.026, yy), 0.0062,
                            facecolor=BLUE_LIGHT, edgecolor=BLUE,
                            linewidth=0.55, zorder=3))
    ax.text(x + 0.026, y + 0.042, "shared tanh\nnode map", ha="center", va="center",
            fontsize=4.7, color=MUTED, linespacing=1.05)
    _arrow(ax, (x + 0.042, y + 0.112), (x + 0.070, y + 0.112), color=BLUE,
           width=0.70, mutation=6)
    for head in range(3):
        yy = y + 0.142 - head * 0.030
        ax.plot([x + 0.078, x + 0.104], [yy, y + 0.118], color=BLUE,
                linewidth=0.65, alpha=0.75)
        ax.add_patch(Circle((x + 0.078, yy), 0.0048,
                            facecolor=BLUE, edgecolor="none", zorder=3))
    ax.add_patch(Circle((x + 0.106, y + 0.118), 0.010,
                        facecolor=BLUE_LIGHT, edgecolor=BLUE,
                        linewidth=0.55, zorder=3))
    ax.text(x + 0.106, y + 0.047, "multi-head attention\n+ masked mean",
            ha="center", va="center", fontsize=4.65, color=MUTED, linespacing=1.05)
    _arrow(ax, (x + 0.120, y + 0.112), (x + 0.145, y + 0.112), color=BLUE,
           width=0.70, mutation=6)
    widths = [3, 4, 3]
    xcols = [x + 0.151, x + 0.170, x + 0.190]
    for col, count in enumerate(widths):
        ys = np.linspace(y + 0.085, y + 0.143, count)
        for yy in ys:
            ax.add_patch(Circle((xcols[col], yy), 0.0042,
                                facecolor="#BED3EA", edgecolor=BLUE,
                                linewidth=0.38, zorder=3))
        if col:
            prev = np.linspace(y + 0.085, y + 0.143, widths[col - 1])
            for yp in prev:
                for yc in ys:
                    ax.plot([xcols[col - 1] + 0.004, xcols[col] - 0.004], [yp, yc],
                            color=BLUE, linewidth=0.25, alpha=0.32, zorder=2)
    ax.text(x + 0.170, y + 0.049, r"geometry MLP $g_\psi$",
            ha="center", va="center", fontsize=4.8, color=INK)
    ax.text(x + 0.170, y + 0.033, "selected depth / width / heads",
            ha="center", va="center", fontsize=4.45, color=LIGHT_TEXT)
    _label_chip(ax, x + 0.010, y + 0.010, r"$y$", face=ORANGE_LIGHT,
                edge=SUPERVISION, color=SUPERVISION, width=0.026)
    ax.plot([x + 0.036, x + 0.076], [y + 0.023, y + 0.023],
            color=SUPERVISION, linewidth=0.65, linestyle="--")
    ax.text(x + 0.082, y + 0.023, "auxiliary BCE; head removed",
            ha="left", va="center", fontsize=4.45, color=SUPERVISION)


def _metric_block(ax, x, y, w, h):
    xnodes = np.linspace(x + 0.024, x + w - 0.024, 4)
    counts = [3, 5, 4, 3]
    for col, (xx, count) in enumerate(zip(xnodes, counts)):
        ys = np.linspace(y + 0.082, y + 0.148, count)
        for yy in ys:
            ax.add_patch(Circle((xx, yy), 0.0043,
                                facecolor=PURPLE_LIGHT if col in (1, 2) else BLUE_LIGHT,
                                edgecolor=PURPLE if col in (1, 2) else BLUE,
                                linewidth=0.42, zorder=3))
        if col:
            prev = np.linspace(y + 0.082, y + 0.148, counts[col - 1])
            for yp in prev:
                for yc in ys:
                    ax.plot([xnodes[col - 1] + 0.004, xx - 0.004], [yp, yc],
                            color=PURPLE, linewidth=0.22, alpha=0.25, zorder=2)
    ax.text(x + w / 2, y + 0.061,
            r"dimension-preserving tanh map $h_\theta$",
            ha="center", va="center", fontsize=4.9, color=INK)
    ax.text(x + w / 2, y + 0.042,
            r"$h_\theta:\;48\;\rightarrow\;96\;\rightarrow\;48$",
            ha="center", va="center", fontsize=5.2, color=PURPLE, weight="bold")
    ax.text(x + w / 2, y + 0.027,
            r"auxiliary BCE fits $h_\theta$; head removed",
            ha="center", va="center", fontsize=4.15, color=SUPERVISION)
    ax.text(x + w / 2, y + 0.011,
            r"standardize $u_i=h_\theta(e_i)\in\mathbb{R}^{48}$",
            ha="center", va="center", fontsize=4.55, color=INK)


def _mini_graph(ax, cx, cy, sx=1.0, sy=1.0, color=PURPLE):
    nodes = np.array([
        [-0.044, 0.010], [-0.026, 0.047], [0.006, 0.052], [0.043, 0.024],
        [-0.032, -0.035], [0.006, -0.018], [0.040, -0.046],
    ])
    edges = [(0, 1), (0, 4), (0, 5), (1, 2), (1, 5), (2, 3),
             (2, 5), (3, 5), (3, 6), (4, 5), (5, 6)]
    for i, j in edges:
        ax.plot([cx + sx * nodes[i, 0], cx + sx * nodes[j, 0]],
                [cy + sy * nodes[i, 1], cy + sy * nodes[j, 1]],
                color=color, linewidth=0.65, alpha=0.72, zorder=3)
    for i, (dx, dy) in enumerate(nodes):
        ax.add_patch(Circle((cx + sx * dx, cy + sy * dy), 0.0048,
                            facecolor=color if i in (0, 2, 5) else "white",
                            edgecolor=color, linewidth=0.55, zorder=4))


def _graph_block(ax, x, y, w, h):
    _mini_graph(ax, x + 0.054, y + 0.137, sx=0.78, sy=0.72)
    ax.text(x + 0.116, y + 0.145, "self-tuning kNN\nin learned $u$-space",
            ha="left", va="center", fontsize=4.7, color=INK, linespacing=1.08)
    ax.text(x + w / 2, y + 0.090,
            r"$w_{ij}=\exp[-\|u_i-u_j\|^2/(2\sigma_i\sigma_j)]$",
            ha="center", va="center", fontsize=4.65, color=PURPLE)
    ax.text(x + w / 2, y + 0.065, r"symmetrize $W$; $d_i=\sum_jw_{ij}$",
            ha="center", va="center", fontsize=4.6, color=MUTED)
    ax.text(x + w / 2, y + 0.039,
            r"reported: $k=32$, $k_\sigma=16$",
            ha="center", va="center", fontsize=4.7, color=LIGHT_TEXT)
    ax.text(x + w / 2, y + 0.016,
            "graph nodes are jets, not individual tracks",
            ha="center", va="center", fontsize=4.5, color=PURPLE, style="italic")


def _potential_block(ax, x, y, w, h):
    _label_chip(ax, x + 0.011, y + 0.145, r"labels $y_j$", face=ORANGE_LIGHT,
                edge=SUPERVISION, color=SUPERVISION, width=0.065)
    ax.text(x + 0.086, y + 0.158, r"$\pi_c=N_c/N$",
            ha="left", va="center", fontsize=4.65, color=MUTED)
    ax.text(x + 0.011, y + 0.116,
            r"$\widehat\eta_{ic}="
            r"\dfrac{\sum_{j\ne i}w_{ij}\mathbf{1}[y_j=c]+\lambda_0\pi_c}"
            r"{d_i+\lambda_0}$",
            ha="left", va="center", fontsize=5.1, color=INK)
    ax.text(x + 0.011, y + 0.073,
            r"$J_i=D_{\rm KL}(\widehat\eta_i\,\|\,\pi)$",
            ha="left", va="center", fontsize=5.2, color=PURPLE)
    ax.text(x + 0.011, y + 0.044,
            r"$B_\pi=\log(1/\min_c\pi_c)$",
            ha="left", va="center", fontsize=4.8, color=MUTED)
    ax.text(x + 0.011, y + 0.018,
            r"$V_{{\rm PI},i}=B_\pi-J_i\geq0$",
            ha="left", va="center", fontsize=5.2, color=PURPLE, weight="bold")


def _operator_block(ax, x, y, w, h):
    ax.text(x + 0.012, y + 0.152,
            r"$L_{\rm sym}=I-D^{-1/2}WD^{-1/2}$",
            ha="left", va="center", fontsize=5.2, color=INK)
    _rounded(ax, x + 0.012, y + 0.091, w - 0.024, 0.043,
             face=PURPLE_LIGHT, edge=PURPLE, radius=0.005,
             linewidth=0.65, zorder=3)
    ax.text(x + w / 2, y + 0.112,
            r"$H_N=L_{\rm sym}+\lambda_{\rm PI}\,\mathrm{diag}(V_{\rm PI})$",
            ha="center", va="center", fontsize=5.7, color=PURPLE, weight="bold")
    ax.text(x + 0.012, y + 0.065,
            r"solve $H_Nu_k=\epsilon_ku_k$ (smallest $\epsilon_k$)",
            ha="left", va="center", fontsize=5.0, color=INK)
    ax.text(x + 0.012, y + 0.038,
            r"recover graph eigenfunctions $\phi_k=D^{-1/2}u_k$",
            ha="left", va="center", fontsize=4.85, color=MUTED)
    ax.text(x + 0.012, y + 0.014,
            r"$\lambda_{\rm PI}$ chooses the predictive operator, not $K_{\rm DG}$",
            ha="left", va="center", fontsize=4.45, color=LIGHT_TEXT)


def _wave(ax, x, y, w, amp, cycles, color, linewidth=0.70):
    t = np.linspace(0.0, 1.0, 100)
    ax.plot(x + w * t, y + amp * np.sin(2 * np.pi * cycles * t),
            color=color, linewidth=linewidth, zorder=3)


def _orbital_block(ax, x, y, w, h):
    for i in range(4):
        yy = y + 0.150 - 0.024 * i
        _rounded(ax, x + 0.012, yy - 0.009, 0.060, 0.018,
                 face="white", edge=PURPLE, radius=0.003,
                 linewidth=0.45, zorder=2)
        _wave(ax, x + 0.017, yy, 0.050, 0.0045, 1.3 + 0.55 * i, PURPLE)
        ax.text(x + 0.078, yy, rf"$\phi_{i+1}$",
                ha="left", va="center", fontsize=4.65, color=PURPLE)
    ax.text(x + 0.050, y + 0.065, r"$\vdots$", ha="center", va="center",
            fontsize=6.0, color=PURPLE)
    ax.text(x + 0.106, y + 0.146,
            "stable low-frequency\npredictive support",
            ha="left", va="center", fontsize=4.7, color=INK, linespacing=1.08)
    ax.text(x + 0.106, y + 0.104,
            r"$\longrightarrow\;K_{\rm PF}$ candidate bank",
            ha="left", va="center", fontsize=4.8, color=PURPLE)
    ax.plot([x + 0.105, x + 0.202], [y + 0.088, y + 0.088],
            color=RULE, linewidth=0.55)
    ax.text(x + 0.012, y + 0.053,
            "for any jet: query affinities",
            ha="left", va="center", fontsize=4.55, color=MUTED)
    ax.text(x + 0.012, y + 0.036,
            "-> analytic orbital extension",
            ha="left", va="center", fontsize=4.55, color=MUTED)
    ax.text(x + 0.012, y + 0.015,
            r"$\Phi(x)=[\phi_1(x),\ldots,\phi_{K_{\rm PF}}(x)]$",
            ha="left", va="center", fontsize=5.25, color=PURPLE, weight="bold")


def _derivative_block(ax, x, y, w, h):
    ax.text(x + 0.012, y + 0.154,
            r"class-occupancy screen: $M_c^{\rm scr}\succeq0$, $\sum_cM_c^{\rm scr}=I$",
            ha="left", va="center", fontsize=4.85, color=INK)
    ax.text(x + 0.012, y + 0.123,
            r"candidate state $a(x)=\Phi(x)/\|\Phi(x)\|_2$",
            ha="left", va="center", fontsize=4.8, color=MUTED)
    _rounded(ax, x + 0.012, y + 0.065, w - 0.024, 0.048,
             face=ORANGE_LIGHT, edge=ORANGE, radius=0.005,
             linewidth=0.65, zorder=3)
    ax.text(x + w / 2, y + 0.096,
            r"$b(z)=\dfrac{z\odot a}{\|z\odot a\|_2}$",
            ha="center", va="center", fontsize=5.35, color=ORANGE, weight="bold")
    ax.text(x + w / 2, y + 0.075,
            r"$S_k=\left\langle(\partial J/\partial z_k)^2\right\rangle$",
            ha="center", va="center", fontsize=5.35, color=ORANGE, weight="bold")
    ax.text(x + 0.012, y + 0.044,
            "rank rotation-safe eigenspace blocks by derivative information",
            ha="left", va="center", fontsize=4.55, color=INK)
    ax.text(x + 0.012, y + 0.020,
            "retain the smallest stable set capturing derivative sensitivity",
            ha="left", va="center", fontsize=4.45, color=LIGHT_TEXT)


def _compact_block(ax, x, y, w, h):
    scores = [0.93, 0.77, 0.58, 0.18]
    for i, score in enumerate(scores):
        yy = y + 0.150 - 0.022 * i
        ax.plot([x + 0.035, x + 0.035 + 0.075 * score], [yy, yy],
                color=ORANGE if i < 3 else RULE,
                linewidth=3.0 if i < 3 else 2.0, solid_capstyle="butt")
        ax.text(x + 0.027, yy, rf"$\phi_{i+1}$", ha="right", va="center",
                fontsize=4.4, color=INK if i < 3 else LIGHT_TEXT)
    ax.text(x + w / 2, y + 0.064, r"$\mathcal{S}=\{1,2,3\},\quad K_{\rm DG}=3$",
            ha="center", va="center", fontsize=5.8, color=ORANGE, weight="bold")
    ax.plot([x + 0.010, x + w - 0.010], [y + 0.051, y + 0.051],
            color=RULE, linewidth=0.55)
    ax.text(x + w / 2, y + 0.032,
            r"$f(x)=\dfrac{[\phi_1,\phi_2,\phi_3]^\top}"
            r"{\sqrt{\phi_1^2+\phi_2^2+\phi_3^2}}\in\mathbb{R}^3$",
            ha="center", va="center", fontsize=4.8, color=INK)
    ax.text(x + w / 2, y + 0.009, "automatic state; no display projection",
            ha="center", va="center", fontsize=4.3, color=LIGHT_TEXT)


def _readout_block(ax, x, y, w, h):
    for c, yy in enumerate([y + 0.145, y + 0.105]):
        ax.add_patch(Rectangle((x + 0.013, yy - 0.013), 0.030, 0.026,
                               facecolor=TEAL_LIGHT, edgecolor=TEAL,
                               linewidth=0.55, zorder=3))
        ax.text(x + 0.028, yy, rf"$B_{c}$", ha="center", va="center",
                fontsize=4.8, color=TEAL)
        _arrow(ax, (x + 0.047, yy), (x + 0.068, yy), color=TEAL,
               width=0.60, mutation=5.5)
        ax.text(x + 0.073, yy, rf"$E_{c}=B_{c}^\top B_{c}+\rho I/2$",
                ha="left", va="center", fontsize=4.55, color=INK)
    ax.text(x + 0.013, y + 0.072,
            r"$S=E_0+E_1,\qquad M_c=S^{-1/2}E_cS^{-1/2}$",
            ha="left", va="center", fontsize=5.0, color=TEAL)
    ax.text(x + 0.013, y + 0.047,
            r"$M_c\succeq0,\qquad M_0+M_1=I$",
            ha="left", va="center", fontsize=5.15, color=INK, weight="bold")
    ax.text(x + 0.013, y + 0.021,
            r"$P(c\mid x)=f(x)^\top M_cf(x)$  (diagonal + cross-orbital terms)",
            ha="left", va="center", fontsize=4.65, color=TEAL)


def _decision_block(ax, x, y, w, h):
    x0, x1, yy = x + 0.020, x + w - 0.020, y + 0.132
    ax.plot([x0, x1], [yy, yy], color=RULE, linewidth=4.2, solid_capstyle="butt")
    ax.plot([x0, x0 + 0.72 * (x1 - x0)], [yy, yy], color=TEAL,
            linewidth=4.2, solid_capstyle="butt")
    ax.plot([x0 + 0.60 * (x1 - x0)] * 2, [yy - 0.017, yy + 0.017],
            color=INK, linewidth=0.75)
    ax.add_patch(Circle((x0 + 0.72 * (x1 - x0), yy), 0.006,
                        facecolor=GOLD, edgecolor=INK, linewidth=0.45, zorder=5))
    ax.text(x0, yy + 0.025, "0", ha="center", va="bottom", fontsize=4.4, color=MUTED)
    ax.text(x1, yy + 0.025, "1", ha="center", va="bottom", fontsize=4.4, color=MUTED)
    ax.text(x0 + 0.60 * (x1 - x0), yy - 0.025, r"$\tau$", ha="center", va="top",
            fontsize=4.8, color=INK)
    ax.text(x + w / 2, y + 0.086, r"$P_b=P(b\mid x)=f^\top M_bf$",
            ha="center", va="center", fontsize=5.25, color=TEAL, weight="bold")
    _rounded(ax, x + 0.014, y + 0.028, w - 0.028, 0.037,
             face=TEAL_LIGHT, edge=TEAL, radius=0.005,
             linewidth=0.65, zorder=3)
    ax.text(x + w / 2, y + 0.046,
            r"$P_b\geq\tau:\ b$ jet   |   $P_b<\tau:$ light jet",
            ha="center", va="center", fontsize=4.7, color=INK)
    ax.text(x + w / 2, y + 0.012, r"$\tau$ sets the operating point",
            ha="center", va="center", fontsize=4.25, color=LIGHT_TEXT)


def draw_workflow(output_base: Path, dpi: int = 600):
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
        "mathtext.fontset": "stixsans",
        "font.size": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })
    fig = plt.figure(figsize=(7.25, 6.15), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _arrow(ax, (0.665, 0.974), (0.705, 0.974), color=INK, width=0.85, mutation=6)
    ax.text(0.713, 0.974, "representation / inference path", ha="left", va="center",
            fontsize=4.8, color=MUTED)
    ax.plot([0.855, 0.895], [0.974, 0.974], color=SUPERVISION,
            linewidth=0.8, linestyle="--")
    ax.text(0.903, 0.974, "fit-only labels", ha="left", va="center",
            fontsize=4.8, color=SUPERVISION)

    _row_header(ax, 0.947, "I. RECONSTRUCTED JET -> SELF-CONFIGURING NEURAL GEOMETRY", BLUE, 0.535)
    _row_header(ax, 0.637, "II. PREDICTIVE ORBITAL CONSTRUCTION", PURPLE, 0.410)
    _row_header(ax, 0.327, "III. AUTOMATIC COMPRESSION -> STRUCTURED CLASS DECISION", ORANGE, 0.555)

    y1, y2, y3, h = 0.690, 0.380, 0.070, 0.225

    _step_box(ax, 0.025, y1, 0.205, h, 1, "Reconstructed inputs", BLUE, face=BLUE_LIGHT)
    _input_block(ax, 0.025, y1, 0.205, h)
    _step_box(ax, 0.255, y1, 0.240, h, 2, "Physics relation branches", BLUE, face=BLUE_LIGHT)
    _relation_block(ax, 0.255, y1, 0.240, h)
    _step_box(ax, 0.520, y1, 0.235, h, 3, "Track-set encoder", BLUE, face=BLUE_LIGHT)
    _encoder_block(ax, 0.520, y1, 0.235, h)
    _step_box(ax, 0.780, y1, 0.195, h, 4, "Task-aware metric", PURPLE, face=PURPLE_LIGHT)
    _metric_block(ax, 0.780, y1, 0.195, h)

    _arrow(ax, (0.230, y1 + 0.112), (0.252, y1 + 0.112), color=BLUE)
    ax.text(0.241, y1 + 0.127, "scaled", ha="center", va="bottom", fontsize=4.1, color=MUTED)
    _arrow(ax, (0.495, y1 + 0.112), (0.517, y1 + 0.112), color=BLUE)
    ax.text(0.506, y1 + 0.127, r"$\oplus$", ha="center", va="bottom", fontsize=4.8, color=BLUE)
    _arrow(ax, (0.755, y1 + 0.112), (0.777, y1 + 0.112), color=PURPLE)
    ax.text(0.766, y1 + 0.127, r"$e\in\mathbb{R}^{48}$", ha="center", va="bottom",
            fontsize=4.25, color=PURPLE)

    _step_box(ax, 0.780, y2, 0.195, h, 5, "Jet neighborhood graph", PURPLE, face=PURPLE_LIGHT)
    _graph_block(ax, 0.780, y2, 0.195, h)
    _step_box(ax, 0.535, y2, 0.220, h, 6, "Local predictive information", PURPLE, face=PURPLE_LIGHT)
    _potential_block(ax, 0.535, y2, 0.220, h)
    _step_box(ax, 0.270, y2, 0.240, h, 7, "Self-adjoint POR operator", PURPLE, face=PURPLE_LIGHT)
    _operator_block(ax, 0.270, y2, 0.240, h)
    _step_box(ax, 0.025, y2, 0.220, h, 8, "Predictive orbital bank", PURPLE, face=PURPLE_LIGHT)
    _orbital_block(ax, 0.025, y2, 0.220, h)

    _arrow(ax, (0.952, y1 - 0.004), (0.952, y2 + h + 0.004), color=PURPLE)
    ax.text(0.938, 0.652, r"$u_i\in\mathbb{R}^{48}$", ha="right", va="center",
            fontsize=4.4, color=PURPLE)
    _arrow(ax, (0.780, y2 + 0.112), (0.758, y2 + 0.112), color=PURPLE)
    _arrow(ax, (0.535, y2 + 0.112), (0.513, y2 + 0.112), color=PURPLE)
    _arrow(ax, (0.270, y2 + 0.112), (0.248, y2 + 0.112), color=PURPLE)

    _step_box(ax, 0.025, y3, 0.260, h, 9, "Analytic derivative gate", ORANGE, face=ORANGE_LIGHT)
    _derivative_block(ax, 0.025, y3, 0.260, h)
    _step_box(ax, 0.310, y3, 0.175, h, 10, "Selected state", ORANGE, face=ORANGE_LIGHT)
    _compact_block(ax, 0.310, y3, 0.175, h)
    _step_box(ax, 0.510, y3, 0.275, h, 11, "Constrained PSD measurement", TEAL, face=TEAL_LIGHT)
    _readout_block(ax, 0.510, y3, 0.275, h)
    _step_box(ax, 0.810, y3, 0.165, h, 12, "Prediction", TEAL, face=TEAL_LIGHT)
    _decision_block(ax, 0.810, y3, 0.165, h)

    _arrow(ax, (0.048, y2 - 0.004), (0.048, y3 + h + 0.004), color=ORANGE)
    ax.text(0.059, 0.342, r"$\Phi(x)$", ha="left", va="center", fontsize=4.6, color=ORANGE)
    _arrow(ax, (0.285, y3 + 0.112), (0.307, y3 + 0.112), color=ORANGE)
    _arrow(ax, (0.485, y3 + 0.112), (0.507, y3 + 0.112), color=TEAL)
    _arrow(ax, (0.785, y3 + 0.112), (0.807, y3 + 0.112), color=TEAL)

    ax.text(0.025, 0.024,
            "Solid arrows define the deployed mapping from reconstructed observables to P(b|x). "
            "Auxiliary neural scores supervise geometry learning but are not fused into the DG-NPOR probability.",
            ha="left", va="center", fontsize=4.55, color=MUTED)

    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.025)
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.025)
    fig.savefig(output_base.with_suffix(".png"), dpi=int(dpi), bbox_inches="tight",
                pad_inches=0.025, facecolor="white")
    plt.close(fig)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-base", default="paper/figures/dg_npor_workflow",
                        help="Output path without an extension.")
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    draw_workflow(Path(args.output_base).expanduser().resolve(), dpi=args.dpi)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
