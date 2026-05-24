#!/usr/bin/env python
"""Generate Fig 2, Fig 6, Fig 8 and Fig 11 in the style of
Ponalagusamy & Priyadharshini (Applied Mathematics and Computation, 2018).

Outputs are written to ``results/figs/fig{2,6,8,11}.png`` (and an
overview combining all four).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon
from matplotlib.lines import Line2D

# ---- import path bootstrap ----
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
sys.path.insert(0, _root)
os.chdir(_root)

from src.geometry import (
    paper_parent_radius,
    paper_axial_grid,
    PAPER_REPORT_Z,
    PAPER_STENOSIS_CENTER,
    PAPER_STENOSIS_LENGTH,
)
from src.casson_axial import (
    SteadyParams,
    axial_sweep,
    calibrate_G,
    PAPER_BASELINE_WSS,
    PAPER_BASELINE_M,
    PAPER_BASELINE_Da,
)


FIG_DIR = Path("results/figs")
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# Calibration cache so we only run the bisection once
# ----------------------------------------------------------------------

_G_CACHE: dict[tuple[float, int], float] = {}


def get_G(theta: float = 0.05, Nr: int = 201) -> float:
    key = (round(theta, 6), Nr)
    if key not in _G_CACHE:
        print(f"  calibrating G (theta={theta}, Nr={Nr}) ...", flush=True)
        _G_CACHE[key] = calibrate_G(theta=theta, Nr=Nr)
        print(f"  G = {_G_CACHE[key]:.6f}", flush=True)
    return _G_CACHE[key]


# ----------------------------------------------------------------------
# Fig 2 -- Schematic of the bifurcated artery with parent stenosis
# ----------------------------------------------------------------------

def fig2_schematic(save_path: Path = FIG_DIR / "fig2_schematic.png") -> None:
    """Geometry schematic: parent vessel + cosine stenosis + 2 daughter
    branches at half-angle ``beta/2``."""

    delta_s = 0.2
    beta_half_deg = 30.0
    beta_half = np.deg2rad(beta_half_deg)

    # Parent grid (half-domain since axisymmetric -> mirror)
    z_p = np.linspace(0.0, 3.6, 400)
    R_p = paper_parent_radius(z_p, delta_s)

    # Daughter axes start at the bifurcation (z = z_b, on the axis)
    z_b = z_p[-1]
    L_d = 2.0  # daughter length (visualization only)
    s_d = np.linspace(0.0, L_d, 200)
    kappa = 0.7  # daughter radius ratio
    R_d = np.full_like(s_d, kappa)

    # Daughter centerline endpoints (rotated by +/- beta/2 from +z axis)
    e_axis = np.array([np.cos(beta_half), np.sin(beta_half)])
    n_perp = np.array([-np.sin(beta_half), np.cos(beta_half)])

    fig, ax = plt.subplots(figsize=(11, 5))

    # Parent walls (top + bottom)
    ax.plot(z_p, R_p, "k-", linewidth=1.8)
    ax.plot(z_p, -R_p, "k-", linewidth=1.8)
    ax.fill_between(z_p, -R_p, R_p, color="lightskyblue", alpha=0.35,
                    edgecolor="none")

    # Daughter branches (upper & lower)
    for sign in (+1, -1):
        e = np.array([np.cos(sign * beta_half), np.sin(sign * beta_half)])
        n = np.array([-np.sin(sign * beta_half), np.cos(sign * beta_half)])
        center = np.array([z_b, 0.0])[:, None] + np.outer(e, s_d)
        upper = center + np.outer(n, R_d)
        lower = center - np.outer(n, R_d)
        ax.plot(upper[0], upper[1], "k-", linewidth=1.8)
        ax.plot(lower[0], lower[1], "k-", linewidth=1.8)
        # Fill
        verts = np.column_stack(
            [np.r_[upper[0], lower[0, ::-1]],
             np.r_[upper[1], lower[1, ::-1]]]
        )
        ax.add_patch(Polygon(verts, closed=True, facecolor="lightskyblue",
                             alpha=0.35, edgecolor="none"))

    # Centerline
    ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)

    # Annotate parent radius R0 at left inlet
    ax.annotate("", xy=(0.05, 1.0), xytext=(0.05, 0.0),
                arrowprops=dict(arrowstyle="<->", color="k"))
    ax.text(0.10, 0.55, r"$R_0$", fontsize=13)

    # Stenosis annotations
    z_c = PAPER_STENOSIS_CENTER
    L_s = PAPER_STENOSIS_LENGTH
    ax.annotate("", xy=(z_c, 1.0), xytext=(z_c, 1 - delta_s),
                arrowprops=dict(arrowstyle="<->", color="red"))
    ax.text(z_c + 0.05, 1 - delta_s / 2, r"$\delta_s$",
            color="red", fontsize=13)
    # Stenosis extent bracket
    ax.annotate("", xy=(z_c - L_s / 2, -1.25), xytext=(z_c + L_s / 2, -1.25),
                arrowprops=dict(arrowstyle="<->", color="darkgreen"))
    ax.text(z_c, -1.42, r"$L_s$", color="darkgreen", fontsize=13,
            ha="center")

    # Bifurcation half-angle
    arc_r = 0.5
    theta_arc = np.linspace(0, beta_half, 30)
    ax.plot(z_b + arc_r * np.cos(theta_arc), arc_r * np.sin(theta_arc),
            "purple", linewidth=1.5)
    ax.text(z_b + 0.55, 0.18, r"$\beta/2$", color="purple", fontsize=13)

    # Daughter radius
    s_mark = 1.2
    pt_center = np.array([z_b, 0.0]) + s_mark * e_axis
    pt_wall = pt_center + kappa * n_perp
    ax.annotate("", xy=tuple(pt_wall), xytext=tuple(pt_center),
                arrowprops=dict(arrowstyle="->", color="k"))
    ax.text(pt_center[0] + 0.05, pt_center[1] + 0.20,
            r"$\kappa R_0$", fontsize=12)

    # Axes
    ax.set_xlim(-0.15, z_b + L_d * np.cos(beta_half) + 0.2)
    ax.set_ylim(-1.7, 2.2)
    ax.set_aspect("equal")
    ax.set_xlabel("axial distance  $z$", fontsize=12)
    ax.set_ylabel("$r$", fontsize=12)
    ax.set_title("Fig. 2 -- Schematic of the bifurcated stenosed artery "
                 r"($\delta_s=0.2$, $\beta/2=30^\circ$)", fontsize=13)
    ax.grid(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {save_path}")


# ----------------------------------------------------------------------
# Fig 6 -- Axial WSS for several stenotic heights delta_s
# ----------------------------------------------------------------------

def fig6_wss_vs_z_delta(save_path: Path = FIG_DIR / "fig6_wss_vs_z_delta.png",
                        theta: float = 0.05,
                        Nr: int = 201) -> None:

    G = get_G(theta=theta, Nr=Nr)
    p = SteadyParams(M=PAPER_BASELINE_M, Da=PAPER_BASELINE_Da,
                     theta=theta, G=G)
    z = paper_axial_grid()

    delta_list = [0.0, 0.05, 0.10, 0.15, 0.20]
    colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(delta_list)))

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for delta_s, c in zip(delta_list, colors):
        res = axial_sweep(delta_s, p, z=z, Nr=Nr)
        ax.plot(res.z, res.tau_w, "-", color=c, linewidth=2.0,
                label=fr"$\delta_s = {delta_s:.2f}$")

    # Mark paper's report z's
    res020 = axial_sweep(0.20, p, z=PAPER_REPORT_Z, Nr=Nr)
    ax.plot(res020.z, res020.tau_w, "ko", markerfacecolor="white",
            markersize=7, label="report points")

    ax.axvspan(PAPER_STENOSIS_CENTER - PAPER_STENOSIS_LENGTH / 2,
               PAPER_STENOSIS_CENTER + PAPER_STENOSIS_LENGTH / 2,
               color="grey", alpha=0.07, label="stenotic region")
    ax.set_xlabel(r"axial distance  $z$", fontsize=12)
    ax.set_ylabel(r"wall shear stress  $\tau_w$", fontsize=12)
    ax.set_title("Fig. 6 -- Effect of stenotic height $\\delta_s$ on WSS\n"
                 fr"($M={p.M:g}$, $Da={p.Da:g}$, $\theta={theta:g}$)",
                 fontsize=12)
    ax.legend(loc="lower center", ncol=3, fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {save_path}")


# ----------------------------------------------------------------------
# Fig 8 -- Axial WSS for several Darcy numbers
# ----------------------------------------------------------------------

def fig8_wss_vs_z_Da(save_path: Path = FIG_DIR / "fig8_wss_vs_z_Da.png",
                     theta: float = 0.05,
                     Nr: int = 201) -> None:

    G = get_G(theta=theta, Nr=Nr)
    z = paper_axial_grid()
    delta_s = 0.2

    Da_list = [np.inf, 1.0, 0.5, 0.1]
    Da_labels = [r"$Da=\infty$", r"$Da=1.0$", r"$Da=0.5$", r"$Da=0.1$"]
    colors = plt.cm.plasma(np.linspace(0.10, 0.85, len(Da_list)))

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for Da, lbl, c in zip(Da_list, Da_labels, colors):
        p = SteadyParams(M=PAPER_BASELINE_M, Da=Da, theta=theta, G=G)
        res = axial_sweep(delta_s, p, z=z, Nr=Nr)
        ax.plot(res.z, res.tau_w, "-", color=c, linewidth=2.0, label=lbl)

    ax.axvspan(PAPER_STENOSIS_CENTER - PAPER_STENOSIS_LENGTH / 2,
               PAPER_STENOSIS_CENTER + PAPER_STENOSIS_LENGTH / 2,
               color="grey", alpha=0.07)
    ax.set_xlabel(r"axial distance  $z$", fontsize=12)
    ax.set_ylabel(r"wall shear stress  $\tau_w$", fontsize=12)
    ax.set_title("Fig. 8 -- Effect of Darcy number $Da$ on WSS\n"
                 fr"($M={PAPER_BASELINE_M:g}$, $\delta_s={delta_s:g}$, "
                 fr"$\theta={theta:g}$)", fontsize=12)
    ax.legend(loc="lower center", ncol=4, fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {save_path}")


# ----------------------------------------------------------------------
# Fig 11 -- Flow resistance Lambda vs delta_s, parametrised by M
# ----------------------------------------------------------------------

def fig11_resistance_vs_delta(save_path: Path = FIG_DIR / "fig11_resistance.png",
                              theta: float = 0.05,
                              Nr: int = 201) -> None:

    G = get_G(theta=theta, Nr=Nr)
    z = paper_axial_grid(n_z=81, z_min=0.0, z_max=3.2)

    delta_list = np.linspace(0.0, 0.30, 7)
    M_list = [0.0, 1.0, 2.0, 3.0]
    Da = PAPER_BASELINE_Da

    # Flow resistance over the stenotic segment z in [z_c - L_s/2, z_c + L_s/2]:
    #   Lambda = (G * L_s) / Q_min   (using the minimum cross-sectional Q)
    # which approaches G*L_s / Q(R=1) for delta_s -> 0.
    #
    # We report Lambda normalised by its value at delta_s=0 so the curves
    # all start at 1 and rise with stenosis severity.

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for M_h in M_list:
        p = SteadyParams(M=M_h, Da=Da, theta=theta, G=G)
        Lambda = []
        for d in delta_list:
            res = axial_sweep(d, p, z=z, Nr=Nr)
            Q_min = float(np.min(res.Q))
            Lambda.append(G * PAPER_STENOSIS_LENGTH / max(Q_min, 1e-12))
        Lambda = np.asarray(Lambda)
        Lambda /= Lambda[0]  # normalise by no-stenosis value
        ax.plot(delta_list, Lambda, "o-", linewidth=2.0, markersize=7,
                label=fr"$M = {M_h:g}$")

    ax.set_xlabel(r"stenotic height  $\delta_s$", fontsize=12)
    ax.set_ylabel(r"normalised flow resistance  "
                  r"$\Lambda(\delta_s)/\Lambda(0)$", fontsize=12)
    ax.set_title("Fig. 11 -- Flow resistance vs stenotic height\n"
                 fr"($Da={Da:g}$, $\theta={theta:g}$)", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {save_path}")


# ----------------------------------------------------------------------
# Combined overview
# ----------------------------------------------------------------------

def overview(save_path: Path = FIG_DIR / "fig_overview.png") -> None:
    images = ["fig2_schematic.png", "fig6_wss_vs_z_delta.png",
              "fig8_wss_vs_z_Da.png", "fig11_resistance.png"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    for ax, name in zip(axes.flat, images):
        path = FIG_DIR / name
        if not path.exists():
            ax.set_visible(False)
            continue
        img = plt.imread(path)
        ax.imshow(img)
        ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {save_path}")


def main():
    print("Generating paper figures (Fig 2, 6, 8, 11)...")
    fig2_schematic()
    fig6_wss_vs_z_delta()
    fig8_wss_vs_z_Da()
    fig11_resistance_vs_delta()
    overview()
    print("All figures saved under", FIG_DIR.resolve())


if __name__ == "__main__":
    main()
