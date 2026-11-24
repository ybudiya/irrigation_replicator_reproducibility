#!/usr/bin/env python3
###############################################################################
# surface_with_heatmap.py
# Critical relative-yield reduction heatmaps
# Author: Yoshi Budiyanto, Arizona State University
# SPDX-FileCopyrightText: 2022-2026 Yoshi Budiyanto
# SPDX-License-Identifier: MIT
###############################################################################

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR.parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

KAPPA_G = 1.0
C_I_MIN = 1.0
C_I_MAX = 12.0
RHO_0_EVAL_MIN = 1.0
RHO_0_MAX = 2000.0
E_I_MIN = 1.0
E_I_MAX = 3.0
HEATMAP_N_GRID = 241
DPI = 600

BASELINE_OMEGA = 1000.0
BASELINE_NU = 100.0
BASELINE_B = 0.50

HEATMAP_COLOR_MIN = 0.0
HEATMAP_COLOR_MAX = 1.375
HEATMAP_COLOR_LEVELS = np.linspace(HEATMAP_COLOR_MIN, HEATMAP_COLOR_MAX, 65)
HEATMAP_COLOR_TICKS = np.linspace(HEATMAP_COLOR_MIN, HEATMAP_COLOR_MAX, 6)

def heatmap_color_mapping():

    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_over("#67001f")
    return Normalize(HEATMAP_COLOR_MIN, HEATMAP_COLOR_MAX, clip=False), cmap

def critical_relative_yield(
    c_i: np.ndarray,
    rho_0: np.ndarray,
    *,
    omega: float,
    nu: float,
    b: float,
) -> np.ndarray:

    if np.any(rho_0 <= 0):
        raise ValueError("rho_0 must be strictly positive for evaluation.")
    if nu <= 0:
        raise ValueError("nu must be positive.")
    if omega < 0:
        raise ValueError("omega must be nonnegative.")
    if b <= 0:
        raise ValueError("b must be positive.")

    kappa_i = c_i * KAPPA_G
    return 1.0 + ((kappa_i - KAPPA_G) / rho_0) * ((nu + omega) ** b)

def save_pdf_eps(fig, filename_stem: str) -> None:

    pdf_path = OUTPUT_DIR / f"{filename_stem}.pdf"

    fig.savefig(pdf_path, format="pdf", dpi=DPI, bbox_inches="tight")

    print(f"Saved {pdf_path}", flush=True)

    plt.close(fig)

def make_reduction_heatmap(
    *,
    changed_omega: float,
    changed_nu: float,
    changed_b: float,
    filename_stem: str,
    title: str,
) -> None:

    c_values = np.linspace(C_I_MIN, C_I_MAX, HEATMAP_N_GRID)
    rho_values = np.linspace(RHO_0_EVAL_MIN, RHO_0_MAX, HEATMAP_N_GRID)
    c_grid, rho_grid = np.meshgrid(c_values, rho_values)

    baseline = critical_relative_yield(
        c_grid,
        rho_grid,
        omega=BASELINE_OMEGA,
        nu=BASELINE_NU,
        b=BASELINE_B,
    )
    changed = critical_relative_yield(
        c_grid,
        rho_grid,
        omega=changed_omega,
        nu=changed_nu,
        b=changed_b,
    )

    reduction = baseline - changed

    visible = (
        (baseline >= E_I_MIN)
        & (baseline <= E_I_MAX)
        & (changed >= E_I_MIN)
        & (changed <= E_I_MAX)
    )
    displayed = np.ma.masked_where(~visible, reduction)

    if displayed.count() == 0:
        raise RuntimeError(f"No visible heatmap values for {filename_stem}.")

    reduction_min = float(displayed.min())
    reduction_max = float(displayed.max())
    if reduction_max <= 0.0:
        raise RuntimeError(
            f"{filename_stem} does not reduce the required relative yield."
        )

    fig, ax = plt.subplots(figsize=(10.5, 8.5))
    ax.set_facecolor("0.90")

    norm, cmap = heatmap_color_mapping()
    image = ax.contourf(
        c_grid,
        rho_grid,
        displayed,
        levels=HEATMAP_COLOR_LEVELS,
        cmap=cmap,
        norm=norm,
        extend="max",
    )

    contour_levels = np.linspace(reduction_max / 6.0, reduction_max, 6)
    contours = ax.contour(
        c_grid,
        rho_grid,
        displayed,
        levels=contour_levels,
        colors="black",
        linewidths=0.65,
    )
    ax.clabel(contours, inline=True, fontsize=9, fmt="%.3f")

    colorbar = fig.colorbar(
        image, ax=ax, pad=0.02, ticks=HEATMAP_COLOR_TICKS,
        format="%.3f",
    )
    colorbar.set_label(
        r"Reduction in required relative yield, "
        r"$e_{i,\mathrm{crit}}^{\mathrm{baseline}}"
        r"-e_{i,\mathrm{crit}}^{\mathrm{changed}}$",
        fontsize=14,
    )
    colorbar.ax.tick_params(labelsize=11)

    ax.set_xlabel(r"Cost ratio, $C_i$", fontsize=17)
    ax.set_ylabel(r"Benchmark gross revenue scale, $\rho_0$", fontsize=17)
    ax.set_title(title, fontsize=16, pad=14)
    ax.set_xlim(C_I_MIN, C_I_MAX)
    ax.set_ylim(RHO_0_EVAL_MIN, RHO_0_MAX)
    ax.set_xticks([1, 3, 5, 7, 9, 11, 12])
    ax.set_yticks([1, 500, 1000, 1500, 2000])
    ax.tick_params(axis="both", labelsize=12)

    fig.text(
        0.50,
        0.01,
        rf"Gray region: at least one threshold is outside "
        rf"${E_I_MIN:g}\leq e_{{i,\mathrm{{crit}}}}\leq {E_I_MAX:g}$."
        "\nAll panels: 0 to 1.375; upper colorbar extension: values above 1.375.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0.0, 0.065, 1.0, 1.0))
    save_pdf_eps(fig, filename_stem)

    print(
        f"  baseline: nu={BASELINE_NU:g}, omega={BASELINE_OMEGA:g}, "
        f"b={BASELINE_B:g}\n"
        f"  changed:  nu={changed_nu:g}, omega={changed_omega:g}, "
        f"b={changed_b:g}\n"
        f"  visible reduction=[{reduction_min:.6f}, {reduction_max:.6f}]\n"
        f"  common color range=[{HEATMAP_COLOR_MIN:.3f}, {HEATMAP_COLOR_MAX:.3f}]; "
        f"overflow points={np.count_nonzero(visible & (reduction > HEATMAP_COLOR_MAX))}",
        flush=True,
    )

def main() -> None:

    make_reduction_heatmap(
        changed_omega=1000,
        changed_nu=100,
        changed_b=0.37,
        filename_stem="heatmap_reduction_b_0.50_to_0.37",
        title=r"Reduction in critical relative yield: $b=0.50\rightarrow0.37$",
    )
    make_reduction_heatmap(
        changed_omega=0,
        changed_nu=100,
        changed_b=0.50,
        filename_stem="heatmap_reduction_omega_1000_to_0",
        title=r"Reduction in critical relative yield: $\omega=1000\rightarrow0$",
    )
    make_reduction_heatmap(
        changed_omega=1000,
        changed_nu=55,
        changed_b=0.50,
        filename_stem="heatmap_reduction_nu_100_to_55",
        title=r"Reduction in critical relative yield: $\nu=100\rightarrow55$",
    )

if __name__ == "__main__":
    main()

