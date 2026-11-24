#!/usr/bin/env python3
###############################################################################
# surface.py
# Critical relative-yield surfaces
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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR.parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

KAPPA_G = 1.0
C_I_MIN = 1.0
C_I_MAX = 12.0
RHO_0_EVAL_MIN = 1.0
RHO_0_DISPLAY_MIN = RHO_0_EVAL_MIN
RHO_0_MAX = 2000.0
E_I_MIN = 1.0
E_I_MAX = 3.0
N_GRID = 121
DPI = 600
BASELINE = {"omega": 1000.0, "nu": 100.0, "b": 0.50}
WIREFRAME_COUNT = 11

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

def baseline_wireframe_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    c = np.linspace(C_I_MIN, C_I_MAX, N_GRID)
    floor = np.maximum(
        RHO_0_EVAL_MIN,
        KAPPA_G * (c - 1.0) * ((BASELINE["nu"] + BASELINE["omega"]) ** BASELINE["b"])
        / (E_I_MAX - 1.0),
    )
    if np.any(floor > RHO_0_MAX):
        raise ValueError("Baseline is outside the requested display domain.")
    fraction = np.linspace(0.0, 1.0, N_GRID)[:, None]
    c_grid = np.broadcast_to(c[None, :], (N_GRID, N_GRID))
    rho_grid = floor[None, :] + fraction * (RHO_0_MAX - floor[None, :])
    return c_grid, rho_grid, critical_relative_yield(c_grid, rho_grid, **BASELINE)

def make_and_save(
    *, omega: float, nu: float, b: float, filename_stem: str,
    baseline_wireframe: bool = False,
) -> None:
    c_values = np.linspace(C_I_MIN, C_I_MAX, N_GRID)

    coefficient = (c_values - 1.0) * ((nu + omega) ** b)
    rho_visible_min = np.maximum(
        RHO_0_EVAL_MIN,
        coefficient / (E_I_MAX - 1.0),
    )
    if np.any(rho_visible_min > RHO_0_MAX):
        raise RuntimeError(f"The requested display range hides part of {filename_stem}.")

    interpolation = np.linspace(0.0, 1.0, N_GRID)[:, np.newaxis]
    c_grid = np.broadcast_to(c_values[np.newaxis, :], (N_GRID, N_GRID))
    rho_grid = rho_visible_min[np.newaxis, :] + interpolation * (
        RHO_0_MAX - rho_visible_min[np.newaxis, :]
    )
    displayed = critical_relative_yield(
        c_grid,
        rho_grid,
        omega=omega,
        nu=nu,
        b=b,
    )
    raw_at_evaluation_floor = critical_relative_yield(
        c_values,
        np.full_like(c_values, RHO_0_EVAL_MIN),
        omega=omega,
        nu=nu,
        b=b,
    )

    fig = plt.figure(figsize=(12.0, 10.0))
    ax = fig.add_subplot(111, projection="3d")
    if baseline_wireframe:
        
        ax.computed_zorder = False
    ax.plot_surface(
        c_grid,
        rho_grid,
        displayed,
        rstride=1,
        cstride=1,
        cmap="viridis",
        vmin=E_I_MIN,
        vmax=E_I_MAX,
        edgecolor="none",
        antialiased=not baseline_wireframe,
    )
    if baseline_wireframe:
        c_base, rho_base, z_base = baseline_wireframe_grid()
        ax.plot_wireframe(
            c_base, rho_base, z_base,
            rcount=WIREFRAME_COUNT, ccount=WIREFRAME_COUNT,
            color="0.25", linewidth=0.95, alpha=1.0, zorder=3,
        )
        fig.legend(
            handles=[
                Line2D([0], [0], color="0.25", linewidth=1.0,
                       label="Common baseline (wireframe)"),
                Patch(facecolor=plt.get_cmap("viridis")(0.55), edgecolor="none",
                      label="Changed scenario (filled surface)"),
            ],
            loc="upper center", bbox_to_anchor=(0.53, 0.98),
            ncol=1, fontsize=13, frameon=False,
        )

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
        axis.pane.set_edgecolor((0.0, 0.0, 0.0, 1.0))
        axis.pane.set_alpha(1.0)
        axis._axinfo["grid"]["color"] = (0.75, 0.75, 0.75, 1.0)

    ax.set_xlim(C_I_MIN, C_I_MAX)
    ax.set_ylim(RHO_0_DISPLAY_MIN, RHO_0_MAX)
    ax.set_zlim(E_I_MIN, E_I_MAX)
    ax.set_xlabel(r"$C_i$", fontsize=22, labelpad=12)
    ax.set_ylabel(r"$\rho_0$", fontsize=22, labelpad=12)
    ax.set_zlabel("")
    fig.text(
        0.025,
        0.53,
        r"$e_{i,\mathrm{crit}}$",
        rotation=90,
        va="center",
        ha="center",
        fontsize=22,
    )
    ax.set_xticks([1, 3, 5, 7, 9, 11, 12])
    ax.set_yticks([RHO_0_DISPLAY_MIN, 500, 1000, 1500, 2000])
    ax.set_zticks(np.linspace(E_I_MIN, E_I_MAX, 9))
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    ax.zaxis.set_tick_params(labelsize=14)
    ax.view_init(elev=18, azim=-135)
    fig.subplots_adjust(left=0.10, right=0.94, bottom=0.06, top=0.98)

    pdf_path = OUTPUT_DIR / f"{filename_stem}.pdf"
    fig.savefig(pdf_path, format="pdf", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    print(
        f"Saved {pdf_path}\n"
        f"  nu={nu}, omega={omega}, b={b}\n"
        f"  raw at rho_0={RHO_0_EVAL_MIN:g}: "
        f"[{np.nanmin(raw_at_evaluation_floor):.6f}, "
        f"{np.nanmax(raw_at_evaluation_floor):.6f}]\n"
        f"  displayed=[{np.nanmin(displayed):.6f}, "
        f"{np.nanmax(displayed):.6f}]"
    )

def main() -> None:
    make_and_save(omega=1000, nu=100, b=0.50, filename_stem="surface_baseline")
    for omega, nu, b, stem in (
        (1000, 100, 0.37, "surface_b_0.37_with_baseline"),
        (0, 100, 0.50, "surface_omega_0_with_baseline"),
        (1000, 55, 0.50, "surface_nu_55_with_baseline"),
    ):
        make_and_save(
            omega=omega, nu=nu, b=b, filename_stem=stem,
            baseline_wireframe=True,
        )

if __name__ == "__main__":
    main()

