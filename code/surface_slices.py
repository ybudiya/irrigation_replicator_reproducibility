#!/usr/bin/env python3
###############################################################################
# surface_slices.py
# Critical relative-yield slices
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

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR.parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

C_VALUES = np.linspace(1.0, 12.0, 500)
RHO_VALUES = (500.0, 1000.0, 2000.0)
Y_MIN = 1.0
Y_MAX = 3.0
DPI = 600

def critical_relative_yield(
    c_i: np.ndarray,
    rho_0: float,
    *,
    nu: float,
    omega: float,
    b: float,
) -> np.ndarray:
    if rho_0 <= 0:
        raise ValueError("rho_0 must be positive.")
    if nu <= 0 or omega < 0 or b <= 0:
        raise ValueError("nu and b must be positive and omega nonnegative.")
    return 1.0 + ((c_i - 1.0) / rho_0) * ((nu + omega) ** b)

def visible(values: np.ndarray) -> np.ndarray:
    return np.where((values >= Y_MIN) & (values <= Y_MAX), values, np.nan)

def save_figure(fig, stem: str) -> None:
    pdf_path = OUTPUT_DIR / f"{stem}.pdf"
    fig.savefig(pdf_path, format="pdf", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {pdf_path}")

def make_comparison(
    *,
    baseline: dict[str, float],
    changed: dict[str, float],
    baseline_label: str,
    changed_label: str,
    stem: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 7.4))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(RHO_VALUES)))

    for rho_0, color in zip(RHO_VALUES, colors):
        baseline_values = critical_relative_yield(
            C_VALUES, rho_0, **baseline
        )
        changed_values = critical_relative_yield(
            C_VALUES, rho_0, **changed
        )
        ax.plot(
            C_VALUES,
            visible(baseline_values),
            color=color,
            linestyle="-",
            linewidth=2.2,
        )
        ax.plot(
            C_VALUES,
            visible(changed_values),
            color=color,
            linestyle="--",
            linewidth=2.2,
        )

    rho_handles = [
        Line2D([0], [0], color=color, linewidth=2.4, label=rf"${rho_0:g}$")
        for rho_0, color in zip(RHO_VALUES, colors)
    ]
    scenario_handles = [
        Line2D([0], [0], color="black", linewidth=2.2, linestyle="-", label=baseline_label),
        Line2D([0], [0], color="black", linewidth=2.2, linestyle="--", label=changed_label),
    ]

    rho_legend = ax.legend(
        handles=rho_handles,
        title=r"$\rho_0$",
        loc="upper left",
        frameon=True,
        framealpha=1.0,
    )
    ax.add_artist(rho_legend)
    ax.legend(
        handles=scenario_handles,
        title="Scenario",
        loc="upper right",
        frameon=True,
        framealpha=1.0,
    )

    ax.set_xlabel(r"Alternative-to-gravity cost ratio, $C_i$", fontsize=14)
    ax.set_ylabel(r"Critical relative yield, $e_{i,\mathrm{crit}}$", fontsize=14)
    ax.set_xlim(1.0, 12.0)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_xticks([1, 3, 5, 7, 9, 11, 12])
    ax.set_yticks(np.linspace(Y_MIN, Y_MAX, 9))
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(True, linestyle=":", linewidth=0.7)
    fig.tight_layout()
    save_figure(fig, stem)

def main() -> None:
    baseline = {"nu": 100.0, "omega": 1000.0, "b": 0.50}

    make_comparison(
        baseline=baseline,
        changed={"nu": 100.0, "omega": 1000.0, "b": 0.37},
        baseline_label=r"$b=0.50$",
        changed_label=r"$b=0.37$",
        stem="slices_b_0.50_to_0.37",
    )
    make_comparison(
        baseline=baseline,
        changed={"nu": 100.0, "omega": 0.0, "b": 0.50},
        baseline_label=r"$\omega=1000$",
        changed_label=r"$\omega=0$",
        stem="slices_omega_1000_to_0",
    )
    make_comparison(
        baseline=baseline,
        changed={"nu": 55.0, "omega": 1000.0, "b": 0.50},
        baseline_label=r"$\nu=100$",
        changed_label=r"$\nu=55$",
        stem="slices_nu_100_to_55",
    )

if __name__ == "__main__":
    main()

