#!/usr/bin/env python3
###############################################################################
# calculations.py
# Irrigation Technology Adoption Replicator-Dynamics Model
# Author: Yoshi Budiyanto, Arizona State University
# SPDX-FileCopyrightText: 2022-2026 Yoshi Budiyanto
# SPDX-License-Identifier: MIT
###############################################################################

from __future__ import annotations

import csv
import math
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PARAMETERS = SCRIPT_DIR.parent / "data" / "scenarios.csv"
OUTPUT_DIR = SCRIPT_DIR.parent / "output" / "tables"


def extract_rows() -> list[dict[str, str]]:
    with PARAMETERS.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def normalize(row: dict[str, str]) -> dict[str, float]:
    return {
        "eg": 1.0,
        "ep": float(row["Ep"]),
        "ec": float(row["Ec"]),
        "nu": float(row["Nu"]),
        "omega": float(row["Omega"]),
        "rho": float(row["Rhozero"]),
        "b": float(row["b"]),
        "kg": 1.0,
        "kp": float(row["KappaP"]),
        "kc": float(row["KappaC"]),
    }


def h(resident: str, p: dict[str, float]) -> float:
    return p["rho"] * (p["nu"] * p["e" + resident] + p["omega"]) ** (-p["b"])


def invasion(invader: str, resident: str, p: dict[str, float]) -> float:
    return (
        h(resident, p) * (p["e" + invader] - p["e" + resident])
        + p["k" + resident]
        - p["k" + invader]
    )


def stable_pair(a: str, b: str, p: dict[str, float]):
    ea, eb = p["e" + a], p["e" + b]
    ka, kb = p["k" + a], p["k" + b]
    if math.isclose(ea, eb):
        return None
    h_star = (kb - ka) / (eb - ea)
    if h_star <= 0:
        return None
    v_star = ((p["rho"] / h_star) ** (1.0 / p["b"]) - p["omega"]) / p["nu"]
    share_b = (v_star - ea) / (eb - ea)
    if not 0.0 < share_b < 1.0:
        return None
    absent = ({"g", "p", "c"} - {a, b}).pop()
    transverse = h_star * (p["e" + absent] - ea) + ka - p["k" + absent]
    if transverse >= 0:
        return None
    return {
        "regime": f"{a.upper()}-{b.upper()}",
        f"x_{a}": 1.0 - share_b,
        f"x_{b}": share_b,
        "transverse": transverse,
    }


def classify(p: dict[str, float]):
    pure = []
    for resident in ("g", "p", "c"):
        values = [
            invasion(invader, resident, p)
            for invader in ("g", "p", "c")
            if invader != resident
        ]
        if all(value < 0 for value in values):
            pure.append(resident.upper())
    pairs = [
        result
        for result in (
            stable_pair("g", "p", p),
            stable_pair("g", "c", p),
            stable_pair("p", "c", p),
        )
        if result is not None
    ]
    regimes = pure + [result["regime"] for result in pairs]
    if len(regimes) != 1:
        raise RuntimeError(f"Expected one locally stable regime, found {regimes}")
    return regimes[0], pairs[0] if pairs else None


def analytical_boundaries() -> list[tuple[str, str, str]]:
    p = normalize(extract_rows()[0])
    eg, ep, ec = p["eg"], p["ep"], p["ec"]
    kg, kp, kc = p["kg"], p["kp"], p["kc"]
    nu, omega, rho, b = p["nu"], p["omega"], p["rho"], p["b"]
    hg = rho * (nu * eg + omega) ** (-b)
    hp = rho * (nu * ep + omega) ** (-b)
    hc = rho * (nu * ec + omega) ** (-b)

    q_gp = (rho * (ep - eg) / (kp - kg)) ** (1.0 / b)
    q_pc = (rho * (ec - ep) / (kc - kp)) ** (1.0 / b)

    def bisect(function, lower, upper, iterations=100):
        f_lower = function(lower)
        for _ in range(iterations):
            mid = (lower + upper) / 2.0
            f_mid = function(mid)
            if f_lower * f_mid <= 0:
                upper = mid
            else:
                lower = mid
                f_lower = f_mid
        return (lower + upper) / 2.0

    ep_p = bisect(
        lambda value: rho * (nu * value + omega) ** (-b) * (eg - value) + kp - kg,
        ep,
        1.5,
    )
    ec_c = bisect(
        lambda value: rho * (nu * value + omega) ** (-b) * (eg - value) + kc - kg,
        ec,
        2.0,
    )
    b_p = bisect(
        lambda value: rho * (nu * ep + omega) ** (-value) * (eg - ep) + kp - kg,
        0.45,
        0.52,
    )
    b_pc = bisect(
        lambda value: rho * (nu * ep + omega) ** (-value) * (ec - ep) + kp - kc,
        0.35,
        0.42,
    )
    b_c = bisect(
        lambda value: rho * (nu * ec + omega) ** (-value) * (ep - ec) + kc - kp,
        0.35,
        0.42,
    )

    return [
        ("e_p increase", f"{1 + (kp - kg) / hg:.6f}; {ep_p:.6f}", "G/G-P; G-P/P"),
        ("e_c increase", f"{1 + (kc - kg) / hg:.6f}; {ec_c:.6f}", "G/G-C; G-C/C"),
        ("kappa_p decrease", f"{kg + hg * (ep - eg):.6f}; {kg + hp * (ep - eg):.6f}", "G/G-P; G-P/P"),
        ("kappa_c decrease", f"{kg + hg * (ec - eg):.6f}; {kg + hc * (ec - eg):.6f}", "G/G-C; G-C/C"),
        ("rho_0 increase", f"{(kp-kg)*(nu*eg+omega)**b/(ep-eg):.6f}; {(kp-kg)*(nu*ep+omega)**b/(ep-eg):.6f}; {(kc-kp)*(nu*ep+omega)**b/(ec-ep):.6f}; {(kc-kp)*(nu*ec+omega)**b/(ec-ep):.6f}", "G/G-P; G-P/P; P/P-C; P-C/C"),
        ("b decrease", f"{math.log(rho*(ep-eg)/(kp-kg))/math.log(nu+omega):.6f}; {b_p:.6f}; {b_pc:.6f}; {b_c:.6f}", "G/G-P; G-P/P; P/P-C; P-C/C"),
        ("omega decrease", f"{q_gp-nu*eg:.6f}; {q_gp-nu*ep:.6f}; {q_pc-nu*ep:.6f}; {q_pc-nu*ec:.6f}", "G/G-P; G-P/P; P/P-C; P-C/C"),
        ("nu decrease", f"{q_gp-omega:.6f}; {(q_gp-omega)/ep:.6f}", "G/G-P; G-P/P"),
    ]


def write_table(filename: str, results: list[dict]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"Wrote {path}")


def main() -> None:
    rows = extract_rows()
    results = []
    for row in rows:
        p = normalize(row)
        observed, pair = classify(p)
        shares = {f"x_{i}": float(observed == i.upper()) for i in "gpc"}
        if pair:
            shares.update({key: value for key, value in pair.items() if key.startswith("x_")})
        results.append({
            "Condition": row["Condition"],
            **p,
            "Regime": observed,
            **shares,
            "lambda_p_at_g": invasion("p", "g", p),
            "lambda_c_at_g": invasion("c", "g", p),
            "transverse": pair["transverse"] if pair else "",
        })
    write_table("scenario_equilibria.csv", results)
    write_table("analytical_boundaries.csv", [
        {"ParameterDirection": parameter, "BoundaryValues": values, "AdjacentRegimes": regimes}
        for parameter, values, regimes in analytical_boundaries()
    ])
    p = normalize(rows[0])
    write_table("baseline_market_hurdles.csv", [
        {"Strategy": i, "H_g": h("g", p),
         "TechnologyHurdle": (p["k"+i]-p["kg"])/(p["e"+i]-p["eg"]) if i != "g" else "",
         "lambda_i_at_g": invasion(i, "g", p),
         "PayoffAtGravityState": p["e"+i]*h("g", p)-p["k"+i]}
        for i in "gpc"
    ])
    eta_g = (0.40 + 0.70) / 2.0
    eta_sprinkler = (0.50 + 0.90) / 2.0
    eta_drip = (0.65 + 0.95) / 2.0
    eta_p = (eta_sprinkler + eta_drip) / 2.0
    w_g = 1.0
    w_p = eta_g / eta_p
    w_c = w_p * (1.0 - 0.13)
    write_table("water_coefficients.csv", [
        {"Strategy": i, "UnroundedCoefficient": value,
         "Coefficient": round(value, 2), "Reduction": round(1.0-round(value, 2), 2),
         "PercentOfGravity": 100.0*round(value, 2)}
        for i, value in zip("gpc", (w_g, w_p, w_c))
    ])
    write_table("application_efficiencies.csv", [
        {"Technology": name, "Efficiency": value}
        for name, value in (("Gravity", eta_g), ("Sprinkler", eta_sprinkler),
                            ("Drip", eta_drip), ("Pressurized", eta_p))
    ])
    results = []
    for scenario, nu, omega, b in (
        ("Baseline", 100.0, 1000.0, 0.50),
        ("b=0.37", 100.0, 1000.0, 0.37),
        ("omega=0", 100.0, 0.0, 0.50),
        ("nu=55", 55.0, 1000.0, 0.50),
    ):
        for C_i in (1.0, 5.12, 11.25, 12.0):
            for rho_0 in (500.0, 1000.0, 2000.0):
                e_i_crit = 1.0 + ((C_i-1.0)/rho_0) * ((nu+omega)**b)
                results.append({"Scenario": scenario, "C_i": C_i, "rho_0": rho_0,
                                "nu": nu, "omega": omega, "b": b,
                                "e_i_crit": e_i_crit,
                                "lambda_at_threshold": rho_0*(nu+omega)**(-b)*(e_i_crit-1.0)+1.0-C_i})
    write_table("critical_yield_thresholds.csv", results)
    write_table("local_supply_comparison.csv", [{
        "BaselineNu": 100.0, "ChangedNu": 55.0, "Omega": 1000.0,
        "BaselineTotalAtGravity": 1100.0, "ChangedTotalAtGravity": 1055.0,
        "SupplyReductionPercent": 100.0*(1.0-1055.0/1100.0),
        "YieldPremiumReductionPercent": 100.0*(1.0-(1055.0/1100.0)**0.50),
    }])


if __name__ == "__main__":
    main()
