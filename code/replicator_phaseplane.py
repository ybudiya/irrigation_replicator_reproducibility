#!/usr/bin/env python3
###############################################################################
# replicator_phaseplane.py
# Irrigation Technology Adoption Replicator-Dynamics Model
# Author: Yoshi Budiyanto, Arizona State University
# SPDX-FileCopyrightText: 2022-2026 Yoshi Budiyanto
# SPDX-License-Identifier: MIT
# Copyright (c) 2022-2026 Yoshi Budiyanto
###############################################################################
from __future__ import annotations

import argparse
import csv
import io
import json
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import Any
import fitz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch, Polygon
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
matplotlib.rcParams['pdf.fonttype'] = 42
HERE = Path(__file__).resolve().parent
OUTPUT = HERE.parent / 'output/replicator_phaseplane'
ADMINISTRATIVE = HERE.parent / '_administrative'

VISIBILITY = {
    'grouped': dict(arrow_mutation_scale=24., arrow_linewidth=3.,
                    trajectory_linewidth=4.5, start_markersize=6.5,
                    start_markeredgewidth=1.6),
    'individual': dict(arrow_mutation_scale=28., arrow_linewidth=3.6,
                       trajectory_linewidth=5.5, start_markersize=8.,
                       start_markeredgewidth=1.8),
}

NULLCLINE_GRID = 321

NULLCLINE_STYLE = {
    'p': dict(color='#0000FF', linestyle='--', zorder=4.),
    'c': dict(color='#FF0000', linestyle=':', zorder=4.),
}

NULLCLINE_WIDTH = {'grouped': 6., 'individual': 6.}

EXPECTED = json.loads((ADMINISTRATIVE / 'reference_results.json').read_text())['grouped_expected']

PAGE_WIDTH = 1070.

AXIS_SIDE = 425.

LEFT = 88.

TOP = 48.

COLUMN_GAP = 60.

ROW_GAP = 60.

FOOTER_GAP = 72.

FOOTER_HEIGHT = 112.

MANUSCRIPT_WIDTH = 5.25 * 72.

SUFFIX = '_solid80'

TRAJECTORY_COLORS = (
    '#00823B', '#D88B00', '#772D91', '#976B49',
    '#008579', '#B13C98', '#727C00', '#333333',
)

STYLE = {
    'grouped': dict(nullcline_linewidth=2.4, trajectory_linewidth=3.3,
                    start_markersize=8.0, arrow_mutation_scale=16.0),
    'individual': dict(nullcline_linewidth=2.8, trajectory_linewidth=4.0,
                       start_markersize=9.2, arrow_mutation_scale=20.0),
}

TRAJECTORY_LINESTYLE = '-'

TRAJECTORY_ALPHA = 0.80

ARROW_STYLE = '->'

ARROW_ALPHA = 0.50

EQUILIBRIUM_SIZE_INCREMENT = 2.0

GROUPED_EQUILIBRIUM_LEGEND_SIZE = 15.0

FOOTER_HEIGHT_WITHOUT_NOTES = 64.0

NULLCLINE_COLORS = {'p': '#0000FF', 'c': '#FF0000'}

GROWTH_TOL = EXTENSION_GROWTH_TOL = 1e-5

RATE_TOL = EXTENSION_RATE_TOL = 1e-6

FEASIBILITY_TOL = GEOMETRY_TOL = 1e-12

RHS_AGREEMENT_TOL = 2e-13

GRAY = '#F2F2F2'

REFINEMENTS = (1, 2, 4, 8)

@dataclass(frozen=True)
class Parameters:
    ep: float = 1.27
    ec: float = 1.45
    rho: float = 500.0
    nu: float = 100.0
    omega: float = 1000.0
    b: float = 0.5
    kg: float = 1.0
    kp: float = 5.12
    kc: float = 11.25

    @property
    def yields(self) -> np.ndarray:
        return np.array([1.0, self.ep, self.ec])

    @property
    def costs(self) -> np.ndarray:
        return np.array([self.kg, self.kp, self.kc])

    def validate(self) -> None:
        if not np.isfinite(list(asdict(self).values())).all():
            raise ValueError('Parameters must all be finite.')
        if min(self.ep, self.ec, self.rho, self.nu, self.b) <= 0:
            raise ValueError('Yield, revenue, local supply and b must be positive.')
        if min(self.omega, self.kg, self.kp, self.kc) < 0:
            raise ValueError('Non-local supply and costs must be nonnegative.')

BASE = Parameters()

STARTS = np.array([
    [0.06, 0.05], [0.84, 0.06], [0.07, 0.83],
    [0.46, 0.06], [0.06, 0.46], [0.47, 0.43],
    [0.22, 0.18], [0.36, 0.24],
])

END_TIME = 75.0

LABEL_COORDINATES = {'g': (-0.035, -0.035),
                     'p': (1.035, -0.035), 'c': (-0.035, 1.035)}

AXIS_LIMITS = (-0.12, 1.12)

PHASER_PCH = ((17, 5, 2), (16, 1, 1))

PCH_MARKERS = {17: ('^', 'full'), 5: ('D', 'none'), 2: ('^', 'none'),
               16: ('o', 'full'), 1: ('o', 'none')}

BASE_FONTS = {
    'grouped': {'vertex': 12., 'tick': 11., 'axis': 12., 'title': 13.,
                'legend': 12.5, 'note': 10.5},
    'individual': {'vertex': 19., 'tick': 17., 'axis': 20., 'title': 23.,
                   'legend': 16., 'note': 11.5},
}

@dataclass(frozen=True)
class Panel:
    letter: str
    parameter: str
    value_text: str
    parameters: Parameters

    @property
    def equation(self) -> str:
        symbols = {'ep': 'e_p', 'ec': 'e_c', 'omega': r'\omega', 'nu': r'\nu',
                   'rho': r'\rho_0', 'b': 'b', 'kp': r'\kappa_p', 'kc': r'\kappa_c'}
        return f'${symbols[self.parameter]} = {self.value_text}$'

def make_panels(specifications: list[tuple[str, str]]) -> tuple[Panel, ...]:
    return tuple(Panel(chr(65+i), key, value, replace(BASE, **{key: float(value)}))
                 for i, (key, value) in enumerate(specifications))

FAMILIES = {
    'ei': make_panels([('ep','1.27'), ('ec','1.45'), ('ep','1.275'),
                       ('ec','1.69'), ('ep','1.30'), ('ec','1.75')]),
    'omega': make_panels([('omega',v) for v in ['1000','960','850','300','80','70']]),
    'nu': make_panels([('nu',v) for v in ('100','70','65','55')]),
    'rhozero': make_panels([('rho',v) for v in ['500','510','600','1100','1148','1200']]),
    'b': make_panels([('b',v) for v in ['0.50','0.498','0.49','0.40','0.382','0.37']]),
    'kappa_i': make_panels([('kp','5.12'), ('kc','11.25'), ('kp','5.05'),
                            ('kc','7.72'), ('kp','5.00'), ('kc','7.50')]),
}

def shares(z: np.ndarray) -> np.ndarray:
    xp, xc = np.asarray(z)
    return np.array([1.0-xp-xc, xp, xc])

def payoffs(z: np.ndarray, p: Parameters) -> np.ndarray:
    x = shares(z)
    theta = p.nu * np.dot(x, p.yields) + p.omega
    if not np.isfinite(theta) or theta <= 0:
        raise ValueError('Nonpositive or nonfinite market quantity.')
    return p.yields * (p.rho * theta**(-p.b)) - p.costs

def rhs(t: float, z: np.ndarray, p: Parameters) -> np.ndarray:
    x = shares(z)
    pi = payoffs(z, p)
    return x[1:] * (pi[1:] - np.dot(x, pi))

def jacobian(z: np.ndarray, p: Parameters) -> np.ndarray:
    x = shares(z)
    e, k = p.yields, p.costs
    V = np.dot(x, e)
    theta = p.nu*V + p.omega
    H = p.rho * theta**(-p.b)
    H_V = -p.b*p.nu*H/theta
    dV = e[1:] - e[0]
    pi = e*H - k
    mean_pi = np.dot(x, pi)
    d_mean_pi = (H+V*H_V)*dV - (k[1:]-k[0])
    d_difference = np.outer(e[1:]*H_V, dV) - d_mean_pi[None, :]
    return np.diag(pi[1:] - mean_pi) + x[1:, None]*d_difference

def independent_rhs(z: np.ndarray, p: Parameters) -> np.ndarray:
    x=np.array([1-z[0]-z[1],z[0],z[1]])
    e=np.array([1.,p.ep,p.ec]);k=np.array([p.kg,p.kp,p.kc])
    pi=e*p.rho/(p.nu*np.dot(x,e)+p.omega)**p.b-k
    return np.array([x[i]*sum(x*(pi[i]-pi)) for i in (1,2)])

@lru_cache(maxsize=None)
def simulate(p: Parameters) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    trajectories = []
    for start in STARTS:
        sol = solve_ivp(lambda t,z: rhs(t,z,p), (0., END_TIME), start,
                        method='RK45', rtol=1e-9, atol=1e-11, max_step=.08)
        if not sol.success or sol.t[-1] != END_TIME:
            raise RuntimeError(f'Integration failed: {sol.message}')
        x = np.vstack((1-sol.y.sum(axis=0), sol.y))
        if not np.isfinite(x).all() or x.min() < -1e-8 or x.max() > 1+1e-8:
            raise RuntimeError('Trajectory violated the feasible simplex.')
        trajectories.append((sol.t, sol.y))
    return tuple(trajectories)

def phaseR_pch(eigenvalues: np.ndarray) -> int:
    ev = np.asarray(eigenvalues, dtype=complex)
    if ev.shape != (2,) or not np.isfinite(ev).all():
        raise ValueError('Exactly two finite eigenvalues are required.')
    return PHASER_PCH[int(ev[0].imag != 0)][int(np.count_nonzero(ev.real > 0))]

def classify(eigenvalues: np.ndarray) -> str:
    ev = np.asarray(eigenvalues)
    cut = 1e-9 * max(1., float(np.max(np.abs(ev))))
    if np.any(np.abs(ev.real) <= cut):
        raise ValueError('Nonhyperbolic equilibrium: nonlinear analysis required.')
    if np.all(ev.real < 0):
        return 'Stable focus' if np.any(np.abs(ev.imag) > cut) else 'Stable node'
    if np.all(ev.real > 0):
        return 'Unstable focus' if np.any(np.abs(ev.imag) > cut) else 'Unstable node'
    return 'Saddle'

@lru_cache(maxsize=None)
def equilibria(p: Parameters) -> tuple[dict[str, Any], ...]:
    p.validate()
    e, k = p.yields, p.costs
    if min(abs(e[j]-e[i]) for i,j in combinations(range(3),2)) < 1e-12:
        raise ValueError('Equal-yield cases need separate analysis.')
    slopes = (k[1:]-k[0])/(e[1:]-e[0])
    if slopes[0] > 0 and np.isclose(slopes[0], slopes[1], rtol=1e-11, atol=1e-12):
        vstar = ((p.rho/slopes[0])**(1/p.b)-p.omega)/p.nu
        if min(e) < vstar < max(e):
            raise ValueError('Equilibrium continuum: do not draw isolated symbols.')
    names = ('g', 'p', 'c')
    candidates = [(names[i], np.eye(3)[i]) for i in range(3)]
    for i,j in combinations(range(3),2):
        hstar = (k[j]-k[i])/(e[j]-e[i])
        if hstar <= 0:
            continue
        vstar = ((p.rho/hstar)**(1/p.b)-p.omega)/p.nu
        xj = (vstar-e[i])/(e[j]-e[i])
        if not 1e-11 < xj < 1-1e-11:
            continue
        x = np.zeros(3)
        x[j], x[i] = xj, 1-xj
        candidates.append((names[i]+names[j], x))
    result = []
    for name, x in candidates:
        residual = float(np.max(np.abs(rhs(0., x[1:], p))))
        if residual > 1e-9:
            raise RuntimeError(f'Equilibrium residual too large: {residual}')
        ev = np.linalg.eigvals(jacobian(x[1:], p))
        result.append(dict(name=name, x=x, eigenvalues=ev, pch=phaseR_pch(ev),
                           classification=classify(ev), residual=residual))
    return tuple(result)

def independent_points(p: Parameters) -> dict:
    e=np.array([1.,p.ep,p.ec]);k=np.array([p.kg,p.kp,p.kc])
    points={name:np.eye(3)[i] for i,name in enumerate('gpc')}
    for i,j in combinations(range(3),2):
        def difference(t):
            return (e[j]-e[i])*p.rho/(p.nu*((1-t)*e[i]+t*e[j])+p.omega)**p.b-(k[j]-k[i])
        if difference(0)*difference(1)<0:
            t=brentq(difference,0,1,xtol=1e-14)
            x=np.zeros(3);x[i]=1-t;x[j]=t;points['gpc'[i]+'gpc'[j]]=x
    return points

def verify_math() -> dict:
    records=[];max_error=0.;count=0
    for family,panels in FAMILIES.items():
        for i,panel in enumerate(panels):
            actual={q['name']:q for q in equilibria(panel.parameters)}
            independent=independent_points(panel.parameters)
            if set(actual)!=set(independent): raise AssertionError('Equilibrium sets differ.')
            stable=[]
            for name,x in independent.items():
                q=actual[name];z=x[1:];h=1e-5
                fd=np.column_stack([(independent_rhs(z+h*d,panel.parameters)-
                                     independent_rhs(z-h*d,panel.parameters))/(2*h) for d in np.eye(2)])
                ev=np.linalg.eigvals(fd)
                np.testing.assert_allclose(x,q['x'],rtol=0,atol=1e-11)
                np.testing.assert_allclose(np.sort(ev),np.sort(q['eigenvalues']),rtol=0,atol=1e-7)
                max_error=max(max_error,float(np.max(abs(fd-jacobian(z,panel.parameters)))))
                if np.all(ev.real<0):stable.append(name)
                count+=1
            if stable != [EXPECTED[family][i]]: raise AssertionError((family,panel.letter,stable))
            q=actual[stable[0]]
            records.append(dict(family=family,panel=panel.letter,parameter=panel.parameter,
                                value=panel.value_text,stable=stable[0],shares=q['x'].tolist(),
                                eigenvalues=q['eigenvalues'].real.tolist()))
    return dict(displayed_scenarios=len(records),equilibrium_records=count,
                unique_parameter_sets=len({p.parameters for panels in FAMILIES.values() for p in panels}),
                initial_conditions=STARTS.tolist(),end_time=END_TIME,
                maximum_jacobian_absolute_difference=max_error,
                method='Analytical roots plus independent bracketing roots and finite-difference Jacobians',
                stable_outcomes=records)

def nullcline_growth(panel: Panel, xp, xc):
    p = panel.parameters
    xp = np.asarray(xp)
    xc = np.asarray(xc)
    xg = 1. - xp - xc
    h = p.rho * (p.nu * (xg + p.ep*xp + p.ec*xc) + p.omega)**(-p.b)
    pi_g = h - p.kg
    pi_p = p.ep*h - p.kp
    pi_c = p.ec*h - p.kc
    mean = xg*pi_g + xp*pi_p + xc*pi_c
    return pi_p-mean, pi_c-mean

def draw_nullclines(ax, panel: Panel, kind: str):
    t = np.linspace(0., 1., NULLCLINE_GRID)
    xp, xc = np.meshgrid(t, t)
    feasible = xp + xc <= 1. + 1e-12
    growth = nullcline_growth(panel, xp, xc)
    contours = {}
    for name, z in zip(('p', 'c'), growth):
        z = np.ma.array(z, mask=~feasible)
        zz = z.compressed()
        if not len(zz) or not (zz.min() < 0. < zz.max()):
            contours[name] = None
            continue
        style = NULLCLINE_STYLE[name]
        artist = ax.contour(
            xp, xc, z, levels=[0.], colors=[style['color']],
            linestyles=[style['linestyle']],
            linewidths=[NULLCLINE_WIDTH[kind]], alpha=1.,
            zorder=style['zorder'],
        )
        artist.set_gid(f'nullcline-{name}')
        contours[name] = artist
    return contours

class ContourPrecisionError(AssertionError):
    pass

def artists(ax, prefix: str) -> list:
    return [a for a in ax.get_children() if (a.get_gid() or '').startswith(prefix)]

def independent_growth(z: np.ndarray, p) -> np.ndarray:
    xp, xc = z
    x = np.array([1.0 - xp - xc, xp, xc])
    e = np.array([1.0, p.ep, p.ec])
    costs = np.array([p.kg, p.kp, p.kc])
    revenue = p.rho / (p.nu * float(x @ e) + p.omega)**p.b
    pi = e * revenue - costs
    return np.array([sum(x * (pi[i] - pi)) for i in (1, 2)])

def contour_checks(ax, panel) -> list[dict]:
    result = []
    for j, name in enumerate(('p', 'c')):
        found = artists(ax, 'nullcline-' + name)
        branches = []
        for cs in found:
            for segment in cs.allsegs[0]:
                if len(segment) < 2:
                    continue
                
                samples = np.vstack([segment] + [
                    (1-a)*segment[:-1] + a*segment[1:] for a in (.25, .5, .75)])
                xg = 1 - samples.sum(axis=1)
                if min(float(samples.min()), float(xg.min())) < -FEASIBILITY_TOL:
                    raise AssertionError('Contour leaves feasible simplex.')
                rates = np.array([independent_rhs(z, panel.parameters)[j]
                                  for z in samples])
                growth = np.array([independent_growth(z, panel.parameters)[j]
                                   for z in samples])
                original_rates = np.array([rhs(0., z, panel.parameters)[j]
                                           for z in samples])
                err_rate = float(np.abs(rates).max())
                err_growth = float(np.abs(growth).max())
                rhs_diff = float(np.abs(rates-original_rates).max())
                if err_rate >= RATE_TOL:
                    raise ContourPrecisionError((panel.letter, name, 'rate', err_rate))
                if err_growth >= GROWTH_TOL:
                    raise ContourPrecisionError((panel.letter, name, 'growth', err_growth))
                assert rhs_diff < RHS_AGREEMENT_TOL, (panel, name, rhs_diff)
                branches.append(dict(vertices=len(segment), checked_points=len(samples),
                    min_xp_xc=segment.min(axis=0).tolist(),
                    max_xp_xc=segment.max(axis=0).tolist(),
                    maximum_abs_replicator_rate=err_rate,
                    maximum_abs_payoff_growth=err_growth,
                    maximum_rhs_agreement_error=rhs_diff))
        result.append(dict(nullcline=name, branches=branches, absent=not branches))
    return result

def is_feasible(z: np.ndarray, tolerance: float = 1e-12) -> np.ndarray:
    z = np.asarray(z)
    return (z[..., 0] >= -tolerance) & (z[..., 1] >= -tolerance) & (z.sum(axis=-1) <= 1+tolerance)

def full_axis_grid(refinement: int = 1) -> np.ndarray:
    lo, hi = AXIS_LIMITS
    original = np.linspace(0., 1., NULLCLINE_GRID)
    step = original[1]-original[0]
    lower = np.linspace(lo, 0., int(np.ceil(-lo/step))+1)[:-1]
    upper = np.linspace(1., hi, int(np.ceil((hi-1)/step))+1)[1:]
    base = np.concatenate([lower, original, upper])
    if refinement == 1:
        return base
    if refinement not in REFINEMENTS:
        raise ValueError('Unsupported contour refinement.')
    return np.concatenate([np.linspace(a, b, refinement+1)[:-1]
                           for a, b in zip(base[:-1], base[1:])] + [base[-1:]])

def outside_segments(path: np.ndarray) -> list[np.ndarray]:
    result: list[np.ndarray] = []
    current: list[np.ndarray] = []
    for a, b in zip(path[:-1], path[1:]):
        if np.array_equal(a, b):
            continue
        fa = np.array([a[0], a[1], 1-a.sum()])
        fb = np.array([b[0], b[1], 1-b.sum()])
        cuts = [0., 1.]
        for f0, f1 in zip(fa, fb):
            if f0*f1 < 0:
                t = float(f0/(f0-f1))
                if 0 < t < 1:
                    cuts.append(t)
        cuts = sorted(set(cuts))
        for t0, t1 in zip(cuts[:-1], cuts[1:]):
            q0 = a+t0*(b-a); q1 = a+t1*(b-a)
            if is_feasible((q0+q1)/2, tolerance=GEOMETRY_TOL):
                if len(current) > 1:
                    result.append(np.array(current))
                current = []
                continue
            if current and np.max(np.abs(current[-1]-q0)) < 1e-13:
                if not np.array_equal(current[-1], q1):
                    current.append(q1)
            else:
                if len(current) > 1:
                    result.append(np.array(current))
                current = [q0, q1]
    if len(current) > 1:
        result.append(np.array(current))
    return result

def _draw_extensions_at_grid(ax, panel, kind='grouped', refinement=1) -> dict:
    t = full_axis_grid(refinement)
    xp, xc = np.meshgrid(t, t)
    p = panel.parameters
    theta = p.nu*(1-xp-xc+p.ep*xp+p.ec*xc)+p.omega
    if not np.isfinite(theta).all() or theta.min() <= 0:
        raise ValueError('Cannot extend the real-valued equations over these axes.')
    growth = nullcline_growth(panel, xp, xc)
    records = {}
    for j, (name, z) in enumerate(zip(('p','c'), growth)):
        segments = []
        if np.min(z) < 0 < np.max(z):
            full = ax.contour(xp, xc, z, levels=[0.], colors=[NULLCLINE_COLORS[name]],
                              linewidths=[STYLE[kind]['nullcline_linewidth']],
                              linestyles=['solid'], alpha=1., zorder=4.)
            for path in full.allsegs[0]:
                if len(path) > 1:
                    segments.extend(outside_segments(path))
            full.remove()  
        if segments:
            lines = LineCollection(segments, colors=[NULLCLINE_COLORS[name]],
                linewidths=[STYLE[kind]['nullcline_linewidth']],
                linestyles=['solid'], alpha=1., zorder=4.,
                gid='nullcline-extension-'+name)
            ax.add_collection(lines)
        checks = []
        for segment in segments:
            samples = np.vstack([segment]+[
                (1-a)*segment[:-1]+a*segment[1:] for a in (.25,.5,.75)])
            rates = np.array([independent_rhs(z,p)[j] for z in samples])
            reference = np.array([rhs(0.,z,p)[j] for z in samples])
            growth_values = np.array([independent_growth(z,p)[j] for z in samples])
            midpoint = (segment[:-1]+segment[1:])/2
            assert not np.any(is_feasible(midpoint, GEOMETRY_TOL))
            assert samples.min() >= AXIS_LIMITS[0]-GEOMETRY_TOL
            assert samples.max() <= AXIS_LIMITS[1]+GEOMETRY_TOL
            max_rate = float(abs(rates).max())
            max_growth = float(abs(growth_values).max())
            agreement = float(abs(rates-reference).max())
            if max_rate >= EXTENSION_RATE_TOL:
                raise ContourPrecisionError((panel.letter, name, 'rate', max_rate))
            if max_growth >= EXTENSION_GROWTH_TOL:
                raise ContourPrecisionError((panel.letter, name, 'growth', max_growth))
            assert agreement < RHS_AGREEMENT_TOL, (panel,name,agreement)
            checks.append(dict(vertices=len(segment), checked_points=len(samples),
                maximum_abs_replicator_rate=max_rate,
                maximum_abs_payoff_growth=max_growth,
                maximum_rhs_agreement_error=agreement,
                min_xp_xc=segment.min(0).tolist(),max_xp_xc=segment.max(0).tolist()))
        records[name] = dict(outside_segments=checks, absent_outside=not checks)
    records['minimum_market_quantity'] = float(theta.min())
    return records

def ensure_feasible_precision(ax, panel, kind='grouped') -> dict:
    attempts = []
    for factor in REFINEMENTS:
        if factor > 1:
            for cs in artists(ax, 'nullcline-'):
                if cs.get_gid() in ('nullcline-p', 'nullcline-c'):
                    cs.remove()
            t = np.linspace(0., 1., (NULLCLINE_GRID-1)*factor+1)
            xp, xc = np.meshgrid(t, t)
            for name, growth in zip(('p', 'c'), nullcline_growth(panel, xp, xc)):
                z = np.ma.array(growth, mask=xp+xc > 1.+GEOMETRY_TOL)
                values = z.compressed()
                if len(values) and values.min() < 0. < values.max():
                    cs = ax.contour(xp, xc, z, levels=[0.], colors=[NULLCLINE_COLORS[name]],
                        linewidths=[STYLE[kind]['nullcline_linewidth']],
                        linestyles=['solid'], alpha=1., zorder=4.)
                    cs.set_gid('nullcline-'+name)
        try:
            contour_checks(ax, panel)
            return dict(grid_refinement=factor, source_vertices_retained=factor==1,
                        failed_precision_attempts=attempts)
        except ContourPrecisionError as error:
            attempts.append(dict(grid_refinement=factor, diagnostic=str(error)))
    raise ContourPrecisionError(f'Feasible contour did not converge: {attempts}')

def draw_extensions(ax, panel, kind='grouped') -> dict:
    attempts = []
    for factor in REFINEMENTS:
        try:
            result = _draw_extensions_at_grid(ax, panel, kind, factor)
            result['outside_grid_refinement'] = factor
            result['failed_precision_attempts'] = attempts
            return result
        except ContourPrecisionError as error:
            for line in artists(ax, 'nullcline-extension-'):
                line.remove()
            attempts.append(dict(grid_refinement=factor, diagnostic=str(error)))
    raise ContourPrecisionError(f'Outside contour did not converge: {attempts}')

def direction_field(p: Parameters) -> tuple[np.ndarray, np.ndarray]:
    xy = np.array([(i/12,j/12) for i in range(13) for j in range(13-i)])
    v = np.array([rhs(0.,z,p) for z in xy])
    n = np.linalg.norm(v,axis=1)
    use = n > 1e-12
    return xy[use], v[use]/n[use,None]

def symbol(pch: int, label: str = '', size: float = 10.5) -> Line2D:
    marker, fill = PCH_MARKERS[pch]
    return Line2D([], [], linestyle='none', marker=marker, fillstyle=fill,
                  markersize=size, markeredgewidth=1.6, label=label)

def font_sizes(kind: str, scale: float) -> dict[str, float]:
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Font scale must be positive and finite.')
    return {key: value*scale for key, value in BASE_FONTS[kind].items()}

def draw_plane(ax, panel: Panel, kind: str, scale: float,
               show_xlabel: bool = True, show_ylabel: bool = True) -> dict:
    f = font_sizes(kind, scale)
    grouped = kind == 'grouped'
    pts = equilibria(panel.parameters)
    locations = np.array([q['x'][1:] for q in pts])
    xy, uv = direction_field(panel.parameters)
    for center, direction in zip(xy, uv):
        if np.min(np.linalg.norm(locations-center,axis=1)) < .045:
            continue

        arrow = FancyArrowPatch(center-.03*direction, center+.03*direction,
                                arrowstyle='->', mutation_scale=10 if grouped else 14,
                                linewidth=1.2 if grouped else 1.8,
                                alpha=1., shrinkA=0, shrinkB=0, zorder=2)
        arrow.set_gid('direction-field')
        ax.add_patch(arrow)
    for i, (_, y) in enumerate(simulate(panel.parameters)):
        line, = ax.plot(y[0],y[1], linewidth=1.35 if grouped else 2.1,
                       zorder=3, marker='x', markevery=[0],
                       markersize=4 if grouped else 5.5,
                       markeredgewidth=.95 if grouped else 1.2)
        line.set_gid(f'trajectory-{i+1}')
    for q in pts:
        artist = symbol(q['pch'], size=10.5 if grouped else 13.)
        artist.set_data([q['x'][1]],[q['x'][2]])
        artist.set_zorder(20)
        artist.set_clip_on(False)
        artist.set_gid('equilibrium-'+q['name'])
        ax.add_line(artist)

    for label, position in LABEL_COORDINATES.items():
        ax.text(*position, label, fontsize=f['vertex'],
                ha='left' if label == 'p' else 'right', va='center',
                transform=ax.transData, gid='vertex-label-'+label)
    ax.set_xlim(*AXIS_LIMITS)
    ax.set_ylim(*AXIS_LIMITS)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks(np.linspace(0,1,6))
    ax.set_yticks(np.linspace(0,1,6))
    ax.tick_params(labelsize=f['tick'], pad=2., length=3.)
    ax.grid(False)
    
    ax.set_title(f'{panel.letter}. {panel.equation}', fontsize=f['title'],
                 fontweight='bold', pad=12.)
    if show_xlabel:
        ax.set_xlabel(r'Pressurized share $x_p$', fontsize=f['axis'], labelpad=8.)
    if show_ylabel:
        ax.set_ylabel(r'Computerized share $x_c$', fontsize=f['axis'], labelpad=10.)
    return f

def draw_plane_visibility(ax, panel: Panel, kind: str, scale: float,
                          show_xlabel: bool = True,
                          show_ylabel: bool = True) -> dict:
    fonts = draw_plane(ax, panel, kind, scale, show_xlabel, show_ylabel)
    config = VISIBILITY[kind]
    arrows = [p for p in ax.patches if p.get_gid() == 'direction-field']
    trajectories = [line for line in ax.lines
                    if (line.get_gid() or '').startswith('trajectory-')]
    if not arrows or len(trajectories) != len(STARTS):
        raise RuntimeError('Unexpected arrow/trajectory artist count.')
    
    draw_nullclines(ax, panel, kind)
    for arrow in arrows:
        arrow.set_mutation_scale(config['arrow_mutation_scale'])
        arrow.set_linewidth(config['arrow_linewidth'])
        arrow.set_arrowstyle('-|>')
        arrow.set_facecolor(arrow.get_edgecolor())
    for line in trajectories:
        line.set_linewidth(config['trajectory_linewidth'])
        line.set_markersize(config['start_markersize'])
        line.set_markeredgewidth(config['start_markeredgewidth'])
    return fonts

def draw_preview_plane(ax, panel, kind='grouped', scale=2.,
                       show_xlabel=True, show_ylabel=True) -> dict:
    fonts = draw_plane_visibility(ax, panel, kind, scale, show_xlabel, show_ylabel)
    conf = STYLE[kind]
    trajectories = artists(ax, 'trajectory-')
    arrows = artists(ax, 'direction-field')
    nullclines = artists(ax, 'nullcline-')
    equilibrium_artists = artists(ax, 'equilibrium-')
    assert len(trajectories) == len(STARTS) == 8
    before_trajectories = [t.get_xydata().copy() for t in trajectories]
    before_arrows = [np.asarray(a._posA_posB).copy() for a in arrows]
    before_nullclines = [[s.copy() for s in c.allsegs[0]] for c in nullclines]
    before_equilibria = [(e.get_gid(), e.get_xydata().copy(), e.get_marker(),
                          e.get_fillstyle(), e.get_zorder(), e.get_color())
                         for e in equilibrium_artists]
    for a in arrows:
        a.set_mutation_scale(conf['arrow_mutation_scale'])
        a.set_arrowstyle(ARROW_STYLE)
        a.set_fill(False)
        a.set_alpha(ARROW_ALPHA)
        
    for i, line in enumerate(trajectories):
        np.testing.assert_array_equal(line.get_xydata()[0], STARTS[i])
        line.set_color(TRAJECTORY_COLORS[i])
        line.set_linestyle(TRAJECTORY_LINESTYLE)
        line.set_alpha(TRAJECTORY_ALPHA)
        line.set_linewidth(conf['trajectory_linewidth'])
        line.set_marker('None')
        line.set_markevery(None)
        marker = Line2D([STARTS[i, 0]], [STARTS[i, 1]],
                        linestyle='None', marker='x', color=TRAJECTORY_COLORS[i],
                        alpha=1.00, markersize=conf['start_markersize'],
                        markeredgewidth=VISIBILITY[kind]['start_markeredgewidth'],
                        zorder=10, gid=f'initial-condition-{i+1}')
        ax.add_line(marker)
    for cs in nullclines:
        name = cs.get_gid().removeprefix('nullcline-')
        cs.set_edgecolor(NULLCLINE_COLORS[name])
        cs.set_linestyle('solid')
        cs.set_linewidth(conf['nullcline_linewidth'])
        cs.set_alpha(1.00)
    for eq in equilibrium_artists:
        eq.set_markersize(eq.get_markersize() + EQUILIBRIUM_SIZE_INCREMENT)
    
    for old, line in zip(before_trajectories, trajectories):
        np.testing.assert_array_equal(old, line.get_xydata())
    for old, arrow in zip(before_arrows, arrows):
        np.testing.assert_array_equal(old, arrow._posA_posB)
    for old, cs in zip(before_nullclines, nullclines):
        assert len(old) == len(cs.allsegs[0])
        for a, b in zip(old, cs.allsegs[0]):
            np.testing.assert_array_equal(a, b)
    for old, eq in zip(before_equilibria, equilibrium_artists):
        assert (old[0], old[2], old[3], old[4], old[5]) == (
            eq.get_gid(), eq.get_marker(), eq.get_fillstyle(), eq.get_zorder(), eq.get_color())
        np.testing.assert_array_equal(old[1], eq.get_xydata())
    return fonts

def add_nullcline_legend(fig, kind: str, top: float, height: float):
    labels = {
        'p': 'Blue: pressurized nullcline',
        'c': 'Red: computerized nullcline',
    }
    handles = [Line2D([], [], color=NULLCLINE_COLORS[n], linestyle='-', alpha=1.,
                      linewidth=STYLE[kind]['nullcline_linewidth'], label=labels[n],
                      gid='nullcline-key-'+n) for n in ('p', 'c')]
    return fig.legend(handles=handles, loc='upper center',
                      bbox_to_anchor=(.5, 1-top/height), ncol=2,
                      fontsize=font_sizes(kind, 2.)['note'],
                      frameon=False, borderaxespad=0., handlelength=2.,
                      handletextpad=.5, columnspacing=1.5, labelcolor='linecolor')

def draw_extended_plane(ax, panel, kind='grouped', scale=2., show_xlabel=True, show_ylabel=True):
    fonts = draw_preview_plane(ax,panel,kind,scale,show_xlabel,show_ylabel)

    ax.set_facecolor(GRAY)
    ax.add_patch(Polygon([(0.,0.),(1.,0.),(0.,1.)],closed=True,
        facecolor='white',edgecolor='none',linewidth=0.,zorder=-10.,
        gid='feasible-background-no-outline'))
    feasible_precision = ensure_feasible_precision(ax, panel, kind)
    before = {c.get_gid(): [s.copy() for s in c.allsegs[0]]
              for c in artists(ax,'nullcline-')}
    record = draw_extensions(ax,panel,kind)
    record['feasible_precision'] = feasible_precision
    for gid, paths in before.items():
        cs = next(c for c in ax.get_children() if c.get_gid() == gid)
        for expected, actual in zip(paths,cs.allsegs[0]):
            np.testing.assert_array_equal(expected,actual)
    
    for text in ax.texts:
        if (text.get_gid() or '').startswith('vertex-label-'):
            text.set_zorder(30.)
    return fonts, record

def layout(name: str) -> dict:
    rows = 2 if name == 'nu' else 3
    axes = [[LEFT + col*(AXIS_SIDE+COLUMN_GAP),
             TOP + row*(AXIS_SIDE+ROW_GAP), AXIS_SIDE, AXIS_SIDE]
            for row in range(rows) for col in range(2)]
    footer_top = axes[-1][1]+AXIS_SIDE+FOOTER_GAP
    return dict(rows=rows, columns=2, page_width_pt=PAGE_WIDTH,
                page_height_pt=footer_top+FOOTER_HEIGHT+4.,
                axes_boxes_pt=axes, footer_top_pt=footer_top)

def preview_layout(family: str) -> dict:
    geometry = dict(layout(family))
    geometry['page_height_pt'] = geometry['footer_top_pt'] + FOOTER_HEIGHT_WITHOUT_NOTES
    return geometry

def build_grouped(family: str):
    if family not in FAMILIES:
        raise ValueError('Unknown parameter family.')
    geometry = preview_layout(family)
    width,height = geometry['page_width_pt'],geometry['page_height_pt']
    fig = plt.figure(figsize=(width/72.,height/72.),dpi=144)
    checks = []
    for i,(panel,box) in enumerate(zip(FAMILIES[family],geometry['axes_boxes_pt'])):
        left,top,side,_ = box
        row,col = divmod(i,2)
        ax = fig.add_axes([left/width,(height-top-side)/height,side/width,side/height])
        _,ext = draw_extended_plane(ax,panel,'grouped',2.,row==geometry['rows']-1,col==0)
        checks.append(dict(panel=panel.letter,parameter=panel.parameter,value=panel.value_text,
            feasible_nullclines=contour_checks(ax,panel),extensions=ext))
    fonts = font_sizes('grouped',2.)
    handles = [symbol(17,'Stable node (sink)',size=GROUPED_EQUILIBRIUM_LEGEND_SIZE),
               symbol(5,'Saddle',size=GROUPED_EQUILIBRIUM_LEGEND_SIZE),
               symbol(2,'Unstable node (source)',size=GROUPED_EQUILIBRIUM_LEGEND_SIZE)]
    fig.legend(handles=handles,loc='upper center',
        bbox_to_anchor=(.5,1-(geometry['footer_top_pt']-10.)/height),ncol=3,
        fontsize=fonts['legend'],frameon=False,borderaxespad=0.,handletextpad=.4,columnspacing=1.1)
    add_nullcline_legend(fig,'grouped',geometry['footer_top_pt']+22.,height)
    texts = assert_text_inside(fig)
    overlaps = text_overlap_checks(texts)
    assert not overlaps, overlaps
    for ax in fig.axes:
        box = ax.get_window_extent(fig.canvas.get_renderer())
        assert abs(box.width-box.height) < 1e-6
    return fig,dict(family=family,**geometry,panels=checks,
        text_checks=len(texts),text_overlaps=overlaps,text_boxes=texts,
        axes_side_pt=AXIS_SIDE,
        manuscript_insertion_width_inches=MANUSCRIPT_WIDTH/72.)

def assert_text_inside(fig) -> list[dict]:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    figure_box = fig.bbox
    records = []
    texts = list(fig.texts)
    for ax in fig.axes:
        texts.extend([ax.title,ax.xaxis.label,ax.yaxis.label])
        texts.extend(ax.texts)
        texts.extend(ax.get_xticklabels())
        texts.extend(ax.get_yticklabels())
    for legend in fig.legends:
        texts.extend(legend.get_texts())
    for text in texts:
        if not text.get_visible() or not text.get_text():
            continue
        bb = text.get_window_extent(renderer)
        if bb.x0 < -.75 or bb.y0 < -.75 or bb.x1 > figure_box.x1+.75 or bb.y1 > figure_box.y1+.75:
            raise RuntimeError(f'Text extends outside canvas: {text.get_text()!r} {bb}')
        records.append({'text':text.get_text(), 'fontsize_pt':text.get_fontsize(),
                        'bbox_pt':(np.array(bb.bounds)*72/fig.dpi).tolist()})
    return records

def save_pdf_png(raw: bytes, pdf_path: Path, dpi: int) -> None:
    pdf_path.parent.mkdir(parents=True,exist_ok=True)
    pdf_path.write_bytes(raw)
    with fitz.open(stream=raw,filetype='pdf') as doc:
        if doc.page_count != 1:
            raise RuntimeError('Each portrait export must have one page.')
        pix = doc[0].get_pixmap(dpi=dpi,alpha=False)
        pix.save(pdf_path.with_suffix('.png'))

def text_overlap_checks(records: list[dict], tolerance: float = .5) -> list[dict]:
    collisions = []
    for i, first in enumerate(records):
        a = first['bbox_pt']; ax0,ay0,ax1,ay1 = a[0],a[1],a[0]+a[2],a[1]+a[3]
        for second in records[i+1:]:
            b = second['bbox_pt']; bx0,by0,bx1,by1 = b[0],b[1],b[0]+b[2],b[1]+b[3]
            if min(ax1,bx1)-max(ax0,bx0)>tolerance and min(ay1,by1)-max(ay0,by0)>tolerance:
                collisions.append({'first':first['text'],'second':second['text']})
    return collisions

def write_manifest(output: Path) -> None:
    rows=[];eqrows=[]
    for name,panels in FAMILIES.items():
        for i,p in enumerate(panels):
            row,col=divmod(i,2)
            rows.append(dict(family=name,panel=p.letter,row=row+1,column=col+1,
                             parameter=p.parameter,value=p.value_text,**asdict(p.parameters)))
            for q in equilibria(p.parameters):
                eqrows.append(dict(family=name,panel=p.letter,equilibrium=q['name'],
                                   x_g=q['x'][0],x_p=q['x'][1],x_c=q['x'][2],
                                   classification=q['classification'],R_pch=q['pch'],
                                   lambda1=float(q['eigenvalues'][0].real),
                                   lambda2=float(q['eigenvalues'][1].real),residual=q['residual']))
    for filename,table in [('scenario_manifest.csv',rows),('equilibrium_checks.csv',eqrows)]:
        with (output/filename).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)

def export(family: str, output: Path, dpi: int=300):
    fig,report = build_grouped(family)
    stem = f'grouped_phaseR_{family}_{report["rows"]}x2_text2x{SUFFIX}'
    pdfpath = output/(stem+'.pdf')
    buf = io.BytesIO()
    fig.savefig(buf,format='pdf',bbox_inches=None,
        metadata={'Creator':'replicator_phaseplane.py',
                  'Title':f'{family}: phase planes'})
    plt.close(fig)
    save_pdf_png(buf.getvalue(),pdfpath,dpi)
    with fitz.open(pdfpath) as document:
        assert len(document) == 1
        assert not document[0].get_images()
        spans = [s for b in document[0].get_text('dict')['blocks'] if 'lines' in b
            for line in b['lines'] for s in line['spans'] if s['text'].strip()]
        assert all(fitz.Rect(s['bbox']) in document[0].rect for s in spans)
        report['pdf_text_spans_in_bounds'] = len(spans)
        report['pdf_raster_images'] = 0
    report.update(pdf=pdfpath.name,png=pdfpath.with_suffix('.png').name,png_dpi=dpi)
    return report

def main():
    parser = argparse.ArgumentParser(description='Generate the six grouped phase-plane figures.')
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    parser.add_argument('--dpi', type=int, default=300)
    args = parser.parse_args()
    if args.dpi < 200:
        parser.error('Use at least 200 dpi.')
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    ADMINISTRATIVE.mkdir(parents=True, exist_ok=True)
    report = dict(math=verify_math(), figures=[])
    write_manifest(ADMINISTRATIVE)
    for family in FAMILIES:
        result = export(family, output, args.dpi)
        report['figures'].append(dict(family=family, pdf=result['pdf'], png=result['png'],
            panels=len(result['panels']), text_overlaps=result['text_overlaps'],
            pdf_text_spans_in_bounds=result['pdf_text_spans_in_bounds'],
            pdf_raster_images=result['pdf_raster_images']))
        print('Generated', family, flush=True)
    (ADMINISTRATIVE / 'phaseplane_validation.json').write_text(
        json.dumps(report, indent=2) + '\n')

if __name__ == '__main__':
    main()

