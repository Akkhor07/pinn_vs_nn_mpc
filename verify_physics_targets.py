"""
verify_physics_targets.py - Reproduces the physics-target accuracy numbers in
Section 3.3 of the manuscript.

The PINN physics target Phi_dt(x_k, u_k) is the CSTR state after one sampling
interval. This script compares several ways of computing it against a
high-accuracy adaptive reference (scipy RK45, rtol=1e-8, atol=1e-10) on random
points of the PINN collocation domain:

  A. single second-order Runge-Kutta (midpoint) step over dt   (first version)
  B. sub-stepped RK4 with temperature clamped at 450 K          (second version)
  C. sub-stepped RK4 without the clamp                           (how many sub-steps are needed)
  D. the method used for all reported results: RK4 with
     PHYSICS_TARGET_SUBSTEPS sub-steps, wide safety bounds only

Usage:
    python verify_physics_targets.py            # 500 points, seed 1
    python verify_physics_targets.py 2000 7     # n_points, seed
"""
import sys
import time
import numpy as np
from config import (CSTR_PARAMS, DT, COLLOCATION_RANGES, PHYSICS_TARGET_SUBSTEPS,
                    RHS_T_CLAMP)
from cstr_plant import simulate_cstr_step

P = CSTR_PARAMS


def rhs(CA, T, Tc, T_bounds, CA_max=None):
    """Vectorized CSTR right-hand side with configurable temperature clamp."""
    T_s = np.clip(T, *T_bounds)
    CA_s = np.clip(CA, 0.0, CA_max) if CA_max is not None else np.maximum(CA, 0.0)
    rA = P['k0'] * np.exp(-P['Ea_over_R'] / T_s) * CA_s
    qV = P['q'] / P['V']
    dCA = qV * (P['CA_f'] - CA_s) - rA
    dT = qV * (P['T_f'] - T_s) + (-P['delta_H'] / P['rho_Cp']) * rA \
        + (P['UA'] / (P['rho_Cp'] * P['V'])) * (Tc - T_s)
    return dCA, dT


def rk4(CA, T, Tc, n_sub, T_bounds, CA_max=None, state_clamp=False):
    h = DT / n_sub
    f = lambda c, t: rhs(c, t, Tc, T_bounds, CA_max)
    with np.errstate(over='ignore', invalid='ignore'):
        for _ in range(n_sub):
            k1 = f(CA, T)
            k2 = f(CA + 0.5 * h * k1[0], T + 0.5 * h * k1[1])
            k3 = f(CA + 0.5 * h * k2[0], T + 0.5 * h * k2[1])
            k4 = f(CA + h * k3[0], T + h * k3[1])
            CA = CA + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            T = T + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
            if state_clamp:   # as in the second (legacy) implementation
                CA, T = np.clip(CA, -1.0, 5.0), np.clip(T, 200.0, 700.0)
    return CA, T


def rk2_single(CA, T, Tc, T_bounds):
    """One explicit midpoint step over the whole sampling interval."""
    with np.errstate(over='ignore', invalid='ignore'):
        k1 = rhs(CA, T, Tc, T_bounds)
        k2 = rhs(CA + 0.5 * DT * k1[0], T + 0.5 * DT * k1[1], Tc, T_bounds)
        return CA + DT * k2[0], T + DT * k2[1]


def report(label, T_pred, T_ref):
    e = np.abs(T_pred - T_ref)
    bad = ~np.isfinite(e)
    e = np.where(bad, np.inf, e)
    print(f"  {label:<38} mean |eT| = {np.mean(e[~bad]):9.4f} K   max = {np.max(e[~bad]):9.4f} K   "
          f"> 5 K: {100 * np.mean(e > 5):5.1f} %   > 1 K: {100 * np.mean(e > 1):5.1f} %"
          + (f"   non-finite: {bad.sum()}" if bad.any() else ""))


def main(n=500, seed=1):
    rng = np.random.default_rng(seed)
    r = COLLOCATION_RANGES
    CA = rng.uniform(*r['CA'], n); T = rng.uniform(*r['T'], n); Tc = rng.uniform(*r['Tc'], n)
    print(f"{n} random points in the collocation domain: CA {r['CA']}, T {r['T']}, Tc {r['Tc']} (seed {seed})")
    t0 = time.time()
    T_ref = np.array([simulate_cstr_step([a, b], c)[1] for a, b, c in zip(CA, T, Tc)])
    print(f"reference: scipy RK45, rtol=1e-8 ({time.time() - t0:.1f} s)\n")

    print("A. single RK2 (midpoint) step over dt:")
    report("1 step", rk2_single(CA, T, Tc, (250.0, 800.0))[1], T_ref)

    print("\nB. sub-stepped RK4, temperature clamped at 450 K (legacy implementation):")
    for ns in (20, 40, 80, 160):
        report(f"{ns} sub-steps", rk4(CA, T, Tc, ns, (250.0, 450.0), CA_max=2.0, state_clamp=True)[1], T_ref)

    print("\nC. sub-stepped RK4 without the 450 K clamp:")
    for ns in (80, 160, 320):
        report(f"{ns} sub-steps", rk4(CA, T, Tc, ns, (250.0, 800.0), CA_max=2.0, state_clamp=True)[1], T_ref)

    print(f"\nD. method used for all reported results (RK4, {PHYSICS_TARGET_SUBSTEPS} sub-steps, "
          f"safety bounds {RHS_T_CLAMP} K):")
    report(f"{PHYSICS_TARGET_SUBSTEPS} sub-steps", rk4(CA, T, Tc, PHYSICS_TARGET_SUBSTEPS, RHS_T_CLAMP)[1], T_ref)


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    main(n, seed)
