"""
run_benchmarks.py
Reproduces Table 1 of the paper: median circle-recovery error vs. arc span
theta, for three estimators (Kasa, known-R geometric fit, normal-based),
N=50 collision points, sigma=0.01, 300 trials, fixed seed.

Usage:
    cd tests && python run_benchmarks.py
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from recovery import make_data, make_data_from_rays, METHODS, THETAS_DEG, N, SIGMA, TRIALS

SEED = 12345


def run_table1():
    rng = np.random.default_rng(SEED)
    results = {k: [] for k in METHODS}
    for th_deg in THETAS_DEG:
        th = np.deg2rad(th_deg)
        errs = {k: [] for k in METHODS}
        for _ in range(TRIALS):
            P, n = make_data(th, rng)
            for k, fn in METHODS.items():
                errs[k].append(np.linalg.norm(fn(P, n)))
        for k in METHODS:
            results[k].append(float(np.median(errs[k])))
    return results


def run_realistic_normals_check():
    """Reproduces the 'realistic normals' robustness check: normals
    reconstructed from noisy ray directions instead of supplied directly."""
    rng = np.random.default_rng(SEED + 1)
    from recovery import estimator_normal
    print("\nRealistic-normals check (incidence angle bins, degrees):")
    for max_inc_deg in (30, 60, 80, 88):
        errs = []
        for th_deg in (1, 5, 20, 90):
            th = np.deg2rad(th_deg)
            trial_errs = []
            for _ in range(TRIALS):
                P, n = make_data_from_rays(th, rng, max_incidence=np.deg2rad(max_inc_deg))
                trial_errs.append(np.linalg.norm(estimator_normal(P, n)))
            errs.append(np.median(trial_errs))
        print(f"  max incidence {max_inc_deg:3d} deg: "
              + " ".join(f"{e:.4f}" for e in errs))


if __name__ == "__main__":
    res = run_table1()
    print(f"{'theta_deg':>10} " + " ".join(f"{k:>20}" for k in METHODS))
    for i, th in enumerate(THETAS_DEG):
        print(f"{th:10.1f} " + " ".join(f"{res[k][i]:20.4e}" for k in METHODS))

    run_realistic_normals_check()
