"""
validation.py
Independent cross-check of orbit stability using the CLASSICAL
curvature-based transfer-matrix formula for billiards (Gutzwiller /
Cvitanovic / Chernov & Markarian), completely independent of the
forward-mode analytic Jacobian in orbits.py.

For a billiard trajectory segment: free flight of length t is
    M_flight(t) = [[1, t], [0, 1]]
Reflection off a circle of radius R (curvature K=1/R) at angle of
incidence phi (angle between the ray and the inward normal) is
    M_reflect = [[-1, 0], [-2K/cos(phi), -1]]
(the standard "wavefront curvature" transfer matrix, e.g. Chernov &
Markarian, "Chaotic Billiards", Ch. 3).

The monodromy matrix of a periodic orbit is the ordered product of these
matrices around the orbit; stability follows from its eigenvalues -- built
here from elementary geometry (bounce points, flight lengths, incidence
angles) read off the actual trajectory, not by differentiating the
simulator the way orbits.py does.
"""
import numpy as np
import json
import os

from orbits import hit_circle, reflect, CONFIGS


def free_flight(t):
    return np.array([[1.0, t], [0.0, 1.0]])


def reflect_matrix(K, phi):
    return np.array([[-1.0, 0.0], [-2 * K / np.cos(phi), -1.0]])


def classical_monodromy(sol, seq, circles):
    """Build the monodromy matrix from elementary geometry of the actual
    trajectory (bounce points, flight lengths, incidence angles)."""
    P = sol[:2].copy()
    D = np.array([np.cos(sol[2]), np.sin(sol[2])])
    L = len(seq)
    M = np.eye(2)
    for k in range(1, L + 1):
        tg = seq[k % L]
        t, C = hit_circle(P, D, circles[tg])
        if t is None:
            return None
        R = circles[tg][2]
        Pn = P + t * D
        n = (Pn - C) / R
        cos_phi = np.dot(D, -n)
        phi = np.arccos(np.clip(cos_phi, -1, 1))
        M = reflect_matrix(1.0 / R, phi) @ free_flight(t) @ M
        D = reflect(D, n)
        P = Pn
    return M


def classical_lambda(sol, seq, circles):
    M = classical_monodromy(sol, seq, circles)
    if M is None:
        return None, None
    det = np.linalg.det(M)
    eigs = np.linalg.eigvals(M)
    lam = np.log(max(abs(eigs))) / len(seq)
    return lam, det


def run_validation(data_path=None):
    if data_path is None:
        data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'orbits_raw.json')
    with open(data_path) as f:
        results = json.load(f)

    max_abs_diff = 0.0
    max_det_err = 0.0
    n_checked = 0
    n_mismatch = 0
    for name, circles in CONFIGS.items():
        for o in results[name]:
            sol = np.array([o['P0'][0], o['P0'][1], o['alpha']])
            seq = tuple(o['seq'])
            lam_c, det_c = classical_lambda(sol, seq, circles)
            if lam_c is None:
                print("FAILED to build monodromy for", name, seq)
                continue
            diff = abs(lam_c - o['lam'])
            max_abs_diff = max(max_abs_diff, diff)
            max_det_err = max(max_det_err, abs(det_c - 1))
            n_checked += 1
            if diff > 1e-6:
                n_mismatch += 1
                print(f"MISMATCH {name} seq={seq}: our_lam={o['lam']:.8f} "
                      f"classical_lam={lam_c:.8f} diff={diff:.2e}")

    print(f"\nChecked {n_checked} orbits against classical curvature-matrix formula.")
    print(f"Max |lambda_ours - lambda_classical| = {max_abs_diff:.3e}")
    print(f"Max |det(M_classical) - 1| = {max_det_err:.3e}")
    print(f"Mismatches (diff>1e-6): {n_mismatch}")
    return max_abs_diff, max_det_err, n_mismatch


if __name__ == "__main__":
    run_validation()
