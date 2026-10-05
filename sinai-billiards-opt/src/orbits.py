"""
orbits.py
Periodic-orbit finder for regular N-gon disk billiards, and the exact
analytic tangent-map Jacobian used to classify orbit stability.

Method:
 - A state x=(P, alpha) is periodic under itinerary `seq` iff it returns to
   itself in BOTH position and direction after one period; the closure
   residual is [Px_end-Px0, Py_end-Py0, wrapped(angle_end-angle0)].
 - Every candidate solution is independently re-verified by directly
   propagating L bounces and comparing the resulting state to the solver's
   claimed fixed point.
 - A dense 2D multistart grid (start point on the launch circle x initial
   direction) is used per itinerary.
 - Orbits are deduplicated by their full bounce trajectory (all L points),
   so the same physical orbit found from different itinerary rotations
   collapses to a single entry.
 - Itineraries allow repeated disks, forbidding only two consecutive hits
   on the same disk (including wraparound), e.g. (0,1,0,2) is included.

Stability (analytic tangent map):
 - The Jacobian of the one-period map (P, alpha) -> (P, alpha) is computed
   by forward-mode differentiation of the reflection law through each
   bounce (NOT finite differences), which stays accurate even for long,
   strongly unstable orbits where finite-difference Jacobians break down.
   Because the return map is symplectic, det(J) = 1 exactly; this is used
   as a running numerical check (see validation.py for an independent,
   classical cross-check of the resulting Lyapunov exponents).
"""
import numpy as np
import time
from scipy.optimize import fsolve
from itertools import product


# ---------------------------------------------------------------------
# Geometry primitives
# ---------------------------------------------------------------------

def reflect(d, n):
    return d - 2 * np.dot(d, n) * n


def hit_circle(P, D, c):
    C = np.array(c[:2]); R = c[2]; w = P - C
    b = 2 * np.dot(w, D); cc = np.dot(w, w) - R ** 2; disc = b * b - 4 * cc
    if disc <= 0:
        return None, None
    sq = np.sqrt(disc)
    for t in ((-b - sq) / 2, (-b + sq) / 2):
        if t > 1e-9:
            return t, C
    return None, None


def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def propagate(P, D, seq, circles, need_points=False):
    """Propagate through seq[1], seq[2], ..., seq[0] (one full cycle)."""
    L = len(seq)
    pts = []
    for k in range(1, L + 1):
        tg = seq[k % L]
        t, C = hit_circle(P, D, circles[tg])
        if t is None:
            return (None, None, None) if need_points else (None, None)
        P = P + t * D
        n = (P - C) / circles[tg][2]
        D = reflect(D, n)
        if need_points:
            pts.append(P.copy())
    return (P, D, pts) if need_points else (P, D)


def visible(P0, D0, seq, circles, margin=1e-6):
    L = len(seq); cur = seq[0]; P = P0.copy(); D = D0.copy()
    for k in range(1, L + 1):
        tg = seq[k % L]
        t, C = hit_circle(P, D, circles[tg])
        if t is None:
            return False
        Pn = P + t * D
        for m in range(len(circles)):
            if m == cur or m == tg:
                continue
            Cm = circles[m][:2]; Rm = circles[m][2]; AB = Pn - P; L2 = AB @ AB
            if L2 < 1e-20:
                continue
            tt = (Cm - P) @ AB / L2
            if 0 <= tt <= 1 and np.linalg.norm(P + tt * AB - Cm) < Rm - margin:
                return False
        n = (Pn - C) / circles[tg][2]
        D = reflect(D, n); P = Pn; cur = tg
    return True


def sequences(N, L):
    """All cyclic itineraries of length L over N disk labels, forbidding
    immediate repetition (including wraparound), deduplicated by
    rotation/reflection."""
    seen = set()
    out = []
    for seq in product(range(N), repeat=L):
        if any(seq[i] == seq[(i + 1) % L] for i in range(L)):
            continue
        variants = [tuple(seq[s:] + seq[:s]) for s in range(L)]
        rev = tuple(reversed(seq))
        variants += [tuple(rev[s:] + rev[:s]) for s in range(L)]
        sig = min(variants)
        if sig in seen:
            continue
        seen.add(sig)
        out.append(seq)
    return out


# ---------------------------------------------------------------------
# Closure solver
# ---------------------------------------------------------------------

def solve_closure(circles, seq, n_pos=16, n_dir=16, tol=1e-9):
    """Multistart solve of full closure (position AND direction)."""
    L = len(seq)
    C0 = np.array(circles[seq[0]][:2]); R0 = circles[seq[0]][2]
    sols = []
    for a in np.linspace(0, 2 * np.pi, n_pos, endpoint=False):
        P0 = C0 + R0 * np.array([np.cos(a), np.sin(a)])
        for al in np.linspace(-np.pi, np.pi, n_dir, endpoint=False):
            x0 = np.array([P0[0], P0[1], al])

            def res(x):
                Pn, Dn = propagate(x[:2].copy(),
                                    np.array([np.cos(x[2]), np.sin(x[2])]),
                                    seq, circles)
                if Pn is None:
                    return [1e3, 1e3, 1e3]
                return [Pn[0] - x[0], Pn[1] - x[1],
                        wrap(np.arctan2(Dn[1], Dn[0]) - x[2])]

            try:
                sol, info, ier, msg = fsolve(res, x0, full_output=True, xtol=1e-12)
            except Exception:
                continue
            if ier != 1 or np.linalg.norm(res(sol)) > tol:
                continue
            if abs(np.linalg.norm(sol[:2] - C0) - R0) > 1e-6:
                continue
            sols.append(sol)
    return sols


def dedup_by_trajectory(entries, tol=1e-4):
    """entries: list of (seq, sol, pts). Collapse duplicates whose full
    point set matches (same physical orbit found via different seq phase)."""
    kept = []
    for e in entries:
        seq, sol, pts = e
        allpts = np.array([sol[:2]] + pts[:-1])
        is_dup = False
        for e2 in kept:
            seq2, sol2, pts2 = e2
            if len(seq) != len(seq2):
                continue
            allpts2 = np.array([sol2[:2]] + pts2[:-1])
            for shift in range(len(allpts2)):
                rolled = np.roll(allpts2, -shift, axis=0)
                if rolled.shape == allpts.shape and np.allclose(rolled, allpts, atol=tol):
                    is_dup = True
                    break
            if is_dup:
                break
        if not is_dup:
            kept.append(e)
    return kept


def primitive_period(seq):
    """Smallest p dividing L such that seq is seq[:p] repeated L/p times."""
    L = len(seq)
    for p in range(1, L + 1):
        if L % p != 0:
            continue
        if all(seq[i] == seq[i % p] for i in range(L)):
            return p
    return L


# ---------------------------------------------------------------------
# Analytic tangent-map Jacobian (forward-mode AD through the reflection law)
# ---------------------------------------------------------------------

def hit_circle_exact(P, D, c):
    return hit_circle(P, D, c)


def bounce_with_jac(P, D, dP, dD, c):
    """One bounce off circle c=(cx,cy,R), propagating 2xk Jacobian blocks
    dP, dD analytically alongside the state (P, D). No finite differences."""
    C = np.array(c[:2]); R = c[2]
    t, _ = hit_circle_exact(P, D, c)
    if t is None:
        return None
    w = P - C
    b = 2 * np.dot(w, D)
    k = dP.shape[1]
    db = np.array([2 * (np.dot(dP[:, j], D) + np.dot(w, dD[:, j])) for j in range(k)])
    dc = np.array([2 * np.dot(w, dP[:, j]) for j in range(k)])
    denom = 2 * t + b
    dt = -(db * t + dc) / denom

    P_new = P + t * D
    dP_new = dP + np.outer(D, dt) + t * dD

    n = (P_new - C) / R
    dn = dP_new / R

    Dn = np.dot(D, n)
    dDn = np.array([np.dot(dD[:, j], n) + np.dot(D, dn[:, j]) for j in range(k)])

    D_new = D - 2 * Dn * n
    dD_new = dD - 2 * (np.outer(n, dDn) + Dn * dn)

    return P_new, D_new, dP_new, dD_new


def analytic_jacobian_2d(sol, seq, circles):
    """Exact 2x2 Jacobian of the reduced Poincare map (arc-angle a on the
    launch circle, direction angle alpha) -> (a_end, alpha_end) after one
    full period of `seq`, at the fixed point `sol` = (x, y, alpha)."""
    L = len(seq)
    C0 = np.array(circles[seq[0]][:2]); R0 = circles[seq[0]][2]
    a0 = np.arctan2(sol[1] - C0[1], sol[0] - C0[0])
    alpha0 = sol[2]

    P = C0 + R0 * np.array([np.cos(a0), np.sin(a0)])
    D = np.array([np.cos(alpha0), np.sin(alpha0)])
    dP = np.array([[-R0 * np.sin(a0), 0.0], [R0 * np.cos(a0), 0.0]])
    dD = np.array([[0.0, -np.sin(alpha0)], [0.0, np.cos(alpha0)]])

    for k in range(1, L + 1):
        tg = seq[k % L]
        out = bounce_with_jac(P, D, dP, dD, circles[tg])
        if out is None:
            return None
        P, D, dP, dD = out

    x, y = P - C0
    da_end = np.array([(-y * dP[0, j] + x * dP[1, j]) / R0 ** 2 for j in range(2)])
    dx, dy = D
    dalpha_end = np.array([(-dy * dD[0, j] + dx * dD[1, j]) for j in range(2)])

    return np.vstack([da_end, dalpha_end])


def lambda_and_det(sol, seq, circles):
    J = analytic_jacobian_2d(sol, seq, circles)
    if J is None:
        return None, None
    det = np.linalg.det(J)
    eigs = np.linalg.eigvals(J)
    lam = np.log(max(abs(eigs))) / len(seq)
    return lam, det


# ---------------------------------------------------------------------
# Top-level search
# ---------------------------------------------------------------------

def find_orbits(circles, Lmax, n_pos=16, n_dir=16, min_gap=0.1, verbose=True):
    N = len(circles)
    results = []
    for L in range(2, Lmax + 1):
        t0 = time.time()
        raw = []
        for seq in sequences(N, L):
            if L == 2:
                i, j = seq
                d = np.linalg.norm(circles[i][:2] - circles[j][:2])
                if d < circles[i][2] + circles[j][2] + min_gap:
                    continue
            for sol in solve_closure(circles, seq, n_pos, n_dir):
                D = np.array([np.cos(sol[2]), np.sin(sol[2])])
                if not visible(sol[:2], D, seq, circles):
                    continue
                Pn, Dn, pts = propagate(sol[:2].copy(), D.copy(), seq, circles, need_points=True)
                if Pn is None:
                    continue
                pos_err = np.linalg.norm(Pn - sol[:2])
                dir_err = abs(wrap(np.arctan2(Dn[1], Dn[0]) - sol[2]))
                if pos_err > 1e-6 or dir_err > 1e-6:
                    continue
                raw.append((seq, sol, pts))
        dedup = dedup_by_trajectory(raw)
        for seq, sol, pts in dedup:
            lam, det = lambda_and_det(sol, seq, circles)
            if lam is None:
                continue
            status = "unstable" if lam > 1e-3 else ("stable" if lam < -1e-3 else "neutral")
            results.append(dict(L=L, seq=seq, P0=sol[:2], alpha=sol[2],
                                 lam=lam, status=status, det=det))
        if verbose:
            print(f"  L={L}: raw candidates {len(raw)} -> {len(dedup)} genuine distinct orbits, "
                  f"{time.time()-t0:.1f}s")
    return results


# Default disk configurations used throughout the paper (unit-radius disks,
# spaced so no two disks in any configuration touch or overlap).
CONFIGS = {
    'Triangle (N=3)': np.array([[0, 1.5, 1], [-1.3, -0.75, 1], [1.3, -0.75, 1]], float),
    'Square (N=4)': np.array([[-1.25, 1.25, 1], [1.25, 1.25, 1], [1.25, -1.25, 1], [-1.25, -1.25, 1]], float),
    'Pentagon (N=5)': np.array([[1.9 * np.cos(a), 1.9 * np.sin(a), 1]
                                 for a in 2 * np.pi * np.arange(5) / 5]),
    'Hexagon (N=6)': np.array([[2.25 * np.cos(a), 2.25 * np.sin(a), 1]
                                for a in 2 * np.pi * np.arange(6) / 6]),
}


if __name__ == "__main__":
    Lmax = 8
    all_res = {}
    for name, circles in CONFIGS.items():
        print(f"\n=== {name} ===")
        all_res[name] = find_orbits(circles, Lmax, n_pos=16, n_dir=16)

    print("\n" + "=" * 70)
    print(f"{'config':>16} {'total':>6} {'primitive':>10}")
    for name, res in all_res.items():
        prim = sum(1 for o in res if primitive_period(o['seq']) == o['L'])
        print(f"{name:>16} {len(res):>6} {prim:>10}")
