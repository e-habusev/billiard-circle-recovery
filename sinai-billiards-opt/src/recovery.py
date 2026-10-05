"""
recovery.py
Linear circle-center recovery from collision normals in dispersing billiards
(Theorem 1 of the paper), plus the two competitor estimators it is compared
against: the algebraic (Kasa) fit and a known-radius geometric fit.

This module exposes the estimators and the data generator; see
tests/run_benchmarks.py to reproduce Table 1 of the paper.
"""
import numpy as np
from scipy.optimize import least_squares

R, SIGMA, N, TRIALS = 1.0, 0.01, 50, 300
THETAS_DEG = np.array([0.5, 1, 2, 5, 10, 20, 45, 90])


def make_data(theta, rng, R=R, sigma=SIGMA, n_points=N):
    """Generate N noisy collision points + normals on an arc of angular
    span `theta` (radians) of a circle of radius R centered at the origin."""
    phi = rng.uniform(np.pi / 2 - theta / 2, np.pi / 2 + theta / 2, n_points)
    n = np.c_[np.cos(phi), np.sin(phi)]
    P = R * n
    return (P + sigma * rng.normal(size=P.shape),
            n + sigma * rng.normal(size=n.shape))


def estimator_normal(P, n, R=R):
    """Eq. (2) of the paper: Chat = (1/N) sum (P_i - R n_i).
    Linear, unbiased, Gauss-Markov optimal (Theorem 1)."""
    return (P - R * n).mean(axis=0)


def estimator_kasa(P, n=None):
    """Algebraic (Kasa) fit; radius is estimated, not assumed known."""
    A = np.c_[2 * P, np.ones(len(P))]
    b = (P ** 2).sum(axis=1)
    u = np.linalg.lstsq(A, b, rcond=None)[0]
    return u[:2]


def estimator_geometric_known_R(P, n=None, R=R):
    """Known-radius geometric (orthogonal) fit, solved iteratively."""
    m = P.mean(axis=0)
    c0 = m - R * np.array([0.0, 1.0])
    f = lambda c: np.linalg.norm(P - c, axis=1) - R
    return least_squares(f, c0).x


METHODS = {
    "kasa": estimator_kasa,
    "geometric_known_R": estimator_geometric_known_R,
    "normal_based": estimator_normal,
}


def make_data_from_rays(theta, rng, R=R, sigma=SIGMA, n_points=N,
                         max_incidence=np.deg2rad(60)):
    """Alternative data generator: normals are reconstructed from noisy
    incoming/outgoing ray directions rather than supplied directly, for a
    given maximum angle of incidence. Used to confirm the arc-independence
    result is not an artifact of idealized additive normal noise."""
    phi = rng.uniform(np.pi / 2 - theta / 2, np.pi / 2 + theta / 2, n_points)
    n = np.c_[np.cos(phi), np.sin(phi)]
    t = np.c_[-n[:, 1], n[:, 0]]
    a = rng.uniform(-max_incidence, max_incidence, n_points)
    d_in = -(np.cos(a)[:, None] * n + np.sin(a)[:, None] * t)
    d_out = d_in - 2 * (d_in * n).sum(1)[:, None] * n
    P = R * n + sigma * rng.normal(size=(n_points, 2))
    d_in_noisy = d_in + sigma * rng.normal(size=(n_points, 2))
    d_out_noisy = d_out + sigma * rng.normal(size=(n_points, 2))
    v = d_out_noisy - d_in_noisy
    n_est = v / np.linalg.norm(v, axis=1)[:, None]
    return P, n_est
