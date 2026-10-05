# sinai-billiards-opt

Code accompanying the paper *"Linear Circle Recovery from Normal Fields, and
a Verified Catalog of Short Periodic Orbits in Dispersing Sinai Billiards"*
(I. Habusev, 2026).

## Structure

```
sinai-billiards-opt/
├── data/
│   └── orbits_raw.json     # Full catalog of isolated periodic orbits, L<=8,
│                            # all four N-gon configurations (sequence, fixed
│                            # point, direction, Lyapunov exponent, det J)
├── src/
│   ├── recovery.py          # Linear normal-based estimator (Theorem 1),
│   │                          plus the Kasa and known-R geometric competitors
│   ├── orbits.py             # Orbit finder: multistart Newton closure solver,
│   │                          visibility filter, deduplication, and the exact
│   │                          analytic tangent-map Jacobian for stability
│   └── validation.py         # Independent cross-check of every orbit's
│                               Lyapunov exponent via the classical
│                               curvature-based transfer-matrix formula
├── tests/
│   └── run_benchmarks.py     # Reproduces Table 1 (circle-recovery comparison)
├── LICENSE                   # MIT
└── README.md
```

## Setup

```bash
pip install numpy scipy
```

## Reproducing the paper

**Table 1** (circle-recovery comparison vs. arc span, and the
realistic-normals robustness check):
```bash
cd tests
python run_benchmarks.py
```

**Table 2** (periodic-orbit catalog, L <= 8, all four configurations).
This is a full re-run of the search that produced `data/orbits_raw.json`;
it takes roughly 30-40 minutes single-threaded (the hexagon L=8 sweep alone
processes 25,395 itineraries):
```bash
cd src
python orbits.py
```

**Independent validation** of the orbit catalog's stability exponents
(classical transfer-matrix cross-check against `data/orbits_raw.json`):
```bash
cd src
python validation.py
```
Expected output: agreement to ~1e-8 between the two independent stability
calculations, and `det(J)` within ~2e-4 of 1 (exact symplecticity) for
every orbit.

## Geometry

Disks have unit radius. Configurations are regular N-gons (triangle,
square, pentagon, hexagon) with spacing chosen so that no two disks touch
or overlap (minimum boundary-to-boundary gap >= 0.23 in every
configuration). Exact coordinates are in `src/orbits.py` (`CONFIGS`).

## License

MIT — see `LICENSE`.
