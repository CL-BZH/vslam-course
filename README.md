# Visual SLAM for an Autonomous Lawnmower

A course on visual SLAM, from geometry and optimisation to
visual-inertial fusion, written with the PG98 HK robot-mower project in
mind. Each chapter has full derivations in LaTeX and a from-scratch
Python practical.

## Contents

| Part | Chapters | Status |
|---|---|---|
| I. Foundations | 1 The SLAM problem · 2 Rigid motion and Lie groups · 3 Cameras · 4 Nonlinear least squares on manifolds | 1–2 drafted |
| II. Visual front-end | 5 Features and correspondence · 6 Two-view geometry · 7 Visual odometry | planned |
| III. Back-end | 8 Bundle adjustment · 9 Factor and pose graphs · 10 Loop closure · 11 Maps for a mower | planned |
| IV. Fusion for the garden | 12 The IMU · 13 Visual-inertial odometry · 14 Wheels and RTK-GNSS · 15 Capstone | planned |

## Layout

```
course/          LaTeX sources (main.tex, preamble.tex, chapters/, figures/)
code/vslam/      the Python package, grown chapter by chapter
code/tests/      pytest suite (closed forms and Jacobian checks)
code/examples/   one script per practical; they also regenerate the figures
```

## Building

```bash
make pdf        # course/main.pdf (needs latexmk + a TeX Live install)
make test       # runs the pytest suite
make figures    # reruns the practicals and rewrites course/figures/
```

Python setup (no GPU needed):

```bash
cd code
pip install -e ".[dev]"
```

## Practicals so far

- `examples/ch01_linear_slam.py` — 1D SLAM as one linear least-squares
  problem: sparsity of the normal equations, fill-in from marginalisation,
  filtered vs smoothed uncertainty, and the gauge null space
  (`--no-prior`).
- `examples/ch02_banana.py` — wheel-odometry uncertainty on SE(3):
  Monte Carlo versus a Gaussian in the Lie algebra (the "banana").
