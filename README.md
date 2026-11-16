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
code/notebooks/  one Jupyter notebook per chapter (theory checks + practicals)
```

## Building

```bash
make pdf        # course/main.pdf (needs latexmk + a TeX Live install)
make test       # runs the pytest suite
make figures    # reruns the practicals and rewrites course/figures/
make notebooks  # re-executes the notebooks in place
```

Python setup (no GPU needed):

```bash
cd code
pip install -e ".[dev]"
pip install jupyter      # for the notebooks
cd notebooks && jupyter lab
```

## Practicals so far

- `examples/ch01_linear_slam.py` — 1D SLAM as one linear least-squares
  problem: sparsity of the normal equations, fill-in from marginalisation,
  filtered vs smoothed uncertainty, and the gauge null space
  (`--no-prior`).
- `examples/ch02_banana.py` — wheel-odometry uncertainty on SE(3):
  Monte Carlo versus a Gaussian in the Lie algebra (the "banana").

## Notebooks

The notebooks walk through each chapter interactively: every statement
that can be checked numerically (identities, closed forms, Jacobians) is
checked, and the practicals are rebuilt step by step with plots. They are
committed with their outputs, so they can be read directly on GitHub.

- `notebooks/ch01_linear_slam.ipynb`: the 1D SLAM problem, sparsity,
  fill-in (including Exercise 1.3), filtering vs smoothing, the gauge.
- `notebooks/ch02_lie_groups.ipynb`: hat identities, Rodrigues and Log
  at the edge angles, the SE(3) screw motion, the adjoint, Jacobians
  against finite differences (including Exercise 2.3), the banana.
