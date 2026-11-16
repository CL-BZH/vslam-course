# Visual SLAM for an Autonomous Lawnmower

A course on visual SLAM, from geometry and optimisation to
visual-inertial fusion, written with the PG98 HK robot-mower project in
mind. Each chapter has full derivations in LaTeX and a from-scratch
Python practical.

## Contents

| Part | Chapters | Status |
|---|---|---|
| I. Foundations | 1 The SLAM problem · 2 Rigid motion and Lie groups · 3 Cameras · 4 Nonlinear least squares on manifolds | drafted |
| II. Visual front-end | 5 Features and correspondence · 6 Two-view geometry · 7 Visual odometry | 5–6 drafted |
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
- `examples/ch03_cameras.py` — lens distortion field, accuracy of Zhang's
  linear calibration versus noise and number of views (with and without
  unmodelled distortion), stereo depth uncertainty.
- `examples/ch04_least_squares.py` — Gauss-Newton's linear rate and its
  failure on a large-residual problem, finishing the Chapter 3
  calibration with GN and LM, robust kernels against outliers.
- `examples/ch05_features.py` — synthetic garden and lawn textures,
  matching precision under viewpoint, lighting and wind, the RANSAC
  iteration law, pyramidal Lucas-Kanade.
- `examples/ch06_twoview.py` — eight-point, five-point and homography
  relative pose on a lawn versus a garden and towards pure rotation;
  triangulation accuracy and the forward-motion blind spot.

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
- `notebooks/ch03_cameras.ipynb`: projecting a garden scene, Newton
  undistortion, reprojection Jacobians (near vs far features), Zhang
  calibration of a simulated wide-angle camera with its residuals, stereo
  depth, and the pure-rotation homography (Exercise 3.2).
- `notebooks/ch04_least_squares.ipynb`: the Gauss-Newton rate, LM's
  damping and gain ratio, the finished calibration, Laplace covariance
  against Monte Carlo, IRLS weights on outliers, optimising on SO(3).
- `notebooks/ch05_features.ipynb`: structure-tensor eigenvalues, steered
  vs plain BRIEF under rotation, the ratio test, the lawn in the wind for
  matching and tracking, the aperture problem, RANSAC's adaptive count,
  and OpenCV's ORB.
- `notebooks/ch06_twoview.ipynb`: epipolar lines, the four decompositions,
  estimators on lawns and gardens, the planar twin solution, choosing
  between homography and essential matrix, parallax.
