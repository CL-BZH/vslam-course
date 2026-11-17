# Visual SLAM for an Autonomous Lawnmower

A course on visual SLAM, from geometry and optimisation to
visual-inertial fusion. The running example is a ride-on mower converted
into an autonomous robot, but the material is general. Each chapter has full derivations in LaTeX and a from-scratch
Python practical.

## Contents

| Part | Chapters | Status |
|---|---|---|
| I. Foundations | 1 The SLAM problem · 2 Rigid motion and Lie groups · 3 Cameras · 4 Nonlinear least squares on manifolds | drafted |
| II. Visual front-end | 5 Features and correspondence · 6 Two-view geometry · 7 Visual odometry | drafted |
| III. Back-end | 8 Bundle adjustment · 9 Factor and pose graphs · 10 Loop closure · 11 Maps for a mower | drafted |
| IV. Fusion for the garden | 12 The IMU · 13 Visual-inertial odometry · 14 Wheels and RTK-GNSS · 15 Capstone | 12–13 drafted |

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
- `examples/ch07_vo.py` — PnP with outliers and on a lawn, stereo VO on a
  simulated garden mown in rows (keyframe vs frame-to-frame), drift.
- `examples/ch08_ba.py` — structure of the normal equations and cost of
  the Schur complement, covariances against Monte Carlo, the VO with a
  sliding-window bundle adjustment.
- `examples/ch09_posegraph.py` — pose-graph optimisation of the VO with
  loop closures between rows, wrong loops and robust kernels,
  marginalisation and the linearisation point.
- `examples/ch10_loops.py` — bag-of-binary-words place recognition on
  simulated descriptors (with an aliased fence), precision and recall,
  geometric verification, and the pose graph with the loops found.
- `examples/ch11_maps.py` — dense stereo on a ray-cast garden, ground
  planes, log-odds occupancy and lawn layers fused with true, VO and
  pose-graph poses, and the export to the
  [Lawnmower](https://github.com/CL-BZH/Lawnmower) coverage planners
  (used if installed: `pip install -e path/to/Lawnmower`).
- `examples/ch12_imu.py` — Allan deviations of a simulated IMU, strapdown
  drift against theory, preintegration covariance against Monte Carlo,
  the first-order bias update and the discretisation error.
- `examples/ch13_vio.py` — sliding-window VIO on a smooth mowing drive
  with an exact IMU: against stereo VO, with the camera blind in a turn,
  the four unobservable directions, the tilt-bias ambiguity of straight
  rows, initialisation, and the cost of a camera-IMU time offset
  (about 6 minutes).

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
- `notebooks/ch07_vo.ipynb`: linear PnP and its planar degeneracy, the
  four P3P solutions, robust refinement, the simulated garden, running
  the VO, the drift law against Monte Carlo, ATE versus RPE.
- `notebooks/ch08_ba.ipynb`: the arrowhead, Schur step against the dense
  step, the gauge, solving and the chi-square check, covariance against
  Monte Carlo, local BA in the VO.
- `notebooks/ch09_posegraph.ipynb`: the SE(3) right Jacobian, a loop
  around a square, the garden pose graph and the odometry noise model,
  wrong loops, marginalisation and the gauge.
- `notebooks/ch10_loops.ipynb`: descriptor aliasing, k-majority, the
  vocabulary and the scores, verification of one loop, all loops and the
  pose graph, under strong aliasing too.
- `notebooks/ch11_maps.ipynb`: the dense garden, stereo point errors
  against Monte Carlo, the ground plane, the vote bound, maps with three
  sets of poses, and the planners on the exported grid.
- `notebooks/ch12_imu.ipynb`: specific force, noise discretisation and
  Allan deviation, strapdown drift, preintegration against direct
  integration, its covariance, and the IMU factor Jacobians.
- `notebooks/ch13_vio.ipynb`: the drive and its IMU, the visual factor
  Jacobians, VIO against VO on 20 s, the null space of the window
  Hessian, tilt against accelerometer bias, the MSCKF projection, and
  initialisation.
