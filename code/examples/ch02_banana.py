"""Chapter 2 practical: uncertainty of a mower driving on wheel odometry.

The mower compounds noisy SE(3) increments T_k = T_{k-1} U Exp(eps_k),
eps_k ~ N(0, Q). We compare three descriptions of where it ends up:

1. Monte Carlo: compound many noisy trajectories (the ground truth).
2. A Gaussian in the Lie algebra, T = T_bar Exp(xi), xi ~ N(0, Sigma),
   with Sigma propagated by the first-order compounding rule
   Sigma_k = Ad(U^-1) Sigma_{k-1} Ad(U^-1)^T + Q (Chapter 2,
   "Uncertainty on groups").
3. A Gaussian on the (x, y) coordinates fitted to the Monte Carlo samples.

The heading uncertainty bends the cloud into a "banana". The Lie-algebra
Gaussian reproduces it; any ellipse in (x, y) cannot.

Usage:
  python examples/ch02_banana.py [--outdir DIR] [--seed N]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402

from vslam import lie  # noqa: E402

# One odometry step: 5 cm forward, slight left turn (twist = (rho, phi)).
STEP = np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.002])
# Per-step noise: forward, lateral, vertical, roll, pitch, yaw (std).
SIGMAS = np.array([0.004, 0.001, 1e-4, 1e-4, 1e-4, 0.03])
N_STEPS = 150
N_SAMPLES = 2000


def monte_carlo(rng: np.random.Generator) -> np.ndarray:
    """Final poses of N_SAMPLES noisy trajectories, shape (N, 4, 4)."""
    U = lie.se3_exp(STEP)
    finals = np.empty((N_SAMPLES, 4, 4))
    for n in range(N_SAMPLES):
        T = np.eye(4)
        for _ in range(N_STEPS):
            T = T @ U @ lie.se3_exp(rng.normal(0.0, SIGMAS))
        finals[n] = T
    return finals


def propagate() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean pose, Lie-algebra covariance and mean trajectory."""
    U = lie.se3_exp(STEP)
    Ad = lie.se3_adjoint(lie.se3_inv(U))
    Q = np.diag(SIGMAS**2)
    T_bar, Sigma = np.eye(4), np.zeros((6, 6))
    path = [T_bar[:3, 3].copy()]
    for _ in range(N_STEPS):
        T_bar = T_bar @ U
        Sigma = Ad @ Sigma @ Ad.T + Q
        path.append(T_bar[:3, 3].copy())
    return T_bar, Sigma, np.array(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outdir", type=Path,
                    default=Path(__file__).resolve().parents[2]
                    / "course" / "figures")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    mc = monte_carlo(rng)
    T_bar, Sigma, path = propagate()

    # Check the propagated covariance against the Monte Carlo one,
    # measured in the same right-perturbation coordinates.
    xi_mc = np.array([lie.se3_log(lie.se3_inv(T_bar) @ T) for T in mc])
    print("std (Lie, propagated):", np.sqrt(np.diag(Sigma)).round(4))
    print("std (Lie, Monte Carlo):", xi_mc.std(axis=0).round(4))

    lie_xy = np.array([
        lie.se3_act(T_bar @ lie.se3_exp(xi), np.zeros(3))[:2]
        for xi in rng.multivariate_normal(np.zeros(6), Sigma, N_SAMPLES)])
    mc_xy = mc[:, :2, 3]

    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.4), sharex=True,
                           sharey=True)
    for a, pts, col, name in [(ax[0], mc_xy, "0.35", "Monte Carlo"),
                              (ax[1], lie_xy, "tab:green",
                               "Gaussian in the Lie algebra")]:
        a.scatter(pts[:, 0], pts[:, 1], s=1.5, color=col, alpha=0.5,
                  label=name)
        a.plot(path[:, 0], path[:, 1], color="k", lw=1.0,
               label="mean path")
        a.set_aspect("equal")
        a.set_xlabel("x [m]")
    # 3-sigma ellipse of a Cartesian Gaussian fitted to the MC samples.
    mu, C = mc_xy.mean(axis=0), np.cov(mc_xy.T)
    w, V = np.linalg.eigh(C)
    ang = np.degrees(np.arctan2(V[1, 1], V[0, 1]))
    for a in ax:
        a.add_patch(Ellipse(mu, 6 * np.sqrt(w[1]), 6 * np.sqrt(w[0]),
                            angle=ang, fill=False, color="tab:orange",
                            lw=1.5, label=r"$(x,y)$ Gaussian, $3\sigma$"))
        a.legend(loc="upper left", fontsize=8, markerscale=6)
    ax[0].set_ylabel("y [m]")
    fig.tight_layout()
    args.outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.outdir / "ch02_banana.pdf")
    print("figure written to", args.outdir / "ch02_banana.pdf")


if __name__ == "__main__":
    main()
