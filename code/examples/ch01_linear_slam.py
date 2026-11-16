"""Chapter 1 practical: one-dimensional linear SLAM.

A robot drives out along a row of landmarks and comes back. Odometry and
landmark measurements are linear and Gaussian, so the MAP estimate is the
solution of one linear least-squares problem and its covariance is exactly
(A^T A)^{-1}.

Produces:
  figures/ch01_sparsity.pdf    sparsity of H and fill-in after
                               marginalising all poses
  figures/ch01_trajectory.pdf  estimation error with +-3 sigma bands

Usage:
  python examples/ch01_linear_slam.py [--outdir DIR] [--seed N] [--no-prior]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@dataclass
class Problem:
    """Ground truth and measurements of the 1D SLAM problem."""

    x_true: np.ndarray  # (K+1,) poses
    m_true: np.ndarray  # (N,) landmarks
    u: np.ndarray  # (K,) odometry, u[k-1] is the step from x_{k-1} to x_k
    obs: list[tuple[int, int, float]]  # (k, j, z_kj)
    sigma_0: float
    sigma_u: float
    sigma_z: float


def simulate(seed: int = 0, n_landmarks: int = 11,
             sensor_range: float = 0.75, step: float = 0.5,
             sigma_0: float = 0.01, sigma_u: float = 0.1,
             sigma_z: float = 0.1) -> Problem:
    """Out-and-back trajectory along a row of landmarks spaced 1 m apart.

    Landmarks sit at 0, 1, ..., n_landmarks-1; the robot drives from 0 to
    the last landmark and back in steps of `step`.
    """
    rng = np.random.default_rng(seed)
    m_true = np.arange(n_landmarks, dtype=float)
    end = float(n_landmarks - 1)
    out = np.arange(0.0, end + 1e-9, step)
    x_true = np.concatenate([out, out[-2::-1]])  # there and back again
    K = len(x_true) - 1
    u = np.diff(x_true) + rng.normal(0.0, sigma_u, K)
    obs = []
    for k, xk in enumerate(x_true):
        for j, mj in enumerate(m_true):
            if abs(mj - xk) <= sensor_range:
                z = mj - xk + rng.normal(0.0, sigma_z)
                obs.append((k, j, z))
    return Problem(x_true, m_true, u, obs, sigma_0, sigma_u, sigma_z)


def build_system(pb: Problem, with_prior: bool = True
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Whitened linear system A theta = b, theta = (x_0..x_K, m_1..m_N)."""
    K1, N = len(pb.x_true), len(pb.m_true)
    rows, rhs = [], []

    def row() -> np.ndarray:
        return np.zeros(K1 + N)

    if with_prior:
        r = row()
        r[0] = 1.0 / pb.sigma_0
        rows.append(r)
        rhs.append(0.0)
    for k in range(1, K1):  # odometry: x_k - x_{k-1} = u_k
        r = row()
        r[k - 1], r[k] = -1.0 / pb.sigma_u, 1.0 / pb.sigma_u
        rows.append(r)
        rhs.append(pb.u[k - 1] / pb.sigma_u)
    for k, j, z in pb.obs:  # observation: m_j - x_k = z_kj
        r = row()
        r[k], r[K1 + j] = -1.0 / pb.sigma_z, 1.0 / pb.sigma_z
        rows.append(r)
        rhs.append(z / pb.sigma_z)
    return np.array(rows), np.array(rhs)


def filtered_sigmas(pb: Problem) -> np.ndarray:
    """Standard deviation of x_k given only the data up to time k.

    This is what a filter reports at time k. We obtain it by solving the
    truncated problem at every k, which is wasteful but transparent.
    Landmarks not yet observed are left out of the truncated problem.
    """
    K1 = len(pb.x_true)
    out = np.empty(K1)
    for k in range(K1):
        obs_k = [o for o in pb.obs if o[0] <= k]
        sub = Problem(pb.x_true[:k + 1], pb.m_true, pb.u[:k], obs_k,
                      pb.sigma_0, pb.sigma_u, pb.sigma_z)
        A, _ = build_system(sub)
        seen = sorted({j for _, j, _ in obs_k})
        idx = list(range(k + 1)) + [k + 1 + j for j in seen]
        Hk = (A.T @ A)[np.ix_(idx, idx)]
        out[k] = np.sqrt(np.linalg.inv(Hk)[k, k])
    return out


def marginalise(Lam: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Schur complement: information of the `keep` block.

    Implements the Gaussian marginalisation lemma of Chapter 1.
    """
    drop = np.setdiff1d(np.arange(Lam.shape[0]), keep)
    Laa = Lam[np.ix_(keep, keep)]
    Lab = Lam[np.ix_(keep, drop)]
    Lbb = Lam[np.ix_(drop, drop)]
    return Laa - Lab @ np.linalg.solve(Lbb, Lab.T)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outdir", type=Path,
                    default=Path(__file__).resolve().parents[2]
                    / "course" / "figures")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-prior", action="store_true",
                    help="drop the prior on x_0 and inspect the gauge")
    args = ap.parse_args()

    pb = simulate(seed=args.seed)
    K1, N = len(pb.x_true), len(pb.m_true)
    A, b = build_system(pb, with_prior=not args.no_prior)
    H = A.T @ A

    if args.no_prior:
        print("smallest eigenvalues of H:", np.linalg.eigvalsh(H)[:3])
        print("|H @ 1| =", np.linalg.norm(H @ np.ones(K1 + N)))
        return

    theta = np.linalg.solve(H, A.T @ b)
    cov = np.linalg.inv(H)
    sigma = np.sqrt(np.diag(cov))
    x_hat, m_hat = theta[:K1], theta[K1:]
    print(f"{K1} poses, {N} landmarks, {len(pb.obs)} observations")
    print("RMS pose error     :",
          np.sqrt(np.mean((x_hat - pb.x_true) ** 2)))
    print("RMS landmark error :",
          np.sqrt(np.mean((m_hat - pb.m_true) ** 2)))

    args.outdir.mkdir(parents=True, exist_ok=True)

    # Figure 1: sparsity of H and fill-in after marginalising the poses.
    Lmm = marginalise(H, np.arange(K1, K1 + N))
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.6))
    ax[0].spy(np.abs(H) > 1e-12, markersize=2.5, color="0.15")
    ax[0].axhline(K1 - 0.5, color="tab:green", lw=0.8)
    ax[0].axvline(K1 - 0.5, color="tab:green", lw=0.8)
    ax[0].set_title(r"$H=A^\top A$ (poses | landmarks)")
    im = ax[1].imshow(np.log10(np.abs(Lmm) + 1e-12), cmap="Greens",
                      vmin=-6)
    ax[1].set_title("landmark information after\n"
                    "marginalising all poses ($\\log_{10}|\\cdot|$)")
    fig.colorbar(im, ax=ax[1], fraction=0.046)
    fig.tight_layout()
    fig.savefig(args.outdir / "ch01_sparsity.pdf")

    # Figure 2: filtered versus smoothed 3-sigma bands, and the error.
    k = np.arange(K1)
    err = x_hat - pb.x_true
    sig_f = filtered_sigmas(pb)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.fill_between(k, -3 * sig_f, 3 * sig_f, color="tab:orange",
                    alpha=0.18, label=r"filtered $\pm3\sigma$ (data up to $k$)")
    ax.fill_between(k, -3 * sigma[:K1], 3 * sigma[:K1], color="tab:green",
                    alpha=0.35, label=r"smoothed $\pm3\sigma$ (all data)")
    ax.plot(k, err, color="0.15", lw=1.2, label="smoothed MAP error")
    ax.axvline((K1 - 1) / 2, color="0.5", ls="--", lw=0.8)
    ax.text((K1 - 1) / 2 + 0.5, 3 * sig_f.max() * 0.85, "turn back",
            color="0.4")
    ax.set_xlabel("time step $k$")
    ax.set_ylabel(r"$\hat x_k - x_k$ [m]")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(args.outdir / "ch01_trajectory.pdf")
    print("figures written to", args.outdir)


if __name__ == "__main__":
    main()
