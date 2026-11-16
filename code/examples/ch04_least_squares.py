"""Chapter 4 practical: Gauss-Newton, Levenberg-Marquardt, robust kernels.

Produces three figures in course/figures:

  ch04_gn_rate.pdf      convergence of Gauss-Newton, Newton and LM on the
                        one-dimensional problem r(x) = (x+1, lam x^2+x-1)
  ch04_calibration.pdf  finishing the calibration of Chapter 3: cost per
                        iteration (GN and LM) and reprojection residuals
                        before and after refinement
  ch04_robust.pdf       robust kernels: IRLS weights, and the error of a
                        camera pose estimated from correspondences
                        contaminated by outliers

Usage:
  python examples/ch04_least_squares.py [--outdir DIR] [--trials N]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from vslam import calib, lie, optim, robust  # noqa: E402
from vslam.camera import PinholeCamera, look_at  # noqa: E402

WIDE = PinholeCamera(458.7, 457.3, 367.2, 248.4,
                     dist=[-0.283, 0.074, 1.8e-4, 1.8e-5, 0.0],
                     width=752, height=480)


# --------------------------------------------------------------------------
# Problems
# --------------------------------------------------------------------------
class MNT:
    """r(x) = (x + 1, lam x^2 + x - 1); minimiser x* = 0 when lam < 1."""

    block_size = 1

    def __init__(self, lam: float) -> None:
        self.lam = lam

    def linearize(self, x):
        x0 = x[0]
        return (np.array([x0 + 1.0, self.lam * x0 * x0 + x0 - 1.0]),
                np.array([[1.0], [2.0 * self.lam * x0 + 1.0]]))

    def retract(self, x, d):
        return x + d

    def newton(self, x0: float, iters: int) -> list[float]:
        """Newton's method with the exact Hessian, for comparison."""
        xs = [x0]
        for _ in range(iters):
            x = xs[-1]
            r, J = self.linearize(np.array([x]))
            g = float(J[:, 0] @ r)
            h = float(J[:, 0] @ J[:, 0] + r[1] * 2.0 * self.lam)
            xs.append(x - g / h)
        return xs


class PoseProblem:
    """Motion-only reprojection: camera pose T_wc from known world points
    and their pixels (the problem of Chapter 7, here with a given start)."""

    block_size = 2

    def __init__(self, cam: PinholeCamera, P: np.ndarray, z: np.ndarray,
                 sigma: float = 1.0) -> None:
        self.cam, self.P, self.z, self.sigma = cam, P, z, sigma

    def linearize(self, T):
        r = np.empty(2 * len(self.P))
        J = np.empty((2 * len(self.P), 6))
        for j, (p, zj) in enumerate(zip(self.P, self.z)):
            u, J_xi, _ = self.cam.project_world(T, p)
            r[2 * j:2 * j + 2] = (zj - u) / self.sigma
            J[2 * j:2 * j + 2] = -J_xi / self.sigma
        return r, J

    def retract(self, T, d):
        return T @ lie.se3_exp(d)


def pose_scene(rng: np.random.Generator, outlier_frac: float,
               n: int = 100, sigma: float = 0.5):
    """A garden-like scene seen by a camera 1.2 m above the ground, with a
    fraction of the correspondences replaced by random pixels."""
    cam = PinholeCamera(500.0, 500.0, 320.0, 240.0)
    T = look_at(np.array([0.0, -5.0, 1.2]), np.array([0.0, 2.0, 0.6]))
    P = np.column_stack([rng.uniform(-4, 4, n), rng.uniform(0, 8, n),
                         rng.uniform(0, 2.5, n)])
    z = np.array([cam.project_world(T, p)[0] for p in P])
    z += rng.normal(scale=sigma, size=z.shape)
    out = rng.random(n) < outlier_frac
    z[out] = rng.uniform([0, 0], [640, 480], size=(out.sum(), 2))
    return cam, T, P, z


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def fig_gn_rate(outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.8))
    x0, n = 0.05, 25
    greens = plt.cm.Greens(np.linspace(0.45, 0.95, 3))
    for lam, col in zip([0.1, 0.5, 0.9], greens):
        res = optim.gauss_newton(MNT(lam), np.array([x0]), max_iters=n,
                                 tol=0.0)
        xs = [abs(h["x"][0]) for h in res.history]
        ax.semilogy(xs, "o-", ms=3, color=col, label=f"GN, $\\lambda$={lam}")
    res = optim.gauss_newton(MNT(-2.0), np.array([x0]), max_iters=n,
                             tol=0.0)
    ax.semilogy([abs(h["x"][0]) for h in res.history], "x-", ms=4,
                color="tab:red", label="GN, $\\lambda$=$-2$")
    lm = optim.levenberg_marquardt(MNT(-2.0), np.array([x0]),
                                   marquardt=False, tol_cost=0.0,
                                   tol_step=0.0, max_iters=n)
    ax.semilogy([abs(h["x"][0]) for h in lm.history], "s-", ms=3,
                color="tab:blue", label="LM, $\\lambda$=$-2$")
    newt = MNT(0.5).newton(x0, 6)
    ax.semilogy(np.abs(newt), "d--", ms=4, color="0.3",
                label="Newton, $\\lambda$=0.5")
    ax.set_ylim(1e-16, 10)
    ax.set_xlabel("iteration $k$")
    ax.set_ylabel("$|x_k - x^\\star|$")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(outdir / "ch04_gn_rate.pdf")


def fig_calibration(outdir: Path) -> None:
    rng = np.random.default_rng(0)
    sigma = 0.3
    board, views = calib.synthetic_views(WIDE, 20, rng, sigma, distance=0.5,
                                         spread=0.1)
    px = [u for _, u in views]
    _, _, res_lm = calib.calibrate(board, px, WIDE.width, WIDE.height,
                                   sigma, method="lm")
    est, poses, res_gn = calib.calibrate(board, px, WIDE.width, WIDE.height,
                                         sigma, method="gn")
    m = 2 * len(board) * len(px)
    n = 8 + 6 * len(px)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].semilogy(res_gn.costs, "o-", ms=3, color="tab:green", label="GN")
    ax[0].semilogy(res_lm.costs, "s-", ms=3, color="tab:blue", label="LM")
    ax[0].axhline(0.5 * (m - n), color="0.4", ls="--", lw=0.8)
    ax[0].text(len(res_lm.costs) * 0.45, 0.5 * (m - n) * 1.25,
               "$(m-n)/2$", color="0.3")
    ax[0].set_xlabel("iteration")
    ax[0].set_ylabel("cost $F$ (whitened)")
    ax[0].legend()
    # residuals before (Zhang, no distortion) and after refinement
    pb = calib.CalibrationProblem(board, px, 1.0, WIDE.width, WIDE.height)
    r_after, _ = pb.linearize(res_gn.x)
    r_before, _ = pb.linearize(res_gn.history[0]["x"])
    bins = np.linspace(0, 12, 61)
    for r, col, lab in [(r_before, "tab:orange", "linear step (Zhang)"),
                        (r_after, "tab:green", "after refinement")]:
        norms = np.linalg.norm(r.reshape(-1, 2), axis=1)
        ax[1].hist(norms, bins=bins, color=col, alpha=0.6,
                   label=f"{lab}: RMS {np.sqrt(np.mean(norms**2)):.2f} px")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("reprojection residual $\\|\\mathbf{e}\\|$ [px]")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "ch04_calibration.pdf")
    print("calibration: true  ", WIDE.params.round(4))
    print("calibration: est.  ", est.params.round(4))
    print(f"iterations GN {res_gn.iterations}, LM {res_lm.iterations}; "
          f"final cost {res_gn.costs[-1]:.1f}, (m-n)/2 = {(m - n) / 2:.1f}")


def fig_robust(outdir: Path, trials: int) -> None:
    kernels = [("least squares", robust.Trivial(), "tab:orange"),
               ("Huber, $k=2$", robust.Huber(2.0), "tab:blue"),
               ("Cauchy, $k=2$", robust.Cauchy(2.0), "tab:green")]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    r = np.linspace(0, 10, 400)
    for name, k, col in kernels:
        ax[0].plot(r, k.weight(r**2), color=col, label=name)
    ax[0].set_xlabel("whitened residual norm $\\|\\mathbf{r}_i\\|$")
    ax[0].set_ylabel("IRLS weight $w=\\rho'(s)$")
    ax[0].legend(fontsize=8)
    rng = np.random.default_rng(3)
    fracs = np.linspace(0.0, 0.6, 7)
    for name, k, col in kernels:
        med = []
        for f in fracs:
            errs = []
            for _ in range(trials):
                cam, T, P, z = pose_scene(rng, f)
                T0 = T @ lie.se3_exp(np.concatenate([
                    rng.normal(scale=0.15, size=3),
                    rng.normal(scale=0.03, size=3)]))
                res = optim.levenberg_marquardt(PoseProblem(cam, P, z, 0.5),
                                                T0, kernel=k)
                errs.append(np.linalg.norm(lie.se3_inv(T)[:3, :3]
                                           @ (res.x[:3, 3] - T[:3, 3])))
            med.append(np.median(errs))
        ax[1].semilogy(100 * fracs, med, "o-", color=col, label=name)
    ax[1].set_xlabel("outliers [%]")
    ax[1].set_ylabel("median position error [m]")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "ch04_robust.pdf")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outdir", type=Path,
                    default=Path(__file__).resolve().parents[2]
                    / "course" / "figures")
    ap.add_argument("--trials", type=int, default=20)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    fig_gn_rate(args.outdir)
    fig_calibration(args.outdir)
    fig_robust(args.outdir, args.trials)
    print("figures written to", args.outdir)


if __name__ == "__main__":
    main()
