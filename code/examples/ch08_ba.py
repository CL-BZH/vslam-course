"""Chapter 8 practical: bundle adjustment.

Produces three figures in course/figures:

  ch08_structure.pdf   sparsity of the normal equations and of the
                       reduced camera system, and the cost of solving them
                       densely versus with the Schur complement
  ch08_covariance.pdf  camera and point uncertainty from the reduced
                       system, against Monte Carlo
  ch08_vo.pdf          the stereo VO of Chapter 7 with and without a
                       sliding-window bundle adjustment

Usage:
  python examples/ch08_ba.py [--outdir DIR] [--seeds N]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from vslam import ba, lie, metrics, sim, vo  # noqa: E402
from vslam.camera import look_at  # noqa: E402

K = np.array([[458.7, 0, 367.2], [0, 457.3, 248.4], [0, 0, 1.0]])
SIGMA = 0.5


def synthetic_ba(n_cams: int = 8, n_pts: int = 200, noise: float = SIGMA,
                 stereo: bool = True, visible: float = 1.0, seed: int = 0,
                 perturb: bool = True):
    """Cameras moving along a line past a cloud of points; each camera
    sees a contiguous band of the points (fraction `visible`). Returns the
    problem (first camera fixed), the true cameras and points."""
    rng = np.random.default_rng(seed)
    xs = np.linspace(-3, 3, n_cams)
    cams = [look_at(np.array([x, -6.0, 1.0]), np.array([x, 0.0, 1.0]))
            for x in xs]
    pts = np.column_stack([rng.uniform(-4, 4, n_pts),
                           rng.uniform(-2, 3, n_pts),
                           rng.uniform(0, 2.5, n_pts)])
    pts = pts[np.argsort(pts[:, 0])]
    ci, pi, zs = [], [], []
    for c, T in enumerate(cams):
        lo = (c / max(n_cams - 1, 1)) * (1 - visible)
        sel = np.arange(int(lo * n_pts), int((lo + visible) * n_pts))
        q = (pts[sel] - T[:3, 3]) @ T[:3, :3]
        z = np.column_stack([q[:, 0] / q[:, 2] * K[0, 0] + K[0, 2],
                             q[:, 1] / q[:, 2] * K[1, 1] + K[1, 2],
                             (q[:, 0] - 0.12) / q[:, 2] * K[0, 0] + K[0, 2]])
        if not stereo:
            z = z[:, :2]
        zs.append(z + rng.normal(0, noise, z.shape))
        ci += [c] * len(sel)
        pi += sel.tolist()
    c0 = [T.copy() for T in cams]
    p0 = pts.copy()
    if perturb:
        c0 = [T @ lie.se3_exp(rng.normal(0, 0.01, 6)) if i else T
              for i, T in enumerate(c0)]
        p0 = p0 + rng.normal(0, 0.05, p0.shape)
    pb = ba.BAProblem(K, c0, p0, np.array(ci), np.array(pi), np.vstack(zs),
                      0.12 if stereo else None, max(noise, 1e-3), {0})
    return pb, cams, pts


def fig_structure(outdir: Path) -> None:
    pb, _, _ = synthetic_ba(n_cams=8, n_pts=60, visible=0.4)
    H, _ = ba.dense_system(pb)
    r, A, B, _ = pb.residuals_jacobians()
    U, V, Wk, g_c, g_p = pb.normal_equations(r, A, B, np.ones(len(r)))
    S, _, _, _ = pb.schur(U, V, Wk, g_c, g_p)
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2),
                           gridspec_kw={"width_ratios": [1, 0.55, 1.2]})
    ax[0].spy(np.abs(H) > 1e-9, markersize=0.8, color="0.15")
    nc = 6 * 7
    ax[0].axhline(nc - 0.5, color="tab:green", lw=0.8)
    ax[0].axvline(nc - 0.5, color="tab:green", lw=0.8)
    ax[0].set_title("$H$: 7 free cameras | 60 points", fontsize=9)
    ax[1].spy(np.abs(S) > 1e-9, markersize=1.5, color="0.15")
    ax[1].set_title("reduced camera system $S$", fontsize=9)
    # timing: dense solve of H versus Schur, growing number of points
    ns = [100, 200, 400, 800, 1600, 3200]
    t_dense, t_schur = [], []
    for n in ns:
        pb, _, _ = synthetic_ba(n_cams=10, n_pts=n, visible=0.5, seed=1)
        r, A, B, _ = pb.residuals_jacobians()
        if n <= 1600:
            H, g = ba.dense_system(pb)
            t0 = time.perf_counter()
            np.linalg.solve(H, -g)
            t_dense.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        U, V, Wk, g_c, g_p = pb.normal_equations(r, A, B, np.ones(len(r)))
        S, b, Vinv, pos = pb.schur(U, V, Wk, g_c, g_p)
        pb.back_substitute(np.linalg.solve(S, b), pos, Vinv, Wk, g_p)
        t_schur.append(time.perf_counter() - t0)
    ax[2].loglog(ns[:len(t_dense)], t_dense, "o-", color="tab:orange",
                 label="dense solve of $H$ (solve only)")
    ax[2].loglog(ns, t_schur, "s-", color="tab:green",
                 label="Schur: build $S$, solve, back-substitute")
    ax[2].set_xlabel("number of points (10 cameras)")
    ax[2].set_ylabel("time [s]")
    ax[2].legend(fontsize=8)
    for a in ax[:2]:
        a.set_xticks([])
        a.set_yticks([])
    fig.tight_layout()
    fig.savefig(outdir / "ch08_structure.pdf")


def fig_covariance(outdir: Path, draws: int = 200) -> None:
    ests = []
    pts_err = []
    for s in range(draws):
        pb, cams, pts = synthetic_ba(n_cams=6, n_pts=80, visible=0.6,
                                     seed=s)
        res = ba.solve(pb, max_iters=20)
        ests.append(res.cams[5][:3, 3] - cams[5][:3, 3])
        pts_err.append(res.points - pts)
    pb, cams, pts = synthetic_ba(n_cams=6, n_pts=80, visible=0.6, seed=0)
    ba.solve(pb, max_iters=20)
    R = pb.cams[5][:3, :3]
    C = R @ ba.camera_covariance(pb)[5][:3, :3] @ R.T
    E = np.array(ests)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.0))
    ax[0].scatter(100 * E[:, 0], 100 * E[:, 1], s=6, color="0.5",
                  label="Monte Carlo (200 noise draws)")
    w, Vv = np.linalg.eigh(C[:2, :2])
    th = np.linspace(0, 2 * np.pi, 200)
    ell = (Vv @ np.diag(np.sqrt(w)) @ np.vstack([np.cos(th), np.sin(th)]))
    for k, ls in [(1, "-"), (2, "--"), (3, ":")]:
        ax[0].plot(100 * k * ell[0], 100 * k * ell[1], ls,
                   color="tab:green", label=f"{k}$\\sigma$ from $S^{{-1}}$")
    ax[0].set_aspect("equal")
    ax[0].set_xlabel("x error of the last camera [cm]")
    ax[0].set_ylabel("y error [cm]")
    ax[0].legend(fontsize=7)
    # points: predicted std against Monte Carlo std, coloured by the
    # number of cameras that see the point
    cov = ba.point_covariance(pb)
    pred = np.sqrt(np.trace(cov, axis1=1, axis2=2))
    mc = np.sqrt(np.sum(np.array(pts_err).var(axis=0), axis=1))
    n_obs = np.bincount(pb.pt_idx, minlength=len(pts))
    sc = ax[1].scatter(100 * pred, 100 * mc, s=10, c=n_obs, cmap="Greens",
                       edgecolors="0.3", linewidths=0.3)
    lim = [100 * pred.min() * 0.8, 100 * pred.max() * 1.25]
    ax[1].plot(lim, lim, "k:", lw=0.8)
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("predicted position std of each point [cm]")
    ax[1].set_ylabel("Monte Carlo std [cm]")
    fig.colorbar(sc, ax=ax[1], label="cameras seeing the point")
    fig.tight_layout()
    fig.savefig(outdir / "ch08_covariance.pdf")


def fig_vo(outdir: Path, seeds: int) -> None:
    world = sim.garden(np.random.default_rng(0))
    S = sim.GardenSim()
    variants = [("keyframes (Ch. 7)", "tab:orange"),
                ("frame-to-frame (Ch. 7)", "tab:blue"),
                ("keyframes + local BA", "tab:green")]
    errs = {v: [] for v, _ in variants}
    rpes = {v: [] for v, _ in variants}
    for seed in range(1, seeds + 1):
        frames = S.run(world, sim.boustrophedon(), sim.camera_mount(),
                       np.random.default_rng(seed))
        gt = np.array([f.T_wc[:3, 3] for f in frames])
        gtT = [f.T_wc for f in frames]
        runs = {
            variants[0][0]: vo.StereoVO(S.stereo, SIGMA,
                                        rng=np.random.default_rng(seed)),
            variants[1][0]: vo.StereoVO(S.stereo, SIGMA,
                                        keyframe_every_frame=True,
                                        rng=np.random.default_rng(seed)),
            variants[2][0]: vo.StereoVOBA(S.stereo, SIGMA,
                                          rng=np.random.default_rng(seed))}
        for name, V in runs.items():
            est = V.run(frames, frames[0].T_wc)
            e = np.array([T[:3, 3] for T in est])
            errs[name].append(np.linalg.norm(e - gt, axis=1))
            rpes[name].append(metrics.rpe(est, gtT, 5.0)[0].mean() / 5.0)
    dist = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(gt, axis=0),
                                                         axis=1))])
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    for name, col in variants:
        rms = np.sqrt(np.mean(np.array(errs[name]) ** 2, axis=0))
        ax.plot(dist, 100 * rms, color=col,
                label=f"{name}: RPE(5 m) {100 * np.mean(rpes[name]):.2f} %")
    ax.set_xlabel("distance driven [m]")
    ax.set_ylabel(f"RMS position error over {seeds} runs [cm]")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "ch08_vo.pdf")
    for name, _ in variants:
        e = np.array(errs[name])
        print(f"{name:>24}: RMS {100 * np.sqrt(np.mean(e ** 2)):.1f} cm, "
              f"RPE(5 m) {100 * np.mean(rpes[name]):.2f} %")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outdir", type=Path,
                    default=Path(__file__).resolve().parents[2]
                    / "course" / "figures")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    fig_structure(args.outdir)
    fig_covariance(args.outdir)
    fig_vo(args.outdir, args.seeds)
    print("figures written to", args.outdir)


if __name__ == "__main__":
    main()
