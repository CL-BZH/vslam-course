"""Chapter 7 practical: PnP and stereo visual odometry in a garden.

Produces three figures in course/figures:

  ch07_pnp.pdf    camera pose from known points: error versus outliers,
                  and versus the relief of the scene (the DLT needs it)
  ch07_vo.pdf     top view of the simulated garden with the true and the
                  estimated mower trajectories (keyframe and
                  frame-to-frame stereo VO)
  ch07_drift.pdf  position error versus distance driven, and while the
                  mower stands still

Usage:
  python examples/ch07_vo.py [--outdir DIR] [--seeds N]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from vslam import lie, metrics, optim, pnp, robust, sim, vo  # noqa: E402
from vslam.camera import look_at  # noqa: E402

K = np.array([[458.7, 0, 367.2], [0, 457.3, 248.4], [0, 0, 1.0]])
SIGMA = 0.5


def pnp_trial(rng, outlier_frac=0.0, relief=1.0, n=100):
    """Errors (metres) of three PnP estimators on one random scene."""
    T_wc = look_at(np.array([0.0, -5.0, 1.0]), np.array([0.0, 2.0, 0.4]))
    X = np.column_stack([rng.uniform(-3, 3, n), rng.uniform(0, 6, n),
                         np.abs(rng.normal(0, relief, n))])
    T_cw = lie.se3_inv(T_wc)
    q = X @ T_cw[:3, :3].T + T_cw[:3, 3]
    z = q[:, :2] / q[:, 2:] @ K[:2, :2].T + K[:2, 2]
    z += rng.normal(0, SIGMA, z.shape)
    out = rng.random(n) < outlier_frac
    z[out] = rng.uniform([0, 0], [752, 480], (out.sum(), 2))
    x = (z - K[:2, 2]) / np.diag(K)[:2]
    res = {}
    T = pnp.pnp_dlt(X, x)
    res["DLT (all points)"] = np.inf if T is None else \
        np.linalg.norm(lie.se3_inv(T)[:3, 3] - T_wc[:3, 3])
    r = pnp.pnp_ransac(X, x, 3 * SIGMA / K[0, 0], rng=rng)
    T0 = lie.se3_inv(r.model)
    res["P3P + RANSAC"] = np.linalg.norm(T0[:3, 3] - T_wc[:3, 3])
    sol = optim.levenberg_marquardt(pnp.MonoPoseProblem(K, X, z, SIGMA), T0,
                                    kernel=robust.Cauchy(2.0))
    res["P3P + RANSAC + LM (Cauchy)"] = np.linalg.norm(sol.x[:3, 3]
                                                       - T_wc[:3, 3])
    return res


def fig_pnp(outdir: Path, trials: int) -> None:
    rng = np.random.default_rng(0)
    cols = {"DLT (all points)": "tab:orange", "P3P + RANSAC": "tab:blue",
            "P3P + RANSAC + LM (Cauchy)": "tab:green"}
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for a, xs, key, xlabel in [
            (ax[0], [0.0, 0.1, 0.2, 0.3, 0.5, 0.7], "outlier_frac",
             "outliers [%]"),
            (ax[1], [0.0, 0.01, 0.03, 0.1, 0.3, 1.0], "relief",
             "relief: std of point height [m] (+3 mm)")]:
        med = {m: [] for m in cols}
        for v in xs:
            runs = [pnp_trial(rng, **{key: v}) for _ in range(trials)]
            for m in cols:
                med[m].append(np.median([r_[m] for r_ in runs]))
        xx = 100 * np.array(xs) if key == "outlier_frac" else \
            np.array(xs) + 0.003
        for m, c in cols.items():
            a.plot(xx, med[m], "o-", color=c, ms=4, label=m)
        a.set_yscale("log")
        if key == "relief":
            a.set_xscale("log")
        a.set_xlabel(xlabel)
        a.set_ylabel("median position error [m]")
    ax[0].set_title("100 points, relief 1 m", fontsize=9)
    ax[1].set_title("100 points, no outliers", fontsize=9)
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "ch07_pnp.pdf")


def run_vo(seed: int, poses, survive: float = 0.97):
    rng = np.random.default_rng(seed)
    world = sim.garden(np.random.default_rng(0))
    S = sim.GardenSim(survive=survive)
    frames = S.run(world, poses, sim.camera_mount(), rng)
    out = {}
    for name, every in [("keyframes", False), ("frame-to-frame", True)]:
        V = vo.StereoVO(S.stereo, SIGMA, keyframe_every_frame=every,
                        rng=np.random.default_rng(seed))
        out[name] = (V.run(frames, frames[0].T_wc), V)
    return world, frames, out


def fig_vo(outdir: Path) -> None:
    world, frames, out = run_vo(0, sim.boustrophedon())
    gt = np.array([f.T_wc[:3, 3] for f in frames])
    fig, ax = plt.subplots(figsize=(9, 6.2))
    s = world.points
    ax.scatter(s[~world.grass, 0], s[~world.grass, 1], s=1, color="0.6",
               label="structure landmarks")
    ax.scatter(s[world.grass, 0], s[world.grass, 1], s=1,
               color="tab:green", alpha=0.3, label="grass landmarks")
    ax.plot(gt[:, 0], gt[:, 1], "k", lw=2, label="truth (camera)")
    for (name, (est, V)), col in zip(out.items(), ["tab:orange",
                                                   "tab:blue"]):
        e = np.array([T[:3, 3] for T in est])
        rmse = np.sqrt(np.mean(np.sum((e - gt) ** 2, axis=1)))
        ax.plot(e[:, 0], e[:, 1], color=col, lw=1.2,
                label=f"{name}: RMS error {100 * rmse:.0f} cm")
        if name == "keyframes":
            kf = e[V.stats.keyframes]
            ax.plot(kf[:, 0], kf[:, 1], ".", color=col, ms=4)
    ax.set_aspect("equal")
    ax.set_xlim(-4, 16)
    ax.set_ylim(-4, 13)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(outdir / "ch07_vo.pdf")
    L = np.sum(np.linalg.norm(np.diff(gt, axis=0), axis=1))
    print(f"trajectory length {L:.1f} m, {len(frames)} frames")
    for name, (est, V) in out.items():
        e = np.array([T[:3, 3] for T in est])
        print(f"{name:>15}: {len(V.stats.keyframes)} keyframes, "
              f"final error {100 * np.linalg.norm(e[-1] - gt[-1]):.1f} cm, "
              f"failures {V.stats.failures}")


def fig_drift(outdir: Path, seeds: int) -> None:
    poses = sim.boustrophedon()
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    errs = {"keyframes": [], "frame-to-frame": []}
    for seed in range(seeds):
        _, frames, out = run_vo(seed + 1, poses)
        gt = np.array([f.T_wc[:3, 3] for f in frames])
        for name, (est, _) in out.items():
            e = np.array([T[:3, 3] for T in est])
            errs[name].append(np.linalg.norm(e - gt, axis=1))
    dist = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(gt, axis=0),
                                                         axis=1))])
    for (name, E), col in zip(errs.items(), ["tab:orange", "tab:blue"]):
        rms = np.sqrt(np.mean(np.array(E) ** 2, axis=0))
        ax[0].loglog(dist[1:], rms[1:], color=col, label=name)
    # reference power laws through the frame-to-frame curve at 2 m
    i2 = np.searchsorted(dist, 2.0)
    anchor = np.sqrt(np.mean(np.array(errs["frame-to-frame"]) ** 2,
                             axis=0))[i2]
    for p_, ls in [(0.5, ":"), (1.0, "-."), (1.5, "--")]:
        ax[0].loglog(dist[1:], anchor * (dist[1:] / 2.0) ** p_, "k" + ls,
                     lw=0.8, label=f"slope {p_:g}")
    ax[0].set_xlabel("distance driven [m]")
    ax[0].set_ylabel(f"RMS position error over {seeds} runs [m]")
    ax[0].set_ylim(1e-3, 1)
    ax[0].legend(fontsize=8)
    # standing still with long-lived tracks
    still = [lie.make_T(np.eye(3), np.array([2.0, 3.0, 0.0]))] * 400
    for (name, every), col in zip([("keyframes", False),
                                   ("frame-to-frame", True)],
                                  ["tab:orange", "tab:blue"]):
        E = []
        for seed in range(seeds):
            world = sim.garden(np.random.default_rng(0))
            S = sim.GardenSim(survive=0.999)
            frames = S.run(world, still, sim.camera_mount(),
                           np.random.default_rng(10 + seed))
            V = vo.StereoVO(S.stereo, SIGMA, keyframe_every_frame=every,
                            rng=np.random.default_rng(seed))
            est = V.run(frames, frames[0].T_wc)
            E.append([np.linalg.norm(T[:3, 3] - frames[0].T_wc[:3, 3])
                      for T in est])
        t = np.arange(400) / 20.0
        ax[1].plot(t, 100 * np.sqrt(np.mean(np.array(E) ** 2, axis=0)),
                   color=col, label=name)
    ax[1].set_xlabel("time standing still [s] (20 Hz)")
    ax[1].set_ylabel("RMS position error [cm]")
    ax[1].set_title("long-lived tracks (survival 0.999 per frame)",
                    fontsize=9)
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "ch07_drift.pdf")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outdir", type=Path,
                    default=Path(__file__).resolve().parents[2]
                    / "course" / "figures")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--drift-only", action="store_true",
                    help="only regenerate ch07_drift.pdf")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    if not args.drift_only:
        fig_pnp(args.outdir, args.trials)
        fig_vo(args.outdir)
    fig_drift(args.outdir, args.seeds)
    print("figures written to", args.outdir)


if __name__ == "__main__":
    main()
