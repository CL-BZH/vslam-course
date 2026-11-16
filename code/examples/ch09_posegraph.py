"""Chapter 9 practical: pose graphs, loop closures and marginalisation.

Produces three figures in course/figures:

  ch09_pgo.pdf     the keyframes of the Chapter 7 VO before and after
                   pose-graph optimisation with loop closures between rows
  ch09_robust.pdf  wrong loop closures: error after optimisation with
                   least squares and with a Cauchy kernel
  ch09_fej.pdf     marginalisation and the linearisation point: the
                   spectrum of the information matrix of a pose graph,
                   with the prior linearised consistently or not

Usage:
  python examples/ch09_posegraph.py [--outdir DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from vslam import lie, optim, posegraph as pg, robust, sim, vo  # noqa: E402

SIGMA = 0.5
# per-edge odometry noise. The empirical spread of the VO's keyframe-to-
# keyframe errors is about (0.6 cm, 0.06 deg), but those errors are
# correlated from edge to edge, and using them as independent noise makes
# the odometry overconfident (see the chapter); we inflate them by ~3-4.
SIG_T, SIG_R = 0.02, 0.004


def keyframe_graph(seed: int = 1, ba_vo: bool = False):
    """Run a VO on the simulated garden and return its keyframe poses, the
    true ones, and the sets of structure landmarks each keyframe sees."""
    world = sim.garden(np.random.default_rng(0))
    S = sim.GardenSim()
    frames = S.run(world, sim.boustrophedon(), sim.camera_mount(),
                   np.random.default_rng(seed))
    if ba_vo:
        V = vo.StereoVOBA(S.stereo, SIGMA, rng=np.random.default_rng(seed))
        V.run(frames, frames[0].T_wc)
        kf = [k["frame"] for k in V.kfs]
        P = [k["T"] for k in V.kfs]
    else:
        V = vo.StereoVO(S.stereo, SIGMA, rng=np.random.default_rng(seed))
        est = V.run(frames, frames[0].T_wc)
        kf = V.stats.keyframes
        P = [est[k] for k in kf]
    G = [frames[k].T_wc for k in kf]
    sets = [set(frames[k].landmark[~world.grass[frames[k].landmark]]
                .tolist()) for k in kf]
    return world, P, G, sets


def rms(P, G) -> float:
    return float(np.sqrt(np.mean([np.sum((a[:3, 3] - b[:3, 3]) ** 2)
                                  for a, b in zip(P, G)])))


def optimise(P, edges, kernel=None):
    return optim.levenberg_marquardt(pg.PoseGraphProblem(edges, len(P)),
                                     P, kernel=kernel).x


def fig_pgo(outdir: Path) -> None:
    world, P, G, sets = keyframe_graph()
    loops = pg.covisible_loops(sets, G, np.random.default_rng(0))
    edges = pg.odometry_edges(P, SIG_T, SIG_R) + loops
    Q = optimise(P, edges)
    fig, ax = plt.subplots(figsize=(9, 6.2))
    s = world.points
    ax.scatter(s[~world.grass, 0], s[~world.grass, 1], s=1, color="0.75")
    g = np.array([T[:3, 3] for T in G])
    p = np.array([T[:3, 3] for T in P])
    q = np.array([T[:3, 3] for T in Q])
    for ed in loops:
        ax.plot(g[[ed.i, ed.j], 0], g[[ed.i, ed.j], 1], color="tab:red",
                lw=0.6, alpha=0.6)
    ax.plot([], [], color="tab:red", lw=0.6, label=f"{len(loops)} loop "
            "closures (between true poses)")
    ax.plot(g[:, 0], g[:, 1], "k", lw=2, label="truth")
    ax.plot(p[:, 0], p[:, 1], ".-", color="tab:orange", ms=3, lw=0.8,
            label=f"VO keyframes: RMS {100 * rms(P, G):.0f} cm")
    ax.plot(q[:, 0], q[:, 1], ".-", color="tab:green", ms=3, lw=0.8,
            label=f"after pose-graph optimisation: RMS {100 * rms(Q, G):.1f}"
                  " cm")
    ax.set_aspect("equal")
    ax.set_xlim(-4, 16)
    ax.set_ylim(-4, 13)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(outdir / "ch09_pgo.pdf")
    print(f"keyframe VO: {len(P)} keyframes, {len(loops)} loops, "
          f"RMS {100 * rms(P, G):.1f} -> {100 * rms(Q, G):.1f} cm")
    _, P2, G2, sets2 = keyframe_graph(ba_vo=True)
    loops2 = pg.covisible_loops(sets2, G2, np.random.default_rng(0))
    Q2 = optimise(P2, pg.odometry_edges(P2, SIG_T / 2, SIG_R / 2) + loops2)
    print(f"VO + local BA: {len(P2)} keyframes, {len(loops2)} loops, "
          f"RMS {100 * rms(P2, G2):.1f} -> {100 * rms(Q2, G2):.1f} cm")


def fig_robust(outdir: Path, trials: int = 10) -> None:
    world, P, G, sets = keyframe_graph()
    loops = pg.covisible_loops(sets, G, np.random.default_rng(0))
    odo = pg.odometry_edges(P, SIG_T, SIG_R)
    n_false = [0, 1, 2, 5, 10, 20]
    res = {"least squares": [], "Cauchy, $k=1$": []}
    rng = np.random.default_rng(5)
    for nf in n_false:
        vals = {k: [] for k in res}
        for _ in range(trials):
            false = []
            for _ in range(nf):   # perceptual aliasing: "same place"
                i, j = sorted(rng.choice(len(P), 2, replace=False))
                Z = lie.se3_exp(np.concatenate([rng.normal(0, 0.3, 3),
                                                rng.normal(0, 0.05, 3)]))
                false.append(pg.Edge(i, j, Z, loops[0].info, loop=True))
            edges = odo + loops + false
            for name, k in [("least squares", None),
                            ("Cauchy, $k=1$", robust.Cauchy(1.0))]:
                vals[name].append(rms(optimise(P, edges, k), G))
        for k in res:
            res[k].append(np.median(vals[k]))
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for (name, v), col in zip(res.items(), ["tab:orange", "tab:green"]):
        ax.semilogy(n_false, 100 * np.array(v), "o-", color=col, label=name)
    ax.axhline(100 * rms(P, G), color="0.5", ls=":", lw=0.8)
    ax.text(12, 100 * rms(P, G) * 1.15, "VO without loops", color="0.4",
            fontsize=8)
    ax.set_xlabel(f"wrong loop closures added (to {len(loops)} correct "
                  "ones)")
    ax.set_ylabel("median RMS error [cm]")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "ch09_robust.pdf")


def fej_spectra():
    """Smallest eigenvalues of a pose-graph information matrix (no fixed
    node) built (a) at one estimate, (b) with node 0 marginalised at the
    same estimate, (c) with node 0 marginalised at a slightly different
    estimate (as happens when the prior is kept while the remaining nodes
    move on)."""
    rng = np.random.default_rng(3)
    truth = [lie.se3_exp(np.concatenate([[0.5 * k, 0.2 * np.sin(k), 0.0],
                                         [0.0, 0.0, 0.1 * k]]))
             for k in range(8)]
    edges = pg.odometry_edges(truth, 0.01, 0.002)
    edges.append(pg.Edge(0, 7, lie.se3_inv(truth[0]) @ truth[7],
                         pg.information(0.01, 0.002), loop=True))
    est_a = [T @ lie.se3_exp(rng.normal(0, 0.05, 6)) for T in truth]
    est_b = [T @ lie.se3_exp(rng.normal(0, 0.05, 6)) for T in est_a]

    def H_of(poses, which):
        sub = [e for e in edges if which(e)]
        pb = pg.PoseGraphProblem(sub, len(poses), fixed=set())
        r, J = pb.linearize(poses)
        return J.T @ J, J.T @ r

    full, _ = H_of(est_b, lambda e: True)
    # factors touching node 0, marginalised at estimate a or b
    out = {}
    for name, lin in [("consistent", est_b), ("inconsistent", est_a)]:
        H0, b0 = H_of(lin, lambda e: 0 in (e.i, e.j))
        Hr, _ = H_of(est_b, lambda e: 0 not in (e.i, e.j))
        keep, drop = np.arange(6, 48), np.arange(6)
        Hp, _ = pg.marginalize(H0 + 1e-12 * np.eye(48), b0, keep, drop)
        out[name] = Hp + Hr[np.ix_(keep, keep)]
    return np.linalg.eigvalsh(full), {k: np.linalg.eigvalsh(v)
                                      for k, v in out.items()}


def fig_fej(outdir: Path) -> None:
    full, marg = fej_spectra()
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    k = np.arange(1, 13)
    ax.semilogy(k, full[:12] / full[-1] + 1e-18, "o-", color="0.3",
                label="full graph (8 nodes)")
    ax.semilogy(k, marg["consistent"][:12] / full[-1] + 1e-18, "s-",
                color="tab:green", label="node 0 marginalised, same "
                "linearisation point")
    ax.semilogy(k, marg["inconsistent"][:12] / full[-1] + 1e-18, "^-",
                color="tab:orange", label="node 0 marginalised at an older "
                "estimate")
    ax.axvline(6.5, color="0.6", ls=":", lw=0.8)
    ax.set_xlabel("eigenvalue index (smallest first)")
    ax.set_ylabel("eigenvalue / largest")
    ax.set_ylim(1e-19, 1)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(outdir / "ch09_fej.pdf")
    for name, ev in [("full", full)] + list(marg.items()):
        print(f"{name:>12}: zero eigenvalues "
              f"{np.sum(ev < 1e-9 * full[-1])}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outdir", type=Path,
                    default=Path(__file__).resolve().parents[2]
                    / "course" / "figures")
    ap.add_argument("--skip-fej", action="store_true")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    if not args.skip_fej:
        fig_fej(args.outdir)
    fig_pgo(args.outdir)
    fig_robust(args.outdir)
    print("figures written to", args.outdir)


if __name__ == "__main__":
    main()
