"""Chapter 10 practical: place recognition and loop closure.

Produces three figures in course/figures:

  ch10_similarity.pdf  bag-of-words similarity between all keyframes of the
                       garden run, next to the true covisibility
  ch10_pr.pdf          precision and recall of the best candidate per
                       keyframe, with and without idf weighting, and after
                       geometric verification
  ch10_loops.pdf       the VO keyframes corrected by the loops found from
                       the images (correct and wrong loops shown), for the
                       default garden and for a strongly aliased fence

Usage:
  python examples/ch10_loops.py [--outdir DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from vslam import bow, lie, loop, optim, posegraph as pg, robust  # noqa
from vslam import sim, vo  # noqa: E402

SIGMA = 0.5
SIG_T, SIG_R = 0.02, 0.004      # odometry factors (Chapter 9)


def train_vocabulary(k: int = 500, seed: int = 99) -> bow.Vocabulary:
    """Learn the vocabulary (and idf) on a DIFFERENT garden, as a real
    system learns it offline on generic images."""
    world = sim.garden(np.random.default_rng(seed))
    base = sim.landmark_descriptors(world, np.random.default_rng(seed - 1))
    frames = sim.GardenSim().run(world, sim.boustrophedon(),
                                 sim.camera_mount(),
                                 np.random.default_rng(seed - 2))
    rng = np.random.default_rng(seed - 3)
    images = [sim.observe_descriptors(base, f, world, rng)
              for f in frames[::12]]
    return bow.Vocabulary.train(images, k=k, iters=8,
                                rng=np.random.default_rng(0))


def garden_keyframes(seed: int = 1, n_fence: int = 40,
                     personal: float = 0.04):
    """The test garden, the Chapter 7 keyframe VO, and per keyframe: its
    measurements, descriptors, VO pose, true pose and structure
    landmarks. `n_fence` and `personal` set the aliasing of the fence."""
    world = sim.garden(np.random.default_rng(0))
    base = sim.landmark_descriptors(world, np.random.default_rng(1),
                                    n_fence=n_fence, personal=personal)
    S = sim.GardenSim()
    frames = S.run(world, sim.boustrophedon(), sim.camera_mount(),
                   np.random.default_rng(seed))
    V = vo.StereoVO(S.stereo, SIGMA, rng=np.random.default_rng(seed))
    est = V.run(frames, frames[0].T_wc)
    kf = V.stats.keyframes
    rng = np.random.default_rng(5)
    desc = [sim.observe_descriptors(base, frames[k], world, rng) for k in kf]
    P = [est[k] for k in kf]
    G = [frames[k].T_wc for k in kf]
    sets = [set(frames[k].landmark[~world.grass[frames[k].landmark]]
                .tolist()) for k in kf]
    return world, S, [frames[k] for k in kf], desc, P, G, sets


def shared_matrix(sets):
    n = len(sets)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            M[i, j] = len(sets[i] & sets[j])
    return M


def rms(A, B) -> float:
    return float(np.sqrt(np.mean([np.sum((a[:3, 3] - b[:3, 3]) ** 2)
                                  for a, b in zip(A, B)])))


def best_candidates(det: loop.LoopDetector):
    """Best (unverified) candidate of every keyframe: (j, i, normalised)."""
    out = []
    for j in range(len(det.kfs)):
        c = det.candidates(j)
        if c:
            out.append((j, c[0].i, c[0].normalised))
    return out


def pr_curve(cands, shared, positives, thresholds, min_shared=30):
    prec, rec = [], []
    for t in thresholds:
        acc = [(j, i) for j, i, s in cands if s >= t]
        good = sum(shared[i, j] >= min_shared for j, i in acc)
        prec.append(good / len(acc) if acc else 1.0)
        rec.append(good / max(positives, 1))
    return np.array(prec), np.array(rec)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outdir", type=Path,
                    default=Path(__file__).resolve().parents[2]
                    / "course" / "figures")
    args = ap.parse_args()
    out = args.outdir
    out.mkdir(parents=True, exist_ok=True)

    voc = train_vocabulary()
    world, S, kfr, desc, P, G, sets = garden_keyframes()
    n = len(P)
    shared = shared_matrix(sets)
    gap = 20
    positives = sum(any(shared[i, j] >= 30 for i in range(0, j - gap + 1))
                    for j in range(n))

    dets = {}
    for name, idf in [("tf-idf", True), ("tf only", False)]:
        d = loop.LoopDetector(voc, S.stereo, use_idf=idf,
                              rng=np.random.default_rng(0))
        for f, ds in zip(kfr, desc):
            d.add(f.z, ds)
        dets[name] = d

    # ---------------------------------------------------- similarity
    V = np.array([k.bow for k in dets["tf-idf"].kfs])
    Sm = bow.score_matrix(V)
    np.fill_diagonal(Sm, np.nan)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    im0 = ax[0].imshow(Sm, cmap="Greens")
    ax[0].set_title("bag-of-words score $s(i, j)$ (tf-idf)", fontsize=9)
    fig.colorbar(im0, ax=ax[0], fraction=0.046)
    im1 = ax[1].imshow(shared, cmap="Greys", vmax=150)
    ax[1].set_title("structure landmarks seen by both keyframes",
                    fontsize=9)
    fig.colorbar(im1, ax=ax[1], fraction=0.046)
    for a in ax:
        a.set_xlabel("keyframe $j$")
        a.set_ylabel("keyframe $i$")
    fig.tight_layout()
    fig.savefig(out / "ch10_similarity.pdf")

    # ---------------------------------------------------- precision/recall
    ths = np.linspace(0.2, 1.2, 41)
    fig, ax = plt.subplots(figsize=(7, 4))
    for (name, d), col in zip(dets.items(), ["tab:green", "tab:orange"]):
        p, r = pr_curve(best_candidates(d), shared, positives, ths)
        ax.plot(r, p, "-", color=col, label=f"best BoW candidate, {name}")
        k = np.argmin(np.abs(ths - d.min_norm))
        ax.plot(r[k], p[k], "o", color=col)
        print(f"best candidate, {name}, threshold {d.min_norm}: "
              f"precision {p[k]:.2f}, recall {r[k]:.2f}")
    det = dets["tf-idf"]
    found = [c for j in range(n) if (c := det.detect(j)) is not None]
    err = np.array([np.linalg.norm((lie.se3_inv(G[c.i]) @ G[c.j])[:3, 3]
                                   - c.Z[:3, 3]) for c in found])
    good = err < 0.5            # a loop is correct if its translation
                                # error is below 0.5 m
    js = {c.j for c, g in zip(found, good) if g}
    print(f"after verification: precision {np.mean(good):.2f}, "
          f"recall {len(js) / positives:.2f}")
    ax.plot(len(js) / positives, np.mean(good), "*", ms=12,
            color="tab:blue", label="after geometric verification")
    ax.set_xlabel("recall (keyframes with a true loop, found)")
    ax.set_ylabel("precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "ch10_pr.pdf")

    # ---------------------------------------------------- pose graph
    # two gardens: the default fence (40 picket prototypes, 4 % personal
    # bits) and a strongly aliased one (5 prototypes, 2 %)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.2))
    for ax, (nf, pers) in zip(axes, [(40, 0.04), (5, 0.02)]):
        if nf == 40:
            res = (world, kfr, P, G, det, found, err)
        else:
            w2, S2, kfr2, desc2, P2, G2, _ = garden_keyframes(
                n_fence=nf, personal=pers)
            d2 = loop.LoopDetector(voc, S2.stereo,
                                   rng=np.random.default_rng(0))
            for f, ds in zip(kfr2, desc2):
                d2.add(f.z, ds)
            f2 = [c for j in range(n) if (c := d2.detect(j)) is not None]
            e2 = np.array([np.linalg.norm(
                (lie.se3_inv(G2[c.i]) @ G2[c.j])[:3, 3] - c.Z[:3, 3])
                for c in f2])
            res = (w2, kfr2, P2, G2, d2, f2, e2)
        wd, kf_, P_, G_, d_, f_, e_ = res
        wrong = e_ > 0.5
        edges = pg.odometry_edges(P_, SIG_T, SIG_R) + loop.loop_edges(f_)
        Q_l2 = optim.levenberg_marquardt(pg.PoseGraphProblem(edges, n),
                                         P_).x
        Q = optim.levenberg_marquardt(pg.PoseGraphProblem(edges, n), P_,
                                      kernel=robust.Cauchy(1.0)).x
        s = wd.points
        ax.scatter(s[~wd.grass, 0], s[~wd.grass, 1], s=1, color="0.75")
        g = np.array([T[:3, 3] for T in G_])
        for c, w in zip(f_, wrong):
            ax.plot(g[[c.i, c.j], 0], g[[c.i, c.j], 1],
                    color="tab:red" if w else "tab:blue",
                    lw=1.2 if w else 0.5, alpha=0.9 if w else 0.4)
        ax.plot([], [], color="tab:blue", lw=0.6,
                label=f"{(~wrong).sum()} correct loops")
        ax.plot([], [], color="tab:red", lw=1.2,
                label=f"{wrong.sum()} wrong loops (error > 0.5 m)")
        ax.plot(g[:, 0], g[:, 1], "k", lw=2, label="truth")
        for X, col, lab in [(P_, "tab:orange", "VO keyframes"),
                            (Q_l2, "0.4", "pose graph, least squares"),
                            (Q, "tab:green", "pose graph, Cauchy")]:
            p = np.array([T[:3, 3] for T in X])
            ax.plot(p[:, 0], p[:, 1], ".-", color=col, ms=2, lw=0.8,
                    label=f"{lab}: RMS {100 * rms(X, G_):.1f} cm")
        ax.set_aspect("equal")
        ax.set_xlim(-4, 16)
        ax.set_ylim(-4, 13)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(f"fence: {nf} picket prototypes, "
                     f"{100 * pers:.0f} % personal bits", fontsize=9)
        ax.legend(fontsize=7, loc="upper center", ncol=2,
                  bbox_to_anchor=(0.5, -0.12))
        # are the inlier matches of the wrong loops wrong?
        right = [np.mean(kf_[c.i].landmark[c.pairs[:, 0]]
                         == kf_[c.j].landmark[c.pairs[:, 1]])
                 for c, w in zip(f_, wrong) if w]
        print(f"[{nf} prototypes] {len(f_)} verified loops, {wrong.sum()} "
              f"wrong, median error of the others "
              f"{100 * np.median(e_[~wrong]):.1f} cm; correct matches "
              f"among the inliers of the wrong loops: "
              f"{100 * np.mean(right) if right else 100:.0f} %")
        print(f"    RMS: VO {100 * rms(P_, G_):.1f} cm, least squares "
              f"{100 * rms(Q_l2, G_):.1f} cm, Cauchy {100 * rms(Q, G_):.1f} cm")
        # temporal consistency (DBoW2): keep a loop (i, j) only if
        # keyframe j - 1 or j + 1 also closed a loop near i
        by_j = {c.j: c for c in f_}
        cons = [w for c, w in zip(f_, wrong)
                if any(k in by_j and abs(by_j[k].i - c.i) <= 4
                       for k in (c.j - 1, c.j + 1))]
        print(f"    temporal consistency keeps {len(cons)} loops, "
              f"{sum(cons)} wrong")
    fig.tight_layout()
    fig.savefig(out / "ch10_loops.pdf")

    print("figures written to", out)


if __name__ == "__main__":
    main()
