"""Chapter 11 practical: dense maps for mowing.

Renders dense stereo depth at the keyframes of the garden run, fits the
ground plane of each keyframe, fuses the classified points into a
log-odds grid with three sets of keyframe poses (ground truth, VO, and
the pose graph corrected by the loops of Chapter 10), and exports a
blade-sized grid for the Lawnmower coverage planners.

Produces in course/figures:

  ch11_frame.pdf   one keyframe: labels, depth, points and ground plane
  ch11_maps.pdf    the fused map with each set of poses, against truth
  ch11_range.pdf   map quality against the maximum mapping range
  ch11_plan.pdf    the planner grid (1 m cells), with a coverage path if
                   the Lawnmower package is installed

Usage:
  python examples/ch11_maps.py [--outdir DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from vslam import lie, loop, maps, optim, posegraph as pg  # noqa: E402
from vslam import robust, scene, sim  # noqa: E402
from ch10_loops import train_vocabulary, garden_keyframes, rms  # noqa
from ch10_loops import SIG_T, SIG_R  # noqa: E402

SIGMA_D = 0.3          # dense stereo disparity noise [px]
CMAP = ListedColormap(["#6abf4b", "#303030", "#d9c89e", "#ffffff"])
SPEC = maps.GridSpec()
EXTENT = (SPEC.x0, SPEC.x1, SPEC.y0, SPEC.y1)


def keyframe_clouds(garden, stereo, T_true, rng, max_range=4.0,
                    keep_frames=False):
    """Render, triangulate, fit the ground and classify, per keyframe.
    The rendering uses the TRUE poses (the world is what it is); the
    clouds are in each camera's frame, so they do not depend on the
    poses used later for fusion."""
    prior = maps.nominal_ground(sim.camera_mount())
    clouds, frames = [], []
    for T in T_true:
        f = scene.render_stereo(garden, stereo, T, rng, sigma_d=SIGMA_D)
        ok = np.isfinite(f.depth)
        q = maps.stereo_points(f.uv[ok], f.depth[ok], stereo)
        grass = scene.segment_grass(f.label[ok], rng)
        plane, _ = maps.ground_plane(q, prior, rng)
        clouds.append(maps.classify_points(q, grass, plane,
                                           max_range=max_range))
        if keep_frames:
            frames.append((f, q, grass, plane, ok))
    return clouds, frames


def inside_fence():
    """Cells inside the fence (the rest cannot be reached)."""
    r = SPEC.res
    xs = SPEC.x0 + r * (np.arange(SPEC.shape[1]) + 0.5)
    ys = SPEC.y0 + r * (np.arange(SPEC.shape[0]) + 0.5)
    X, Y = np.meshgrid(xs, ys)
    return (X > -1) & (X < 13) & (Y > -1) & (Y < 9)


def scores(cls, truth):
    """Inside the fence: share of the lawn found mowable, cells wrongly
    called mowable (obstacle or not lawn in truth), and cells wrongly
    called obstacles."""
    lawn = truth == maps.MOWABLE
    ins = inside_fence()
    mow = ins & (cls == maps.MOWABLE)
    return {"lawn found": float(np.mean(cls[lawn] == maps.MOWABLE)),
            "obstacle -> mowable": int(np.sum(mow & (truth
                                                     == maps.OBSTACLE))),
            "not lawn -> mowable": int(np.sum(mow & (truth
                                                     == maps.NOT_LAWN))),
            "false obstacle": int(np.sum(ins & (cls == maps.OBSTACLE)
                                         & (truth != maps.OBSTACLE)))}


def corrected_poses(stereo, kfr, desc, P):
    """The pose graph of Chapter 10: VO odometry plus detected loops,
    Cauchy kernel."""
    voc = train_vocabulary()
    det = loop.LoopDetector(voc, stereo, rng=np.random.default_rng(0))
    for f, ds in zip(kfr, desc):
        det.add(f.z, ds)
    found = [c for j in range(len(P)) if (c := det.detect(j)) is not None]
    edges = pg.odometry_edges(P, SIG_T, SIG_R) + loop.loop_edges(found)
    return optim.levenberg_marquardt(pg.PoseGraphProblem(edges, len(P)), P,
                                     kernel=robust.Cauchy(1.0)).x


def draw(ax, cls, title):
    ax.imshow(cls, origin="lower", cmap=CMAP, vmin=0, vmax=3,
              extent=EXTENT, interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outdir", type=Path,
                    default=Path(__file__).resolve().parents[2]
                    / "course" / "figures")
    out = ap.parse_args().outdir
    out.mkdir(parents=True, exist_ok=True)

    garden = scene.Garden()
    truth = garden.truth_grid(EXTENT, SPEC.res)
    world, S, kfr, desc, P, G, _ = garden_keyframes()
    rng = np.random.default_rng(11)

    # ---------------------------------------------------- one keyframe
    k = 37
    clouds, frames = keyframe_clouds(garden, S.stereo, G, rng,
                                     keep_frames=True)
    f, q, grass, plane, ok = frames[k]
    cam = S.stereo.cam
    H, W = len(np.unique(f.uv[:, 1])), len(np.unique(f.uv[:, 0]))
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.4),
                           gridspec_kw={"width_ratios": [1, 1, 1.3]})
    lab_cmap = ListedColormap(["#bcd7f0", "#6abf4b", "#d9c89e", "#b0a8a0",
                               "#8c6d46", "#c0504d", "#7f7f7f", "#5b3a1e",
                               "#d070b0"])
    ax[0].imshow(f.label.reshape(H, W), cmap=lab_cmap, vmin=0, vmax=8,
                 interpolation="nearest")
    ax[0].set_title("surface labels (truth)", fontsize=9)
    dimg = f.depth.reshape(H, W)
    im = ax[1].imshow(np.minimum(dimg, 15), cmap="viridis_r",
                      interpolation="nearest")
    fig.colorbar(im, ax=ax[1], fraction=0.035, label="depth [m]")
    ax[1].set_title(f"stereo depth ($\\sigma_d$ = {SIGMA_D} px, "
                    "5 % dropouts)", fontsize=9)
    for a in ax[:2]:
        a.set_xticks([])
        a.set_yticks([])
    h = plane.height(q)
    near = q[:, 2] < 8
    ground = near & (np.abs(h) < 0.15)
    obst = near & (h >= 0.15) & (h <= 1.5)
    ax[2].scatter(q[ground, 2], h[ground], s=1, color="#6abf4b",
                  label="ground")
    ax[2].scatter(q[obst, 2], h[obst], s=1, color="#c0504d",
                  label="obstacle (0.15 m $\\leq h \\leq$ 1.5 m)")
    ax[2].axhline(0.15, color="k", lw=0.6, ls="--")
    ax[2].axvline(4.0, color="k", lw=0.6, ls=":")
    ax[2].text(4.1, 1.2, "max. range", fontsize=7)
    ax[2].set_xlabel("depth Z [m]")
    ax[2].set_ylabel("height above the fitted plane [m]")
    ax[2].set_ylim(-0.3, 1.6)
    ax[2].legend(fontsize=7, markerscale=5, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "ch11_frame.pdf")

    # ---------------------------------------------------- poses
    Q = corrected_poses(S.stereo, kfr, desc, P)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.4))
    draw(axes[0, 0], truth, "truth")
    for a, (poses, name) in zip(axes.ravel()[1:],
                                [(G, "true poses"), (P, "VO poses"),
                                 (Q, "pose graph with loops")]):
        cls = maps.fuse(SPEC, poses, clouds).classes()
        s = scores(cls, truth)
        a_err = 100 * rms(poses, G)
        draw(a, cls, f"{name} (RMS {a_err:.1f} cm): lawn found "
             f"{100 * s['lawn found']:.0f} %, wrongly mowable "
             f"{s['obstacle -> mowable'] + s['not lawn -> mowable']}")
        print(f"{name:22s} pose RMS {a_err:5.1f} cm  {s}")
    handles = [plt.Rectangle((0, 0), 1, 1, color=CMAP(i)) for i in range(4)]
    axes[0, 0].legend(handles, ["mowable lawn", "obstacle",
                                "drivable, not lawn", "unknown"],
                      fontsize=7, loc="lower left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out / "ch11_maps.pdf")

    # ---------------------------------------------------- range
    ranges = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 12.0]
    res = []
    for r in ranges:
        cl, _ = keyframe_clouds(garden, S.stereo, G,
                                np.random.default_rng(11), max_range=r)
        cls = maps.fuse(SPEC, G, cl).classes()
        s = scores(cls, truth)
        fat = s["false obstacle"]
        res.append((s["lawn found"], s["obstacle -> mowable"]
                    + s["not lawn -> mowable"], fat))
        print(f"max range {r:4.1f} m: {s}")
    res = np.array(res)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(ranges, 100 * res[:, 0], "o-", color="tab:green",
            label="lawn found mowable [%]")
    ax.set_xlabel("maximum mapping range [m]")
    ax.set_ylabel("lawn found [%]")
    ax2 = ax.twinx()
    ax2.plot(ranges, res[:, 2], "s-", color="0.3",
             label="false obstacle cells")
    ax2.plot(ranges, res[:, 1], "^-", color="tab:red",
             label="cells wrongly mowable")
    ax2.set_ylabel("cells (0.1 m)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")
    fig.tight_layout()
    fig.savefig(out / "ch11_range.pdf")

    # ---------------------------------------------------- planner grid
    cls = maps.fuse(SPEC, Q, clouds).classes()
    factor = int(round(1.0 / SPEC.res))            # 1 m cells
    mow = maps.coarsen(cls, factor)
    mow_true = maps.coarsen(truth, factor)
    start = (3, 3)                                  # cell of (1.5, 1.5)
    text = maps.to_lawn_text(mow, start)
    print(text)
    print(f"planner cells: {mow.sum()} mowable (truth {mow_true.sum()}), "
          f"{np.sum(mow & ~mow_true)} wrongly mowable")
    fig, ax = plt.subplots(figsize=(7, 5.4))
    x0, y0 = SPEC.x0, SPEC.y0
    Hc, Wc = mow.shape
    ax.imshow(np.where(mow, 0, 1), origin="lower", cmap=CMAP, vmin=0,
              vmax=3, extent=(x0, x0 + Wc, y0, y0 + Hc), alpha=0.85)
    try:
        from lawnmower import Lawn, plan
        lawn = Lawn.from_string(text)
        r = plan(lawn, "stc")
        rows = np.array([c[0] for c in r.path])
        cols = np.array([c[1] for c in r.path])
        # text row 0 is the top border; mowable row i is text row Hc - i
        px = x0 + (cols - 1) + 0.5
        py = y0 + (Hc - rows) + 0.5
        ax.plot(px, py, "-", color="tab:blue", lw=1.2,
                label=f"STC coverage path: {r.metrics.as_dict()}")
        ax.plot(px[0], py[0], "o", color="tab:blue")
        print("STC:", r.metrics.as_dict())
        ax.set_title("planner grid (1 m cells) and the STC path of the "
                     "Lawnmower package", fontsize=9)
    except ImportError:
        ax.set_title("planner grid (1 m cells)", fontsize=9)
    ax.set_xticks(np.arange(x0, x0 + Wc + 1, 1), minor=True)
    ax.set_yticks(np.arange(y0, y0 + Hc + 1, 1), minor=True)
    ax.grid(which="minor", color="w", lw=0.4)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    fig.tight_layout()
    fig.savefig(out / "ch11_plan.pdf")
    print("figures written to", out)


if __name__ == "__main__":
    main()
