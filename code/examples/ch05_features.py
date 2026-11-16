"""Chapter 5 practical: corners, matching, the lawn, RANSAC and LK.

Produces four figures in course/figures:

  ch05_textures.pdf  the garden and lawn textures with their corners, the
                     Shi-Tomasi response, and a patch of lawn before and
                     after the wind
  ch05_matching.pdf  precision of oriented-BRIEF matching under viewpoint
                     and illumination change, and on a lawn in the wind
  ch05_ransac.pdf    the RANSAC iteration law, and homography RANSAC on
                     the lawn matches
  ch05_lk.pdf        Lucas-Kanade tracking error versus displacement for
                     one to four pyramid levels

Usage:
  python examples/ch05_features.py [--outdir DIR] [--seeds N]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from vslam import features as ft  # noqa: E402
from vslam import flow, image, ransac  # noqa: E402

K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])


def corners_and_descriptors(img: np.ndarray, n: int = 500):
    kps = ft.detect(ft.shi_tomasi(img), n)
    return kps, ft.describe(img, kps, ft.orientation(img, kps))


def match_views(img0, img1, H, n: int = 500, tol: float = 3.0):
    """Match two views related by H; return the matched keypoints and
    which matches are correct (transfer error below tol pixels)."""
    k0, d0 = corners_and_descriptors(img0, n)
    k1, d1 = corners_and_descriptors(img1, n)
    m = ft.match(d0, d1)
    p0, p1 = k0[m[:, 0]], k1[m[:, 1]]
    err = np.linalg.norm(image.apply_homography(H, p0) - p1, axis=1)
    return p0, p1, err < tol


def fig_textures(outdir: Path) -> None:
    garden = image.garden_texture(rng=np.random.default_rng(0))
    lawn = image.lawn_texture(rng=np.random.default_rng(0))
    windy = image.lawn_texture(rng=np.random.default_rng(0), sway=0.2,
                               sway_rng=np.random.default_rng(9))
    fig, ax = plt.subplots(1, 4, figsize=(13, 3.0),
                           gridspec_kw={"width_ratios": [4, 4, 4, 3]})
    for a, img, title in [(ax[0], garden, "garden"), (ax[1], lawn, "lawn")]:
        kps = ft.detect(ft.shi_tomasi(img), 300)
        a.imshow(img, cmap="gray", vmin=0, vmax=1)
        a.plot(kps[:, 0], kps[:, 1], "o", ms=3, mfc="none",
               color="tab:orange", mew=0.8)
        a.set_title(f"{title}: 300 strongest corners", fontsize=9)
    resp = ft.shi_tomasi(garden)
    ax[2].imshow(np.sqrt(np.maximum(resp, 0)), cmap="Greens")
    ax[2].set_title("garden: $\\sqrt{\\lambda_{\\min}}$", fontsize=9)
    crop = np.s_[200:280, 300:360]
    ax[3].imshow(np.hstack([lawn[crop], np.ones((80, 4)), windy[crop]]),
                 cmap="gray", vmin=0, vmax=1)
    ax[3].set_title("lawn patch: calm | windy", fontsize=9)
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    fig.tight_layout()
    fig.savefig(outdir / "ch05_textures.pdf")


def fig_matching(outdir: Path, seeds: int) -> None:
    angles = [0, 10, 20, 30, 40]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for tex, name, col in [(image.garden_texture, "garden", "tab:blue"),
                           (image.lawn_texture, "lawn", "tab:green")]:
        for photo, ls in [(False, "-"), (True, "--")]:
            prec = []
            for a in angles:
                vals = []
                for s in range(seeds):
                    rng = np.random.default_rng(s)
                    img0 = tex(rng=rng)
                    H = image.plane_homography(K, np.radians(a))
                    img1 = image.warp(img0, H)
                    if photo:
                        img1 = image.illumination(img1, 0.8, 0.05, 1.2,
                                                  0.01, rng)
                    _, _, ok = match_views(img0, img1, H)
                    vals.append(ok.mean())
                prec.append(np.mean(vals))
            ax[0].plot(angles, prec, "o" + ls, color=col, ms=4,
                       label=name + (", lighting change" if photo else ""))
    ax[0].set_xlabel("viewpoint change (orbit angle) [deg]")
    ax[0].set_ylabel("precision of matches")
    ax[0].set_ylim(0, 1.02)
    ax[0].legend(fontsize=8)
    ax[0].set_title("rigid planar textures", fontsize=9)

    sways = [0.0, 0.025, 0.05, 0.1, 0.2, 0.4]
    H = image.plane_homography(K, np.radians(10))
    n_ok, prec = [], []
    for sw in sways:
        a, b = [], []
        for s in range(seeds):
            img0 = image.lawn_texture(rng=np.random.default_rng(s))
            img1 = image.warp(image.lawn_texture(
                rng=np.random.default_rng(s), sway=sw,
                sway_rng=np.random.default_rng(100 + s)), H)
            _, _, ok = match_views(img0, img1, H)
            a.append(ok.sum())
            b.append(ok.mean() if len(ok) else 0.0)
        n_ok.append(np.mean(a))
        prec.append(np.mean(b))
    deg = np.degrees(sways)
    ax[1].plot(deg, n_ok, "o-", color="tab:green", ms=4,
               label="correct matches")
    ax[1].set_xlabel("wind: std of blade rotation [deg]")
    ax[1].set_ylabel("correct matches (of 500 corners)")
    tw = ax[1].twinx()
    tw.plot(deg, prec, "s--", color="tab:orange", ms=4, label="precision")
    tw.set_ylabel("precision", color="tab:orange")
    tw.set_ylim(0, 1.02)
    ax[1].set_title("lawn, 10 deg viewpoint change, wind", fontsize=9)
    h1, l1 = ax[1].get_legend_handles_labels()
    h2, l2 = tw.get_legend_handles_labels()
    ax[1].legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(outdir / "ch05_matching.pdf")


def fig_ransac(outdir: Path) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    w = np.linspace(0.1, 0.95, 200)
    for s, col in zip([3, 4, 5, 8], plt.cm.Greens(np.linspace(0.4, 1, 4))):
        ax[0].semilogy(w, [ransac.required_iterations(x, s) for x in w],
                       color=col, label=f"$s={s}$")
    ax[0].set_xlabel("inlier ratio $w$")
    ax[0].set_ylabel("samples $N$ for $p=0.99$")
    ax[0].legend(fontsize=8)
    ax[0].grid(True, which="major", lw=0.3)
    # homography RANSAC on a windy lawn
    rng = np.random.default_rng(4)
    H = image.plane_homography(K, np.radians(10))
    img0 = image.lawn_texture(rng=np.random.default_rng(4))
    img1 = image.warp(image.lawn_texture(rng=np.random.default_rng(4),
                                         sway=0.1,
                                         sway_rng=np.random.default_rng(7)),
                      H)
    p0, p1, ok = match_views(img0, img1, H)
    res = ransac.homography_ransac(p0, p1, 3.0, rng=rng)
    ax[1].imshow(img1, cmap="gray", vmin=0, vmax=1)
    for a_, b_, inl in zip(p0, p1, res.inliers):
        col = "tab:green" if inl else "tab:red"
        ax[1].plot([a_[0], b_[0]], [a_[1], b_[1]], color=col, lw=1.2)
        ax[1].plot(b_[0], b_[1], "o", color=col, ms=3)
    ax[1].set_title(f"{len(p0)} matches, {res.inliers.sum()} RANSAC "
                    f"inliers ({ok[res.inliers].mean():.0%} correct), "
                    f"{res.iterations} samples", fontsize=9)
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    fig.tight_layout()
    fig.savefig(outdir / "ch05_ransac.pdf")
    print(f"RANSAC on windy lawn: {len(p0)} matches, {ok.sum()} correct; "
          f"{res.inliers.sum()} inliers, {res.iterations} iterations")


def fig_lk(outdir: Path) -> None:
    img0 = image.garden_texture(rng=np.random.default_rng(1))
    pts = ft.detect(ft.shi_tomasi(img0), 150, border=80)
    shifts = np.array([0.5, 1, 2, 4, 6, 8, 12, 16, 24, 32])
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for levels, col in zip([1, 2, 3, 4],
                           plt.cm.Greens(np.linspace(0.4, 1, 4))):
        med = []
        for s in shifts:
            d = s * np.array([np.cos(0.6), np.sin(0.6)])
            T = np.array([[1, 0, d[0]], [0, 1, d[1]], [0, 0, 1.0]])
            new, ok = flow.lucas_kanade(img0, image.warp(img0, T), pts,
                                        levels=levels)
            med.append(np.median(np.linalg.norm(new - pts - d, axis=1)))
        ax.loglog(shifts, med, "o-", color=col, ms=4,
                  label=f"{levels} level{'s' if levels > 1 else ''}")
    ax.set_xlabel("true displacement [px]")
    ax.set_ylabel("median tracking error [px]")
    ax.axhline(0.1, color="0.5", ls=":", lw=0.8)
    ax.legend(fontsize=8)
    ax.set_title("Lucas-Kanade, 15x15 window", fontsize=9)
    fig.tight_layout()
    fig.savefig(outdir / "ch05_lk.pdf")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outdir", type=Path,
                    default=Path(__file__).resolve().parents[2]
                    / "course" / "figures")
    ap.add_argument("--seeds", type=int, default=4)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    fig_textures(args.outdir)
    fig_matching(args.outdir, args.seeds)
    fig_ransac(args.outdir)
    fig_lk(args.outdir)
    print("figures written to", args.outdir)


if __name__ == "__main__":
    main()
