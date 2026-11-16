"""Chapter 3 practical: lens distortion, Zhang calibration, stereo depth.

Produces three figures in course/figures:

  ch03_distortion.pdf  displacement field of a wide-angle lens and the
                       image of a straight-line grid
  ch03_zhang.pdf       accuracy of the linear step of Zhang's method versus
                       pixel noise and number of views, without and with
                       (unmodelled) lens distortion
  ch03_stereo.pdf      depth uncertainty of a rectified stereo pair versus
                       depth, closed form and Monte Carlo

Usage:
  python examples/ch03_cameras.py [--outdir DIR] [--trials N]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from vslam import calib  # noqa: E402
from vslam.camera import PinholeCamera, StereoCamera  # noqa: E402

# A wide-angle camera similar to those on small robots (EuRoC-like).
WIDE = PinholeCamera(458.7, 457.3, 367.2, 248.4,
                     dist=[-0.283, 0.074, 1.8e-4, 1.8e-5, 0.0],
                     width=752, height=480)


def fig_distortion(outdir: Path) -> None:
    cam = WIDE
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.9))
    # (a) displacement field: where does each ideal pixel end up?
    us, vs = np.meshgrid(np.linspace(20, cam.width - 20, 19),
                         np.linspace(20, cam.height - 20, 13))
    x = np.column_stack([(us.ravel() - cam.cx) / cam.fx,
                         (vs.ravel() - cam.cy) / cam.fy])
    ud = cam.pixel(cam.distort(x))
    d = ud - np.column_stack([us.ravel(), vs.ravel()])
    ax[0].quiver(us.ravel(), vs.ravel(), d[:, 0], d[:, 1], angles="xy",
                 scale_units="xy", scale=1, color="tab:green", width=0.003)
    ax[0].set_title("displacement ideal $\\to$ distorted (true scale)")
    # (b) a grid of straight lines in the world, as the lens sees it
    for c in np.linspace(-1.0, 1.0, 13):
        t = np.linspace(-1.0, 1.0, 200)
        for line in (np.column_stack([np.full_like(t, c * 0.95), t * 0.7]),
                     np.column_stack([t * 0.95, np.full_like(t, c * 0.7)])):
            u = cam.pixel(cam.distort(line))
            ax[1].plot(u[:, 0], u[:, 1], color="0.25", lw=0.7)
    ax[1].set_title("straight lines through the lens")
    for a in ax:
        a.set_xlim(0, cam.width)
        a.set_ylim(cam.height, 0)
        a.set_aspect("equal")
        a.set_xlabel("$u$ [px]")
        a.add_patch(plt.Rectangle((0, 0), cam.width, cam.height, fill=False,
                                  color="tab:orange", lw=1))
    ax[0].set_ylabel("$v$ [px]")
    fig.tight_layout()
    fig.savefig(outdir / "ch03_distortion.pdf")


def zhang_error(cam: PinholeCamera, n_views: int, noise: float,
                trials: int, rng: np.random.Generator) -> float:
    """Median relative error of the focal length fx over `trials` runs."""
    errs = []
    for _ in range(trials):
        board, views = calib.synthetic_views(cam, n_views, rng, noise)
        Hs = [calib.homography_dlt(board[:, :2], u) for _, u in views]
        try:
            K = calib.zhang_intrinsics(Hs)
        except np.linalg.LinAlgError:  # B not positive definite
            errs.append(np.inf)
            continue
        errs.append(abs(K[0, 0] - cam.fx) / cam.fx)
    return float(np.median(errs))


def fig_zhang(outdir: Path, trials: int) -> None:
    rng = np.random.default_rng(0)
    noises = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
    views = [3, 5, 10, 20]
    ideal = PinholeCamera(WIDE.fx, WIDE.fy, WIDE.cx, WIDE.cy,
                          width=WIDE.width, height=WIDE.height)
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.8), sharey=True)
    for a, cam, title in [(ax[0], ideal, "no lens distortion"),
                          (ax[1], WIDE, "wide-angle lens, distortion "
                                        "ignored")]:
        for n, col in zip(views, plt.cm.Greens(np.linspace(0.45, 1, 4))):
            e = [zhang_error(cam, n, s, trials, rng) for s in noises]
            a.loglog(noises, 100 * np.array(e), "o-", color=col,
                     label=f"{n} views")
        a.set_title(title)
        a.set_xlabel("pixel noise $\\sigma$ [px]")
        a.grid(True, which="both", lw=0.3)
    ax[0].set_ylabel("median $|\\hat f_x - f_x| / f_x$ [%]")
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "ch03_zhang.pdf")


def fig_stereo(outdir: Path) -> None:
    rng = np.random.default_rng(1)
    cam = PinholeCamera(700.0, 700.0, 640.0, 360.0, width=1280, height=720)
    sigma_d = 0.25
    Z = np.linspace(0.5, 15.0, 100)
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for b, col in zip([0.06, 0.12, 0.30], ["tab:orange", "tab:green",
                                           "tab:blue"]):
        st = StereoCamera(cam, b)
        ax.plot(Z, st.depth_std(Z, sigma_d), color=col,
                label=f"baseline {100 * b:.0f} cm")
        # Monte Carlo: add disparity noise and triangulate
        Zs = np.array([1, 3, 6, 10, 15], dtype=float)
        mc = []
        for z in Zs:
            m, _ = st.project(np.array([0.0, 0.0, z]))
            noisy = m[None, :] + np.column_stack([
                np.zeros(4000), np.zeros(4000),
                rng.normal(scale=sigma_d, size=4000)])
            depths = np.array([st.triangulate(n)[2] for n in noisy])
            mc.append(depths.std())
        ax.plot(Zs, mc, "o", color=col, ms=4)
    ax.axvspan(0.5, 10, color="tab:green", alpha=0.06)
    ax.text(0.7, 0.9 * ax.get_ylim()[1], "useful range on a lawn",
            color="tab:green", fontsize=8)
    ax.set_xlabel("depth $Z$ [m]")
    ax.set_ylabel("$\\sigma_Z$ [m]")
    ax.set_title(f"$f$ = 700 px, disparity noise {sigma_d} px "
                 "(lines: first order, dots: Monte Carlo)", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "ch03_stereo.pdf")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outdir", type=Path,
                    default=Path(__file__).resolve().parents[2]
                    / "course" / "figures")
    ap.add_argument("--trials", type=int, default=40)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    fig_distortion(args.outdir)
    fig_zhang(args.outdir, args.trials)
    fig_stereo(args.outdir)
    print("figures written to", args.outdir)


if __name__ == "__main__":
    main()
