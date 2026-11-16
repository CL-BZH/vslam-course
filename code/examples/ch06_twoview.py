"""Chapter 6 practical: two-view geometry and its degeneracies.

Produces two figures in course/figures:

  ch06_degeneracy.pdf     relative-pose error of the eight-point and
                          five-point algorithms and of the homography
                          model, (a) as the scene flattens into a lawn and
                          (b) as the motion shrinks to a pure rotation
  ch06_triangulation.pdf  (a) depth error versus parallax angle, and
                          (b) the parallax of a forward-moving camera
                          across the image

Usage:
  python examples/ch06_twoview.py [--outdir DIR] [--trials N]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from vslam import lie, twoview as tv  # noqa: E402
from vslam.camera import look_at  # noqa: E402

F_PX = 500.0          # focal length used to express noise in pixels
SIGMA_PX = 0.5


def mower_scene(rng, relief: float, forward: float = 0.3,
                yaw_deg: float = 3.0, n: int = 150):
    """A camera 0.8 m above a lawn, pitched down, moving forward and
    turning slightly. Points lie on the ground plus a random height of
    standard deviation `relief` (0: a perfect lawn).

    Returns normalised points x1, x2 (noisy), the true (R, t) = T_21 and
    the ground normal in camera-1 coordinates.
    """
    T1 = look_at(np.array([0.0, 0.0, 0.8]), np.array([0.0, 4.0, 0.0]))
    yaw = np.radians(yaw_deg)
    T_move = lie.make_T(lie.so3_exp(np.array([0.0, 0.0, yaw])),
                        np.array([0.0, forward, 0.0]))
    T2 = T_move @ T1
    T2[:3, 3] = T1[:3, 3] + np.array([0.0, forward, 0.0])
    P = np.column_stack([rng.uniform(-3, 3, n), rng.uniform(1.5, 9, n),
                         np.abs(rng.normal(0.0, relief, n))])
    q1 = (P - T1[:3, 3]) @ T1[:3, :3]
    q2 = (P - T2[:3, 3]) @ T2[:3, :3]
    keep = (q1[:, 2] > 0.5) & (q2[:, 2] > 0.5)
    q1, q2 = q1[keep], q2[keep]
    noise = SIGMA_PX / F_PX
    x1 = q1[:, :2] / q1[:, 2:] + rng.normal(0, noise, (len(q1), 2))
    x2 = q2[:, :2] / q2[:, 2:] + rng.normal(0, noise, (len(q2), 2))
    T21 = lie.se3_inv(T2) @ T1
    # ground normal in camera 1, oriented so that n^T q = d > 0 for ground
    # points q (it points from the camera down to the lawn)
    n_c1 = T1[:3, :3].T @ np.array([0.0, 0.0, -1.0])
    return x1, x2, T21[:3, :3], T21[:3, 3], n_c1


def rot_err_deg(Ra, Rb) -> float:
    return float(np.degrees(np.linalg.norm(lie.so3_log(Ra.T @ Rb))))


def dir_err_deg(ta, tb) -> float:
    if np.linalg.norm(ta) < 1e-12 or np.linalg.norm(tb) < 1e-12:
        return np.nan
    c = ta @ tb / np.linalg.norm(ta) / np.linalg.norm(tb)
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def estimate_all(x1, x2, n_ground, rng):
    """Relative pose by eight-point RANSAC, five-point RANSAC and the
    homography model (choosing the decomposition whose plane normal is
    closest to the known ground normal)."""
    thr = 2.0 / F_PX
    out = {}
    res = tv.essential_ransac(x1, x2, thr, rng=rng)
    rp = tv.recover_pose(res.model, x1[res.inliers], x2[res.inliers])
    out["eight-point"] = (rp.R, rp.t)
    E5, m5 = tv.five_point_ransac(x1, x2, thr)
    rp = tv.recover_pose(E5, x1[m5], x2[m5])
    out["five-point"] = (rp.R, rp.t)
    H, mh = cv2.findHomography(x1, x2, cv2.RANSAC, thr)
    mh = mh.ravel().astype(bool)
    sols = tv.decompose_homography(H, x1[mh], x2[mh])
    if sols:
        R, t, n = max(sols, key=lambda s: s[2] @ n_ground)
        out["homography"] = (R, t)
    else:
        out["homography"] = (np.eye(3), np.zeros(3))
    return out


def fig_degeneracy(outdir: Path, trials: int) -> None:
    methods = [("eight-point", "tab:orange"), ("five-point", "tab:blue"),
               ("homography", "tab:green")]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.9))
    rng = np.random.default_rng(0)
    reliefs = [0.0, 0.01, 0.03, 0.1, 0.3, 1.0]
    res = {m: [] for m, _ in methods}
    for rel in reliefs:
        fails = {m: 0 for m, _ in methods}
        for _ in range(2 * trials):
            x1, x2, R, t, n = mower_scene(rng, rel)
            for m, (Re, te) in estimate_all(x1, x2, n, rng).items():
                fails[m] += rot_err_deg(Re, R) > 1.0
        for m in fails:
            res[m].append(100.0 * fails[m] / (2 * trials))
    xs = np.array(reliefs) + 0.003
    for m, col in methods:
        ax[0].semilogx(xs, res[m], "o-", color=col, ms=4, label=m)
    ax[0].set_xlabel("relief: std of point height above the lawn [m]"
                     " (+3 mm)")
    ax[0].set_ylabel("failures: rotation error > 1 deg [%]")
    ax[0].set_ylim(-3, 103)
    ax[0].set_title("forward 30 cm, yaw 3 deg", fontsize=9)
    ax[0].legend(fontsize=8)

    forwards = [0.3, 0.1, 0.03, 0.01, 0.003, 0.001]
    rot = {m: [] for m, _ in methods}
    tra = {m: [] for m, _ in methods}
    for f in forwards:
        er = {m: [] for m, _ in methods}
        et = {m: [] for m, _ in methods}
        for _ in range(trials):
            x1, x2, R, t, n = mower_scene(rng, 0.3, forward=f)
            for m, (Re, te) in estimate_all(x1, x2, n, rng).items():
                er[m].append(rot_err_deg(Re, R))
                et[m].append(dir_err_deg(te, t))
        for m in er:
            rot[m].append(np.median(er[m]))
            tra[m].append(np.nanmedian(et[m]))
    for m, col in methods[:2]:
        ax[1].loglog(forwards, rot[m], "o-", color=col, ms=4,
                     label=f"{m}: rotation")
        ax[1].loglog(forwards, tra[m], "s--", color=col, ms=4,
                     label=f"{m}: translation direction")
    ax[1].invert_xaxis()
    ax[1].set_xlabel("forward motion between the views [m]")
    ax[1].set_ylabel("median error [deg]")
    ax[1].set_title("relief 0.3 m, yaw 3 deg: towards pure rotation",
                    fontsize=9)
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(outdir / "ch06_degeneracy.pdf")
    for m, _ in methods:
        print(f"{m:>12}: failure rate [%] vs relief", np.round(res[m], 1))


def fig_triangulation(outdir: Path, trials: int) -> None:
    rng = np.random.default_rng(1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.9))
    Z = 5.0
    angles = np.radians([0.1, 0.3, 1, 3, 10, 30])
    med = []
    for th in angles:
        b = 2 * Z * np.tan(th / 2)                 # sideways baseline
        R, t = np.eye(3), np.array([-b, 0.0, 0.0])
        X = np.column_stack([rng.uniform(-0.5, 0.5, 400) + b / 2,
                             rng.uniform(-0.5, 0.5, 400),
                             np.full(400, Z)])
        x1 = X[:, :2] / X[:, 2:]
        Y = X + t
        x2 = Y[:, :2] / Y[:, 2:]
        s = SIGMA_PX / F_PX
        Xh = tv.triangulate(R, t, x1 + rng.normal(0, s, x1.shape),
                            x2 + rng.normal(0, s, x2.shape))
        med.append(np.median(np.abs(Xh[:, 2] - Z) / Z))
    deg = np.degrees(angles)
    ax[0].loglog(deg, med, "o", color="tab:green", label="Monte Carlo")
    pred = 0.6745 * np.sqrt(2) * SIGMA_PX / F_PX / angles
    ax[0].loglog(deg, pred, "-", color="0.3",
                 label="$0.67\\,\\sqrt{2}\\,\\sigma/(f\\,\\theta)$")
    ax[0].set_xlabel("parallax angle $\\theta$ [deg]")
    ax[0].set_ylabel("median relative depth error")
    ax[0].set_title(f"depth 5 m, {SIGMA_PX} px noise, f = {F_PX:.0f} px",
                    fontsize=9)
    ax[0].legend(fontsize=8)
    # forward motion: parallax of a wall 5 m ahead after moving 30 cm
    u, v = np.meshgrid(np.linspace(-0.64, 0.64, 161),
                       np.linspace(-0.48, 0.48, 121))
    X = np.column_stack([u.ravel() * Z, v.ravel() * Z, np.full(u.size, Z)])
    par = tv.parallax_angle(np.eye(3), np.array([0.0, 0.0, -0.3]), X)
    im = ax[1].imshow(np.degrees(par).reshape(u.shape), cmap="Greens",
                      extent=[0, 640, 480, 0])
    ax[1].contour(np.linspace(0, 640, 161), np.linspace(0, 480, 121),
                  np.degrees(par).reshape(u.shape), levels=[0.5, 1.0, 1.5],
                  colors="k", linewidths=0.6)
    fig.colorbar(im, ax=ax[1], label="parallax [deg]")
    ax[1].set_title("forward motion 30 cm, wall at 5 m: parallax per pixel",
                    fontsize=9)
    ax[1].set_xlabel("$u$ [px]")
    ax[1].set_ylabel("$v$ [px]")
    fig.tight_layout()
    fig.savefig(outdir / "ch06_triangulation.pdf")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outdir", type=Path,
                    default=Path(__file__).resolve().parents[2]
                    / "course" / "figures")
    ap.add_argument("--trials", type=int, default=20)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    fig_degeneracy(args.outdir, args.trials)
    fig_triangulation(args.outdir, args.trials)
    print("figures written to", args.outdir)


if __name__ == "__main__":
    main()
