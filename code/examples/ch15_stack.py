"""Chapter 15 practical: the localisation stack, evaluated.

The garden drive of Chapters 13-14, with the camera, the IMU, the wheels
and the RTK receiver. Produces in course/figures:

  ch15_eval.pdf     position errors and NEES of the stereo VO, the VIO,
                    VIO + wheels and the full stack (VIO + wheels + gated
                    RTK-GNSS)
  ch15_failure.pdf  error against the reported 3-sigma: camera blind in
                    the first turn (VIO, VIO + wheels), and camera blind
                    under the tree's canopy (full stack, with the GNSS
                    gate's decisions)
  ch15_map.pdf      the mowing map fused with the stack's poses, and the
                    planner grid with a coverage path

Usage:
  python examples/ch15_stack.py [--outdir DIR]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import chi2  # noqa: E402

from vslam import evaluation, fusion, lie, maps, scene, stack, vio  # noqa
from ch13_vio import setup, stereo_vo, CAM_RATE, KF_EVERY  # noqa: E402
from ch14_fusion import sensors, LEVER, ENC_RATE  # noqa: E402
from ch11_maps import keyframe_clouds, scores, draw, SPEC, EXTENT  # noqa
from ch11_maps import CMAP  # noqa: E402


def run_stack(m, P, data, S, Tbc, ts, frames, sens, wheels=True, gnss=True,
              gate=0.999, blind=None, t_end=None, inflate=1.0):
    """The stack on the drive; the VIO starts in the true frame after a
    static initialisation (Chapter 13)."""
    k0 = int(round(m.still * CAM_RATE))
    n0 = int(round(m.still / P.dt))
    R0, bg = vio.static_init(data.gyro[:n0], data.accel[:n0])
    yaw = np.arctan2(m.R(m.still)[1, 0], m.R(m.still)[0, 0])
    R0 = lie.so3_exp(np.array([0.0, 0.0, yaw])) @ R0
    s0 = vio.KFState(R0, np.zeros(3), m.p(m.still), bg, np.zeros(3))
    st = stack.LocalisationStack(S.stereo, Tbc, P, LEVER, wheels=wheels,
                                 gnss=gnss, gate=gate, inflate=inflate)
    sig = np.r_[[1e-2] * 2, [1e-6], [1e-3] * 3, [1e-6] * 3, [1e-3] * 3,
                [0.1] * 3]

    def speed(t):
        i = int(round(t * ENC_RATE))
        return fusion.diff_drive(sens["vl"][i], sens["vr"][i],
                                 sens["enc"].track)[0]

    st.start(ts[k0], s0, sig, frames[k0].track_ids, frames[k0].z,
             speed(ts[k0]))
    last = n0
    tic = time.time()
    for k in range(k0 + KF_EVERY, len(frames), KF_EVERY):
        t = ts[k]
        if t_end and t > t_end:
            break
        n = int(round(t / P.dt))
        f = frames[k]
        ids, z = f.track_ids, f.z
        if blind and blind[0] <= t < blind[1]:
            ids, z = ids[:0], z[:0]
        j = int(round(t * sens["rx"].rate))
        st.step(t, data.gyro[last:n], data.accel[last:n], ids, z, speed(t),
                (sens["z"][j], sens["sig"][j]))
        last = n
    st.runtime = time.time() - tic
    return st


def summary(m, st):
    t = np.array([r.t for r in st.reports])
    e = np.array([r.state.p - m.p(r.t) for r in st.reports])
    C = np.array([r.cov_p for r in st.reports])
    sig_h = np.array([np.sqrt(np.linalg.eigvalsh(c[:2, :2]).max())
                      for c in C])
    return t, e, evaluation.nees(e, C), sig_h


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outdir", type=Path,
                    default=Path(__file__).resolve().parents[2]
                    / "course" / "figures")
    out = ap.parse_args().outdir
    out.mkdir(parents=True, exist_ok=True)

    m, P, data, S, Tbc, ts, frames = setup()
    sens = sensors(m)
    q99 = chi2.ppf(0.99, 3)

    # ---------------------------------------------------- configurations
    vo_est = stereo_vo(m, S, Tbc, ts, frames)
    res = {}
    for name, kw in [("VIO", dict(wheels=False, gnss=False)),
                     ("VIO + wheels", dict(wheels=True, gnss=False)),
                     ("full stack", dict(wheels=True, gnss=True))]:
        st = run_stack(m, P, data, S, Tbc, ts, frames, sens, **kw)
        res[name] = (st, summary(m, st))
    print(f"{'':14s} {'ATE none':>9s} {'ATE 4dof':>9s} {'RPE 5 m':>8s} "
          f"{'ANEES':>6s} {'med.NEES':>8s} {'>99%':>5s} {'time':>5s}")
    # the stereo VO
    tvo = np.array([t for t, _, _ in vo_est])
    pvo = np.array([p for _, _, p in vo_est])
    gvo = np.array([m.p(t) for t in tvo])
    Tvo = [lie.make_T(R, p) for _, R, p in vo_est]
    Tg = [m.T(t) for t in tvo]
    rt, _ = evaluation.rpe_distance(Tvo, Tg, 5.0)
    rows = {"stereo VO": (100 * np.sqrt(np.mean(evaluation.ate(pvo, gvo) ** 2)),
                          100 * np.sqrt(np.mean(evaluation.ate(pvo, gvo,
                                                               "4dof") ** 2)),
                          100 * np.median(rt), np.nan, np.nan, np.nan,
                          np.nan)}
    for name, (st, (t, e, ne, _)) in res.items():
        est = np.array([r.state.p for r in st.reports])
        gt = np.array([m.p(r.t) for r in st.reports])
        Te = [lie.make_T(r.state.R, r.state.p) for r in st.reports]
        Tgt = [m.T(r.t) for r in st.reports]
        rt, _ = evaluation.rpe_distance(Te, Tgt, 5.0)
        rows[name] = (100 * np.sqrt(np.mean(evaluation.ate(est, gt) ** 2)),
                      100 * np.sqrt(np.mean(evaluation.ate(est, gt, "4dof")
                                            ** 2)),
                      100 * np.median(rt), ne.mean(), np.median(ne),
                      100 * np.mean(ne > q99), st.runtime)
    for name, r in rows.items():
        print(f"{name:14s} {r[0]:8.1f}c {r[1]:8.1f}c {r[2]:7.1f}c "
              f"{r[3]:6.2f} {r[4]:8.2f} {r[5]:4.0f}% {r[6]:5.0f}")
    lo, hi = evaluation.anees_bounds(len(res["VIO"][1][2]), 3)
    print(f"ANEES 95 % bounds (independent samples): [{lo:.2f}, {hi:.2f}]; "
          f"median of chi2(3): {chi2.ppf(0.5, 3):.2f}")
    st_full = res["full stack"][0]
    dec = [r.gnss for r in st_full.reports]
    print(f"GNSS decisions: {dec.count('accepted')} accepted, "
          f"{dec.count('rejected')} rejected")
    # inflation that brings the median NEES of VIO + wheels to that of chi2
    for name in ("VIO", "VIO + wheels"):
        ne = res[name][1][2]
        print(f"{name}: inflation for a median NEES of chi2(3): "
              f"{np.median(ne) / chi2.ppf(0.5, 3):.2f}")

    fig, ax = plt.subplots(1, 2, figsize=(12, 3.9))
    ax[0].semilogy(tvo, 100 * np.linalg.norm(pvo - gvo, axis=1),
                   color="tab:orange", lw=0.8, label="stereo VO")
    for (name, (_, (t, e, ne, _))), c in zip(res.items(),
                                             ["tab:blue", "tab:purple",
                                              "tab:green"]):
        ax[0].semilogy(t, 100 * np.linalg.norm(e, axis=1), color=c, lw=0.8,
                       label=name)
        ax[1].semilogy(t, ne, ".", ms=2, color=c, label=name)
    ax[0].set_xlabel("time [s]")
    ax[0].set_ylabel("position error [cm]")
    ax[0].legend(fontsize=7)
    a, b = chi2.ppf([0.025, 0.975], 3)
    ax[1].axhspan(a, b, color="0.9", label=r"95 % of $\chi^2_3$")
    ax[1].axhline(3, color="k", lw=0.6)
    ax[1].set_xlabel("time [s]")
    ax[1].set_ylabel("NEES of the position")
    ax[1].legend(fontsize=7, markerscale=4, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "ch15_eval.pdf")

    # ---------------------------------------------------- failures
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
    for a_, (name, kw, blind, t_end) in zip(ax, [
            ("VIO, camera blind in the first turn",
             dict(wheels=False, gnss=False), (12.5, 16.5), 20.0),
            ("VIO + wheels, camera blind in the first turn",
             dict(wheels=True, gnss=False), (12.5, 16.5), 20.0),
            ("full stack, camera blind under the tree",
             dict(wheels=True, gnss=True), (5.5, 9.5), 14.0)]):
        st = run_stack(m, P, data, S, Tbc, ts, frames, sens, blind=blind,
                       t_end=t_end, **kw)
        t, e, ne, sh = summary(m, st)
        eh = np.linalg.norm(e[:, :2], axis=1)
        a_.semilogy(t, 100 * eh, color="tab:blue", label="horizontal error")
        a_.semilogy(t, 300 * sh, color="tab:red", ls="--",
                    label=r"reported $3\sigma$ (largest axis)")
        a_.axvspan(*blind, color="0.9", label="camera blind")
        a_.axhline(15, color="k", lw=0.6, ls=":",
                   label="alarm: 3σ = 15 cm")
        if kw["gnss"]:
            for r in st.reports:
                if r.gnss == "rejected":
                    a_.plot(r.t, 100 * np.sqrt(r.cov_p[0, 0]) * 3, "kx",
                            ms=6)
            a_.plot([], [], "kx", label="GNSS fix rejected by the gate")
        a_.set_title(name, fontsize=9)
        a_.set_xlabel("time [s]")
        a_.set_ylabel("cm")
        a_.legend(fontsize=6.5, loc="upper left")
        kb = (t >= blind[0]) & (t <= blind[1] + 0.3)
        alarm = t[(3 * sh > 0.15)]
        print(f"{name}: max error {100 * eh[kb].max():.1f} cm, reported "
              f"3 sigma {300 * sh[kb][np.argmax(eh[kb])]:.1f} cm there; "
              f"alarm (3 sigma > 15 cm) from "
              f"{alarm.min() if len(alarm) else 'never'}")
    fig.tight_layout()
    fig.savefig(out / "ch15_failure.pdf")

    # ---------------------------------------------------- the map
    garden = scene.Garden()
    truth = garden.truth_grid(EXTENT, SPEC.res)
    reps = st_full.reports
    T_true = [m.T(r.t) @ Tbc for r in reps]
    T_est = [lie.make_T(r.state.R, r.state.p) @ Tbc for r in reps]
    clouds, _ = keyframe_clouds(garden, S.stereo, T_true,
                                np.random.default_rng(11))
    cls = maps.fuse(SPEC, T_est, clouds).classes()
    sc = scores(cls, truth)
    print("map from the stack's poses:", sc)
    mow = maps.coarsen(cls, 10)
    text = maps.to_lawn_text(mow, start=(3, 3))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    draw(ax[0], cls, f"mowing map (stack poses): lawn found "
         f"{100 * sc['lawn found']:.0f} %, wrongly mowable "
         f"{sc['obstacle -> mowable'] + sc['not lawn -> mowable']}")
    x0, y0 = SPEC.x0, SPEC.y0
    Hc, Wc = mow.shape
    ax[1].imshow(np.where(mow, 0, 1), origin="lower", cmap=CMAP, vmin=0,
                 vmax=3, extent=(x0, x0 + Wc, y0, y0 + Hc), alpha=0.85)
    try:
        from lawnmower import Lawn, plan
        r = plan(Lawn.from_string(text), "stc")
        rows_, cols = np.array([c[0] for c in r.path]), \
            np.array([c[1] for c in r.path])
        ax[1].plot(x0 + cols - 0.5, y0 + (Hc - rows_) + 0.5, "-",
                   color="tab:blue", lw=1.2)
        md = r.metrics.as_dict()
        ax[1].set_title(f"planner grid (1 m cells) and STC path: "
                        f"{md['coverage_%']:.0f} % covered in {md['steps']} "
                        f"moves", fontsize=9)
        print("STC:", r.metrics.as_dict())
    except ImportError:
        ax[1].set_title("planner grid (1 m cells)", fontsize=9)
    ax[1].set_xlabel("x [m]")
    ax[1].set_ylabel("y [m]")
    fig.tight_layout()
    fig.savefig(out / "ch15_map.pdf")
    print(f"{mow.sum()} mowable planner cells")
    print("figures written to", out)


if __name__ == "__main__":
    main()
