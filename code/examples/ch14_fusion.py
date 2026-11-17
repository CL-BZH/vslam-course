"""Chapter 14 practical: wheels and RTK-GNSS with the VIO.

Same drive as Chapter 13 (sim.GardenMotion), with wheel encoders on a
differential drive and an RTK-GNSS antenna on a mast. Produces in
course/figures:

  ch14_wheels.pdf  dead reckoning from the wheels alone and with the
                   gyroscope; VIO with the camera blind in the first turn,
                   with and without the wheel factor
  ch14_gnss.pdf    the GNSS measurements (fix, float under the tree and
                   near the house, multipath outliers) and the errors of
                   VIO, VIO + GNSS (least squares, Cauchy, no lever arm)
  ch14_align.pdf   aligning a VIO started with a wrong heading to ENU:
                   heading error against the length of the alignment

Usage:
  python examples/ch14_fusion.py [--outdir DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from vslam import fusion, lie, robust, vio  # noqa: E402
from ch13_vio import setup, errors, CAM_RATE, KF_EVERY  # noqa: E402

LEVER = np.array([-0.4, 0.0, 1.4])      # antenna: behind and above
ENC_RATE = 50.0
BLIND = (12.5, 16.5)


def body_speed_rate(m, t):
    """Forward speed of the body origin and yaw rate (what a differential
    drive's wheels see)."""
    R, _, v, _, w = m.state(t)
    return (R.T @ v)[0], (R @ w)[2]


def sensors(m, seed=7):
    """Encoder speeds at 50 Hz and RTK fixes at the keyframe epochs."""
    rng = np.random.default_rng(seed)
    enc = fusion.Encoders()
    te = np.arange(0, m.duration - 0.05, 1 / ENC_RATE)
    vw = np.array([body_speed_rate(m, t) for t in te])
    vl, vr = enc.measure(vw[:, 0], vw[:, 1], rng)
    rx = fusion.RTKReceiver()
    tg = np.arange(0, m.duration - 0.05, 1 / rx.rate)
    ant = np.array([m.p(t) + m.R(t) @ LEVER for t in tg])
    z, sig, status = rx.measure(ant, rng)
    return dict(te=te, vw=vw, vl=vl, vr=vr, enc=enc, tg=tg, z=z, sig=sig,
                status=status, ant=ant, rx=rx)


def run(m, P, data, S, Tbc, ts, frames, sens, wheels=False, gnss=None,
        lever=LEVER, blind=None, yaw_err=0.0, p_err=(0.0, 0.0, 0.0),
        align_at=None):
    """VIO of Chapter 13 with optional wheel and GNSS factors. gnss: None,
    "ls" (least squares) or "cauchy". With yaw_err/p_err the VIO starts in
    a wrong frame; with align_at it is aligned to ENU at that time by the
    4-DoF alignment of its antenna positions to the GNSS fixes, after
    which the GNSS factors are used."""
    k0 = int(round(m.still * CAM_RATE))
    n0 = int(round(m.still / P.dt))
    R0, bg = vio.static_init(data.gyro[:n0], data.accel[:n0])
    yaw = np.arctan2(m.R(m.still)[1, 0], m.R(m.still)[0, 0]) + yaw_err
    R0 = lie.so3_exp(np.array([0.0, 0.0, yaw])) @ R0
    s0 = vio.KFState(R0, np.zeros(3), m.p(m.still) + np.asarray(p_err),
                     bg, np.zeros(3))
    V = vio.SlidingWindowVIO(S.stereo, Tbc, P)
    sig = np.r_[[1e-2] * 2, [1e-6], [1e-3] * 3, [1e-6] * 3, [1e-3] * 3,
                [0.1] * 3]
    kern = robust.Cauchy(2.0) if gnss == "cauchy" else None
    aligned = align_at is None
    pairs = []

    def unary(t):
        out = []
        if wheels:
            i = int(round(t * ENC_RATE))
            v, _ = fusion.diff_drive(sens["vl"][i], sens["vr"][i],
                                     sens["enc"].track)
            out.append(fusion.WheelFactor(v, kernel=robust.Huber(2.0)))
        if gnss and aligned:
            j = int(round(t * sens["rx"].rate))
            out.append(fusion.GnssFactor(sens["z"][j], sens["sig"][j], lever,
                                         kernel=kern))
        return out

    V.initialise(ts[k0], s0, sig, frames[k0].track_ids, frames[k0].z,
                 unary(ts[k0]))
    out = [(ts[k0], s0.copy())]
    last = n0
    for k in range(k0 + KF_EVERY, len(frames), KF_EVERY):
        t = ts[k]
        n = int(round(t / P.dt))
        f = frames[k]
        ids, z = f.track_ids, f.z
        if blind and blind[0] <= t < blind[1]:
            ids, z = ids[:0], z[:0]
        s = V.add_keyframe(t, data.gyro[last:n], data.accel[last:n], ids, z,
                           unary(t))
        last = n
        if not aligned:
            j = int(round(t * sens["rx"].rate))
            if sens["status"][j] == 0:
                pairs.append((s.p + s.R @ lever, sens["z"][j]))
            if t >= align_at:
                src, dst = map(np.array, zip(*pairs))
                a, tr = fusion.align_4dof(src, dst)
                V.transform(lie.so3_exp(np.array([0.0, 0.0, a])), tr)
                aligned = True
                V.align = (t, a, tr, len(pairs))
        out.append((t, V.kfs[-1].copy()))
    return V, out


def rms(e):
    return 100 * np.sqrt(np.mean(np.asarray(e) ** 2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outdir", type=Path,
                    default=Path(__file__).resolve().parents[2]
                    / "course" / "figures")
    out = ap.parse_args().outdir
    out.mkdir(parents=True, exist_ok=True)

    m, P, data, S, Tbc, ts, frames = setup()
    sens = sensors(m)
    tt = np.linspace(0, m.duration - 0.05, 3000)
    gt = np.array([m.p(t) for t in tt])

    # ---------------------------------------------------- wheels alone
    te, vw = sens["te"], sens["vw"]
    v_e, w_e = fusion.diff_drive(sens["vl"], sens["vr"], sens["enc"].track)
    psi0 = np.arctan2(m.R(0)[1, 0], m.R(0)[0, 0])
    x0 = (m.p(0)[0], m.p(0)[1], psi0)
    dt = 1 / ENC_RATE
    odo, _ = fusion.dead_reckon_2d(v_e, w_e, dt, x0)
    # heading from the gyroscope (bias known from the still phase)
    n0 = int(round(m.still / P.dt))
    bgz = data.gyro[:n0, 2].mean()
    gz = np.array([(m.R(t) @ data.gyro[int(round(t / P.dt))])[2] - bgz
                   for t in te])
    odo_g, _ = fusion.dead_reckon_2d(v_e, gz, dt, x0)
    e_odo = np.linalg.norm(odo[:-1, :2] - np.array([m.p(t)[:2] for t in te]),
                           axis=1)
    e_odg = np.linalg.norm(odo_g[:-1, :2] - np.array([m.p(t)[:2]
                                                      for t in te]), axis=1)
    psi_t = np.unwrap([np.arctan2(m.R(t)[1, 0], m.R(t)[0, 0]) for t in te])
    print(f"wheels alone: final error {e_odo[-1]:.2f} m, heading error "
          f"{np.degrees(odo[-2, 2] - psi_t[-1]):.1f} deg; wheels + gyro: "
          f"final {e_odg[-1]:.2f} m, heading "
          f"{np.degrees(odo_g[-2, 2] - psi_t[-1]):.1f} deg")

    # ---------------------------------------------------- VIO blind turn
    runs = {}
    for name, kw in [("VIO", {}),
                     ("VIO, blind", dict(blind=BLIND)),
                     ("VIO + wheels, blind", dict(blind=BLIND, wheels=True)),
                     ("VIO + wheels", dict(wheels=True))]:
        V, est = run(m, P, data, S, Tbc, ts, frames, sens, **kw)
        e, tilt, yaw = errors(m, [(t, s.R, s.p) for t, s in est])
        runs[name] = (np.array([t for t, _ in est]), e, yaw)
        kb = (runs[name][0] >= BLIND[0]) & (runs[name][0] < BLIND[1] + 0.3)
        print(f"{name:22s} RMS {rms(e):5.1f} cm, final {100 * e[-1]:5.1f}, "
              f"max {100 * e.max():5.1f}, max in blackout window "
              f"{100 * e[kb].max():5.1f} cm")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4),
                           gridspec_kw={"width_ratios": [1.1, 1]})
    ax[0].plot(gt[:, 0], gt[:, 1], "k", lw=2, label="truth")
    ax[0].plot(odo[:, 0], odo[:, 1], color="tab:orange", lw=1,
               label=f"wheels alone: {e_odo[-1]:.1f} m at the end")
    ax[0].plot(odo_g[:, 0], odo_g[:, 1], color="tab:green", lw=1,
               label=f"wheel speed + gyroscope: {e_odg[-1]:.1f} m")
    ax[0].set_aspect("equal")
    ax[0].set_xlabel("x [m]")
    ax[0].set_ylabel("y [m]")
    ax[0].legend(fontsize=7, loc="upper center", ncol=3,
                 bbox_to_anchor=(0.5, -0.17))
    for name, c in [("VIO", "tab:blue"), ("VIO, blind", "tab:red"),
                    ("VIO + wheels, blind", "tab:green")]:
        t, e, _ = runs[name]
        ax[1].semilogy(t, 100 * e, color=c, label=name, lw=1)
    ax[1].axvspan(*BLIND, color="0.9", label="camera blind")
    ax[1].set_xlabel("time [s]")
    ax[1].set_ylabel("position error [cm]")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "ch14_wheels.pdf", bbox_inches="tight")

    # ---------------------------------------------------- GNSS
    z, st, tg = sens["z"], sens["status"], sens["tg"]
    e_g = np.linalg.norm(z - sens["ant"], axis=1)
    print(f"GNSS: {np.mean(st == 0) * 100:.0f} % fix, "
          f"{np.mean(st == 1) * 100:.0f} % float, {np.sum(st == 2)} outliers;"
          f" error fix {100 * np.median(e_g[st == 0]):.1f} cm (median), "
          f"float {100 * np.median(e_g[st == 1]):.0f} cm, outliers "
          f"{np.median(e_g[st == 2]):.1f} m")
    for name, kw in [("VIO + GNSS, least squares", dict(gnss="ls")),
                     ("VIO + GNSS, Cauchy", dict(gnss="cauchy")),
                     ("VIO + GNSS, Cauchy, no lever arm",
                      dict(gnss="cauchy", lever=np.zeros(3))),
                     ("VIO + wheels + GNSS, Cauchy",
                      dict(gnss="cauchy", wheels=True))]:
        V, est = run(m, P, data, S, Tbc, ts, frames, sens, **kw)
        e, tilt, yaw = errors(m, [(t, s.R, s.p) for t, s in est])
        runs[name] = (np.array([t for t, _ in est]), e, yaw)
        print(f"{name:34s} RMS {rms(e):5.1f} cm, final {100 * e[-1]:5.1f}, "
              f"max {100 * e.max():5.1f}; |yaw| final "
              f"{abs(yaw[-1]):.3f} deg")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4),
                           gridspec_kw={"width_ratios": [1.1, 1]})
    ax[0].plot(gt[:, 0], gt[:, 1], color="0.7", lw=1)
    for s_, c, lab in [(0, "tab:green", "RTK fix"),
                       (1, "tab:orange", "float"),
                       (2, "tab:red", "multipath outlier")]:
        sel = st == s_
        ax[0].plot(z[sel, 0], z[sel, 1], ".", ms=3, color=c,
                   label=f"{lab} ({sel.sum()})")
    th = np.linspace(0, 2 * np.pi, 100)
    for cx, cy in sens["rx"].canopy:
        r = sens["rx"].canopy_r
        ax[0].plot(cx + r * np.cos(th), cy + r * np.sin(th), "k:", lw=0.8)
    ax[0].axhline(sens["rx"].house_y, xmax=0.33, color="k", ls=":", lw=0.8)
    ax[0].set_aspect("equal")
    ax[0].set_xlabel("x [m]")
    ax[0].set_ylabel("y [m]")
    ax[0].legend(fontsize=7, loc="upper center", ncol=3,
                 bbox_to_anchor=(0.5, -0.17))
    for name, c in [("VIO", "tab:blue"),
                    ("VIO + GNSS, least squares", "tab:red"),
                    ("VIO + GNSS, Cauchy", "tab:green"),
                    ("VIO + GNSS, Cauchy, no lever arm", "tab:purple")]:
        t, e, _ = runs[name]
        ax[1].semilogy(t, 100 * e, color=c, label=name, lw=1)
    deg = np.array([sens["rx"].degraded(p) for p in sens["ant"]])
    for a in np.nonzero(np.diff(deg.astype(int)) == 1)[0]:
        b = a + 1 + np.argmax(~deg[a + 1:])
        ax[1].axvspan(tg[a + 1], tg[b], color="0.92")
    ax[1].set_xlabel("time [s]")
    ax[1].set_ylabel("position error [cm]")
    ax[1].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "ch14_gnss.pdf", bbox_inches="tight")

    # ---------------------------------------------------- alignment
    yaw_err, p_err = np.radians(30.0), (3.0, -2.0, 0.0)
    Va, est_a = run(m, P, data, S, Tbc, ts, frames, sens, gnss="cauchy",
                    yaw_err=yaw_err, p_err=p_err, align_at=8.0)
    ta = np.array([t for t, _ in est_a])
    ea, _, ya = errors(m, [(t, s.R, s.p) for t, s in est_a])
    t_al, a, tr, npairs = Va.align
    print(f"aligned at {t_al:.2f} s on {npairs} fixes: yaw correction "
          f"{np.degrees(a):.2f} deg (true {-30:.0f}); after: final error "
          f"{100 * ea[-1]:.1f} cm, RMS after 10 s "
          f"{rms(ea[ta > 10]):.1f} cm, |yaw| final {abs(ya[-1]):.3f} deg")
    # heading error of the alignment against its length, from a VIO run
    # in the wrong frame (no GNSS factors)
    Vw, est_w = run(m, P, data, S, Tbc, ts, frames, sens, yaw_err=yaw_err,
                    p_err=p_err)
    lengths, yerr, theory = [], [], []
    tw = np.array([t for t, _ in est_w])
    for T in [2.6, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0]:
        src, dst = [], []
        for t, s in est_w:
            j = int(round(t * sens["rx"].rate))
            if t <= T and sens["status"][j] == 0:
                src.append(s.p + s.R @ LEVER)
                dst.append(sens["z"][j])
        a, _ = fusion.align_4dof(np.array(src), np.array(dst))
        # the residual heading error of the aligned VIO
        k = np.argmin(np.abs(tw - T))
        Rk = lie.so3_exp(np.array([0, 0, a])) @ est_w[k][1].R
        dR = Rk @ m.R(tw[k]).T
        lengths.append(np.linalg.norm(m.p(T) - m.p(m.still)))
        yerr.append(abs(np.degrees(np.arctan2(dR[1, 0], dR[0, 0]))))
        # Proposition 14.7: fixes spread along a straight segment of length
        # L; the fixes taken at rest add no direction information
        n_mov = sum(1 for t, _ in est_w
                    if m.still < t <= T and
                    sens["status"][int(round(t * sens["rx"].rate))] == 0)
        theory.append(np.degrees(sens["rx"].sigma_fix * np.sqrt(12) /
                                 (lengths[-1] * np.sqrt(max(n_mov, 1)))))
        print(f"  alignment on {lengths[-1]:4.1f} m of driving: heading "
              f"error {yerr[-1]:.3f} deg")
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    ax[0].loglog(lengths, yerr, "o-", label="alignment of the VIO")
    ax[0].loglog(lengths, theory, "k--", lw=0.8,
                 label=r"$\sigma\sqrt{12}/(L\sqrt{n})$, Proposition 14.7")
    ax[0].legend(fontsize=7)
    ax[0].set_xlabel("distance driven before the alignment [m]")
    ax[0].set_ylabel("heading error after alignment [deg]")
    ax[1].semilogy(ta, 100 * ea, color="tab:green",
                   label="VIO started 30° and 3.6 m off, aligned at 8 s, "
                         "then + GNSS")
    t, e, _ = runs["VIO + GNSS, Cauchy"]
    ax[1].semilogy(t, 100 * e, color="0.5", lw=0.8,
                   label="VIO + GNSS started in the right frame")
    ax[1].axvline(t_al, color="k", ls=":", lw=0.8)
    ax[1].set_xlabel("time [s]")
    ax[1].set_ylabel("position error [cm]")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "ch14_align.pdf")
    print("figures written to", out)


if __name__ == "__main__":
    main()
