"""Chapter 13 practical: sliding-window visual-inertial odometry.

The mower drives the garden pattern (sim.GardenMotion: 2 s at rest, then
the rows and turns, 65 m in 68 s) with a 200 Hz IMU and a 20 Hz stereo
camera. Produces in course/figures:

  ch13_traj.pdf   trajectories and position errors: stereo VO (Chapter 7),
                  VIO, and VIO with the camera blind for 4 s in a turn
  ch13_obs.pdf    the four unobservable directions (Hessian spectra), and
                  tilt and yaw errors along the drive
  ch13_td.pdf     error against an uncalibrated camera-IMU time offset

Usage:
  python examples/ch13_vio.py [--outdir DIR]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from vslam import imu, lie, sim, vio, vo  # noqa: E402

CAM_RATE, KF_EVERY = 20.0, 5          # keyframes at 4 Hz
EZ = np.array([0.0, 0.0, 1.0])


def setup(seed: int = 0, td: float = 0.0):
    """Motion, IMU data and stereo frames. Frame k is stamped ts[k] but
    taken at ts[k] + td (an uncalibrated time offset)."""
    m = sim.GardenMotion()
    T = m.duration - 0.05
    P = imu.ImuParams()
    rng = np.random.default_rng(seed)
    bg0, ba0 = rng.normal(0, 0.005, 3), rng.normal(0, 0.05, 3)
    data = imu.simulate(m, T, P, rng, bg0=bg0, ba0=ba0)
    world = sim.garden(np.random.default_rng(0))
    S = sim.GardenSim()
    Tbc = sim.camera_mount()
    ts = np.arange(0, T - abs(td), 1 / CAM_RATE)
    frames = S.run(world, [m.T(max(t + td, 0.0)) for t in ts], Tbc,
                   np.random.default_rng(seed + 1))
    return m, P, data, S, Tbc, ts, frames


def run_vio(m, P, data, S, Tbc, ts, frames, blind=None, use_imu=True,
            window=8):
    """Static initialisation on the first 2 s, then a keyframe every
    KF_EVERY frames. `blind` = (t0, t1): no visual measurements."""
    k0 = int(round(m.still * CAM_RATE))
    n0 = int(round(m.still / P.dt))
    R0, bg = vio.static_init(data.gyro[:n0], data.accel[:n0])
    yaw = np.arctan2(m.R(m.still)[1, 0], m.R(m.still)[0, 0])   # gauge
    R0 = lie.so3_exp(np.array([0.0, 0.0, yaw])) @ R0
    s0 = vio.KFState(R0, np.zeros(3), m.p(m.still), bg, np.zeros(3))
    V = vio.SlidingWindowVIO(S.stereo, Tbc, P, window=window,
                             use_imu=use_imu)
    sig = np.r_[[1e-2] * 2, [1e-6], [1e-3] * 3, [1e-6] * 3, [1e-3] * 3,
                [0.1] * 3]
    V.initialise(ts[k0], s0, sig, frames[k0].track_ids, frames[k0].z)
    out = [(ts[k0], s0.copy())]
    last = n0
    tic = time.time()
    for k in range(k0 + KF_EVERY, len(frames), KF_EVERY):
        n = int(round(ts[k] / P.dt))
        f = frames[k]
        ids, z = f.track_ids, f.z
        if blind and blind[0] <= ts[k] < blind[1]:
            ids, z = ids[:0], z[:0]
        s = V.add_keyframe(ts[k], data.gyro[last:n], data.accel[last:n],
                           ids, z)
        last = n
        out.append((ts[k], s.copy()))
    V.runtime = time.time() - tic
    return V, out


def errors(m, est):
    """Position error, tilt error (deg) and yaw error (deg) along est =
    [(t, R, p)]."""
    e, tilt, yaw = [], [], []
    for t, R, p in est:
        Rt = m.R(t)
        e.append(np.linalg.norm(p - m.p(t)))
        tilt.append(np.degrees(np.arccos(np.clip((R.T @ EZ) @ (Rt.T @ EZ),
                                                 -1, 1))))
        dR = R @ Rt.T                       # rotation error, world frame
        yaw.append(np.degrees(np.arctan2(dR[1, 0], dR[0, 0])))
    return np.array(e), np.array(tilt), np.array(yaw)


def stereo_vo(m, S, Tbc, ts, frames):
    """The keyframe stereo VO of Chapter 7, started from the true pose at
    the end of the still phase; body poses at the VIO keyframe times."""
    k0 = int(round(m.still * CAM_RATE))
    V = vo.StereoVO(S.stereo, 0.5, rng=np.random.default_rng(1))
    est = V.run(frames[k0:], m.T(m.still) @ Tbc)
    Tcb = lie.se3_inv(Tbc)
    out = []
    for k in range(0, len(est), KF_EVERY):
        T = est[k] @ Tcb
        out.append((ts[k0 + k], T[:3, :3], T[:3, 3]))
    return out


def window_hessian(V, use_imu=True):
    """Full Gauss-Newton Hessian of the current window, without prior."""
    V.use_imu = use_imu
    Hs, W, Vb, _, _, pid = V.normal_equations(use_prior=False)
    V.use_imu = True
    n, P = len(Hs), len(pid)
    H = np.zeros((n + 3 * P, n + 3 * P))
    H[:n, :n], H[:n, n:], H[n:, :n] = Hs, W, W.T
    for j in range(P):
        H[n + 3 * j:n + 3 * j + 3, n + 3 * j:n + 3 * j + 3] = Vb[j]
    # eliminate the points (Schur complement)
    Vi = np.linalg.inv(H[n:, n:] + 1e-12 * np.eye(3 * P))
    return Hs - W @ Vi @ W.T


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outdir", type=Path,
                    default=Path(__file__).resolve().parents[2]
                    / "course" / "figures")
    out = ap.parse_args().outdir
    out.mkdir(parents=True, exist_ok=True)

    m, P, data, S, Tbc, ts, frames = setup()
    V, est = run_vio(m, P, data, S, Tbc, ts, frames)
    print(f"VIO: {len(est)} keyframes in {V.runtime:.0f} s, "
          f"{np.mean(V.stats.n_points):.0f} points and "
          f"{np.mean(V.stats.n_obs):.0f} observations per window")
    blind = (12.5, 16.5)                       # the first turn
    blind2 = (26.4, 30.4)                      # the second turn
    Vb, est_b = run_vio(m, P, data, S, Tbc, ts, frames, blind=blind)
    Vb2, est_b2 = run_vio(m, P, data, S, Tbc, ts, frames, blind=blind2)
    vo_est = stereo_vo(m, S, Tbc, ts, frames)

    tv = np.array([t for t, _ in est])
    ev, tilt_v, yaw_v = errors(m, [(t, s.R, s.p) for t, s in est])
    eb, _, _ = errors(m, [(t, s.R, s.p) for t, s in est_b])
    eb2, _, _ = errors(m, [(t, s.R, s.p) for t, s in est_b2])
    to = np.array([t for t, _, _ in vo_est])
    eo, tilt_o, yaw_o = errors(m, vo_est)
    for name, e in [("stereo VO", eo), ("VIO", ev), ("VIO, blind 1", eb),
                    ("VIO, blind 2", eb2)]:
        print(f"{name:15s} RMS {100 * np.sqrt(np.mean(e ** 2)):5.1f} cm, "
              f"final {100 * e[-1]:5.1f} cm, max {100 * e.max():5.1f} cm")
    kb = (tv >= blind[0]) & (tv < blind[1] + 0.3)
    kb2 = (tv >= blind2[0]) & (tv < blind2[1] + 0.3)
    print(f"largest error during blackout 1: {100 * eb[kb].max():.1f} cm, "
          f"blackout 2: {100 * eb2[kb2].max():.1f} cm")
    bga = np.array([[np.linalg.norm(s.bg - data.bg[int(round(t / P.dt))]),
                     np.linalg.norm(s.ba - data.ba[int(round(t / P.dt))])]
                    for t, s in est])
    for T in (12.0, 17.0, 40.0, 65.0):
        k = np.argmin(np.abs(tv - T))
        print(f"  t = {T:4.1f} s: gyro bias error "
              f"{np.degrees(bga[k, 0]):.4f} deg/s, accel bias error "
              f"{bga[k, 1]:.4f} m/s^2")

    # ---------------------------------------------------- trajectories
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4),
                           gridspec_kw={"width_ratios": [1.1, 1]})
    tt = np.linspace(0, m.duration - 0.05, 2000)
    gt = np.array([m.p(t) for t in tt])
    ax[0].plot(gt[:, 0], gt[:, 1], "k", lw=2, label="truth")
    po = np.array([p for _, _, p in vo_est])
    pv = np.array([s.p for _, s in est])
    pb = np.array([s.p for _, s in est_b])
    ax[0].plot(po[:, 0], po[:, 1], color="tab:orange", lw=1,
               label="stereo VO")
    ax[0].plot(pv[:, 0], pv[:, 1], color="tab:blue", lw=1, label="VIO")
    pb2 = np.array([s.p for _, s in est_b2])
    ax[0].plot(pb[kb, 0], pb[kb, 1], ".", color="tab:red", ms=4,
               label="VIO, camera blind (turn 1)")
    ax[0].plot(pb2[kb2, 0], pb2[kb2, 1], ".", color="tab:purple", ms=4,
               label="VIO, camera blind (turn 2)")
    ax[0].set_ylim(-1.5, 8)
    ax[0].set_aspect("equal")
    ax[0].set_xlabel("x [m]")
    ax[0].set_ylabel("y [m]")
    ax[0].legend(fontsize=6.5, loc="lower center", ncol=3)
    ax[1].plot(to, 100 * eo, color="tab:orange", label="stereo VO")
    ax[1].plot(tv, 100 * ev, color="tab:blue", label="VIO")
    ax[1].plot(tv, 100 * eb, color="tab:red", lw=0.8,
               label="VIO, camera blind 12.5-16.5 s")
    ax[1].plot(tv, 100 * eb2, color="tab:purple", lw=0.8,
               label="VIO, camera blind 26.4-30.4 s")
    ax[1].axvspan(*blind, color="0.9")
    ax[1].axvspan(*blind2, color="0.9")
    ax[1].set_xlabel("time [s]")
    ax[1].set_ylabel("position error [cm]")
    ax[1].set_yscale("log")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "ch13_traj.pdf")

    # ---------------------------------------------------- observability
    m2, P2, d2, S2, T2, ts2, fr2 = setup(seed=4)
    spectra = {}
    for name, T_end in [("straight row (window ends at 9 s)", 9.0),
                        ("in a turn (window ends at 15 s)", 15.0)]:
        short = fr2[:int(T_end * CAM_RATE)]
        W, _ = run_vio(m2, P2, d2, S2, T2, ts2[:len(short)], short)
        ev = np.sort(np.linalg.eigvalsh(window_hessian(W, use_imu=True)))
        spectra[name] = ev / ev[-1]
        print(f"smallest eigenvalues, VI, {name}:", spectra[name][:7])
        if T_end == 9.0:                  # stereo only, same window
            H_v = window_hessian(W, use_imu=False)
            pose = np.concatenate([np.r_[15 * k:15 * k + 3,
                                         15 * k + 6:15 * k + 9]
                                   for k in range(len(W.kfs))])
    ev_v = np.sort(np.linalg.eigvalsh(H_v[np.ix_(pose, pose)]))
    ev_v /= ev_v[-1]
    print("smallest eigenvalues, stereo poses only:", ev_v[:8])
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    ax[0].semilogy(np.arange(1, 13), np.abs(ev_v[:12]) + 1e-18, "s-",
                   color="tab:orange", label="stereo only (poses)")
    for (name, ev), c in zip(spectra.items(), ["tab:cyan", "tab:blue"]):
        ax[0].semilogy(np.arange(1, 13), np.abs(ev[:12]) + 1e-18, "o-",
                       color=c, label=f"visual-inertial, {name}")
    ax[0].set_xlabel("eigenvalue rank (smallest first)")
    ax[0].set_ylabel("eigenvalue / largest")
    ax[0].set_title("window Hessian without prior", fontsize=9)
    ax[0].legend(fontsize=6.5)
    ax[1].plot(to, tilt_o, color="tab:orange", ls="--", label="VO tilt")
    ax[1].plot(to, np.abs(yaw_o), color="tab:orange", label="VO |yaw|")
    ax[1].plot(tv, tilt_v, color="tab:blue", ls="--", label="VIO tilt")
    ax[1].plot(tv, np.abs(yaw_v), color="tab:blue", label="VIO |yaw|")
    ax[1].set_xlabel("time [s]")
    ax[1].set_ylabel("orientation error [deg]")
    ax[1].legend(fontsize=7)
    ax[2].plot(tv, np.degrees(bga[:, 0]) * 100, label=r"gyro [$0.01^\circ$/s]")
    ax[2].plot(tv, 100 * bga[:, 1], label=r"accel. [cm/s$^2$]")
    for k, (a, b) in enumerate(zip([12.5, 26.4, 40.2, 54.1],
                                   [16.4, 30.2, 44.1, 57.9])):
        ax[2].axvspan(a, b, color="0.92", label="turns" if k == 0 else None)
    ax[2].set_xlabel("time [s]")
    ax[2].set_ylabel("VIO bias error")
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "ch13_obs.pdf")
    print(f"tilt error: VIO max {tilt_v.max():.2f} deg, final "
          f"{tilt_v[-1]:.2f}; VO max {tilt_o.max():.2f}, final "
          f"{tilt_o[-1]:.2f}")
    print(f"yaw error: VIO final {yaw_v[-1]:.2f} deg; VO final "
          f"{yaw_o[-1]:.2f} deg")

    # ---------------------------------------------------- dynamic init
    # VO started from the identity: its frame is the first camera frame,
    # in which gravity is unknown; align on 2 s of motion after the ramp
    k1 = int(round((m.still + m.ramp) * CAM_RATE))
    V0 = vo.StereoVO(S.stereo, 0.5, rng=np.random.default_rng(1))
    est0 = V0.run(frames[k1:k1 + int(2 * CAM_RATE) + 1], np.eye(4))
    Tcb = lie.se3_inv(Tbc)
    idx = list(range(0, len(est0), 2))                    # 10 Hz
    Rs = [(est0[i] @ Tcb)[:3, :3] for i in idx]
    ps = [(est0[i] @ Tcb)[:3, 3] for i in idx]
    n_of = [int(round(ts[k1 + i] / P.dt)) for i in idx]
    pres = [imu.Preintegration.from_data(data, np.zeros(3), np.zeros(3),
                                         P, a, b)
            for a, b in zip(n_of[:-1], n_of[1:])]
    bg = vio.gyro_bias_from_rotations(Rs, pres)
    pres = [imu.Preintegration.from_data(data, bg, np.zeros(3), P, a, b)
            for a, b in zip(n_of[:-1], n_of[1:])]
    g, vel = vio.align_gravity_velocity(Rs, ps, pres)
    R_vw = (m.T(ts[k1]) @ Tbc)[:3, :3].T                 # world -> VO
    g_true = R_vw @ imu.GRAVITY
    ang = np.degrees(np.arccos(g @ g_true / np.linalg.norm(g) / 9.81))
    k_mid = int(round(np.mean(n_of)))
    print(f"dynamic alignment on 2 s: gravity direction error {ang:.2f} "
          f"deg, |g| = {np.linalg.norm(g):.3f}; gyro bias error "
          f"{np.degrees(np.linalg.norm(bg - data.bg[k_mid])):.4f} deg/s "
          f"(true bias {np.degrees(np.linalg.norm(data.bg[k_mid])):.3f})")

    # ---------------------------------------------------- time offset
    tds = [0.0, 0.005, 0.01, 0.02, 0.04]
    rms = []
    for td in tds:
        mm, PP, dd, SS, TT, tts, ff = setup(td=td)
        _, e_td = run_vio(mm, PP, dd, SS, TT, tts, ff)
        e, _, y = errors(mm, [(t, s.R, s.p) for t, s in e_td])
        rms.append((np.sqrt(np.mean(e ** 2)), e[-1], np.abs(y).max()))
        print(f"time offset {1000 * td:4.0f} ms: RMS {100 * rms[-1][0]:.1f} "
              f"cm, final {100 * rms[-1][1]:.1f} cm, max |yaw| "
              f"{rms[-1][2]:.2f} deg")
    rms = np.array(rms)
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.plot(1000 * np.array(tds), 100 * rms[:, 0], "o-",
            label="RMS position error [cm]")
    ax.plot(1000 * np.array(tds), 100 * rms[:, 1], "s-",
            label="final position error [cm]")
    ax.set_xlabel("camera-IMU time offset, not modelled [ms]")
    ax.set_ylabel("cm")
    ax2 = ax.twinx()
    ax2.plot(1000 * np.array(tds), rms[:, 2], "^-", color="tab:red",
             label="max |yaw error| [deg]")
    ax2.set_ylabel("deg")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "ch13_td.pdf")
    print("figures written to", out)


if __name__ == "__main__":
    main()
