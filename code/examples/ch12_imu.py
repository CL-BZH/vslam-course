"""Chapter 12 practical: the IMU and preintegration.

Produces in course/figures:

  ch12_allan.pdf   Allan deviations of a simulated stationary IMU (two
                   hours) against the noise model
  ch12_drift.pdf   dead reckoning: position error growth (Monte Carlo)
                   against the drift formulas
  ch12_preint.pdf  preintegrated measurements on the lawn trajectory:
                   covariance from the recursion against Monte Carlo
  ch12_bias.pdf    first-order bias correction against re-integration,
                   and the discretisation error against the IMU rate

Usage:
  python examples/ch12_imu.py [--outdir DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from vslam import imu  # noqa: E402
from vslam.lie import so3_log  # noqa: E402

P = imu.ImuParams()
G = 9.81


def allan(rng, hours: float = 2.0):
    """Stationary IMU, level: gyro reads bias + noise, accel reads -g
    along z plus bias and noise."""
    traj = imu.Trajectory(lambda t: (np.eye(3), np.zeros(3)))
    n = int(hours * 3600 * P.rate)
    sg, sa, sbg, sba = P.discrete()
    bg = np.cumsum(rng.normal(0, sbg, n))
    ba = np.cumsum(rng.normal(0, sba, n))
    gyro = bg + rng.normal(0, sg, n)
    acc = ba + rng.normal(0, sa, n)
    taus = np.logspace(-2, 3.3, 40)
    return (imu.allan_deviation(gyro, P.dt, taus),
            imu.allan_deviation(acc, P.dt, taus))


def drift_theory(t, sigma_a, sigma_g, dba=0.0, dbg=0.0):
    """Per horizontal axis: accel white noise, gyro white noise through
    tilt, and constant bias errors (Proposition 12.6)."""
    var = sigma_a ** 2 * t ** 3 / 3 + G ** 2 * sigma_g ** 2 * t ** 5 / 20
    return np.sqrt(var), 0.5 * dba * t ** 2, G * dbg * t ** 3 / 6


def drift_mc(rng, T=30.0, runs=200):
    """Stationary, level IMU, biases known exactly: only white noise.
    Returns times and the RMS horizontal error per axis."""
    n = int(T * P.rate)
    sg, sa, _, _ = P.discrete()
    f = np.tile(-imu.GRAVITY, (n, 1))
    err = np.zeros((runs, n + 1))
    for r in range(runs):
        data = imu.ImuData(P.dt * np.arange(n), rng.normal(0, sg, (n, 3)),
                           f + rng.normal(0, sa, (n, 3)), None, None, None,
                           None)
        s = imu.NavState(np.eye(3), np.zeros(3), np.zeros(3))
        out = imu.dead_reckon(s, data, np.zeros(3), np.zeros(3))
        err[r] = [o.p[0] for o in out]
    return P.dt * np.arange(n + 1), np.sqrt(np.mean(err ** 2, axis=0))


def preint_mc(rng, T=1.0, runs=1000):
    traj = imu.lawn_trajectory()
    clean = imu.simulate(traj, T, P, rng, t0=2.0, noise=False)
    zeros = np.zeros(3)
    n = len(clean.t)
    # the recursion, keeping the covariance at every step
    pre = imu.Preintegration(zeros, zeros, P)
    sd = [np.zeros(9)]
    for w, a in zip(clean.gyro, clean.accel):
        pre.integrate(w, a, P.dt)
        sd.append(np.sqrt(np.diag(pre.cov)))
    sd = np.array(sd)
    sg, sa, _, _ = P.discrete()
    errs = np.zeros((runs, 9))
    for r in range(runs):
        g = clean.gyro + rng.normal(0, sg, clean.gyro.shape)
        a = clean.accel + rng.normal(0, sa, clean.accel.shape)
        q = imu.Preintegration(zeros, zeros, P)
        for wk, ak in zip(g, a):
            q.integrate(wk, ak, P.dt)
        errs[r] = np.concatenate([so3_log(pre.dR.T @ q.dR), q.dv - pre.dv,
                                  q.dp - pre.dp])
    return P.dt * np.arange(n + 1), sd, errs, pre


def bias_correction(rng, T=0.5):
    """Error of the first-order bias update against re-integration, for
    gyro bias changes of growing size."""
    traj = imu.lawn_trajectory()
    data = imu.simulate(traj, T, P, rng, t0=2.0)
    z = np.zeros(3)
    pre = imu.Preintegration.from_data(data, z, z, P)
    sizes = np.radians(np.logspace(-2, 1, 13))     # 0.01 to 10 deg/s
    eR, ev, ep = [], [], []
    for s in sizes:
        d = s * np.array([1.0, -1.0, 1.0]) / np.sqrt(3)
        exact = imu.Preintegration.from_data(data, d, z, P)
        dR, dv, dp = pre.corrected(d, z)
        eR.append(np.linalg.norm(so3_log(exact.dR.T @ dR)))
        ev.append(np.linalg.norm(exact.dv - dv))
        ep.append(np.linalg.norm(exact.dp - dp))
    sd = np.sqrt(np.diag(pre.cov))
    return np.degrees(sizes), np.array(eR), np.array(ev), np.array(ep), sd


def discretisation(T=1.0):
    """Noise-free preintegration at several rates against the truth on the
    lawn trajectory: the error of the first-order scheme."""
    traj = imu.lawn_trajectory()
    rates = np.array([50, 100, 200, 400, 800, 1600])
    out = []
    for r in rates:
        p = imu.ImuParams(rate=float(r))
        data = imu.simulate(traj, T, p, None, t0=2.0, noise=False)
        pre = imu.Preintegration.from_data(data, np.zeros(3), np.zeros(3), p)
        Ri = traj.R(2.0)
        true_dv = Ri.T @ (traj.v(2.0 + T) - traj.v(2.0) - imu.GRAVITY * T)
        true_dR = Ri.T @ traj.R(2.0 + T)
        out.append((np.linalg.norm(pre.dv - true_dv),
                    np.linalg.norm(so3_log(true_dR.T @ pre.dR))))
    return rates, np.array(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outdir", type=Path,
                    default=Path(__file__).resolve().parents[2]
                    / "course" / "figures")
    out = ap.parse_args().outdir
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(12)

    # ---------------------------------------------------- Allan
    (tg, ag), (ta, aa) = allan(rng)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for a, t, s, white, walk, unit, name in [
            (ax[0], tg, np.degrees(ag) * 3600, P.sigma_g, P.sigma_bg,
             "deg/h", "gyroscope"),
            (ax[1], ta, aa, P.sigma_a, P.sigma_ba, "m/s$^2$",
             "accelerometer")]:
        k = np.degrees(1) * 3600 if unit == "deg/h" else 1.0
        a.loglog(t, s, "o", ms=3, label="simulated (2 h)")
        a.loglog(t, k * white / np.sqrt(t), "--",
                 label=r"white noise: $\sigma/\sqrt{\tau}$")
        a.loglog(t, k * walk * np.sqrt(t / 3), ":",
                 label=r"bias random walk: $\sigma_b\sqrt{\tau/3}$")
        a.loglog(t, k * np.sqrt(white ** 2 / t + walk ** 2 * t / 3), "k-",
                 lw=0.8, label="sum")
        a.set_xlabel(r"averaging time $\tau$ [s]")
        a.set_ylabel(f"Allan deviation [{unit}]")
        a.set_title(name, fontsize=9)
        a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "ch12_allan.pdf")
    i1 = np.argmin(np.abs(tg - 1.0))
    print(f"gyro Allan dev at 1 s: {ag[i1]:.3e} rad/s "
          f"(sigma_g = {P.sigma_g:.3e}); accel {aa[i1]:.3e} "
          f"(sigma_a = {P.sigma_a:.3e})")

    # ---------------------------------------------------- drift
    t, mc = drift_mc(rng)
    sd, _, _ = drift_theory(t, P.sigma_a, P.sigma_g)
    tt = np.linspace(0, 30, 200)
    _, bias_a, bias_g = drift_theory(tt, 0, 0, dba=0.02,
                                     dbg=np.radians(0.05))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(t[1:], mc[1:], color="tab:blue",
                label="white noise only: Monte Carlo (200 runs)")
    ax.semilogy(t[1:], sd[1:], "k--", lw=0.8,
                label=r"$\sqrt{\sigma_a^2t^3/3+g^2\sigma_g^2t^5/20}$")
    ax.semilogy(tt[1:], bias_a[1:], color="tab:orange",
                label=r"accel. bias error 0.02 m/s$^2$: $\frac{1}{2}\delta b_a"
                      r"\,t^2$")
    ax.semilogy(tt[1:], bias_g[1:], color="tab:red",
                label=r"gyro bias error 0.05$^\circ$/s: "
                      r"$g\,\delta b_g t^3/6$")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("horizontal position error, one axis [m]")
    ax.set_ylim(1e-4, 1e3)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "ch12_drift.pdf")
    for T in (1, 10, 30):
        k = np.argmin(np.abs(t - T))
        print(f"dead reckoning at {T:2d} s: noise MC {mc[k]:.3f} m, "
              f"theory {sd[k]:.3f} m; bias errors "
              f"{0.5 * 0.02 * T ** 2:.2f} m (accel), "
              f"{G * np.radians(0.05) * T ** 3 / 6:.2f} m (gyro)")

    # ---------------------------------------------------- preintegration
    tp, sdp, errs, pre = preint_mc(rng)
    mcsd = errs.std(axis=0)
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    for a, (blk, scale, unit, name) in zip(ax, [
            (slice(0, 3), np.degrees(1), "deg", r"$\delta\phi$"),
            (slice(3, 6), 100, "cm/s", r"$\delta v$"),
            (slice(6, 9), 1000, "mm", r"$\delta p$")]):
        for k, c in zip(range(blk.start, blk.stop),
                        ["tab:blue", "tab:orange", "tab:green"]):
            a.plot(tp, scale * sdp[:, k], color=c,
                   label=f"axis {'xyz'[k - blk.start]}: recursion")
            a.plot(tp[-1], scale * mcsd[k], "o", color=c)
        a.set_xlabel("time since keyframe i [s]")
        a.set_ylabel(f"std of {name} [{unit}]")
        a.legend(fontsize=7, title="dots: Monte Carlo at 1 s",
                 title_fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "ch12_preint.pdf")
    print("preintegration std after 1 s, recursion vs MC:")
    print("  ", np.round(sdp[-1] / mcsd, 3))

    # ---------------------------------------------------- bias + rate
    sz, eR, ev, ep, sdb = bias_correction(rng)
    rates, dis = discretisation()
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].loglog(sz, np.degrees(eR), "o-", label=r"$\Delta R$ [deg]")
    ax[0].loglog(sz, 100 * ev, "s-", label=r"$\Delta v$ [cm/s]")
    ax[0].loglog(sz, 1000 * ep, "^-", label=r"$\Delta p$ [mm]")
    ax[0].axhline(np.degrees(sdb[0]), color="tab:blue", ls=":", lw=0.8)
    ax[0].axhline(100 * sdb[3], color="tab:orange", ls=":", lw=0.8)
    ax[0].axhline(1000 * sdb[6], color="tab:green", ls=":", lw=0.8)
    ax[0].set_xlabel(r"change of gyro bias $|\delta b_g|$ [deg/s]")
    ax[0].set_ylabel("error of the first-order update")
    ax[0].set_title("bias update over 0.5 s (dotted: noise std)", fontsize=9)
    ax[0].legend(fontsize=7)
    ax[1].loglog(rates, 100 * dis[:, 0], "o-", label=r"$\Delta v$ [cm/s]")
    ax[1].loglog(rates, 100 * dis[0, 0] * rates[0] / rates, "k--", lw=0.6,
                 label=r"slope $-1$")
    ax[1].set_xlabel("IMU rate [Hz]")
    ax[1].set_ylabel(r"error of $\Delta v$ after 1 s, no noise")
    ax[1].set_title("discretisation error of the first-order scheme",
                    fontsize=9)
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "ch12_bias.pdf")
    for s, a, b in zip(sz, eR, ev):
        print(f"  dbg {s:6.3f} deg/s: dR err {np.degrees(a):.2e} deg, "
              f"dv err {100 * b:.2e} cm/s")
    print("noise std over 0.5 s: dR", np.degrees(sdb[0]), "deg, dv",
          100 * sdb[3], "cm/s")
    for r, (v, R) in zip(rates, dis):
        print(f"  {r:5d} Hz: dv err {100 * v:.3f} cm/s, "
              f"dR err {np.degrees(R) * 1e3:.3f} mdeg")
    print("figures written to", out)


if __name__ == "__main__":
    main()
