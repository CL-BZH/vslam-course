"""Tests for Chapter 13: sliding-window VIO, observability, MSCKF,
initialisation."""

import numpy as np
import pytest

from vslam import imu, lie, sim, vio

RNG = np.random.default_rng(13)


@pytest.fixture(scope="module")
def drive():
    """Six seconds of the garden drive, with IMU and stereo frames."""
    m = sim.GardenMotion()
    P = imu.ImuParams()
    rng = np.random.default_rng(0)
    bg0, ba0 = rng.normal(0, 0.005, 3), rng.normal(0, 0.05, 3)
    T = 6.0
    data = imu.simulate(m, T, P, rng, bg0=bg0, ba0=ba0)
    world = sim.garden(np.random.default_rng(0))
    S = sim.GardenSim()
    Tbc = sim.camera_mount()
    ts = np.arange(0, T, 0.05)
    frames = S.run(world, [m.T(t) for t in ts], Tbc,
                   np.random.default_rng(1))
    return m, P, data, S, Tbc, ts, frames


def run(drive, kf_every=5, window=8):
    m, P, data, S, Tbc, ts, frames = drive
    n0 = int(round(m.still / P.dt))
    k0 = int(round(m.still / 0.05))
    R0, bg = vio.static_init(data.gyro[:n0], data.accel[:n0])
    yaw = np.arctan2(m.R(m.still)[1, 0], m.R(m.still)[0, 0])
    R0 = lie.so3_exp(np.array([0, 0, yaw])) @ R0
    s0 = vio.KFState(R0, np.zeros(3), m.p(m.still), bg, np.zeros(3))
    V = vio.SlidingWindowVIO(S.stereo, Tbc, P, window=window)
    V.initialise(ts[k0], s0, np.r_[[1e-2] * 2, [1e-6], [1e-3] * 3,
                                   [1e-6] * 3, [1e-3] * 3, [0.1] * 3],
                 frames[k0].track_ids, frames[k0].z)
    last = n0
    for k in range(k0 + kf_every, len(frames), kf_every):
        n = int(round(ts[k] / P.dt))
        V.add_keyframe(ts[k], data.gyro[last:n], data.accel[last:n],
                       frames[k].track_ids, frames[k].z)
        last = n
    return V


def test_stereo_jacobians_numerically():
    S = sim.GardenSim().stereo
    Tbc = sim.camera_mount()
    R = lie.so3_exp(RNG.normal(0, 0.3, 3))
    p = RNG.normal(0, 1, 3)
    Rc, c = R @ Tbc[:3, :3], p + R @ Tbc[:3, 3]
    X = Rc @ np.array([0.4, -0.2, 5.0]) + c
    z = np.array([[400.0, 250.0, 390.0]])

    def res(R, p, X):
        return vio.stereo_residuals(R[None], p[None], X[None], z, Tbc, S,
                                    0.5)[0][0]

    r0, A, B, _ = vio.stereo_residuals(R[None], p[None], X[None], z, Tbc,
                                       S, 0.5)
    eps = 1e-6
    for k in range(15):
        d = np.zeros(15)
        d[k] = eps
        s = vio.boxplus(vio.KFState(R, np.zeros(3), p, np.zeros(3),
                                    np.zeros(3)), d)
        num = (res(s.R, s.p, X) - r0[0]) / eps
        assert np.allclose(num, A[0, :, k], atol=1e-4)
    for k in range(3):
        d = np.zeros(3)
        d[k] = eps
        num = (res(R, p, X + d) - r0[0]) / eps
        assert np.allclose(num, B[0, :, k], atol=1e-4)


def test_short_drive_is_accurate_and_prior_is_psd(drive):
    m = drive[0]
    V = run(drive)
    for t, s in zip(V.times, V.kfs):
        assert np.linalg.norm(s.p - m.p(t)) < 0.03
    ev = np.linalg.eigvalsh(V.prior.H)
    assert ev.min() > -1e-6 * ev.max()


def test_four_unobservable_directions(drive):
    """Without a prior, the Gauss-Newton Hessian of the window annihilates
    the translations and the rotation about gravity, but not roll."""
    V = run(drive)
    Hs, W, Vb, _, _, pid = V.normal_equations(use_prior=False)
    P = len(pid)
    n = len(Hs)
    H = np.zeros((n + 3 * P, n + 3 * P))
    H[:n, :n] = Hs
    H[:n, n:] = W
    H[n:, :n] = W.T
    for j in range(P):
        H[n + 3 * j:n + 3 * j + 3, n + 3 * j:n + 3 * j + 3] = Vb[j]
    U = vio.gauge_directions(V.kfs, V.points)
    U /= np.linalg.norm(U, axis=0)
    scale = np.linalg.norm(H, 2)
    assert np.linalg.norm(H @ U, axis=0).max() < 1e-6 * scale
    # a rotation about the world x axis (roll) is not a null direction
    ex = np.array([1.0, 0.0, 0.0])
    u = np.zeros(n + 3 * P)
    for k, s in enumerate(V.kfs):
        u[15 * k:15 * k + 3] = s.R.T @ ex
        u[15 * k + 3:15 * k + 6] = np.cross(ex, s.v)
        u[15 * k + 6:15 * k + 9] = s.R.T @ np.cross(ex, s.p)
    for j, i in enumerate(pid):
        u[n + 3 * j:n + 3 * j + 3] = np.cross(ex, V.points[i])
    u /= np.linalg.norm(u)
    assert u @ H @ u > 1e3 * np.max(np.abs(U.T @ H @ U))


def test_msckf_projection_equals_marginalisation():
    m, nx, nf = 12, 9, 3
    Hx = RNG.normal(size=(m, nx))
    Hf = RNG.normal(size=(m, nf))
    r = RNG.normal(size=m)
    Ho, ro = vio.msckf_project(Hx, Hf, r)
    assert Ho.shape == (m - nf, nx)
    # information and information vector on x after eliminating f
    H = np.block([[Hx.T @ Hx, Hx.T @ Hf], [Hf.T @ Hx, Hf.T @ Hf]])
    g = np.r_[Hx.T @ r, Hf.T @ r]
    S = H[:nx, :nx] - H[:nx, nx:] @ np.linalg.solve(H[nx:, nx:], H[nx:, :nx])
    gs = g[:nx] - H[:nx, nx:] @ np.linalg.solve(H[nx:, nx:], g[nx:])
    assert np.allclose(Ho.T @ Ho, S)
    assert np.allclose(Ho.T @ ro, gs)


def test_static_init_recovers_tilt_and_gyro_bias():
    R = imu.euler_zyx(0.0, np.radians(3.0), np.radians(-2.0))
    traj = imu.Trajectory(lambda t: (R, np.zeros(3)))
    P = imu.ImuParams()
    bg = np.array([0.01, -0.005, 0.002])
    data = imu.simulate(traj, 2.0, P, np.random.default_rng(2), bg0=bg)
    R0, bg0 = vio.static_init(data.gyro, data.accel)
    up = np.array([0.0, 0.0, 1.0])
    ang = np.degrees(np.arccos(np.clip((R0.T @ up) @ (R.T @ up), -1, 1)))
    # limited by the accelerometer bias: tilt error ~ |b_a| / g, and the
    # bias random walk reaches a few 1e-3 m/s^2 in 2 s (Exercise 12.3)
    assert ang < 0.1
    # the mean of 2 s of samples: per-axis error sigma_g / sqrt(2 s)
    sd = P.sigma_g / np.sqrt(2.0)
    assert np.all(np.abs(bg0 - data.bg.mean(axis=0)) < 4 * sd)


def test_alignment_recovers_gravity_velocity_and_gyro_bias():
    m = sim.GardenMotion(still=0.0, ramp=0.5)
    P = imu.ImuParams()
    bg = np.array([0.004, -0.003, 0.006])
    data = imu.simulate(m, 4.0, P, np.random.default_rng(3), bg0=bg,
                        noise=False)
    kf = np.arange(0, len(data.t) + 1, 25)            # every 0.125 s
    Rs = [m.R(k * P.dt) for k in kf]
    ps = [m.p(k * P.dt) for k in kf]
    pres = [imu.Preintegration.from_data(data, np.zeros(3), np.zeros(3), P,
                                         a, b) for a, b in zip(kf[:-1],
                                                               kf[1:])]
    bg_est = vio.gyro_bias_from_rotations(Rs, pres)
    assert np.linalg.norm(bg_est - bg) < 1e-4
    pres = [imu.Preintegration.from_data(data, bg_est, np.zeros(3), P, a, b)
            for a, b in zip(kf[:-1], kf[1:])]
    g, v = vio.align_gravity_velocity(Rs, ps, pres)
    assert np.linalg.norm(g - imu.GRAVITY) < 0.02
    vt = np.array([m.v(k * P.dt) for k in kf])
    assert np.abs(v - vt).max() < 0.02
