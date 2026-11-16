"""Tests for Chapter 12: IMU model, Allan deviation, preintegration."""

import numpy as np
import pytest

from vslam import imu
from vslam.lie import so3_exp, so3_log

RNG = np.random.default_rng(12)
P = imu.ImuParams()


def state_at(traj, t):
    return imu.NavState(traj.R(t), traj.v(t), traj.p(t))


def test_noise_free_integration_follows_the_trajectory():
    traj = imu.wavy_trajectory()
    data = imu.simulate(traj, 5.0, P, RNG, noise=False)
    out = imu.dead_reckon(state_at(traj, 0.0), data, np.zeros(3),
                          np.zeros(3))
    end = state_at(traj, 5.0)
    # rotation: exact up to the sampling of w; velocity and position: the
    # scheme is first order, error ~ dt/2 x (change of acceleration)
    assert np.linalg.norm(so3_log(out[-1].R.T @ end.R)) < 1e-4
    assert np.linalg.norm(out[-1].v - end.v) < 0.02
    assert np.linalg.norm(out[-1].p - end.p) < 0.05


def test_preintegration_predicts_like_direct_integration():
    traj = imu.wavy_trajectory()
    bg, ba = np.array([0.01, -0.02, 0.005]), np.array([0.1, 0.05, -0.08])
    data = imu.simulate(traj, 1.0, P, RNG, bg0=bg, ba0=ba)
    s0 = state_at(traj, 0.0)
    direct = imu.dead_reckon(s0, data, bg, ba)[-1]
    pre = imu.Preintegration.from_data(data, bg, ba, P)
    pred = pre.predict(s0)
    assert np.allclose(pred.R, direct.R, atol=1e-12)
    assert np.allclose(pred.v, direct.v, atol=1e-10)
    assert np.allclose(pred.p, direct.p, atol=1e-10)


def test_preintegration_covariance_against_monte_carlo():
    traj = imu.wavy_trajectory()
    clean = imu.simulate(traj, 0.5, P, RNG, noise=False)
    ref = imu.Preintegration.from_data(clean, np.zeros(3), np.zeros(3), P)
    sg, sa, _, _ = P.discrete()
    errs = []
    rng = np.random.default_rng(3)
    for _ in range(3000):
        noisy = imu.ImuData(clean.t, clean.gyro + rng.normal(0, sg,
                                                             clean.gyro.shape),
                            clean.accel + rng.normal(0, sa,
                                                     clean.accel.shape),
                            clean.bg, clean.ba, clean.w_true, clean.f_true)
        pre = imu.Preintegration.from_data(noisy, np.zeros(3), np.zeros(3),
                                           P)
        errs.append(np.concatenate([so3_log(ref.dR.T @ pre.dR),
                                    pre.dv - ref.dv, pre.dp - ref.dp]))
    C = np.cov(np.array(errs).T)
    d_mc, d_th = np.sqrt(np.diag(C)), np.sqrt(np.diag(ref.cov))
    assert np.all(np.abs(d_mc / d_th - 1) < 0.08)
    # correlation between attitude and velocity errors is captured too
    corr_mc = C[0, 4] / (d_mc[0] * d_mc[4])
    corr_th = ref.cov[0, 4] / (d_th[0] * d_th[4])
    assert abs(corr_mc - corr_th) < 0.08


def test_bias_jacobians_against_reintegration():
    traj = imu.wavy_trajectory()
    data = imu.simulate(traj, 1.0, P, RNG)
    bg, ba = np.zeros(3), np.zeros(3)
    pre = imu.Preintegration.from_data(data, bg, ba, P)
    eps = 1e-6
    for k in range(3):
        d = np.zeros(3)
        d[k] = eps
        g = imu.Preintegration.from_data(data, bg + d, ba, P)
        a = imu.Preintegration.from_data(data, bg, ba + d, P)
        assert np.allclose(so3_log(pre.dR.T @ g.dR) / eps, pre.J_R_bg[:, k],
                           atol=1e-5)
        assert np.allclose((g.dv - pre.dv) / eps, pre.J_v_bg[:, k], atol=1e-5)
        assert np.allclose((g.dp - pre.dp) / eps, pre.J_p_bg[:, k], atol=1e-5)
        assert np.allclose((a.dv - pre.dv) / eps, pre.J_v_ba[:, k], atol=1e-5)
        assert np.allclose((a.dp - pre.dp) / eps, pre.J_p_ba[:, k], atol=1e-5)


def _perturb(s, d):
    return imu.NavState(s.R @ so3_exp(d[0:3]), s.v + d[3:6],
                        s.p + s.R @ d[6:9])


def test_factor_jacobians_numerically():
    traj = imu.wavy_trajectory()
    data = imu.simulate(traj, 0.8, P, RNG)
    pre = imu.Preintegration.from_data(data, np.zeros(3), np.zeros(3), P)
    si = state_at(traj, 0.0)
    sj = state_at(traj, 0.8)
    # move away from the optimum so that every term is exercised
    si = _perturb(si, RNG.normal(0, 0.05, 9))
    sj = _perturb(sj, RNG.normal(0, 0.05, 9))
    bg, ba = np.array([0.02, -0.01, 0.03]), np.array([0.05, 0.1, -0.05])
    Ji, Jj, Jb = pre.jacobians(si, sj, bg, ba)
    r0 = pre.residual(si, sj, bg, ba)
    eps = 1e-6
    for k in range(9):
        d = np.zeros(9)
        d[k] = eps
        num_i = (pre.residual(_perturb(si, d), sj, bg, ba) - r0) / eps
        num_j = (pre.residual(si, _perturb(sj, d), bg, ba) - r0) / eps
        assert np.allclose(num_i, Ji[:, k], atol=1e-4)
        assert np.allclose(num_j, Jj[:, k], atol=1e-4)
    for k in range(6):
        d = np.zeros(6)
        d[k] = eps
        num = (pre.residual(si, sj, bg + d[:3], ba + d[3:]) - r0) / eps
        assert np.allclose(num, Jb[:, k], atol=1e-4)


def test_residual_is_zero_at_truth_without_noise():
    traj = imu.wavy_trajectory()
    bg = np.array([0.01, 0.0, -0.01])
    data = imu.simulate(traj, 1.0, P, RNG, bg0=bg, noise=False)
    pre = imu.Preintegration.from_data(data, bg, np.zeros(3), P)
    r = pre.residual(state_at(traj, 0.0), state_at(traj, 1.0), bg,
                     np.zeros(3))
    assert np.linalg.norm(r[:3]) < 1e-5          # rotation: exact
    assert np.linalg.norm(r[3:]) < 1e-2          # first-order scheme


@pytest.mark.parametrize("kind", ["white", "walk"])
def test_allan_deviation_slopes(kind):
    dt, n, s = 0.01, 400_000, 0.02
    rng = np.random.default_rng(7)
    if kind == "white":
        x = rng.normal(0, s / np.sqrt(dt), n)       # density s
        taus = np.array([0.1, 1.0, 10.0])
        t, a = imu.allan_deviation(x, dt, taus)
        assert np.allclose(a, s / np.sqrt(t), rtol=0.05)
    else:
        x = np.cumsum(rng.normal(0, s * np.sqrt(dt), n))   # density s
        taus = np.array([10.0, 30.0, 100.0])
        t, a = imu.allan_deviation(x, dt, taus)
        assert np.allclose(a, s * np.sqrt(t / 3), rtol=0.2)
