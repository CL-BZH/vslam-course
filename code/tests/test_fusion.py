"""Tests for Chapter 14: geodesy, wheels, GNSS factors, alignment."""

import numpy as np
import pytest

from vslam import fusion, imu, lie, sim, vio

RNG = np.random.default_rng(14)


def test_geodetic_round_trip_and_enu():
    lat, lon, h = np.radians(45.0), np.radians(7.0), 300.0
    x = fusion.geodetic_to_ecef(lat, lon, h)
    lat2, lon2, h2 = fusion.ecef_to_geodetic(x)
    assert abs(lat2 - lat) < 1e-11 and abs(lon2 - lon) < 1e-12
    assert abs(h2 - h) < 1e-4
    F = fusion.LocalFrame(lat, lon, h)
    # 1 m of extra height is 1 m up; a small step in latitude is north
    up = F.to_enu(lat, lon, h + 1.0)
    assert np.allclose(up, [0, 0, 1], atol=1e-9)
    dlat = 10.0 / 6.37e6
    north = F.to_enu(lat + dlat, lon, h)
    assert abs(north[0]) < 1e-6 and 9.9 < north[1] < 10.1
    assert np.allclose(F.R @ F.R.T, np.eye(3))
    p = np.array([12.3, -4.5, 0.7])
    assert np.allclose(F.to_enu(*F.to_geodetic(p)), p, atol=1e-6)


def test_dead_reckoning_covariance_against_monte_carlo():
    n, dt = 400, 0.02
    v = np.full(n, 1.0)
    w = 0.3 * np.sin(np.linspace(0, 6, n))
    sv, sw = 0.05, 0.02                            # densities
    _, P = fusion.dead_reckon_2d(v, w, dt, sigma_v=sv, sigma_w=sw)
    ref, _ = fusion.dead_reckon_2d(v, w, dt)
    rng = np.random.default_rng(1)
    end = []
    for _ in range(2000):
        vn = v + rng.normal(0, sv / np.sqrt(dt), n)
        wn = w + rng.normal(0, sw / np.sqrt(dt), n)
        x, _ = fusion.dead_reckon_2d(vn, wn, dt)
        end.append(x[-1] - ref[-1])
    C = np.cov(np.array(end).T)
    assert np.all(np.abs(np.sqrt(np.diag(C)) / np.sqrt(np.diag(P[-1])) - 1)
                  < 0.1)


def _state():
    return vio.KFState(lie.so3_exp(RNG.normal(0, 0.4, 3)),
                       RNG.normal(0, 1, 3), RNG.normal(0, 2, 3),
                       np.zeros(3), np.zeros(3))


@pytest.mark.parametrize("which", ["wheel", "gnss"])
def test_factor_jacobians(which):
    s = _state()
    if which == "wheel":
        f = fusion.WheelFactor(0.8)
    else:
        f = fusion.GnssFactor(np.array([1.0, 2.0, 0.5]),
                              np.array([0.02, 0.02, 0.04]),
                              np.array([-0.4, 0.1, 1.4]))
    r0, J = f(s)
    eps = 1e-6
    for k in range(15):
        d = np.zeros(15)
        d[k] = eps
        num = (f(vio.boxplus(s, d))[0] - r0) / eps
        assert np.allclose(num, J[:, k], atol=1e-4)


def test_align_4dof():
    src = RNG.normal(0, 5, (50, 3))
    yaw, t = 0.7, np.array([3.0, -2.0, 0.5])
    dst = src @ lie.so3_exp(np.array([0, 0, yaw])).T + t
    dst += RNG.normal(0, 0.01, dst.shape)
    y, tt = fusion.align_4dof(src, dst)
    assert abs(y - yaw) < 1e-3 and np.allclose(tt, t, atol=0.02)


def test_window_transform_leaves_cost_unchanged():
    m = sim.GardenMotion()
    P = imu.ImuParams()
    rng = np.random.default_rng(0)
    data = imu.simulate(m, 4.5, P, rng, bg0=rng.normal(0, 0.005, 3))
    world = sim.garden(np.random.default_rng(0))
    S = sim.GardenSim()
    Tbc = sim.camera_mount()
    ts = np.arange(0, 4.5, 0.05)
    frames = S.run(world, [m.T(t) for t in ts], Tbc,
                   np.random.default_rng(1))
    V = vio.SlidingWindowVIO(S.stereo, Tbc, P, window=4)
    s0 = vio.KFState(m.R(2.0), np.zeros(3), m.p(2.0), np.zeros(3),
                     np.zeros(3))
    V.initialise(2.0, s0, np.full(15, 0.1), frames[40].track_ids,
                 frames[40].z)
    last = 400
    for k in range(45, len(frames), 5):
        n = int(round(ts[k] / P.dt))
        V.add_keyframe(ts[k], data.gyro[last:n], data.accel[last:n],
                       frames[k].track_ids, frames[k].z,
                       [fusion.WheelFactor(np.linalg.norm(m.v(ts[k])))])
        last = n
    assert V.prior is not None and len(V.prior.ids) > 1
    # perturb so that the prior's gradient term is not zero
    V.kfs = [vio.boxplus(s, rng.normal(0, 0.01, 15)) for s in V.kfs]
    F0 = V.cost()
    Q = lie.so3_exp(np.array([0, 0, 0.9]))
    V.transform(Q, np.array([5.0, -3.0, 0.0]))
    assert np.isclose(V.cost(), F0, rtol=1e-9)
