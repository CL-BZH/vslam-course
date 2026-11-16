"""Tests for Chapter 11: dense scene, ground plane, grids, export."""

import numpy as np
import pytest

from vslam import lie, maps, scene, sim

RNG = np.random.default_rng(111)


def test_tls_plane_and_orientation():
    n = np.array([0.1, -0.95, 0.2])
    n /= np.linalg.norm(n)
    d = -0.8
    # points on n^T q = d plus noise
    u = np.cross(n, [1, 0, 0])
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    a, b = RNG.uniform(-3, 3, (2, 500))
    P = d * n + a[:, None] * u + b[:, None] * v + RNG.normal(0, 0.01,
                                                              (500, 3))
    pl = maps.fit_plane_tls(P)
    assert abs(abs(pl.n @ n) - 1) < 1e-3
    assert pl.height(np.zeros((1, 3)))[0] > 0      # camera above
    assert abs(pl.height(np.zeros((1, 3)))[0] - 0.8) < 0.01


def test_point_covariance_monte_carlo():
    S = sim.GardenSim().stereo
    cam = S.cam
    q = np.array([[0.8, 0.3, 4.0]])
    u = cam.fx * q[0, 0] / q[0, 2] + cam.cx
    v = cam.fy * q[0, 1] / q[0, 2] + cam.cy
    fb = cam.fx * S.baseline
    disp = fb / q[0, 2] + RNG.normal(0, 0.3, 200_000)
    Z = fb / disp
    pts = maps.stereo_points(np.tile([u, v], (len(Z), 1)), Z, S)
    C_mc = np.cov(pts.T)
    C = maps.point_covariance(q, S, 0.3)[0]
    assert np.allclose(C_mc, C, rtol=0.05, atol=1e-6)
    assert np.linalg.matrix_rank(C, tol=1e-12) == 1


def test_raycast_ground_and_fence():
    g = scene.Garden(grass_height=0.0)
    o = np.array([6.0, 4.0, 0.8])
    down = np.array([1.0, 0.0, -1.0]) / np.sqrt(2)
    t, lab = g.raycast(o, down[None])
    assert np.isclose(t[0], 0.8 * np.sqrt(2)) and lab[0] == scene.L["lawn"]
    east = np.array([[1.0, 0.0, 0.0]])
    t, lab = g.raycast(o, east)
    assert np.isclose(t[0], 13 - 0.05 - 6.0) and lab[0] == scene.L["fence"]
    up = np.array([[0.0, 0.0, 1.0]])
    t, lab = g.raycast(o, up)
    assert np.isinf(t[0]) and lab[0] == scene.L["sky"]


def test_ground_plane_with_obstacles():
    S = sim.GardenSim()
    g = scene.Garden()
    T = sim.boustrophedon()[100] @ sim.camera_mount()
    f = scene.render_stereo(g, S.stereo, T, np.random.default_rng(0))
    ok = np.isfinite(f.depth)
    q = maps.stereo_points(f.uv[ok], f.depth[ok], S.stereo)
    prior = maps.nominal_ground(sim.camera_mount())
    pl, inl = maps.ground_plane(q, prior, np.random.default_rng(1))
    up_c = T[:3, :3].T @ np.array([0, 0, 1.0])
    assert np.degrees(np.arccos(pl.n @ up_c)) < 1.0
    # camera height above the fitted plane: 0.8 m minus about half the
    # grass height (the plane sits on the grass tops)
    assert 0.8 - 0.08 < -pl.d < 0.8 + 0.01


def test_log_odds_is_bayes():
    spec = maps.GridSpec(0, 1, 0, 1, 0.5)
    grid = maps.LogOddsGrid(spec, p_hit=0.8, clamp=100)
    kf = maps.KeyframeCloud(np.array([[0.2, 0.2, 0.0]]), np.array([1]),
                            np.array([False]))
    p = 0.5
    for _ in range(3):
        grid.update(np.eye(4), kf)
        p = 0.8 * p / (0.8 * p + 0.2 * (1 - p))      # Bayes, by hand
    assert np.isclose(1 / (1 + np.exp(-grid.occ[0, 0])), p)
    assert grid.occ[1, 1] == 0.0                     # never seen


def test_coarsen_is_conservative_and_export():
    cls = np.full((4, 6), maps.MOWABLE)
    cls[0, 0] = maps.OBSTACLE
    cls[3, 5] = maps.UNKNOWN
    mow = maps.coarsen(cls, 2, min_mowable=0.9)
    assert mow.shape == (2, 3)
    assert not mow[0, 0] and not mow[1, 2] and mow[0, 1]
    text = maps.to_lawn_text(mow, start=(0, 1))
    rows = text.splitlines()
    assert rows[0] == "+++++" and rows[-1] == "+++++"
    assert rows[2] == "++SL+"         # south row (row 0) is near the bottom
    assert rows[1] == "+LL++"


def test_export_parses_in_lawnmower():
    lm = pytest.importorskip("lawnmower")
    mow = np.ones((3, 4), dtype=bool)
    mow[1, 2] = False
    lawn = lm.Lawn.from_string(maps.to_lawn_text(mow, start=(0, 0)))
    res = lm.plan(lawn, "boustrophedon")
    assert res.metrics.as_dict()["coverage_%"] == 100.0
