"""Tests for Chapter 7: PnP, refinement problems, metrics, simulator, VO."""

import numpy as np
import pytest

from vslam import lie, metrics, optim, pnp, robust, sim, vo
from vslam.camera import look_at

RNG = np.random.default_rng(31)
K = np.array([[458.7, 0, 367.2], [0, 457.3, 248.4], [0, 0, 1.0]])


def scene(n=60, planar=False, rng=RNG):
    T_wc = look_at(np.array([0.0, -5.0, 1.0]), np.array([0.0, 2.0, 0.5]))
    X = np.column_stack([rng.uniform(-3, 3, n), rng.uniform(0, 6, n),
                         rng.uniform(0, 2.5, n)])
    if planar:
        X[:, 2] = 0.0
    T_cw = lie.se3_inv(T_wc)
    q = X @ T_cw[:3, :3].T + T_cw[:3, 3]
    return T_wc, X, q[:, :2] / q[:, 2:]


def pose_err(Ta, Tb):
    return np.linalg.norm(lie.se3_log(lie.se3_inv(Ta) @ Tb))


def test_pnp_dlt_exact_and_planar_degeneracy():
    T_wc, X, x = scene()
    assert pose_err(lie.se3_inv(pnp.pnp_dlt(X, x)), T_wc) < 1e-8
    T_wc, X, x = scene(planar=True)
    rows = []
    for p, (u, v) in zip(X, x):
        ph = np.append(p, 1.0)
        rows += [np.concatenate([ph, np.zeros(4), -u * ph]),
                 np.concatenate([np.zeros(4), ph, -v * ph])]
    s = np.linalg.svd(np.array(rows), compute_uv=False)
    assert s[-4] / s[0] < 1e-10  # three extra null directions on a plane


def test_p3p_contains_the_truth():
    T_wc, X, x = scene(n=3)
    sols = pnp.p3p(X, x)
    assert 1 <= len(sols) <= 4
    assert min(pose_err(lie.se3_inv(T), T_wc) for T in sols) < 1e-6


def test_p3p_works_on_a_plane():
    T_wc, X, x = scene(n=4, planar=True)
    res = pnp.pnp_ransac(X, x, 1e-6, rng=np.random.default_rng(0))
    assert pose_err(lie.se3_inv(res.model), T_wc) < 1e-6


def test_pnp_ransac_and_refinement_with_outliers():
    rng = np.random.default_rng(2)
    T_wc, X, x = scene(n=150, rng=rng)
    z = x @ K[:2, :2].T + K[:2, 2] + rng.normal(0, 0.5, x.shape)
    out = rng.random(150) < 0.4
    z[out] = rng.uniform([0, 0], [752, 480], (out.sum(), 2))
    xn = (z - K[:2, 2]) / np.diag(K)[:2]
    res = pnp.pnp_ransac(X, xn, 1.5 / 458.7, rng=rng)
    assert res.inliers[out].mean() < 0.05
    pb = pnp.MonoPoseProblem(K, X, z, 0.5)
    sol = optim.levenberg_marquardt(pb, lie.se3_inv(res.model),
                                    kernel=robust.Cauchy(2.0))
    assert pose_err(sol.x, T_wc) < 0.01


@pytest.mark.parametrize("stereo", [False, True])
def test_pose_problem_jacobians(stereo):
    T_wc, X, x = scene(n=10)
    z = np.zeros((10, 3 if stereo else 2))
    pb = (pnp.StereoPoseProblem(K, 0.12, X, z) if stereo
          else pnp.MonoPoseProblem(K, X, z))
    r0, J = pb.linearize(T_wc)
    eps = 1e-6
    num = np.column_stack([
        (pb.linearize(T_wc @ lie.se3_exp(e))[0]
         - pb.linearize(T_wc @ lie.se3_exp(-e))[0]) / (2 * eps)
        for e in eps * np.eye(6)])
    assert np.allclose(J, num, atol=1e-4)


def test_align_and_ate():
    P = RNG.normal(size=(50, 3))
    R, t, s = lie.so3_exp(RNG.normal(size=3)), RNG.normal(size=3), 2.5
    Q = s * P @ R.T + t
    Re, te, se = metrics.align(P, Q, with_scale=True)
    assert np.allclose(Re, R) and np.allclose(te, t) and np.isclose(se, s)
    assert metrics.ate(P, P @ R.T + t).max() < 1e-10


def test_rpe_of_a_perfect_trajectory_is_zero():
    poses = [T @ sim.camera_mount() for T in sim.boustrophedon()[:200]]
    te, re = metrics.rpe(poses, poses, 1.0)
    assert len(te) > 0 and te.max() < 1e-12 and re.max() < 1e-12


def test_boustrophedon_is_continuous_and_covers_the_lawn():
    poses = sim.boustrophedon()
    p = np.array([T[:3, 3] for T in poses])
    steps = np.linalg.norm(np.diff(p, axis=0), axis=1)
    # rows from x = 1 to 11, U-turns of radius 0.75 m beyond them
    assert steps.max() < 0.08 and p[:, 0].min() >= 0.2 and \
        p[:, 0].max() <= 11.8


def test_simulator_noise_free_stereo_is_exact():
    world = sim.garden(np.random.default_rng(0))
    S = sim.GardenSim(sigma=0.0, outlier=0.0, outlier_grass=0.0)
    frames = S.run(world, sim.boustrophedon()[:3], sim.camera_mount(),
                   np.random.default_rng(0))
    f = frames[-1]
    assert len(f.z) > 50
    q = np.array([S.stereo.triangulate(m) for m in f.z])
    P = q @ f.T_wc[:3, :3].T + f.T_wc[:3, 3]
    assert np.allclose(P, world.points[f.landmark], atol=1e-8)


@pytest.mark.parametrize("sigma", [0.0, 0.5])
def test_stereo_vo_on_a_short_run(sigma):
    rng = np.random.default_rng(4)
    world = sim.garden(rng)
    S = sim.GardenSim(sigma=sigma)
    frames = S.run(world, sim.boustrophedon()[:150], sim.camera_mount(), rng)
    V = vo.StereoVO(S.stereo, sigma=max(sigma, 0.05),
                    rng=np.random.default_rng(0))
    est = V.run(frames, frames[0].T_wc)
    err = np.linalg.norm(est[-1][:3, 3] - frames[-1].T_wc[:3, 3])
    assert V.stats.failures == 0
    # exact without noise; with noise, below 5 % of the 7.5 m driven
    assert err < (1e-5 if sigma == 0 else 0.375)
