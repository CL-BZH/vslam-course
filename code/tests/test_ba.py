"""Tests for vslam.ba (Chapter 8)."""

import numpy as np
import pytest

from vslam import ba, lie, robust
from vslam.camera import look_at

K = np.array([[458.7, 0, 367.2], [0, 457.3, 248.4], [0, 0, 1.0]])


def make_problem(n_cams=5, n_pts=80, stereo=True, noise=0.0, seed=0,
                 outliers=0.0):
    """Cameras along a line looking at a cloud of points; every camera
    sees every point."""
    rng = np.random.default_rng(seed)
    cams = [look_at(np.array([x, -6.0, 1.0]), np.array([0.0, 0.0, 1.0]))
            for x in np.linspace(-1.5, 1.5, n_cams)]
    pts = rng.uniform([-3, -2, 0], [3, 2, 2.5], size=(n_pts, 3))
    ci, pi, zs = [], [], []
    for c, T in enumerate(cams):
        q = (pts - T[:3, 3]) @ T[:3, :3]
        u = q[:, 0] / q[:, 2] * K[0, 0] + K[0, 2]
        v = q[:, 1] / q[:, 2] * K[1, 1] + K[1, 2]
        ur = (q[:, 0] - 0.12) / q[:, 2] * K[0, 0] + K[0, 2]
        z = np.column_stack([u, v, ur] if stereo else [u, v])
        z = z + rng.normal(0, noise, z.shape)
        bad = rng.random(n_pts) < outliers
        z[bad] += rng.uniform(-80, 80, (bad.sum(), z.shape[1]))
        ci += [c] * n_pts
        pi += list(range(n_pts))
        zs.append(z)
    pb = ba.BAProblem(K, [T.copy() for T in cams], pts.copy(),
                      np.array(ci), np.array(pi), np.vstack(zs),
                      0.12 if stereo else None, max(noise, 1.0), {0})
    return pb, cams, pts


@pytest.mark.parametrize("stereo", [False, True])
def test_jacobians(stereo):
    pb, cams, pts = make_problem(n_cams=2, n_pts=5, stereo=stereo)
    r, A, B, _ = pb.residuals_jacobians()
    eps = 1e-6
    for k in [0, 7]:
        c, p = pb.cam_idx[k], pb.pt_idx[k]
        for i in range(6):
            d = np.zeros(6)
            d[i] = eps
            cp = [T.copy() for T in pb.cams]
            cm = [T.copy() for T in pb.cams]
            cp[c] = cp[c] @ lie.se3_exp(d)
            cm[c] = cm[c] @ lie.se3_exp(-d)
            num = (pb.residuals_jacobians(cp)[0][k]
                   - pb.residuals_jacobians(cm)[0][k]) / (2 * eps)
            assert np.allclose(A[k][:, i], num, atol=1e-5)
        for i in range(3):
            P1, P2 = pb.points.copy(), pb.points.copy()
            P1[p, i] += eps
            P2[p, i] -= eps
            num = (pb.residuals_jacobians(None, P1)[0][k]
                   - pb.residuals_jacobians(None, P2)[0][k]) / (2 * eps)
            assert np.allclose(B[k][:, i], num, atol=1e-5)


def test_schur_step_equals_dense_step():
    pb, cams, pts = make_problem(noise=0.5)
    rng = np.random.default_rng(1)
    pb.cams = [T @ lie.se3_exp(rng.normal(0, 0.01, 6)) if i else T
               for i, T in enumerate(pb.cams)]
    pb.points = pb.points + rng.normal(0, 0.05, pb.points.shape)
    r, A, B, _ = pb.residuals_jacobians()
    U, V, Wk, g_c, g_p = pb.normal_equations(r, A, B, np.ones(len(r)))
    S, b, Vinv, pos = pb.schur(U, V, Wk, g_c, g_p)
    dc, dp = pb.back_substitute(np.linalg.solve(S, b), pos, Vinv, Wk, g_p)
    H, g = ba.dense_system(pb)
    delta = np.linalg.solve(H, -g)
    nc = 6 * len(pos)
    assert np.allclose(delta[:nc], dc[1:].ravel(), atol=1e-9)
    assert np.allclose(delta[nc:], dp.ravel(), atol=1e-9)


def test_gauge_freedom():
    for stereo, expected in [(True, 6), (False, 7)]:
        pb, _, _ = make_problem(stereo=stereo)
        pb.fixed = set()
        H, _ = ba.dense_system(pb)
        ev = np.linalg.eigvalsh(H)
        assert np.sum(ev < 1e-7 * ev[-1]) == expected
        pb.fixed = {0}
        H, _ = ba.dense_system(pb)
        ev = np.linalg.eigvalsh(H)
        assert np.sum(ev < 1e-7 * ev[-1]) == expected - 6  # mono: scale


def test_ba_converges_to_truth_without_noise():
    pb, cams, pts = make_problem(stereo=True)
    rng = np.random.default_rng(2)
    pb.cams = [T @ lie.se3_exp(rng.normal(0, 0.02, 6)) if i else T
               for i, T in enumerate(pb.cams)]
    pb.points = pb.points + rng.normal(0, 0.1, pb.points.shape)
    res = ba.solve(pb, max_iters=50)
    assert res.costs[-1] < 1e-12
    for Te, Tt in zip(res.cams, cams):
        assert np.linalg.norm(lie.se3_log(lie.se3_inv(Tt) @ Te)) < 1e-7
    assert np.abs(res.points - pts).max() < 1e-7


def test_robust_ba_with_outliers():
    errs = {}
    for name, out, k in [("clean", 0.0, None), ("l2", 0.15, None),
                         ("cauchy", 0.15, robust.Cauchy(2.0))]:
        pb, cams, pts = make_problem(noise=0.5, outliers=out, seed=3)
        res = ba.solve(pb, kernel=k, max_iters=50)
        errs[name] = max(np.linalg.norm(Te[:3, 3] - Tt[:3, 3])
                         for Te, Tt in zip(res.cams, cams))
    assert errs["l2"] > 20 * errs["cauchy"]
    assert errs["cauchy"] < 2 * errs["clean"]


def test_camera_covariance_matches_monte_carlo():
    est = []
    for s in range(150):
        pb, cams, pts = make_problem(n_cams=4, n_pts=40, noise=0.5, seed=s)
        res = ba.solve(pb, max_iters=20)
        est.append(res.cams[3][:3, 3] - cams[3][:3, 3])
    pb, cams, pts = make_problem(n_cams=4, n_pts=40, noise=0.5, seed=0)
    pb.sigma = 0.5
    ba.solve(pb, max_iters=20)
    cov = ba.camera_covariance(pb)[3]
    # translational block of a right perturbation = R * (position error)
    R = pb.cams[3][:3, :3]
    pred = np.sqrt(np.diag(R @ cov[:3, :3] @ R.T))
    mc = np.array(est).std(axis=0)
    assert np.all(np.abs(mc / pred - 1) < 0.3)
