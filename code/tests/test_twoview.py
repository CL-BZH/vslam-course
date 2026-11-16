"""Tests for vslam.twoview (Chapter 6)."""

import numpy as np
import pytest

from vslam import lie, twoview as tv

RNG = np.random.default_rng(21)


def scene(n=100, rng=RNG, depth=(2.0, 8.0), planar=False, R=None, t=None):
    """Points in front of camera 1, their normalised images in both
    cameras, and the relative pose (R, t) = T_21."""
    R = lie.so3_exp(rng.normal(scale=0.1, size=3)) if R is None else R
    t = rng.normal(size=3) if t is None else t
    t = t / np.linalg.norm(t) if np.linalg.norm(t) > 0 else t
    X = np.column_stack([rng.uniform(-2, 2, n), rng.uniform(-1.5, 1.5, n),
                         rng.uniform(*depth, n)])
    if planar:  # the plane z = 5 - 0.3 y, seen obliquely
        X[:, 2] = 5.0 - 0.3 * X[:, 1]
    Y = X @ R.T + t
    return X, X[:, :2] / X[:, 2:], Y[:, :2] / Y[:, 2:], R, t


def rot_err(R1, R2):
    return np.linalg.norm(lie.so3_log(R1.T @ R2))


def test_epipolar_constraint_and_singular_values():
    X, x1, x2, R, t = scene()
    E = tv.essential_from_pose(R, t)
    assert np.abs(tv.epipolar_residual(E, x1, x2)).max() < 1e-12
    s = np.linalg.svd(E, compute_uv=False)
    assert np.isclose(s[0], s[1]) and abs(s[2]) < 1e-12
    assert np.abs(tv.demazure(E)).max() < 1e-12


def test_fundamental_matrix_in_pixels():
    X, x1, x2, R, t = scene()
    K = np.array([[500.0, 0, 320], [0, 480, 240], [0, 0, 1]])
    F = tv.fundamental_from_essential(tv.essential_from_pose(R, t), K)
    u1, u2 = tv.homog(x1) @ K.T, tv.homog(x2) @ K.T
    assert np.abs(np.einsum("ij,jk,ik->i", u2, F, u1)).max() < 1e-9
    e2 = K @ t  # image of camera 1's centre in camera 2
    assert np.allclose(e2 @ F, 0, atol=1e-9)


def test_eight_point_noise_free():
    X, x1, x2, R, t = scene()
    E = tv.eight_point(x1, x2)
    E_true = tv.essential_from_pose(R, t)
    E_true /= np.linalg.norm(E_true)
    E /= np.linalg.norm(E)
    assert min(np.abs(E - E_true).max(), np.abs(E + E_true).max()) < 1e-9


def test_decomposition_contains_truth_and_cheirality_picks_it():
    X, x1, x2, R, t = scene()
    E = tv.essential_from_pose(R, t)
    cands = tv.decompose_essential(E)
    assert any(rot_err(Rc, R) < 1e-9 and np.allclose(tc, t, atol=1e-9)
               for Rc, tc in cands)
    for Rc, tc in cands:
        assert np.allclose(np.abs(tv.essential_from_pose(Rc, tc)
                                  / np.linalg.norm(E)), np.abs(E)
                           / np.linalg.norm(E), atol=1e-9)
    rp = tv.recover_pose(E, x1, x2)
    assert rot_err(rp.R, R) < 1e-9 and np.allclose(rp.t, t, atol=1e-9)
    assert rp.in_front.all()
    assert np.allclose(rp.points, X, atol=1e-8)


def test_sampson_matches_geometric_distance_for_small_noise():
    X, x1, x2, R, t = scene(n=50)
    E = tv.essential_from_pose(R, t)
    d = 1e-3 * RNG.normal(size=(50, 2))
    s = tv.sampson(E, x1, x2 + d)
    # geometric distance of x2 + d to the epipolar line E x1~
    l = tv.homog(x1) @ E.T
    geo = np.abs(np.sum(l * tv.homog(x2 + d), 1)) / np.hypot(l[:, 0],
                                                              l[:, 1])
    # Sampson spreads the error over both images: s <= geo, same order
    assert np.all(s <= geo + 1e-15) and np.all(s > 0.5 * geo)


def test_essential_ransac_with_outliers():
    rng = np.random.default_rng(3)
    X, x1, x2, R, t = scene(n=200, rng=rng)
    x2 = x2 + rng.normal(scale=0.5 / 500, size=x2.shape)
    out = rng.random(200) < 0.3
    x2[out] = rng.uniform(-0.6, 0.6, size=(out.sum(), 2))
    res = tv.essential_ransac(x1, x2, threshold=2.0 / 500, rng=rng)
    rp = tv.recover_pose(res.model, x1[res.inliers], x2[res.inliers])
    assert rot_err(rp.R, R) < 0.01 and np.arccos(rp.t @ t) < 0.05
    assert res.inliers[out].mean() < 0.05


def test_planar_scene_is_degenerate_for_the_essential_matrix():
    rng = np.random.default_rng(4)
    X, x1, x2, R, t = scene(n=100, rng=rng, planar=True)
    h1, h2 = tv.homog(x1), tv.homog(x2)
    A = np.einsum("ni,nj->nij", h2, h1).reshape(-1, 9)
    s = np.linalg.svd(A, compute_uv=False)
    assert s[6] / s[0] < 1e-10  # rank 6: a 3-dimensional null space
    # the five-point solver returns an E that explains every
    # correspondence -- but it may be the "twin" solution, not the truth
    E5, mask = tv.five_point_ransac(x1, x2, 1e-6)
    assert mask.all() and np.abs(tv.sampson(E5, x1, x2)).max() < 1e-8


def test_homography_of_a_plane_and_its_decomposition():
    rng = np.random.default_rng(5)
    R = lie.so3_exp(np.array([0.05, -0.2, 0.03]))
    t = np.array([0.4, 0.0, 0.1])
    n, d = np.array([0.0, -0.3, 1.0]) / np.hypot(0.3, 1.0), 4.0
    X = np.column_stack([rng.uniform(-2, 2, 60), rng.uniform(-1.5, 1.5, 60),
                         np.zeros(60)])
    X[:, 2] = (d - n[0] * X[:, 0] - n[1] * X[:, 1]) / n[2]
    Y = X @ R.T + t
    x1, x2 = X[:, :2] / X[:, 2:], Y[:, :2] / Y[:, 2:]
    H = tv.homography_from_plane(R, t, n, d)
    h = tv.homog(x1) @ H.T
    assert np.allclose(h[:, :2] / h[:, 2:], x2)
    sols = tv.decompose_homography(H, x1, x2)
    assert any(rot_err(Rs, R) < 1e-8 and np.allclose(ts, t / d, atol=1e-8)
               for Rs, ts, _ in sols)


def test_pure_rotation_kills_the_essential_matrix():
    X, x1, x2, R, _ = scene(t=np.zeros(3))
    assert np.allclose(tv.essential_from_pose(R, np.zeros(3)), 0)
    # every image pair is related by the homography R (normalised coords)
    h = tv.homog(x1) @ R.T
    assert np.allclose(h[:, :2] / h[:, 2:], x2)


def test_parallax_angle():
    R, t = np.eye(3), np.array([-1.0, 0.0, 0.0])  # camera 2 at x = +1
    X = np.array([[0.5, 0.0, np.sqrt(3) / 2]])    # equidistant, 60 deg
    assert np.isclose(tv.parallax_angle(R, t, X)[0], np.pi / 3)
