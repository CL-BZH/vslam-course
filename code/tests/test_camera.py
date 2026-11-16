"""Tests for vslam.camera and vslam.calib (Chapter 3)."""

import numpy as np
import pytest

from vslam import calib, lie
from vslam.camera import PinholeCamera, StereoCamera, look_at, perspective

RNG = np.random.default_rng(3)
DIST = np.array([-0.28, 0.07, 1e-3, -5e-4, -0.008])


def cam(dist=DIST):
    return PinholeCamera(458.0, 457.0, 367.0, 248.0, dist, 752, 480)


def num_jac(f, x, eps=1e-6):
    return np.column_stack([(f(x + e) - f(x - e)) / (2 * eps)
                            for e in eps * np.eye(len(x))])


def test_perspective_jacobian():
    q = np.array([0.3, -0.2, 2.5])
    _, J = perspective(q)
    assert np.allclose(J, num_jac(lambda v: perspective(v)[0], q), atol=1e-8)


def test_distortion_jacobian_and_inverse():
    c = cam()
    for _ in range(20):
        x = RNG.uniform(-0.6, 0.6, size=2)
        assert np.allclose(c.distort_jacobian(x),
                           num_jac(c.distort, x), atol=1e-7)
        assert np.allclose(c.undistort(c.distort(x)), x, atol=1e-10)


def test_project_matches_projection_matrix():
    """Without distortion, u ~ K [R|t] p (Section 3.1)."""
    c = cam(np.zeros(5))
    T_wc = look_at(np.array([1.0, -2.0, 1.5]), np.zeros(3))
    T_cw = lie.se3_inv(T_wc)
    P = c.K @ T_cw[:3, :]
    for p in RNG.normal(scale=0.5, size=(10, 3)):
        h = P @ np.append(p, 1.0)
        u, _, _ = c.project_world(T_wc, p)
        assert np.allclose(u, h[:2] / h[2])


def test_project_world_jacobians():
    c = cam()
    T_wc = look_at(np.array([0.5, -3.0, 1.2]), np.array([0.0, 0.0, 0.3]))
    p = np.array([0.2, 0.1, 0.4])
    _, J_xi, J_p = c.project_world(T_wc, p)
    f_xi = lambda d: c.project_world(T_wc @ lie.se3_exp(d), p)[0]
    f_p = lambda v: c.project_world(T_wc, v)[0]
    assert np.allclose(J_xi, num_jac(f_xi, np.zeros(6)), atol=1e-5)
    assert np.allclose(J_p, num_jac(f_p, p), atol=1e-5)


def test_unproject_is_inverse_of_project():
    c = cam()
    q = np.array([0.4, -0.3, 1.7])
    b = c.unproject(c.project(q))
    assert np.allclose(b * q[2], q, atol=1e-9)


def test_look_at_points_axis_at_target():
    eye, target = np.array([2.0, 1.0, 1.0]), np.array([0.0, 0.5, 0.0])
    T = look_at(eye, target)
    q = lie.se3_act(lie.se3_inv(T), target)
    assert np.allclose(q[:2], 0.0) and q[2] > 0
    assert np.isclose(np.linalg.det(T[:3, :3]), 1.0)


def test_stereo():
    st = StereoCamera(cam(np.zeros(5)), baseline=0.12)
    q = np.array([0.5, -0.2, 4.0])
    z, J = st.project(q)
    assert np.allclose(st.triangulate(z), q)
    assert np.allclose(J, num_jac(lambda v: st.project(v)[0], q), atol=1e-6)
    # first-order depth std against the exact derivative dZ/dd
    d = z[0] - z[2]
    assert np.isclose(st.depth_std(4.0, 1.0), st.cam.fx * 0.12 / d**2)


# ------------------------------------------------------------- calibration
def test_homography_dlt_exact():
    H = np.array([[1.2, 0.1, 30.0], [-0.05, 0.9, -12.0], [1e-3, 2e-4, 1.0]])
    src = RNG.uniform(-50, 50, size=(12, 2))
    h = np.column_stack([src, np.ones(12)]) @ H.T
    dst = h[:, :2] / h[:, 2:]
    for norm in (True, False):
        assert np.allclose(calib.homography_dlt(src, dst, norm), H, atol=1e-8)


@pytest.mark.parametrize("zero_skew,n", [(False, 3), (True, 2), (False, 8)])
def test_zhang_recovers_K_and_poses(zero_skew, n):
    c = cam(np.zeros(5))
    board, views = calib.synthetic_views(c, n, np.random.default_rng(7))
    Hs = [calib.homography_dlt(board[:, :2], u) for _, u in views]
    K = calib.zhang_intrinsics(Hs, zero_skew=zero_skew)
    assert np.allclose(K, c.K, atol=1e-5)
    for (T_true, _), H in zip(views, Hs):
        assert np.allclose(calib.zhang_extrinsics(K, H), T_true, atol=1e-7)


def test_nearest_rotation():
    R = lie.so3_exp(RNG.normal(size=3))
    M = R + 1e-3 * RNG.normal(size=(3, 3))
    Rn = calib.nearest_rotation(M)
    assert np.allclose(Rn.T @ Rn, np.eye(3)) and np.isclose(
        np.linalg.det(Rn), 1.0)
    # Rn is the closest rotation to M, so in particular closer than R
    assert np.linalg.norm(Rn - M) <= np.linalg.norm(R - M) + 1e-15
