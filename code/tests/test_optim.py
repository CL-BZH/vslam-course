"""Tests for vslam.optim, vslam.robust and calibration refinement
(Chapter 4)."""

import numpy as np
import pytest

from vslam import calib, lie, optim, robust
from vslam.camera import PinholeCamera, look_at

RNG = np.random.default_rng(11)


class Linear:
    block_size = 1

    def __init__(self, A, b):
        self.A, self.b = A, b

    def linearize(self, x):
        return self.A @ x - self.b, self.A

    def retract(self, x, d):
        return x + d


class MNT:
    """r(x) = (x + 1, lam x^2 + x - 1): Gauss-Newton converges linearly
    with rate |lam| near the minimiser x = 0 (Example 4.4)."""

    block_size = 1

    def __init__(self, lam):
        self.lam = lam

    def linearize(self, x):
        x0 = x[0]
        return (np.array([x0 + 1.0, self.lam * x0 * x0 + x0 - 1.0]),
                np.array([[1.0], [2.0 * self.lam * x0 + 1.0]]))

    def retract(self, x, d):
        return x + d


class Rotation:
    """Find R with R a_i ~ b_i (Wahba's problem), right perturbation."""

    block_size = 3

    def __init__(self, a, b):
        self.a, self.b = a, b

    def linearize(self, R):
        r = (self.a @ R.T - self.b).ravel()
        J = np.vstack([-R @ lie.hat(ai) for ai in self.a])
        return r, J

    def retract(self, R, d):
        return R @ lie.so3_exp(d)


class Pose:
    """Motion-only reprojection problem: camera pose from known points."""

    block_size = 2

    def __init__(self, cam, P, z):
        self.cam, self.P, self.z = cam, P, z

    def linearize(self, T):
        r, J = [], []
        for p, zj in zip(self.P, self.z):
            u, J_xi, _ = self.cam.project_world(T, p)
            r.append(zj - u)
            J.append(-J_xi)
        return np.concatenate(r), np.vstack(J)

    def retract(self, T, d):
        return T @ lie.se3_exp(d)


def test_linear_problem_one_step():
    A, b = RNG.normal(size=(30, 4)), RNG.normal(size=30)
    res = optim.gauss_newton(Linear(A, b), np.zeros(4))
    x_ls = np.linalg.lstsq(A, b, rcond=None)[0]
    assert np.allclose(res.history[1]["x"] if len(res.history) > 1
                       else res.x, x_ls)
    assert np.allclose(res.x, x_ls)


def test_gauss_newton_linear_rate():
    lam = 0.5
    res = optim.gauss_newton(MNT(lam), np.array([0.01]), max_iters=8)
    xs = np.array([h["x"][0] for h in res.history])
    ratios = xs[2:6] / xs[1:5]
    assert np.allclose(ratios, lam, atol=0.02)


def test_lm_succeeds_where_gn_fails():
    gn = optim.gauss_newton(MNT(-2.0), np.array([0.05]), max_iters=40)
    lm = optim.levenberg_marquardt(MNT(-2.0), np.array([0.05]),
                                   marquardt=False)
    assert not gn.converged and abs(gn.x[0]) > 1e-3
    assert lm.converged and abs(lm.x[0]) < 1e-4


def test_lm_cost_is_monotone():
    lm = optim.levenberg_marquardt(MNT(0.9), np.array([0.5]))
    c = lm.costs
    assert np.all(np.diff(c) <= 1e-15)


def test_rotation_on_manifold_matches_kabsch():
    R_true = lie.so3_exp(RNG.normal(size=3))
    a = RNG.normal(size=(20, 3))
    b = a @ R_true.T + 0.01 * RNG.normal(size=(20, 3))
    res = optim.levenberg_marquardt(Rotation(a, b),
                                    lie.so3_exp(RNG.normal(size=3)))
    U, _, Vt = np.linalg.svd(b.T @ a)
    R_kabsch = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
    assert np.allclose(res.x, R_kabsch, atol=1e-8)


@pytest.mark.parametrize("kernel", [robust.Huber(1.5), robust.Cauchy(2.0)])
def test_kernel_weight_is_derivative(kernel):
    s = np.linspace(0.01, 30.0, 200)
    eps = 1e-6
    num = (kernel.rho(s + eps) - kernel.rho(s - eps)) / (2 * eps)
    assert np.allclose(kernel.weight(s), num, atol=1e-6)
    assert np.allclose(kernel.rho(s[:3] * 1e-4), s[:3] * 1e-4, rtol=1e-3)


def test_robust_pose_with_outliers():
    cam = PinholeCamera(500.0, 500.0, 320.0, 240.0)
    T_true = look_at(np.array([0.0, -6.0, 1.5]), np.array([0.0, 0.0, 0.5]))
    P = RNG.uniform([-3, -1, 0], [3, 3, 2], size=(80, 3))
    z = np.array([cam.project_world(T_true, p)[0] for p in P])
    z += RNG.normal(scale=0.5, size=z.shape)
    out = RNG.random(80) < 0.3
    z[out] = RNG.uniform([0, 0], [640, 480], size=(out.sum(), 2))
    T0 = T_true @ lie.se3_exp(np.array([0.1, -0.1, 0.1, 0.02, -0.02, 0.03]))
    pb = Pose(cam, P, z)
    err = {}
    for name, k in [("l2", None), ("cauchy", robust.Cauchy(2.0))]:
        res = optim.levenberg_marquardt(pb, T0, kernel=k)
        err[name] = np.linalg.norm(lie.se3_log(lie.se3_inv(T_true) @ res.x))
    assert err["cauchy"] < 0.02 < err["l2"]


def test_calibration_refinement():
    cam = PinholeCamera(458.7, 457.3, 367.2, 248.4,
                        [-0.283, 0.074, 1.8e-4, 1.8e-5, 0.0], 752, 480)
    board, views = calib.synthetic_views(cam, 15, np.random.default_rng(2),
                                         noise=0.2, distance=0.5, spread=0.1)
    est, poses, res = calib.calibrate(board, [u for _, u in views], 752,
                                      480, sigma=0.2)
    assert res.converged
    # errors consistent with the Laplace covariance (Proposition 4.12)
    pb = calib.CalibrationProblem(board, [u for _, u in views], 0.2, 752,
                                  480, fix_k3=True)
    sig = np.sqrt(np.diag(optim.covariance(pb, res.x))[:8])
    assert np.all(np.abs(est.params[:8] - cam.params[:8]) < 4 * sig)
    # final cost ~ chi^2 / 2 with m - n degrees of freedom
    m = 2 * len(board) * 15
    assert 0.8 < res.costs[-1] / (0.5 * (m - 9 - 6 * 15)) < 1.2
    for (T_cb, _), T_bc in zip(views, poses):
        assert np.linalg.norm(lie.se3_log(T_cb @ T_bc)) < 0.03
