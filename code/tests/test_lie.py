"""Tests for vslam.lie: identities of Chapter 2 and Jacobian checks."""

import numpy as np
import pytest
from scipy.linalg import expm

from vslam import lie

RNG = np.random.default_rng(42)
ANGLES = [0.0, 1e-9, 1e-5, 1e-3 * 0.999, 1e-3 * 1.001, 0.3, 1.5, 3.0,
          np.pi - 1e-4, np.pi - 1e-8]


def random_axis() -> np.ndarray:
    a = RNG.normal(size=3)
    return a / np.linalg.norm(a)


def random_T() -> np.ndarray:
    return lie.se3_exp(RNG.normal(size=6))


def num_jac_group(f, T, eps=1e-6):
    """Right Jacobian of a group-valued f by central differences."""
    fT_inv = lie.se3_inv(f(T))
    J = np.zeros((6, 6))
    for i in range(6):
        d = np.zeros(6)
        d[i] = eps
        plus = lie.se3_log(fT_inv @ f(T @ lie.se3_exp(d)))
        minus = lie.se3_log(fT_inv @ f(T @ lie.se3_exp(-d)))
        J[:, i] = (plus - minus) / (2 * eps)
    return J


def test_hat_is_cross_product():
    a, b = RNG.normal(size=3), RNG.normal(size=3)
    assert np.allclose(lie.hat(a) @ b, np.cross(a, b))
    assert np.allclose(lie.vee(lie.hat(a)), a)


def test_hat_cubed():
    a = RNG.normal(size=3)
    K = lie.hat(a)
    assert np.allclose(K @ K, np.outer(a, a) - (a @ a) * np.eye(3))
    assert np.allclose(K @ K @ K, -(a @ a) * K)


@pytest.mark.parametrize("theta", ANGLES)
def test_so3_exp_matches_expm_and_is_rotation(theta):
    phi = theta * random_axis()
    R = lie.so3_exp(phi)
    assert np.allclose(R, expm(lie.hat(phi)), atol=1e-12)
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(R), 1.0)


@pytest.mark.parametrize("theta", ANGLES)
def test_so3_log_inverts_exp(theta):
    phi = theta * random_axis()
    assert np.allclose(lie.so3_log(lie.so3_exp(phi)), phi, atol=1e-7)


def test_so3_log_at_pi():
    a = random_axis()
    R = lie.so3_exp(np.pi * a)
    phi = lie.so3_log(R)
    assert np.isclose(np.linalg.norm(phi), np.pi)
    assert np.allclose(lie.so3_exp(phi), R, atol=1e-12)  # +-a both valid


@pytest.mark.parametrize("theta", ANGLES[:-2])
def test_right_jacobian_definition(theta):
    """Exp(phi + d) ~= Exp(phi) Exp(J_r d), checked column by column."""
    phi = theta * random_axis()
    eps = 1e-6
    R_inv = lie.so3_exp(phi).T
    J_num = np.zeros((3, 3))
    for i in range(3):
        d = np.zeros(3)
        d[i] = eps
        J_num[:, i] = (lie.so3_log(R_inv @ lie.so3_exp(phi + d))
                       - lie.so3_log(R_inv @ lie.so3_exp(phi - d))) / (2 * eps)
    assert np.allclose(J_num, lie.so3_right_jacobian(phi), atol=1e-7)


@pytest.mark.parametrize("theta", ANGLES[:-2])
def test_jacobian_identities(theta):
    phi = theta * random_axis()
    Jr, Jl = lie.so3_right_jacobian(phi), lie.so3_left_jacobian(phi)
    assert np.allclose(Jr @ lie.so3_right_jacobian_inv(phi), np.eye(3))
    assert np.allclose(Jl @ lie.so3_left_jacobian_inv(phi), np.eye(3))
    assert np.allclose(Jl, lie.so3_exp(phi) @ Jr)


def test_rotation_conjugation():
    """R hat(a) R^T = hat(R a) and R Exp(a) R^T = Exp(R a)."""
    R, a = lie.so3_exp(RNG.normal(size=3)), RNG.normal(size=3)
    assert np.allclose(R @ lie.hat(a) @ R.T, lie.hat(R @ a))
    assert np.allclose(R @ lie.so3_exp(a) @ R.T, lie.so3_exp(R @ a))


@pytest.mark.parametrize("theta", [0.0, 1e-6, 0.7, 3.0])
def test_se3_exp_log(theta):
    xi = np.concatenate([RNG.normal(size=3), theta * random_axis()])
    T = lie.se3_exp(xi)
    assert np.allclose(T, expm(lie.se3_hat(xi)), atol=1e-12)
    assert np.allclose(lie.se3_log(T), xi, atol=1e-9)
    assert np.allclose(lie.se3_inv(T) @ T, np.eye(4))


def test_adjoint():
    T, xi = random_T(), RNG.normal(size=6)
    lhs = T @ lie.se3_hat(xi) @ lie.se3_inv(T)
    assert np.allclose(lhs, lie.se3_hat(lie.se3_adjoint(T) @ xi))
    assert np.allclose(T @ lie.se3_exp(xi) @ lie.se3_inv(T),
                       lie.se3_exp(lie.se3_adjoint(T) @ xi))
    assert np.allclose(lie.se3_adjoint(lie.se3_inv(T)),
                       np.linalg.inv(lie.se3_adjoint(T)))


def test_jac_act():
    T, p, eps = random_T(), RNG.normal(size=3), 1e-6
    J_T, J_p = lie.jac_act(T, p)
    num_T = np.zeros((3, 6))
    for i in range(6):
        d = np.zeros(6)
        d[i] = eps
        num_T[:, i] = (lie.se3_act(T @ lie.se3_exp(d), p)
                       - lie.se3_act(T @ lie.se3_exp(-d), p)) / (2 * eps)
    num_p = np.column_stack([
        (lie.se3_act(T, p + e) - lie.se3_act(T, p - e)) / (2 * eps)
        for e in eps * np.eye(3)])
    assert np.allclose(J_T, num_T, atol=1e-7)
    assert np.allclose(J_p, num_p, atol=1e-7)


def test_jac_compose_and_inverse():
    T1, T2 = random_T(), random_T()
    J1, J2 = lie.jac_compose(T1, T2)
    assert np.allclose(J1, num_jac_group(lambda X: X @ T2, T1), atol=1e-6)
    assert np.allclose(J2, num_jac_group(lambda X: T1 @ X, T2), atol=1e-6)
    assert np.allclose(lie.jac_inv(T1), num_jac_group(lie.se3_inv, T1),
                       atol=1e-6)


@pytest.mark.parametrize("scale", [1e-6, 0.1, 1.0])
def test_se3_right_jacobian_and_inverse(scale):
    xi = scale * RNG.normal(size=6)
    Jr = lie.se3_right_jacobian(xi)
    eps = 1e-6
    T_inv = lie.se3_inv(lie.se3_exp(xi))
    num = np.column_stack([
        (lie.se3_log(T_inv @ lie.se3_exp(xi + d))
         - lie.se3_log(T_inv @ lie.se3_exp(xi - d))) / (2 * eps)
        for d in eps * np.eye(6)])
    assert np.allclose(Jr, num, atol=1e-6)
    assert np.allclose(lie.se3_right_jacobian_inv(xi) @ Jr, np.eye(6),
                       atol=1e-8)
    # the SO(3) block is the SO(3) right Jacobian
    assert np.allclose(Jr[3:, 3:], lie.so3_right_jacobian(xi[3:]))


def test_se3_ad_is_derivative_of_adjoint():
    xi, x = RNG.normal(size=6), RNG.normal(size=6)
    lhs = lie.se3_hat(xi) @ lie.se3_hat(x) - lie.se3_hat(x) @ lie.se3_hat(xi)
    assert np.allclose(lhs, lie.se3_hat(lie.se3_ad(xi) @ x))
