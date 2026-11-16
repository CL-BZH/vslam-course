"""Lie groups SO(3) and SE(3) for state estimation (Chapter 2).

Conventions (same as the course notes, Barfoot and Sophus):

* A rotation vector is phi in R^3, with angle theta = |phi|.
* A twist is xi = (rho, phi) in R^6: translational part FIRST.
* hat(xi) = [[hat(phi), rho], [0, 0]] in se(3).
* Perturbations are on the RIGHT unless stated otherwise:
  T (+) xi = T @ Exp(xi), and Jacobians are defined by
  f(T Exp(xi)) ~= f(T) Exp(J xi) (group-valued f) or
  f(T Exp(xi)) ~= f(T) + J xi (vector-valued f).

Every closed form below is derived in Chapter 2, and every Jacobian is
checked against finite differences in tests/test_lie.py.
"""

from __future__ import annotations

import numpy as np

# Below this angle the closed forms lose precision through cancellation;
# we switch to Taylor series, whose truncation error is then ~1e-16.
_SMALL = 1e-3


# --------------------------------------------------------------------------
# so(3)
# --------------------------------------------------------------------------
def hat(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix of a 3-vector: hat(a) @ b == cross(a, b)."""
    x, y, z = v
    return np.array([[0.0, -z, y],
                     [z, 0.0, -x],
                     [-y, x, 0.0]])


def vee(S: np.ndarray) -> np.ndarray:
    """Inverse of `hat` (uses the antisymmetric part of S)."""
    return 0.5 * np.array([S[2, 1] - S[1, 2],
                           S[0, 2] - S[2, 0],
                           S[1, 0] - S[0, 1]])


def _coeffs(theta: float) -> tuple[float, float, float]:
    """A = sin t / t, B = (1 - cos t) / t^2, C = (t - sin t) / t^3."""
    if theta < _SMALL:
        t2 = theta * theta
        A = 1.0 - t2 / 6.0 * (1.0 - t2 / 20.0)
        B = 0.5 - t2 / 24.0 * (1.0 - t2 / 30.0)
        C = 1.0 / 6.0 - t2 / 120.0 * (1.0 - t2 / 42.0)
    else:
        s, c = np.sin(theta), np.cos(theta)
        A = s / theta
        B = (1.0 - c) / (theta * theta)
        C = (theta - s) / theta**3
    return A, B, C


# --------------------------------------------------------------------------
# SO(3)
# --------------------------------------------------------------------------
def so3_exp(phi: np.ndarray) -> np.ndarray:
    """Rodrigues' formula: Exp(phi) = I + A hat(phi) + B hat(phi)^2."""
    phi = np.asarray(phi, dtype=float)
    A, B, _ = _coeffs(np.linalg.norm(phi))
    K = hat(phi)
    return np.eye(3) + A * K + B * (K @ K)


def so3_log(R: np.ndarray) -> np.ndarray:
    """Rotation vector phi with |phi| in [0, pi] such that Exp(phi) = R."""
    cos_t = np.clip(0.5 * (np.trace(R) - 1.0), -1.0, 1.0)
    theta = np.arccos(cos_t)
    w = vee(R - R.T)  # = 2 sin(theta) * axis
    if theta < _SMALL:
        # R - R^T = 2 A hat(phi), and A ~ 1 - t^2/6 here.
        A, _, _ = _coeffs(theta)
        return w / (2.0 * A)
    if np.pi - theta > 1e-6:
        return theta / (2.0 * np.sin(theta)) * w
    # Near pi, sin(theta) ~ 0: read the axis from the symmetric part,
    # sym(R) = cos(t) I + (1 - cos t) a a^T, and fix its sign with w.
    aaT = (0.5 * (R + R.T) - cos_t * np.eye(3)) / (1.0 - cos_t)
    i = int(np.argmax(np.diag(aaT)))
    a = aaT[:, i] / np.sqrt(aaT[i, i])
    if a @ w < 0.0:
        a = -a
    return theta * a


def so3_right_jacobian(phi: np.ndarray) -> np.ndarray:
    """J_r(phi) = I - B hat(phi) + C hat(phi)^2.

    Exp(phi + d) ~= Exp(phi) Exp(J_r(phi) d) for small d.
    """
    _, B, C = _coeffs(np.linalg.norm(phi))
    K = hat(phi)
    return np.eye(3) - B * K + C * (K @ K)


def so3_left_jacobian(phi: np.ndarray) -> np.ndarray:
    """J_l(phi) = J_r(-phi) = I + B hat(phi) + C hat(phi)^2."""
    return so3_right_jacobian(-np.asarray(phi, dtype=float))


def so3_right_jacobian_inv(phi: np.ndarray) -> np.ndarray:
    """J_r(phi)^{-1} = I + hat(phi)/2 + D hat(phi)^2.

    D = 1/t^2 - (1 + cos t) / (2 t sin t). Log(Exp(phi) Exp(d)) ~=
    phi + J_r(phi)^{-1} d. Singular at t = 2 pi (never reached by Log).
    """
    theta = np.linalg.norm(phi)
    if theta < _SMALL:
        t2 = theta * theta
        D = 1.0 / 12.0 + t2 / 720.0
    else:
        D = (1.0 / theta**2
             - (1.0 + np.cos(theta)) / (2.0 * theta * np.sin(theta)))
    K = hat(phi)
    return np.eye(3) + 0.5 * K + D * (K @ K)


def so3_left_jacobian_inv(phi: np.ndarray) -> np.ndarray:
    """J_l(phi)^{-1} = J_r(-phi)^{-1}."""
    return so3_right_jacobian_inv(-np.asarray(phi, dtype=float))


# --------------------------------------------------------------------------
# se(3) and SE(3)
# --------------------------------------------------------------------------
def se3_hat(xi: np.ndarray) -> np.ndarray:
    """4x4 matrix of a twist xi = (rho, phi)."""
    M = np.zeros((4, 4))
    M[:3, :3] = hat(xi[3:])
    M[:3, 3] = xi[:3]
    return M


def se3_vee(M: np.ndarray) -> np.ndarray:
    """Inverse of `se3_hat`."""
    return np.concatenate([M[:3, 3], vee(M[:3, :3])])


def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Homogeneous 4x4 matrix from rotation and translation."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def se3_exp(xi: np.ndarray) -> np.ndarray:
    """Exp(rho, phi) = [[Exp(phi), J_l(phi) rho], [0, 1]]."""
    xi = np.asarray(xi, dtype=float)
    rho, phi = xi[:3], xi[3:]
    return make_T(so3_exp(phi), so3_left_jacobian(phi) @ rho)


def se3_log(T: np.ndarray) -> np.ndarray:
    """Twist (rho, phi) with Exp(rho, phi) = T."""
    phi = so3_log(T[:3, :3])
    rho = so3_left_jacobian_inv(phi) @ T[:3, 3]
    return np.concatenate([rho, phi])


def se3_inv(T: np.ndarray) -> np.ndarray:
    """T^{-1} = [[R^T, -R^T t], [0, 1]] (cheaper than a 4x4 inverse)."""
    R, t = T[:3, :3], T[:3, 3]
    return make_T(R.T, -R.T @ t)


def se3_adjoint(T: np.ndarray) -> np.ndarray:
    """Ad_T = [[R, hat(t) R], [0, R]]; T hat(xi) T^-1 = hat(Ad_T xi)."""
    R, t = T[:3, :3], T[:3, 3]
    Ad = np.zeros((6, 6))
    Ad[:3, :3] = R
    Ad[:3, 3:] = hat(t) @ R
    Ad[3:, 3:] = R
    return Ad


def se3_act(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Apply T to a 3D point: R p + t."""
    return T[:3, :3] @ p + T[:3, 3]


# --------------------------------------------------------------------------
# Jacobians with respect to right perturbations
# --------------------------------------------------------------------------
def jac_act(T: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Jacobians of q = T p w.r.t. T (right perturbation) and p.

    d q / d xi = R [I, -hat(p)]  (3x6),   d q / d p = R  (3x3).
    """
    R = T[:3, :3]
    J_T = np.hstack([R, -R @ hat(p)])
    return J_T, R


def jac_compose(T1: np.ndarray, T2: np.ndarray
                ) -> tuple[np.ndarray, np.ndarray]:
    """Jacobians of T1 T2 w.r.t. T1 and T2: (Ad_{T2^-1}, I)."""
    return se3_adjoint(se3_inv(T2)), np.eye(6)


def jac_inv(T: np.ndarray) -> np.ndarray:
    """Jacobian of T^{-1} w.r.t. T: -Ad_T."""
    return -se3_adjoint(T)
