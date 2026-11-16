"""Camera pose from known 3D points: PnP (Chapter 7).

Conventions: X are world points (N, 3); x are their NORMALISED image
coordinates (N, 2); the solvers return T_cw (world to camera), so that
q = R_cw X + t_cw. The refinement problems estimate T_wc (camera to
world) with a right perturbation, like everywhere else in the course.
"""

from __future__ import annotations

import cv2
import numpy as np

from . import lie
from .calib import nearest_rotation
from .ransac import RansacResult, ransac


# --------------------------------------------------------------------------
# Linear and minimal solvers
# --------------------------------------------------------------------------
def pnp_dlt(X: np.ndarray, x: np.ndarray) -> np.ndarray | None:
    """Linear PnP from N >= 6 non-coplanar points: solve x~ ~ [R|t] X~ for
    the 12 entries by SVD, then fix scale and sign and project the left
    block onto SO(3). Returns T_cw."""
    rows = []
    for (Xw, Yw, Zw), (u, v) in zip(X, x):
        p = np.array([Xw, Yw, Zw, 1.0])
        z = np.zeros(4)
        rows.append(np.concatenate([p, z, -u * p]))
        rows.append(np.concatenate([z, p, -v * p]))
    _, s, Vt = np.linalg.svd(np.array(rows))
    P = Vt[-1].reshape(3, 4)
    M = P[:, :3]
    scale = np.cbrt(np.linalg.det(M))  # det(lambda R) = lambda^3
    if abs(scale) < 1e-12:
        return None
    P = P / scale                       # det(M) = 1: fixes scale and sign
    R = nearest_rotation(P[:, :3])
    t = P[:, 3]
    if np.median(X @ R[2] + t[2]) < 0:  # points must be in front
        return None
    return lie.make_T(R, t)


def p3p(X: np.ndarray, x: np.ndarray) -> list[np.ndarray]:
    """All poses T_cw from exactly three correspondences (up to four),
    using OpenCV's P3P solver on normalised coordinates."""
    n, rvecs, tvecs = cv2.solveP3P(X.astype(np.float64),
                                   x.astype(np.float64), np.eye(3), None,
                                   flags=cv2.SOLVEPNP_P3P)
    out = []
    for rv, tv in zip(rvecs or [], tvecs or []):
        R, _ = cv2.Rodrigues(rv)
        out.append(lie.make_T(R, tv.ravel()))
    return out


def reprojection_errors(T_cw: np.ndarray, X: np.ndarray, x: np.ndarray
                        ) -> np.ndarray:
    """|pi(R X + t) - x| in normalised units; inf behind the camera."""
    q = X @ T_cw[:3, :3].T + T_cw[:3, 3]
    err = np.full(len(X), np.inf)
    ok = q[:, 2] > 1e-6
    err[ok] = np.linalg.norm(q[ok, :2] / q[ok, 2:] - x[ok], axis=1)
    return err


def pnp_ransac(X: np.ndarray, x: np.ndarray, threshold: float,
               p: float = 0.99, rng: np.random.Generator | None = None
               ) -> RansacResult:
    """P3P inside RANSAC: each sample has four points, three for the
    solver and the fourth to choose among its solutions. The final model
    is refitted on the inliers with the DLT (if they are not coplanar) or
    left to the nonlinear refinement. Threshold in normalised units."""

    def fit(idx):
        if len(idx) == 4:
            best, best_err = None, np.inf
            for T in p3p(X[idx[:3]], x[idx[:3]]):
                e = reprojection_errors(T, X[idx[3:]], x[idx[3:]])[0]
                if e < best_err:
                    best, best_err = T, e
            return best
        return None  # the refit is done by the nonlinear refinement

    return ransac(len(X), fit, lambda T: reprojection_errors(T, X, x), 4,
                  threshold, p, rng=rng)


# --------------------------------------------------------------------------
# Vectorised refinement problems (motion-only bundle adjustment)
# --------------------------------------------------------------------------
class MonoPoseProblem:
    """Reprojection error of known world points in one undistorted
    pinhole camera; state T_wc. Residuals whitened by sigma (pixels)."""

    block_size = 2

    def __init__(self, K: np.ndarray, X: np.ndarray, z: np.ndarray,
                 sigma: float = 1.0) -> None:
        self.fx, self.fy = K[0, 0], K[1, 1]
        self.cx, self.cy = K[0, 2], K[1, 2]
        self.X, self.z, self.sigma = X, z, sigma

    def _project(self, T_wc):
        q = (self.X - T_wc[:3, 3]) @ T_wc[:3, :3]
        X, Y, Z = q.T
        iz = 1.0 / Z
        u = np.column_stack([self.fx * X * iz + self.cx,
                             self.fy * Y * iz + self.cy])
        Jq = np.zeros((len(q), 2, 3))
        Jq[:, 0, 0], Jq[:, 0, 2] = self.fx * iz, -self.fx * X * iz * iz
        Jq[:, 1, 1], Jq[:, 1, 2] = self.fy * iz, -self.fy * Y * iz * iz
        return q, u, Jq

    @staticmethod
    def _dq_dxi(q):
        """[-I, hat(q)] for each point, shape (N, 3, 6)."""
        J = np.zeros((len(q), 3, 6))
        J[:, :, :3] = -np.eye(3)
        X, Y, Z = q.T
        J[:, 0, 4], J[:, 0, 5] = -Z, Y
        J[:, 1, 3], J[:, 1, 5] = Z, -X
        J[:, 2, 3], J[:, 2, 4] = -Y, X
        return J

    def linearize(self, T_wc):
        q, u, Jq = self._project(T_wc)
        r = ((self.z - u) / self.sigma).ravel()
        J = -(Jq @ self._dq_dxi(q)).reshape(-1, 6) / self.sigma
        return r, J

    def retract(self, T, d):
        return T @ lie.se3_exp(d)


class StereoPoseProblem(MonoPoseProblem):
    """Same with rectified stereo measurements (u_l, v, u_r)."""

    block_size = 3

    def __init__(self, K: np.ndarray, baseline: float, X: np.ndarray,
                 z: np.ndarray, sigma: float = 1.0) -> None:
        super().__init__(K, X, z, sigma)
        self.b = baseline

    def linearize(self, T_wc):
        q, u, Jq = self._project(T_wc)
        X, Y, Z = q.T
        iz = 1.0 / Z
        ur = self.fx * (X - self.b) * iz + self.cx
        pred = np.column_stack([u, ur])
        J3 = np.zeros((len(q), 3, 3))
        J3[:, :2] = Jq
        J3[:, 2, 0] = self.fx * iz
        J3[:, 2, 2] = -self.fx * (X - self.b) * iz * iz
        r = ((self.z - pred) / self.sigma).ravel()
        J = -(J3 @ self._dq_dxi(q)).reshape(-1, 6) / self.sigma
        return r, J
