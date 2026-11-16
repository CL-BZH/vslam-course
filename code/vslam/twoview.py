"""Two-view geometry (Chapter 6).

Conventions. x1, x2 are matched points in NORMALISED coordinates (N, 2),
i.e. pixels mapped through K^{-1} and undistorted. The relative pose
(R, t) maps camera-1 coordinates to camera-2 coordinates,
q2 = R q1 + t, so that T_21 = T_wc2^{-1} T_wc1. The essential matrix is
E = hat(t) R and satisfies x2~^T E x1~ = 0 with x~ = (x, 1).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .lie import hat
from .ransac import RansacResult, ransac


def homog(x: np.ndarray) -> np.ndarray:
    """Append a column of ones: (N, 2) -> (N, 3)."""
    return np.column_stack([x, np.ones(len(x))])


# --------------------------------------------------------------------------
# Essential and fundamental matrices
# --------------------------------------------------------------------------
def essential_from_pose(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return hat(t) @ R


def fundamental_from_essential(E: np.ndarray, K1: np.ndarray,
                               K2: np.ndarray | None = None) -> np.ndarray:
    """F = K2^{-T} E K1^{-1}, so that u2~^T F u1~ = 0 in pixels."""
    K2 = K1 if K2 is None else K2
    return np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)


def project_to_essential(M: np.ndarray) -> np.ndarray:
    """Closest essential matrix in Frobenius norm: equalise the two
    largest singular values and zero the third (unit scale)."""
    U, s, Vt = np.linalg.svd(M)
    return U @ np.diag([1.0, 1.0, 0.0]) @ Vt


def epipolar_residual(E: np.ndarray, x1: np.ndarray, x2: np.ndarray
                      ) -> np.ndarray:
    """Algebraic residuals x2~^T E x1~."""
    return np.einsum("ij,jk,ik->i", homog(x2), E, homog(x1))


def sampson(E: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Sampson distance: first-order geometric distance of each
    correspondence to the epipolar constraint (same units as x)."""
    h1, h2 = homog(x1), homog(x2)
    Ex1 = h1 @ E.T
    Etx2 = h2 @ E
    num = np.einsum("ij,ij->i", h2, Ex1)
    den = Ex1[:, 0] ** 2 + Ex1[:, 1] ** 2 + Etx2[:, 0] ** 2 + Etx2[:, 1] ** 2
    return np.abs(num) / np.sqrt(den)


def eight_point(x1: np.ndarray, x2: np.ndarray,
                normalise: bool = True) -> np.ndarray:
    """Linear estimate of E from N >= 8 correspondences, projected onto
    the essential manifold. With `normalise`, both point sets are first
    centred and scaled (Hartley), which matters for pixel coordinates and
    helps a little for normalised ones."""
    from .calib import normalise_points

    if normalise:
        x1n, T1 = normalise_points(x1)
        x2n, T2 = normalise_points(x2)
    else:
        x1n, x2n, T1, T2 = x1, x2, np.eye(3), np.eye(3)
    h1, h2 = homog(x1n), homog(x2n)
    A = np.einsum("ni,nj->nij", h2, h1).reshape(len(x1), 9)
    _, _, Vt = np.linalg.svd(A)
    En = Vt[-1].reshape(3, 3)
    if normalise:
        # enforce rank 2 before de-normalising (as for F), then project
        U, s, Vt2 = np.linalg.svd(En)
        En = U @ np.diag([s[0], s[1], 0.0]) @ Vt2
    E = T2.T @ En @ T1
    return project_to_essential(E)


def demazure(E: np.ndarray) -> np.ndarray:
    """The cubic constraints 2 E E^T E - tr(E E^T) E, zero for essential
    matrices."""
    return 2.0 * E @ E.T @ E - np.trace(E @ E.T) * E


# --------------------------------------------------------------------------
# Decomposition, triangulation, cheirality
# --------------------------------------------------------------------------
_W = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def decompose_essential(E: np.ndarray) -> list[tuple[np.ndarray,
                                                      np.ndarray]]:
    """The four (R, t), |t| = 1, with hat(t) R proportional to +-E."""
    U, _, Vt = np.linalg.svd(E)
    if np.linalg.det(U) < 0:
        U = -U
    if np.linalg.det(Vt) < 0:
        Vt = -Vt
    t = U[:, 2]
    Ra, Rb = U @ _W @ Vt, U @ _W.T @ Vt
    return [(Ra, t), (Ra, -t), (Rb, t), (Rb, -t)]


def triangulate(R: np.ndarray, t: np.ndarray, x1: np.ndarray,
                x2: np.ndarray) -> np.ndarray:
    """Linear (DLT) triangulation with P1 = [I|0], P2 = [R|t]; returns
    points (N, 3) in the frame of camera 1."""
    P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = np.hstack([R, t.reshape(3, 1)])
    X = np.empty((len(x1), 3))
    for i, (a, b) in enumerate(zip(x1, x2)):
        A = np.vstack([a[0] * P1[2] - P1[0], a[1] * P1[2] - P1[1],
                       b[0] * P2[2] - P2[0], b[1] * P2[2] - P2[1]])
        _, _, Vt = np.linalg.svd(A)
        X[i] = Vt[-1, :3] / Vt[-1, 3]
    return X


def parallax_angle(R: np.ndarray, t: np.ndarray, X: np.ndarray
                   ) -> np.ndarray:
    """Angle between the two viewing rays of each point X (camera-1
    frame), in radians."""
    c2 = -R.T @ t  # centre of camera 2 in the frame of camera 1
    r1 = X / np.linalg.norm(X, axis=1, keepdims=True)
    r2 = X - c2
    r2 /= np.linalg.norm(r2, axis=1, keepdims=True)
    return np.arccos(np.clip(np.sum(r1 * r2, axis=1), -1.0, 1.0))


def in_front(R: np.ndarray, t: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Cheirality: positive depth in both cameras."""
    return (X[:, 2] > 0) & ((X @ R.T + t)[:, 2] > 0)


@dataclass
class RelativePose:
    R: np.ndarray
    t: np.ndarray
    points: np.ndarray       # triangulated, camera-1 frame (unit baseline)
    in_front: np.ndarray     # cheirality mask over the input points


def recover_pose(E: np.ndarray, x1: np.ndarray, x2: np.ndarray
                 ) -> RelativePose:
    """Choose among the four decompositions of E the one that puts the
    most points in front of both cameras."""
    best = None
    for R, t in decompose_essential(E):
        X = triangulate(R, t, x1, x2)
        mask = in_front(R, t, X)
        if best is None or mask.sum() > best.in_front.sum():
            best = RelativePose(R, t, X, mask)
    return best


# --------------------------------------------------------------------------
# Robust estimation
# --------------------------------------------------------------------------
def essential_ransac(x1: np.ndarray, x2: np.ndarray, threshold: float,
                     p: float = 0.99,
                     rng: np.random.Generator | None = None
                     ) -> RansacResult:
    """E by eight-point RANSAC; `threshold` is a Sampson distance in
    normalised units (pixels / focal length)."""

    def fit(idx):
        if len(idx) < 8:
            return None
        return eight_point(x1[idx], x2[idx])

    return ransac(len(x1), fit, lambda E: sampson(E, x1, x2), 8, threshold,
                  p, rng=rng)


def five_point_ransac(x1: np.ndarray, x2: np.ndarray, threshold: float,
                      p: float = 0.999) -> tuple[np.ndarray, np.ndarray]:
    """Nister's five-point solver inside RANSAC, from OpenCV. Returns
    (E, inlier mask). Inputs in normalised coordinates."""
    E, mask = cv2.findEssentialMat(x1.astype(np.float64),
                                   x2.astype(np.float64), np.eye(3),
                                   method=cv2.RANSAC, prob=p,
                                   threshold=threshold)
    if E is None:
        return None, np.zeros(len(x1), dtype=bool)
    return E[:3] / np.linalg.norm(E[:3]), mask.ravel().astype(bool)


# --------------------------------------------------------------------------
# The planar model
# --------------------------------------------------------------------------
def homography_from_plane(R: np.ndarray, t: np.ndarray, n: np.ndarray,
                          d: float) -> np.ndarray:
    """Homography (normalised coordinates) induced by the plane
    n^T q1 = d of camera 1: x2~ ~ (R + t n^T / d) x1~."""
    return R + np.outer(t, n) / d


def decompose_homography(H: np.ndarray, x1: np.ndarray, x2: np.ndarray
                         ) -> list[tuple[np.ndarray, np.ndarray,
                                         np.ndarray]]:
    """Physically valid (R, t/d, n) of a homography in normalised
    coordinates, via OpenCV's decomposition (Malis and Vargas), keeping
    the solutions with the observed points in front of both cameras."""
    _, Rs, ts, ns = cv2.decomposeHomographyMat(H, np.eye(3))
    out = []
    for R, t, n in zip(Rs, ts, ns):
        t, n = t.ravel(), n.ravel()
        # plane points: n^T q1 = 1 (t is t/d) => depth of x1 is 1/(n.x1~)
        depth1 = 1.0 / (homog(x1) @ n)
        q1 = homog(x1) * depth1[:, None]
        if np.all(depth1 > 0) and np.all((q1 @ R.T + t)[:, 2] > 0):
            out.append((R, t, n))
    return out
