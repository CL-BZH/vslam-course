"""Planar-target calibration: homographies and Zhang's method (Chapter 3).

A planar target (checkerboard) lies in the plane Z = 0 of its own frame
b. A point (X, Y, 0) of the board projects, ignoring distortion, to

    [u, v, 1]^T  ~  K [r1  r2  t] [X, Y, 1]^T  =  H [X, Y, 1]^T,

where (R, t) = T_cb maps board coordinates to camera coordinates and r1,
r2 are the first two columns of R. Zhang's method estimates one
homography H per view, extracts K linearly from the constraints that r1
and r2 are orthonormal, and then recovers each (R, t). The result is the
starting point of a nonlinear refinement (Chapter 4), which also
estimates the distortion.
"""

from __future__ import annotations

import numpy as np

from . import lie
from .camera import PinholeCamera, look_at


def checkerboard(nx: int = 9, ny: int = 6, square: float = 0.03
                 ) -> np.ndarray:
    """Inner corners of a checkerboard, shape (nx*ny, 3), in the plane Z=0
    of the board frame, centred on the origin."""
    xs = (np.arange(nx) - (nx - 1) / 2) * square
    ys = (np.arange(ny) - (ny - 1) / 2) * square
    X, Y = np.meshgrid(xs, ys)
    return np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)])


def normalise_points(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hartley normalisation: a similarity T moving the centroid of the 2D
    points x (N, 2) to the origin with mean distance sqrt(2).

    Returns the normalised points and the 3x3 matrix T.
    """
    c = x.mean(axis=0)
    s = np.sqrt(2.0) / np.mean(np.linalg.norm(x - c, axis=1))
    T = np.array([[s, 0.0, -s * c[0]],
                  [0.0, s, -s * c[1]],
                  [0.0, 0.0, 1.0]])
    return (x - c) * s, T


def homography_dlt(src: np.ndarray, dst: np.ndarray,
                   normalise: bool = True) -> np.ndarray:
    """Homography H with dst ~ H src, from N >= 4 correspondences (N, 2).

    Direct linear transform on dst x (H src) = 0, solved by SVD. With
    `normalise`, both point sets are first Hartley-normalised and H is
    de-normalised afterwards. The result is scaled so that H[2, 2] = 1.
    """
    if normalise:
        s, Ts = normalise_points(src)
        d, Td = normalise_points(dst)
    else:
        s, d = src, dst
        Ts = Td = np.eye(3)
    rows = []
    for (x, y), (u, v) in zip(s, d):
        p = np.array([x, y, 1.0])
        z = np.zeros(3)
        rows.append(np.concatenate([z, -p, v * p]))
        rows.append(np.concatenate([p, z, -u * p]))
    _, _, Vt = np.linalg.svd(np.array(rows))
    Hn = Vt[-1].reshape(3, 3)
    H = np.linalg.solve(Td, Hn @ Ts)
    return H / H[2, 2]


def _v(H: np.ndarray, i: int, j: int) -> np.ndarray:
    """Row vector v_ij with h_i^T B h_j = v_ij . b, b = (B11, B12, B22,
    B13, B23, B33) and h_i the i-th column of H."""
    a, c = H[:, i], H[:, j]
    return np.array([a[0] * c[0],
                     a[0] * c[1] + a[1] * c[0],
                     a[1] * c[1],
                     a[2] * c[0] + a[0] * c[2],
                     a[2] * c[1] + a[1] * c[2],
                     a[2] * c[2]])


def zhang_intrinsics(Hs: list[np.ndarray],
                     zero_skew: bool = False) -> np.ndarray:
    """Intrinsic matrix K from >= 3 board homographies (>= 2 with
    zero_skew), by the linear step of Zhang's method."""
    rows = []
    for H in Hs:
        rows.append(_v(H, 0, 1))                 # r1 . r2 = 0
        rows.append(_v(H, 0, 0) - _v(H, 1, 1))   # |r1| = |r2|
    if zero_skew:
        rows.append(np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]))  # B12 = 0
    _, _, Vt = np.linalg.svd(np.array(rows))
    b = Vt[-1]
    B = np.array([[b[0], b[1], b[3]],
                  [b[1], b[2], b[4]],
                  [b[3], b[4], b[5]]])
    if B[0, 0] < 0:  # b is defined up to sign; B = K^-T K^-1 is PD
        B = -B
    L = np.linalg.cholesky(B)  # B = L L^T with L = K^{-T} (up to scale)
    K = np.linalg.inv(L.T)
    return K / K[2, 2]


def nearest_rotation(M: np.ndarray) -> np.ndarray:
    """Rotation closest to M in Frobenius norm: U diag(1, 1, det) V^T."""
    U, _, Vt = np.linalg.svd(M)
    D = np.diag([1.0, 1.0, np.linalg.det(U @ Vt)])
    return U @ D @ Vt


def zhang_extrinsics(K: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Board pose T_cb from K and the board homography H."""
    A = np.linalg.solve(K, H)  # ~ [r1 r2 t] up to scale
    lam = 1.0 / np.linalg.norm(A[:, 0])
    if A[2, 2] * lam < 0:  # the board must be in front of the camera
        lam = -lam
    r1, r2, t = lam * A[:, 0], lam * A[:, 1], lam * A[:, 2]
    R = nearest_rotation(np.column_stack([r1, r2, np.cross(r1, r2)]))
    return lie.make_T(R, t)


def synthetic_views(cam: PinholeCamera, n_views: int,
                    rng: np.random.Generator, noise: float = 0.0,
                    board: np.ndarray | None = None,
                    distance: float = 0.45, spread: float = 0.02
                    ) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """Simulate views of a checkerboard for calibration experiments.

    The camera is placed at random around the point (0, 0, -distance) of
    the board frame, looking at the board centre, and views where some
    corner falls outside the image are rejected. Returns the board points
    and a list of (T_cb, pixels) pairs, with Gaussian pixel noise of
    standard deviation `noise`. Distortion of `cam` is applied.
    """
    board = checkerboard() if board is None else board
    views = []
    while len(views) < n_views:
        eye = rng.normal(size=3) * np.array([0.15, 0.15, 0.1]) \
            + np.array([0.0, 0.0, -distance])
        T_bc = look_at(eye, rng.normal(scale=spread, size=3),
                       up=np.array([0.0, -1.0, 0.0]))
        T_cb = lie.se3_inv(T_bc)
        q = board @ T_cb[:3, :3].T + T_cb[:3, 3]
        u = cam.project(q)
        if np.all(cam.in_image(u, margin=5.0)):
            views.append((T_cb, u + rng.normal(scale=noise, size=u.shape)))
    return board, views
