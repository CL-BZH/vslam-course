"""Camera models (Chapter 3).

Frames: the camera frame has z pointing forward along the optical axis,
x to the right and y down, as in OpenCV. A camera pose is T_wc (camera
to world); a world point p_w is seen in the camera frame as
q = T_wc^{-1} p_w.

The projection chain is

    q  --pi-->  x = (q_x/q_z, q_y/q_z)  --d-->  x_d  --K-->  u

(perspective division, lens distortion on normalised coordinates, affine
map to pixels). Every Jacobian is given for this chain and is checked
against finite differences in tests/test_camera.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import lie


# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------
def perspective(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """x = (q_x/q_z, q_y/q_z) and its 2x3 Jacobian with respect to q."""
    X, Y, Z = q
    iz = 1.0 / Z
    x = np.array([X * iz, Y * iz])
    J = np.array([[iz, 0.0, -X * iz * iz],
                  [0.0, iz, -Y * iz * iz]])
    return x, J


def look_at(eye: np.ndarray, target: np.ndarray,
            up: np.ndarray = np.array([0.0, 0.0, 1.0])) -> np.ndarray:
    """Pose T_wc of a camera at `eye` whose optical axis points at `target`.

    `up` is a world direction that should appear "up" in the image (the
    camera's y axis points down, i.e. roughly along -up).
    """
    z = np.asarray(target, float) - np.asarray(eye, float)
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    if np.linalg.norm(x) < 1e-9:  # looking along `up`: pick any x
        x = np.cross(z, np.array([1.0, 0.0, 0.0]))
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return lie.make_T(np.column_stack([x, y, z]), np.asarray(eye, float))


# --------------------------------------------------------------------------
# Pinhole camera with radial-tangential (Brown-Conrady) distortion
# --------------------------------------------------------------------------
@dataclass
class PinholeCamera:
    """Pinhole camera with intrinsics (fx, fy, cx, cy) and OpenCV-style
    distortion coefficients dist = (k1, k2, p1, p2, k3)."""

    fx: float
    fy: float
    cx: float
    cy: float
    dist: np.ndarray = field(default_factory=lambda: np.zeros(5))
    width: int = 640
    height: int = 480

    def __post_init__(self) -> None:
        self.dist = np.asarray(self.dist, dtype=float)

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx],
                         [0.0, self.fy, self.cy],
                         [0.0, 0.0, 1.0]])

    # ---------------------------------------------------------- distortion
    def distort(self, x: np.ndarray) -> np.ndarray:
        """Distort normalised coordinates x (shape (2,) or (N, 2))."""
        k1, k2, p1, p2, k3 = self.dist
        x = np.asarray(x, dtype=float)
        u, v = x[..., 0], x[..., 1]
        r2 = u * u + v * v
        radial = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
        ud = u * radial + 2.0 * p1 * u * v + p2 * (r2 + 2.0 * u * u)
        vd = v * radial + p1 * (r2 + 2.0 * v * v) + 2.0 * p2 * u * v
        return np.stack([ud, vd], axis=-1)

    def distort_jacobian(self, x: np.ndarray) -> np.ndarray:
        """2x2 Jacobian of `distort` at a single point x."""
        k1, k2, p1, p2, k3 = self.dist
        u, v = x
        r2 = u * u + v * v
        radial = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
        # d(radial)/du = g u and d(radial)/dv = g v
        g = 2.0 * k1 + 4.0 * k2 * r2 + 6.0 * k3 * r2 * r2
        return np.array([
            [radial + g * u * u + 2.0 * p1 * v + 6.0 * p2 * u,
             g * u * v + 2.0 * p1 * u + 2.0 * p2 * v],
            [g * u * v + 2.0 * p1 * u + 2.0 * p2 * v,
             radial + g * v * v + 6.0 * p1 * v + 2.0 * p2 * u]])

    def undistort(self, xd: np.ndarray, iters: int = 20,
                  tol: float = 1e-14) -> np.ndarray:
        """Invert `distort` at a single point by Newton's method."""
        xd = np.asarray(xd, dtype=float)
        x = xd.copy()  # the undistorted point is a good initial guess
        for _ in range(iters):
            r = self.distort(x) - xd
            if r @ r < tol * tol:
                break
            x = x - np.linalg.solve(self.distort_jacobian(x), r)
        return x

    # ---------------------------------------------------------- projection
    def pixel(self, xd: np.ndarray) -> np.ndarray:
        """Affine map from distorted normalised coordinates to pixels."""
        xd = np.asarray(xd, dtype=float)
        return np.stack([self.fx * xd[..., 0] + self.cx,
                         self.fy * xd[..., 1] + self.cy], axis=-1)

    def project(self, q: np.ndarray) -> np.ndarray:
        """Pixel of a point (or of N points, shape (N, 3)) in camera frame."""
        q = np.asarray(q, dtype=float)
        x = q[..., :2] / q[..., 2:3]
        return self.pixel(self.distort(x))

    def project_jacobian(self, q: np.ndarray
                         ) -> tuple[np.ndarray, np.ndarray]:
        """Pixel u of a single point q and the 2x3 Jacobian du/dq."""
        x, J_pi = perspective(q)
        J_d = self.distort_jacobian(x)
        J_k = np.diag([self.fx, self.fy])
        return self.pixel(self.distort(x)), J_k @ J_d @ J_pi

    def project_world(self, T_wc: np.ndarray, p_w: np.ndarray
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Reprojection of a world point and its Jacobians.

        Returns u (2,), du/dxi (2x6, right perturbation of T_wc) and
        du/dp_w (2x3). Uses dq/dxi = [-I, hat(q)] (Exercise 2.3).
        """
        q = lie.se3_act(lie.se3_inv(T_wc), p_w)
        u, J_q = self.project_jacobian(q)
        J_xi = J_q @ np.hstack([-np.eye(3), lie.hat(q)])
        J_p = J_q @ T_wc[:3, :3].T
        return u, J_xi, J_p

    # ---------------------------------------------------------- intrinsics
    @property
    def params(self) -> np.ndarray:
        """(fx, fy, cx, cy, k1, k2, p1, p2, k3)."""
        return np.concatenate([[self.fx, self.fy, self.cx, self.cy],
                               self.dist])

    @classmethod
    def from_params(cls, p: np.ndarray, width: int = 640,
                    height: int = 480) -> "PinholeCamera":
        return cls(p[0], p[1], p[2], p[3], np.array(p[4:9]), width, height)

    def intrinsics_jacobian(self, q: np.ndarray) -> np.ndarray:
        """2x9 Jacobian of the pixel of q (camera frame) with respect to
        (fx, fy, cx, cy, k1, k2, p1, p2, k3)."""
        x, y = q[0] / q[2], q[1] / q[2]
        xd, yd = self.distort(np.array([x, y]))
        r2 = x * x + y * y
        r4, r6 = r2 * r2, r2 * r2 * r2
        Jd = np.array([[x * r2, x * r4, 2 * x * y, r2 + 2 * x * x, x * r6],
                       [y * r2, y * r4, r2 + 2 * y * y, 2 * x * y, y * r6]])
        J = np.zeros((2, 9))
        J[0, 0], J[0, 2] = xd, 1.0
        J[1, 1], J[1, 3] = yd, 1.0
        J[:, 4:] = np.diag([self.fx, self.fy]) @ Jd
        return J

    def unproject(self, u: np.ndarray) -> np.ndarray:
        """Bearing (x, y, 1) of the ray through pixel u."""
        xd = np.array([(u[0] - self.cx) / self.fx,
                       (u[1] - self.cy) / self.fy])
        x = self.undistort(xd)
        return np.array([x[0], x[1], 1.0])

    def in_image(self, u: np.ndarray, margin: float = 0.0) -> np.ndarray:
        """Boolean mask of pixels inside the image."""
        u = np.asarray(u)
        return ((u[..., 0] >= margin) & (u[..., 0] < self.width - margin)
                & (u[..., 1] >= margin) & (u[..., 1] < self.height - margin))


# --------------------------------------------------------------------------
# Rectified stereo pair
# --------------------------------------------------------------------------
@dataclass
class StereoCamera:
    """Rectified stereo pair: two identical undistorted pinholes, the right
    one translated by `baseline` along the left camera's x axis."""

    cam: PinholeCamera
    baseline: float

    def project(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Measurement (u_l, v, u_r) of a point q in the LEFT camera frame,
        and its 3x3 Jacobian with respect to q."""
        fx, fy, cx, cy, b = (self.cam.fx, self.cam.fy, self.cam.cx,
                             self.cam.cy, self.baseline)
        X, Y, Z = q
        iz = 1.0 / Z
        z = np.array([fx * X * iz + cx, fy * Y * iz + cy,
                      fx * (X - b) * iz + cx])
        J = np.array([[fx * iz, 0.0, -fx * X * iz * iz],
                      [0.0, fy * iz, -fy * Y * iz * iz],
                      [fx * iz, 0.0, -fx * (X - b) * iz * iz]])
        return z, J

    def triangulate(self, z: np.ndarray) -> np.ndarray:
        """Point in the left camera frame from (u_l, v, u_r)."""
        ul, v, ur = z
        d = ul - ur  # disparity
        Z = self.cam.fx * self.baseline / d
        X = (ul - self.cam.cx) * Z / self.cam.fx
        Y = (v - self.cam.cy) * Z / self.cam.fy
        return np.array([X, Y, Z])

    def depth_std(self, Z: np.ndarray, sigma_d: float) -> np.ndarray:
        """First-order standard deviation of depth: Z^2 sigma_d / (f b)."""
        return np.asarray(Z) ** 2 * sigma_d / (self.cam.fx * self.baseline)
