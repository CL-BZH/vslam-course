"""Wheels and satellites (Chapter 14).

* WGS84 geodetic coordinates, ECEF and a local east-north-up (ENU) frame.
* Differential-drive odometry: kinematics, dead reckoning and its
  covariance.
* Factors on one VIO keyframe state (see vio.SlidingWindowVIO, `unary`):
  - WheelFactor: wheel speed along the body x axis and the non-holonomic
    constraint (no sideways or vertical velocity);
  - GnssFactor: RTK position of an antenna at a lever arm from the body
    origin.
* A simulated RTK receiver (fix, float under canopy, multipath), wheel
  encoders with slip, and the 4-DoF alignment of a local frame to ENU.

States are perturbed as in vio: R Exp(dphi), v + dv, p + R dp.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .lie import hat, so3_exp

# --------------------------------------------------------------------------
# Geodesy
# --------------------------------------------------------------------------
WGS84_A = 6378137.0                    # semi-major axis [m]
WGS84_F = 1 / 298.257223563            # flattening
WGS84_E2 = WGS84_F * (2 - WGS84_F)     # first eccentricity squared


def geodetic_to_ecef(lat: float, lon: float, h: float) -> np.ndarray:
    """Latitude, longitude (radians) and ellipsoidal height to ECEF
    (Proposition 14.4)."""
    s, c = np.sin(lat), np.cos(lat)
    N = WGS84_A / np.sqrt(1 - WGS84_E2 * s * s)
    return np.array([(N + h) * c * np.cos(lon), (N + h) * c * np.sin(lon),
                     (N * (1 - WGS84_E2) + h) * s])


def ecef_to_geodetic(x: np.ndarray, iters: int = 6
                     ) -> tuple[float, float, float]:
    """Inverse of geodetic_to_ecef by fixed-point iteration on the
    latitude (converges to below a millimetre in a few steps)."""
    X, Y, Z = x
    lon = np.arctan2(Y, X)
    r = np.hypot(X, Y)
    lat = np.arctan2(Z, r * (1 - WGS84_E2))
    for _ in range(iters):
        s = np.sin(lat)
        N = WGS84_A / np.sqrt(1 - WGS84_E2 * s * s)
        h = r / np.cos(lat) - N
        lat = np.arctan2(Z, r * (1 - WGS84_E2 * N / (N + h)))
    s = np.sin(lat)
    N = WGS84_A / np.sqrt(1 - WGS84_E2 * s * s)
    return lat, lon, r / np.cos(lat) - N


def R_enu_ecef(lat: float, lon: float) -> np.ndarray:
    """Rotation from ECEF to the local east-north-up frame at (lat, lon):
    rows are the east, north and up unit vectors in ECEF."""
    sl, cl = np.sin(lat), np.cos(lat)
    so, co = np.sin(lon), np.cos(lon)
    return np.array([[-so, co, 0.0],
                     [-sl * co, -sl * so, cl],
                     [cl * co, cl * so, sl]])


@dataclass
class LocalFrame:
    """A local ENU frame anchored at a geodetic reference point."""
    lat: float
    lon: float
    h: float

    def __post_init__(self) -> None:
        self.o = geodetic_to_ecef(self.lat, self.lon, self.h)
        self.R = R_enu_ecef(self.lat, self.lon)

    def to_enu(self, lat, lon, h) -> np.ndarray:
        return self.R @ (geodetic_to_ecef(lat, lon, h) - self.o)

    def to_geodetic(self, enu: np.ndarray) -> tuple[float, float, float]:
        return ecef_to_geodetic(self.o + self.R.T @ enu)


# --------------------------------------------------------------------------
# Differential drive
# --------------------------------------------------------------------------
def diff_drive(v_l, v_r, track: float):
    """Forward speed and yaw rate from the speeds of the two wheels."""
    return 0.5 * (v_l + v_r), (v_r - v_l) / track


def wheel_speeds(v, w, track: float):
    """Inverse of diff_drive."""
    return v - 0.5 * w * track, v + 0.5 * w * track


def dead_reckon_2d(v: np.ndarray, w: np.ndarray, dt: float,
                   x0=(0.0, 0.0, 0.0), sigma_v: float = 0.0,
                   sigma_w: float = 0.0):
    """Integrate (x, y, yaw) with the midpoint heading, and propagate the
    covariance of white speed and yaw-rate errors (Proposition 14.2).
    Returns poses (n + 1, 3) and covariances (n + 1, 3, 3)."""
    x = np.array(x0, float)
    P = np.zeros((3, 3))
    out, covs = [x.copy()], [P.copy()]
    Q = np.diag([sigma_v ** 2, sigma_w ** 2]) / dt      # density -> sample
    for vk, wk in zip(v, w):
        th = x[2] + 0.5 * wk * dt
        c, s = np.cos(th), np.sin(th)
        F = np.array([[1, 0, -vk * dt * s], [0, 1, vk * dt * c], [0, 0, 1]])
        G = np.array([[dt * c, -0.5 * vk * dt ** 2 * s],
                      [dt * s, 0.5 * vk * dt ** 2 * c], [0, dt]])
        x = x + np.array([vk * dt * c, vk * dt * s, wk * dt])
        P = F @ P @ F.T + G @ Q @ G.T
        out.append(x.copy())
        covs.append(P.copy())
    return np.array(out), np.array(covs)


# --------------------------------------------------------------------------
# Factors on one keyframe state
# --------------------------------------------------------------------------
class WheelFactor:
    """Body-frame velocity of the wheel axle centre (the body origin):
    forward speed measured by the encoders, zero sideways and vertical
    speed (Proposition 14.3). r = diag(1/sigma) (R^T v - (s, 0, 0))."""

    def __init__(self, speed: float, sigma=(0.02, 0.05, 0.05),
                 kernel=None) -> None:
        self.s = float(speed)
        self.Wh = np.diag(1.0 / np.asarray(sigma, float))
        self.kernel = kernel

    def __call__(self, st):
        vb = st.R.T @ st.v
        r = self.Wh @ (vb - np.array([self.s, 0.0, 0.0]))
        J = np.zeros((3, 15))
        J[:, 0:3] = self.Wh @ hat(vb)
        J[:, 3:6] = self.Wh @ st.R.T
        return r, J


class GnssFactor:
    """ENU position z of the antenna at lever arm t_ant (body frame):
    r = L^T (p + R t_ant - z), L L^T the information (Proposition 14.5)."""

    def __init__(self, z: np.ndarray, sigma, lever: np.ndarray,
                 kernel=None) -> None:
        self.z = np.asarray(z, float)
        cov = np.diag(np.asarray(sigma, float) ** 2) if np.ndim(sigma) == 1 \
            else np.asarray(sigma, float)
        self.Wh = np.linalg.cholesky(np.linalg.inv(cov)).T
        self.t = np.asarray(lever, float)
        self.kernel = kernel

    def __call__(self, st):
        r = self.Wh @ (st.p + st.R @ self.t - self.z)
        J = np.zeros((3, 15))
        J[:, 0:3] = -self.Wh @ st.R @ hat(self.t)
        J[:, 6:9] = self.Wh @ st.R
        return r, J


# --------------------------------------------------------------------------
# Simulated sensors
# --------------------------------------------------------------------------
@dataclass
class Encoders:
    """Wheel encoders of a differential drive: per-wheel scale errors (tyre
    radius), white noise, and slip on wet grass in tight turns (the wheels
    turn faster than the mower moves)."""
    track: float = 1.0
    scale_l: float = 0.01
    scale_r: float = -0.008
    sigma: float = 0.01                 # m/s per sample
    slip: float = 0.06                  # fraction, when |yaw rate| > w_slip
    w_slip: float = 0.8

    def measure(self, v: np.ndarray, w: np.ndarray,
                rng: np.random.Generator):
        vl, vr = wheel_speeds(v, w, self.track)
        k = 1 + self.slip * (np.abs(w) > self.w_slip)
        vl_m = (1 + self.scale_l) * k * vl + rng.normal(0, self.sigma, len(v))
        vr_m = (1 + self.scale_r) * k * vr + rng.normal(0, self.sigma, len(v))
        return vl_m, vr_m


@dataclass
class RTKReceiver:
    """An RTK-GNSS receiver at `rate` Hz.

    Open sky: RTK fix, white errors (sigma_fix horizontal, twice that
    vertically). Under the canopy of a tree (within `canopy_r` of a trunk)
    or near the house: float solution, a slowly wandering error of a few
    decimetres (random walk) and a reported sigma that is too small; and
    multipath outliers of one to three metres with probability
    `p_outlier` per sample in those zones."""
    sigma_fix: float = 0.015
    sigma_float: float = 0.25
    walk: float = 0.08                 # m / sqrt(s), float error
    reported_float: float = 0.10       # what the receiver claims
    p_outlier: float = 0.15
    canopy: tuple = ((5.5, 3.25),)     # tree positions (x, y)
    canopy_r: float = 2.5
    house_y: float = 6.2               # near the house: y > house_y, x < 4
    rate: float = 4.0

    def degraded(self, p: np.ndarray) -> bool:
        near_tree = any(np.hypot(p[0] - c[0], p[1] - c[1]) < self.canopy_r
                        for c in self.canopy)
        return near_tree or (p[1] > self.house_y and p[0] < 4.0)

    def measure(self, positions: np.ndarray, rng: np.random.Generator):
        """Positions of the antenna (n, 3) at the receiver's epochs ->
        (z (n, 3), reported sigma (n, 3), status (n,): 0 fix, 1 float,
        2 outlier)."""
        n = len(positions)
        z = np.empty((n, 3))
        sig = np.empty((n, 3))
        status = np.zeros(n, dtype=int)
        bias = np.zeros(3)
        dt = 1.0 / self.rate
        for k, p in enumerate(positions):
            if self.degraded(p):
                bias = bias + rng.normal(0, self.walk * np.sqrt(dt), 3)
                bias = np.clip(bias, -2 * self.sigma_float,
                               2 * self.sigma_float)
                e = bias + rng.normal(0, 0.03, 3)
                status[k] = 1
                if rng.random() < self.p_outlier:
                    d = rng.normal(size=3)
                    d[2] *= 0.5
                    e = e + rng.uniform(1, 3) * d / np.linalg.norm(d)
                    status[k] = 2
                rep = self.reported_float
            else:
                bias = 0.8 * bias                    # back to fix
                e = rng.normal(0, self.sigma_fix, 3) * [1, 1, 2]
                rep = self.sigma_fix
            z[k] = p + e
            sig[k] = np.array([rep, rep, 2 * rep])
        return z, sig, status


# --------------------------------------------------------------------------
# Frame alignment
# --------------------------------------------------------------------------
def align_4dof(src: np.ndarray, dst: np.ndarray
               ) -> tuple[float, np.ndarray]:
    """Yaw angle and translation minimising sum |dst - (Rz(yaw) src + t)|^2
    (Proposition 14.6). Returns (yaw, t)."""
    ms, md = src.mean(axis=0), dst.mean(axis=0)
    a, b = src - ms, dst - md
    num = np.sum(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])
    den = np.sum(a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1])
    yaw = np.arctan2(num, den)
    Rz = so3_exp(np.array([0.0, 0.0, yaw]))
    return yaw, md - Rz @ ms
