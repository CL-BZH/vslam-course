"""Inertial measurement units and preintegration (Chapter 12).

Conventions:

* World frame W with z up and gravity g = (0, 0, -9.81).
* The body (IMU) pose is (R, p) with R = R_WB, p = p_WB; v = velocity of
  the body in W.
* Measurements: gyro  w~ = w + b_g + n_g      (w: angular rate, body frame)
                accel a~ = R^T (a - g) + b_a + n_a   (specific force)
* Perturbations: R <- R Exp(dphi), v <- v + dv, p <- p + R dp,
  b <- b + db. A 15-dim perturbation is ordered (dphi, dv, dp, dbg, dba);
  the preintegration noise and the IMU residual are ordered (phi, v, p).

Preintegration follows C. Forster, L. Carlone, F. Dellaert, D. Scaramuzza,
"On-manifold preintegration for real-time visual-inertial odometry",
IEEE T-RO 33(1), 2017. Every Jacobian is checked numerically in
tests/test_imu.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .lie import (hat, so3_exp, so3_log, so3_right_jacobian,
                  so3_right_jacobian_inv)

GRAVITY = np.array([0.0, 0.0, -9.81])


# --------------------------------------------------------------------------
# Noise parameters
# --------------------------------------------------------------------------
@dataclass
class ImuParams:
    """Continuous-time noise densities.

    Defaults: the values distributed with the EuRoC MAV dataset for its
    ADIS16448 (gyro rad/s/sqrt(Hz), accel m/s^2/sqrt(Hz), bias random
    walks rad/s^2/sqrt(Hz) and m/s^3/sqrt(Hz)), at 200 Hz.
    """
    sigma_g: float = 1.6968e-4
    sigma_a: float = 2.0e-3
    sigma_bg: float = 1.9393e-5
    sigma_ba: float = 3.0e-3
    rate: float = 200.0

    @property
    def dt(self) -> float:
        return 1.0 / self.rate

    def discrete(self) -> tuple[float, float, float, float]:
        """Standard deviations of one sample: white noise sigma/sqrt(dt),
        bias increments sigma_b sqrt(dt)."""
        dt = self.dt
        return (self.sigma_g / np.sqrt(dt), self.sigma_a / np.sqrt(dt),
                self.sigma_bg * np.sqrt(dt), self.sigma_ba * np.sqrt(dt))


# --------------------------------------------------------------------------
# Trajectories and simulated measurements
# --------------------------------------------------------------------------
def euler_zyx(yaw: float, pitch: float, roll: float) -> np.ndarray:
    return (so3_exp(np.array([0.0, 0.0, yaw]))
            @ so3_exp(np.array([0.0, pitch, 0.0]))
            @ so3_exp(np.array([roll, 0.0, 0.0])))


@dataclass
class Trajectory:
    """A smooth body trajectory t -> (R(t), p(t)). Angular rate and
    acceleration are obtained by central differences of the closed forms
    (step 1e-3 s: truncation ~1e-7, far below any IMU noise)."""
    pose: Callable[[float], tuple[np.ndarray, np.ndarray]]
    h: float = 1e-3

    def R(self, t: float) -> np.ndarray:
        return self.pose(t)[0]

    def p(self, t: float) -> np.ndarray:
        return self.pose(t)[1]

    def v(self, t: float) -> np.ndarray:
        h = self.h
        return (self.p(t + h) - self.p(t - h)) / (2 * h)

    def acc(self, t: float) -> np.ndarray:
        h = self.h
        return (self.p(t + h) - 2 * self.p(t) + self.p(t - h)) / h ** 2

    def omega(self, t: float) -> np.ndarray:
        """Body-frame angular rate: R^T R' = hat(w)."""
        h = self.h
        return so3_log(self.R(t - h).T @ self.R(t + h)) / (2 * h)


def lawn_trajectory(speed: float = 1.0, row: float = 10.0,
                    bump: float = 0.02, bump_len: float = 0.7,
                    sway_deg: float = 1.5) -> Trajectory:
    """Driving along a lawn row and back, smoothly: x oscillates between
    0 and `row` (x = row/2 (1 - cos(pi s t / row))... with a smooth turn),
    the ground is bumpy (height `bump`, wavelength `bump_len`), and the
    body pitches and rolls with the bumps by about `sway_deg`."""
    w = np.pi * speed / row             # rad/s: one row in pi/w seconds

    def pose(t):
        x = 0.5 * row * (1 - np.cos(w * t))
        y = 0.75 * (1 - np.cos(w * t / 2))       # drift sideways (rows)
        k = 2 * np.pi / bump_len
        z = bump * np.sin(k * x) * np.cos(0.8 * k * y)
        dxdt = 0.5 * row * w * np.sin(w * t)
        dydt = 0.75 * 0.5 * w * np.sin(w * t / 2)
        yaw = np.arctan2(dydt, dxdt + 1e-3)
        s = np.radians(sway_deg)
        pitch = -s * np.cos(k * x) * np.cos(0.8 * k * y)
        roll = s * np.sin(0.8 * k * y + 0.3) * np.sin(k * x)
        return euler_zyx(yaw, pitch, roll), np.array([x, y, z])

    return Trajectory(pose)


def wavy_trajectory() -> Trajectory:
    """A generic smooth 3D motion with all six degrees of freedom excited
    (used by the tests)."""
    def pose(t):
        R = euler_zyx(0.5 * t + 0.3 * np.sin(1.3 * t), 0.2 * np.sin(0.7 * t),
                      0.25 * np.cos(1.1 * t))
        p = np.array([2 * np.sin(0.4 * t), np.cos(0.5 * t) + 0.3 * t,
                      0.2 * np.sin(0.9 * t)])
        return R, p
    return Trajectory(pose)


@dataclass
class ImuData:
    t: np.ndarray            # (n,) sample times
    gyro: np.ndarray         # (n, 3)
    accel: np.ndarray        # (n, 3)
    bg: np.ndarray           # (n, 3) true gyro bias
    ba: np.ndarray           # (n, 3) true accel bias
    w_true: np.ndarray       # (n, 3) noise-free angular rate
    f_true: np.ndarray       # (n, 3) noise-free specific force


def simulate(traj: Trajectory, duration: float, params: ImuParams,
             rng: np.random.Generator, bg0=None, ba0=None, t0: float = 0.0,
             noise: bool = True) -> ImuData:
    """Sample the IMU at params.rate over [t0, t0 + duration). Sample k
    is taken at t_k and is held constant over [t_k, t_k + dt)."""
    dt = params.dt
    n = int(round(duration / dt))
    t = t0 + dt * np.arange(n)
    # sample at the middle of each interval: the best constant value
    w = np.array([traj.omega(tk + dt / 2) for tk in t])
    f = np.array([traj.R(tk + dt / 2).T @ (traj.acc(tk + dt / 2) - GRAVITY)
                  for tk in t])
    sg, sa, sbg, sba = params.discrete()
    bg = np.zeros((n, 3)) if bg0 is None else np.tile(bg0, (n, 1)).astype(
        float)
    ba = np.zeros((n, 3)) if ba0 is None else np.tile(ba0, (n, 1)).astype(
        float)
    if noise:
        bg = bg + np.cumsum(np.r_[np.zeros((1, 3)),
                                  rng.normal(0, sbg, (n - 1, 3))], axis=0)
        ba = ba + np.cumsum(np.r_[np.zeros((1, 3)),
                                  rng.normal(0, sba, (n - 1, 3))], axis=0)
        gyro = w + bg + rng.normal(0, sg, (n, 3))
        accel = f + ba + rng.normal(0, sa, (n, 3))
    else:
        gyro, accel = w + bg, f + ba
    return ImuData(t, gyro, accel, bg, ba, w, f)


# --------------------------------------------------------------------------
# Allan deviation
# --------------------------------------------------------------------------
def allan_deviation(x: np.ndarray, dt: float, taus: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Overlapping Allan deviation of a rate signal x (n,) sampled every
    dt, at averaging times tau = m dt (rounded to whole samples)."""
    theta = np.r_[0.0, np.cumsum(x)] * dt        # integrated signal
    n = len(x)
    out_t, out_s = [], []
    for tau in taus:
        m = int(round(tau / dt))
        if m < 1 or 2 * m >= n:
            continue
        d = theta[2 * m:] - 2 * theta[m:-m] + theta[:-2 * m]
        out_t.append(m * dt)
        out_s.append(np.sqrt(np.mean(d ** 2) / (2 * (m * dt) ** 2)))
    return np.array(out_t), np.array(out_s)


# --------------------------------------------------------------------------
# Strapdown integration
# --------------------------------------------------------------------------
@dataclass
class NavState:
    R: np.ndarray
    v: np.ndarray
    p: np.ndarray


def step(s: NavState, w: np.ndarray, a: np.ndarray, dt: float,
         g: np.ndarray = GRAVITY) -> NavState:
    """One step with constant (bias-corrected) body rate w and specific
    force a over dt. The rotation is exact for a constant body rate; the
    velocity and position use the rotation at the start of the step."""
    acc = s.R @ a + g
    return NavState(s.R @ so3_exp(w * dt), s.v + acc * dt,
                    s.p + s.v * dt + 0.5 * acc * dt ** 2)


def dead_reckon(s0: NavState, data: ImuData, bg: np.ndarray,
                ba: np.ndarray, g: np.ndarray = GRAVITY) -> list[NavState]:
    dt = data.t[1] - data.t[0]
    out = [s0]
    for w, a in zip(data.gyro, data.accel):
        out.append(step(out[-1], w - bg, a - ba, dt, g))
    return out


# --------------------------------------------------------------------------
# Preintegration
# --------------------------------------------------------------------------
@dataclass
class Preintegration:
    """Preintegrated measurements between keyframes i and j, at the
    linearisation biases bg, ba."""
    bg: np.ndarray
    ba: np.ndarray
    params: ImuParams = field(default_factory=ImuParams)
    dR: np.ndarray = field(default_factory=lambda: np.eye(3))
    dv: np.ndarray = field(default_factory=lambda: np.zeros(3))
    dp: np.ndarray = field(default_factory=lambda: np.zeros(3))
    dt: float = 0.0
    cov: np.ndarray = field(default_factory=lambda: np.zeros((9, 9)))
    J_R_bg: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    J_v_bg: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    J_v_ba: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    J_p_bg: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    J_p_ba: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))

    def integrate(self, w_meas: np.ndarray, a_meas: np.ndarray,
                  dt: float) -> None:
        """Add one sample (Theorem 12.7 for the measurements,
        Proposition 12.8 for the covariance, Proposition 12.9 for the
        bias Jacobians). Order matters: everything uses the values at k
        before they are updated to k + 1."""
        w = w_meas - self.bg
        a = a_meas - self.ba
        dRk = so3_exp(w * dt)
        Jr = so3_right_jacobian(w * dt)
        ah = hat(a)
        R = self.dR
        # covariance: eta_{k+1} = A eta_k + B n_k
        A = np.eye(9)
        A[0:3, 0:3] = dRk.T
        A[3:6, 0:3] = -R @ ah * dt
        A[6:9, 0:3] = -0.5 * R @ ah * dt ** 2
        A[6:9, 3:6] = np.eye(3) * dt
        B = np.zeros((9, 6))
        B[0:3, 0:3] = Jr * dt
        B[3:6, 3:6] = R * dt
        B[6:9, 3:6] = 0.5 * R * dt ** 2
        sg2 = self.params.sigma_g ** 2 / dt
        sa2 = self.params.sigma_a ** 2 / dt
        Q = np.diag([sg2] * 3 + [sa2] * 3)
        self.cov = A @ self.cov @ A.T + B @ Q @ B.T
        # bias Jacobians (p first: it uses the v Jacobians at k)
        self.J_p_ba = self.J_p_ba + self.J_v_ba * dt - 0.5 * R * dt ** 2
        self.J_p_bg = (self.J_p_bg + self.J_v_bg * dt
                       - 0.5 * R @ ah @ self.J_R_bg * dt ** 2)
        self.J_v_ba = self.J_v_ba - R * dt
        self.J_v_bg = self.J_v_bg - R @ ah @ self.J_R_bg * dt
        self.J_R_bg = dRk.T @ self.J_R_bg - Jr * dt
        # measurements
        self.dp = self.dp + self.dv * dt + 0.5 * R @ a * dt ** 2
        self.dv = self.dv + R @ a * dt
        self.dR = R @ dRk
        self.dt += dt

    @classmethod
    def from_data(cls, data: ImuData, bg, ba, params: ImuParams,
                  k0: int = 0, k1: int | None = None) -> "Preintegration":
        pre = cls(np.asarray(bg, float), np.asarray(ba, float), params)
        dt = params.dt
        for w, a in zip(data.gyro[k0:k1], data.accel[k0:k1]):
            pre.integrate(w, a, dt)
        return pre

    # ------------------------------------------------------------ bias
    def corrected(self, bg: np.ndarray, ba: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """First-order update of the measurements for new biases."""
        dbg, dba = bg - self.bg, ba - self.ba
        dR = self.dR @ so3_exp(self.J_R_bg @ dbg)
        dv = self.dv + self.J_v_bg @ dbg + self.J_v_ba @ dba
        dp = self.dp + self.J_p_bg @ dbg + self.J_p_ba @ dba
        return dR, dv, dp

    # ------------------------------------------------------------ predict
    def predict(self, s: NavState, bg=None, ba=None,
                g: np.ndarray = GRAVITY) -> NavState:
        """State at j from the state at i (Theorem 12.7)."""
        bg = self.bg if bg is None else bg
        ba = self.ba if ba is None else ba
        dR, dv, dp = self.corrected(bg, ba)
        T = self.dt
        return NavState(s.R @ dR, s.v + g * T + s.R @ dv,
                        s.p + s.v * T + 0.5 * g * T ** 2 + s.R @ dp)

    # ------------------------------------------------------------ factor
    def residual(self, si: NavState, sj: NavState, bg: np.ndarray,
                 ba: np.ndarray, g: np.ndarray = GRAVITY) -> np.ndarray:
        """r = (r_R, r_v, r_p) of the IMU factor (Definition 12.10)."""
        dR, dv, dp = self.corrected(bg, ba)
        T = self.dt
        Ri = si.R
        rR = so3_log(dR.T @ Ri.T @ sj.R)
        rv = Ri.T @ (sj.v - si.v - g * T) - dv
        rp = Ri.T @ (sj.p - si.p - si.v * T - 0.5 * g * T ** 2) - dp
        return np.concatenate([rR, rv, rp])

    def jacobians(self, si: NavState, sj: NavState, bg: np.ndarray,
                  ba: np.ndarray, g: np.ndarray = GRAVITY
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Jacobians of the residual (9) with respect to the perturbations
        of state i (dphi, dv, dp: 9), state j (9) and the biases at i
        (dbg, dba: 6) (Proposition 12.11)."""
        T = self.dt
        Ri, Rj = si.R, sj.R
        r = self.residual(si, sj, bg, ba, g)
        rR = r[:3]
        Jrinv = so3_right_jacobian_inv(rR)
        dbg = bg - self.bg
        Ji = np.zeros((9, 9))
        Jj = np.zeros((9, 9))
        Jb = np.zeros((9, 6))
        # rotation
        Ji[0:3, 0:3] = -Jrinv @ Rj.T @ Ri
        Jj[0:3, 0:3] = Jrinv
        Jb[0:3, 0:3] = (-Jrinv @ so3_exp(rR).T
                        @ so3_right_jacobian(self.J_R_bg @ dbg) @ self.J_R_bg)
        # velocity
        Ji[3:6, 0:3] = hat(Ri.T @ (sj.v - si.v - g * T))
        Ji[3:6, 3:6] = -Ri.T
        Jj[3:6, 3:6] = Ri.T
        Jb[3:6, 0:3] = -self.J_v_bg
        Jb[3:6, 3:6] = -self.J_v_ba
        # position
        Ji[6:9, 0:3] = hat(Ri.T @ (sj.p - si.p - si.v * T
                                   - 0.5 * g * T ** 2))
        Ji[6:9, 3:6] = -Ri.T * T
        Ji[6:9, 6:9] = -np.eye(3)
        Jj[6:9, 6:9] = Ri.T @ Rj
        Jb[6:9, 0:3] = -self.J_p_bg
        Jb[6:9, 3:6] = -self.J_p_ba
        return Ji, Jj, Jb
