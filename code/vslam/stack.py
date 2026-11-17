"""The mower's localisation stack (Chapter 15).

One object that runs, for every keyframe:

1. the sliding-window VIO (Chapter 13) with the wheel factor (Chapter 14);
2. a chi-square gate on the RTK-GNSS position, against the VIO's own
   prediction and covariance (Proposition 15.4); an accepted fix becomes
   a GNSS factor (robust kernel as a second line of defence) and the
   window is re-optimised;
3. a health report: the position covariance, the visual support, the
   GNSS decision, and a flag that a mower would use to stop.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import evaluation, fusion, vio
from .lie import hat
from .robust import Cauchy, Huber


@dataclass
class Report:
    t: float
    state: vio.KFState
    cov_p: np.ndarray            # 3 x 3 world-frame position covariance
    var_yaw: float               # variance of the heading (rad^2)
    n_obs: int                   # visual observations of the keyframe
    gnss: str                    # "none", "accepted" or "rejected"
    nis: float                   # NIS of the GNSS fix (nan if none)
    healthy: bool


def world_position_cov(s: vio.KFState, C: np.ndarray) -> np.ndarray:
    """Position covariance in the world frame (p <- p + R dp)."""
    return s.R @ C[6:9, 6:9] @ s.R.T


def yaw_variance(s: vio.KFState, C: np.ndarray) -> float:
    """Variance of the rotation about the world vertical (R Exp(dphi):
    the world-frame rotation error is R dphi)."""
    return float((s.R @ C[0:3, 0:3] @ s.R.T)[2, 2])


class LocalisationStack:
    def __init__(self, stereo, T_bc, params, lever: np.ndarray,
                 wheels: bool = True, gnss: bool = True,
                 gate: float | None = 0.999, sigma_alarm: float = 0.10,
                 inflate: float = 1.0, gnss_kernel=Cauchy(2.0),
                 **vio_kw) -> None:
        self.vio = vio.SlidingWindowVIO(stereo, T_bc, params, **vio_kw)
        self.lever = np.asarray(lever, float)
        self.wheels, self.gnss = wheels, gnss
        self.gate, self.alarm = gate, sigma_alarm
        self.inflate = inflate          # covariance inflation (from NEES)
        self.kernel = gnss_kernel
        self.reports: list[Report] = []

    def _wheel(self, speed):
        return [fusion.WheelFactor(speed, kernel=Huber(2.0))] \
            if self.wheels and speed is not None else []

    def start(self, t, state, sigmas, ids, z, speed=None) -> None:
        self.vio.initialise(t, state, sigmas, ids, z, self._wheel(speed))

    def step(self, t, gyro, accel, ids, z, speed=None, fix=None) -> Report:
        """One keyframe. `fix` = (z_enu (3,), sigma (3,)) or None."""
        V = self.vio
        V.add_keyframe(t, gyro, accel, ids, z, self._wheel(speed))
        status, q = "none", np.nan
        if self.gnss and fix is not None:
            zg, sg = fix
            s = V.kfs[-1]
            ok = True
            if self.gate is not None:
                C = self.inflate * V.covariance(-1)
                J = np.zeros((3, 15))
                J[:, 0:3] = -s.R @ hat(self.lever)
                J[:, 6:9] = s.R
                S = J @ C @ J.T + np.diag(np.asarray(sg, float) ** 2)
                r = s.p + s.R @ self.lever - zg
                q, ok = evaluation.nis_gate(r, S, self.gate)
            if ok:
                V.add_unary(fusion.GnssFactor(zg, sg, self.lever,
                                              kernel=self.kernel), iters=3)
                status = "accepted"
            else:
                status = "rejected"
        s = V.kfs[-1]
        C = self.inflate * V.covariance(-1)
        cp = world_position_cov(s, C)
        sig_h = np.sqrt(np.linalg.eigvalsh(cp[:2, :2]).max())
        rep = Report(t, s.copy(), cp, yaw_variance(s, C), len(V.obs[-1]),
                     status, q, bool(sig_h < self.alarm))
        self.reports.append(rep)
        return rep
