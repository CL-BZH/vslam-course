"""Evaluating a localisation system (Chapter 15).

* ATE with an explicit choice of alignment: none (the estimate claims to
  be in the reference frame), 4-DoF (a VIO: yaw and translation are its
  gauge), SE(3) or Sim(3) (vision only, Chapter 7).
* RPE over a travelled distance, as in the KITTI benchmark.
* NEES and its average over a run (ANEES), with chi-square bounds: does
  the estimator know how wrong it is?
"""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2

from . import fusion, lie, metrics


def ate(est: np.ndarray, gt: np.ndarray, alignment: str = "none"
        ) -> np.ndarray:
    """Per-pose absolute position errors (N,) after the chosen alignment:
    "none", "4dof", "se3" or "sim3"."""
    if alignment == "none":
        a = est
    elif alignment == "4dof":
        yaw, t = fusion.align_4dof(est, gt)
        a = est @ lie.so3_exp(np.array([0.0, 0.0, yaw])).T + t
    elif alignment in ("se3", "sim3"):
        R, t, s = metrics.align(est, gt, with_scale=alignment == "sim3")
        a = s * est @ R.T + t
    else:
        raise ValueError(alignment)
    return np.linalg.norm(gt - a, axis=1)


def rpe_distance(T_est: list[np.ndarray], T_gt: list[np.ndarray],
                 d: float) -> tuple[np.ndarray, np.ndarray]:
    """Relative errors over segments of travelled length d: for each pose
    i, the first j with path length(i, j) >= d; returns the translation
    errors (m) and rotation errors (rad) of T_i^-1 T_j."""
    p = np.array([T[:3, 3] for T in T_gt])
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    et, er = [], []
    for i in range(len(T_gt)):
        j = np.searchsorted(s, s[i] + d)
        if j >= len(T_gt):
            break
        Ee = lie.se3_inv(T_est[i]) @ T_est[j]
        Eg = lie.se3_inv(T_gt[i]) @ T_gt[j]
        D = lie.se3_inv(Eg) @ Ee
        et.append(np.linalg.norm(D[:3, 3]))
        er.append(np.linalg.norm(lie.so3_log(D[:3, :3])))
    return np.array(et), np.array(er)


def nees(err: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """e^T P^-1 e for errors (N, d) and covariances (N, d, d)."""
    return np.einsum("ni,ni->n", err, np.linalg.solve(cov, err[..., None])
                     [..., 0])


def anees_bounds(n: int, d: int, conf: float = 0.95
                 ) -> tuple[float, float]:
    """Two-sided bounds on the average of n independent chi^2_d variables
    (Proposition 15.3)."""
    a = (1 - conf) / 2
    return chi2.ppf(a, n * d) / n, chi2.ppf(1 - a, n * d) / n


def nis_gate(r: np.ndarray, S: np.ndarray, p: float = 0.999
             ) -> tuple[float, bool]:
    """Normalised innovation squared of a residual r with predicted
    covariance S, and whether it passes the chi^2 gate at level p
    (Proposition 15.4)."""
    q = float(r @ np.linalg.solve(S, r))
    return q, q <= chi2.ppf(p, len(r))
