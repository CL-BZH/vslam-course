"""Trajectory evaluation (Chapter 7): alignment, ATE and RPE."""

from __future__ import annotations

import numpy as np

from . import lie
from .calib import nearest_rotation


def align(est: np.ndarray, gt: np.ndarray, with_scale: bool = False
          ) -> tuple[np.ndarray, np.ndarray, float]:
    """Least-squares similarity (R, t, s) minimising
    sum |gt_i - (s R est_i + t)|^2 (Horn/Umeyama). Positions (N, 3)."""
    me, mg = est.mean(axis=0), gt.mean(axis=0)
    E, G = est - me, gt - mg
    C = G.T @ E                       # sum (g_i - mg)(e_i - me)^T
    R = nearest_rotation(C)
    s = np.trace(R.T @ C) / np.sum(E * E) if with_scale else 1.0
    t = mg - s * R @ me
    return R, t, s


def ate(est: np.ndarray, gt: np.ndarray, with_scale: bool = False
        ) -> np.ndarray:
    """Absolute trajectory errors (per pose, metres) after alignment."""
    R, t, s = align(est, gt, with_scale)
    return np.linalg.norm(gt - (s * est @ R.T + t), axis=1)


def rpe(T_est: list[np.ndarray], T_gt: list[np.ndarray], delta: float
        ) -> tuple[np.ndarray, np.ndarray]:
    """Relative pose errors over segments of `delta` metres travelled.

    For each i, j is the first index with path length from i at least
    delta; the error is E = (T_gt_i^-1 T_gt_j)^-1 (T_est_i^-1 T_est_j).
    Returns (translation errors in metres, rotation errors in radians).
    """
    pos = np.array([T[:3, 3] for T in T_gt])
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pos, axis=0),
                                                        axis=1))])
    te, re = [], []
    for i in range(len(T_gt)):
        j = np.searchsorted(s, s[i] + delta)
        if j >= len(T_gt):
            break
        D_gt = lie.se3_inv(T_gt[i]) @ T_gt[j]
        D_est = lie.se3_inv(T_est[i]) @ T_est[j]
        E = lie.se3_inv(D_gt) @ D_est
        te.append(np.linalg.norm(E[:3, 3]))
        re.append(np.linalg.norm(lie.so3_log(E[:3, :3])))
    return np.array(te), np.array(re)
