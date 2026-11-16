"""Robust kernels (M-estimators) for least squares (Chapter 4).

A kernel rho acts on the squared whitened norm s = |r|^2 of a residual
block; the robust cost is F = 1/2 sum_i rho(s_i). Each kernel provides
rho(s) and the weight w(s) = rho'(s) used by iteratively reweighted least
squares: since grad F = sum_i w(s_i) J_i^T r_i, a stationary point of F is
a stationary point of the weighted problem 1/2 sum_i w_i |r_i|^2 with the
weights frozen at that point.
"""

from __future__ import annotations

import numpy as np


class Trivial:
    """Plain least squares: rho(s) = s."""

    def rho(self, s: np.ndarray) -> np.ndarray:
        return np.asarray(s, dtype=float)

    def weight(self, s: np.ndarray) -> np.ndarray:
        return np.ones_like(np.asarray(s, dtype=float))


class Huber:
    """Quadratic up to |r| = k, linear beyond.

    rho(s) = s for s <= k^2, 2 k sqrt(s) - k^2 otherwise.
    """

    def __init__(self, k: float) -> None:
        self.k = k

    def rho(self, s: np.ndarray) -> np.ndarray:
        s = np.asarray(s, dtype=float)
        k = self.k
        return np.where(s <= k * k, s, 2.0 * k * np.sqrt(s) - k * k)

    def weight(self, s: np.ndarray) -> np.ndarray:
        s = np.asarray(s, dtype=float)
        k = self.k
        return np.where(s <= k * k, 1.0, k / np.sqrt(np.maximum(s, 1e-300)))


class Cauchy:
    """rho(s) = k^2 log(1 + s / k^2): logarithmic growth, redescending
    influence. Not convex: needs a reasonable starting point."""

    def __init__(self, k: float) -> None:
        self.k = k

    def rho(self, s: np.ndarray) -> np.ndarray:
        k2 = self.k * self.k
        return k2 * np.log1p(np.asarray(s, dtype=float) / k2)

    def weight(self, s: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.asarray(s, dtype=float) / (self.k * self.k))
