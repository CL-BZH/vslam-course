"""RANSAC (Chapter 5).

Random sample consensus fits a model from minimal random samples and keeps
the one with the largest consensus set. With an inlier ratio w and samples
of size s, a sample is outlier-free with probability w^s, so

    N = log(1 - p) / log(1 - w^s)

samples give at least one outlier-free sample with probability p. The
number is updated as better consensus sets reveal a higher w (adaptive
termination). The final model is refitted on all inliers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .calib import homography_dlt
from .image import apply_homography


def required_iterations(w: float, s: int, p: float = 0.99) -> int:
    """Number of samples needed to draw one all-inlier sample of size s
    with probability p when the inlier ratio is w."""
    if w >= 1.0:
        return 1
    ws = w ** s if w > 0.0 else 0.0
    if ws <= 0.0:
        return np.iinfo(np.int64).max
    # log1p keeps precision when w^s is tiny (1 - w^s would round to 1)
    return int(np.ceil(np.log(1.0 - p) / np.log1p(-ws)))


@dataclass
class RansacResult:
    model: Any
    inliers: np.ndarray
    iterations: int


def ransac(n: int, fit: Callable[[np.ndarray], Any],
           residuals: Callable[[Any], np.ndarray], sample_size: int,
           threshold: float, p: float = 0.99, max_iters: int = 10000,
           rng: np.random.Generator | None = None) -> RansacResult:
    """Generic RANSAC over n data indices.

    `fit(idx)` returns a model from the data at indices idx (or None if
    degenerate); `residuals(model)` returns the n residual magnitudes; a
    datum is an inlier when its residual is below `threshold`.
    """
    rng = rng or np.random.default_rng()
    best_model, best_in = None, np.zeros(n, dtype=bool)
    needed, it = max_iters, 0
    while it < min(needed, max_iters):
        it += 1
        idx = rng.choice(n, size=sample_size, replace=False)
        model = fit(idx)
        if model is None:
            continue
        inl = residuals(model) < threshold
        if inl.sum() > best_in.sum():
            best_model, best_in = model, inl
            needed = required_iterations(inl.mean(), sample_size, p)
    if best_model is not None and best_in.sum() >= sample_size:
        refit = fit(np.nonzero(best_in)[0])
        if refit is not None:
            best_model = refit
            best_in = residuals(refit) < threshold
    return RansacResult(best_model, best_in, it)


def homography_ransac(src: np.ndarray, dst: np.ndarray,
                      threshold: float = 2.0, p: float = 0.99,
                      rng: np.random.Generator | None = None
                      ) -> RansacResult:
    """Homography dst ~ H src from noisy correspondences with outliers,
    using the transfer error |dst - H(src)| in pixels."""

    def fit(idx):
        try:
            H = homography_dlt(src[idx], dst[idx])
        except np.linalg.LinAlgError:
            return None
        return H if np.all(np.isfinite(H)) else None

    def residuals(H):
        return np.linalg.norm(apply_homography(H, src) - dst, axis=1)

    return ransac(len(src), fit, residuals, 4, threshold, p, rng=rng)
