"""Pyramidal Lucas-Kanade tracking (Chapter 5).

For a window W around each point p of the first image I0 we seek the
displacement d minimising sum_{x in W} (I1(x + d) - I0(x))^2. One
Gauss-Newton step, with the gradient of I0 standing in for that of I1, is

    M delta = -sum_x grad I0(x) (I1(x + d) - I0(x)),
    M = sum_x grad I0(x) grad I0(x)^T          (the structure tensor),

and d <- d + delta. A Gaussian pyramid handles displacements larger than
the window: the estimate from a coarse level, doubled, initialises the
next finer level.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as nd

from .features import gradients


def pyramid(img: np.ndarray, levels: int) -> list[np.ndarray]:
    """Gaussian pyramid, finest first; each level halves the resolution."""
    pyr = [img]
    for _ in range(levels - 1):
        pyr.append(nd.gaussian_filter(pyr[-1], 1.0)[::2, ::2])
    return pyr


def lucas_kanade(img0: np.ndarray, img1: np.ndarray, pts: np.ndarray,
                 half_win: int = 7, levels: int = 3, iters: int = 20,
                 eps: float = 0.01, min_eig: float = 1e-4
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Track points (N, 2) of img0 into img1.

    Returns the tracked positions (N, 2) and a status mask: False when the
    window is textureless (smallest eigenvalue of M per pixel below
    `min_eig`), the iteration did not converge, or the point left the
    image.
    """
    pts = np.asarray(pts, dtype=float)
    du, dv = np.meshgrid(np.arange(-half_win, half_win + 1),
                         np.arange(-half_win, half_win + 1))
    du, dv = du.ravel().astype(float), dv.ravel().astype(float)
    P0, P1 = pyramid(img0, levels), pyramid(img1, levels)
    d = np.zeros_like(pts)
    ok = np.ones(len(pts), dtype=bool)
    for lev in range(levels - 1, -1, -1):
        scale = 2.0 ** lev
        I0, I1 = P0[lev], P1[lev]
        Iu, Iv = gradients(I0)
        p = pts / scale
        # template, its gradients and the structure tensor, per point
        uu, vv = p[:, :1] + du, p[:, 1:] + dv
        T = nd.map_coordinates(I0, [vv, uu], order=1, mode="nearest")
        gu = nd.map_coordinates(Iu, [vv, uu], order=1, mode="nearest")
        gv = nd.map_coordinates(Iv, [vv, uu], order=1, mode="nearest")
        a, b, c = (gu * gu).sum(1), (gu * gv).sum(1), (gv * gv).sum(1)
        det = a * c - b * b
        lmin = 0.5 * (a + c) - np.sqrt(0.25 * (a - c) ** 2 + b * b)
        textured = lmin / du.size > min_eig
        dl = d / scale
        converged = np.zeros(len(pts), dtype=bool)
        for _ in range(iters):
            W = nd.map_coordinates(I1, [vv + dl[:, 1:], uu + dl[:, :1]],
                                   order=1, mode="nearest")
            e = W - T
            bu, bv = (gu * e).sum(1), (gv * e).sum(1)
            safe = np.where(textured, det, 1.0)
            step_u = -(c * bu - b * bv) / safe
            step_v = -(-b * bu + a * bv) / safe
            step = np.column_stack([step_u, step_v]) * textured[:, None]
            dl = dl + step * ~converged[:, None]
            converged |= np.hypot(step_u, step_v) < eps
        d = dl * scale
        if lev == 0:
            ok &= textured & converged
    new = pts + d
    h, w = img1.shape
    ok &= ((new[:, 0] >= 0) & (new[:, 0] <= w - 1)
           & (new[:, 1] >= 0) & (new[:, 1] <= h - 1))
    return new, ok
