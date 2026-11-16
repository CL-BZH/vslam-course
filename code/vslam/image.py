"""Synthetic images for the front-end chapters (Chapter 5 onwards).

Two procedural textures stand in for what the mower's camera sees:

* `garden_texture`: man-made structure (rectangles, discs, lines) on a
  smooth background -- fences, a shed, flower beds, a house wall;
* `lawn_texture`: thousands of short, similar blades of grass with random
  shading -- a high-frequency, repetitive texture.

Images are float arrays in [0, 1], indexed img[row, col] = img[v, u].
A second view of a planar texture is obtained with a homography, so that
the ground-truth correspondence of every pixel is known exactly.
"""

from __future__ import annotations

import cv2
import numpy as np


def _smooth_background(h: int, w: int, rng: np.random.Generator,
                       lo: float, hi: float) -> np.ndarray:
    """Low-frequency shading: a few broad Gaussian blobs."""
    v, u = np.mgrid[0:h, 0:w]
    img = np.full((h, w), 0.5 * (lo + hi))
    for _ in range(6):
        cu, cv = rng.uniform(0, w), rng.uniform(0, h)
        s = rng.uniform(0.2, 0.6) * max(h, w)
        img += rng.uniform(-1, 1) * 0.5 * (hi - lo) * np.exp(
            -((u - cu) ** 2 + (v - cv) ** 2) / (2 * s * s))
    return np.clip(img, lo, hi)


def garden_texture(h: int = 480, w: int = 640,
                   rng: np.random.Generator | None = None) -> np.ndarray:
    """Structured scene: shapes of many sizes and contrasts."""
    rng = rng or np.random.default_rng(0)
    img = _smooth_background(h, w, rng, 0.3, 0.7)
    for _ in range(60):
        kind = rng.integers(3)
        col = float(rng.uniform(0.0, 1.0))
        if kind == 0:
            x0, y0 = rng.integers(0, w), rng.integers(0, h)
            x1 = x0 + rng.integers(10, 120)
            y1 = y0 + rng.integers(10, 120)
            cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), col,
                          -1, lineType=cv2.LINE_AA)
        elif kind == 1:
            c = (int(rng.integers(0, w)), int(rng.integers(0, h)))
            cv2.circle(img, c, int(rng.integers(5, 40)), col, -1,
                       lineType=cv2.LINE_AA)
        else:
            p0 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
            p1 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
            cv2.line(img, p0, p1, col, int(rng.integers(1, 5)),
                     lineType=cv2.LINE_AA)
    img += rng.normal(0, 0.01, img.shape)
    return np.clip(cv2.GaussianBlur(img, (0, 0), 0.7), 0, 1)


def lawn_texture(h: int = 480, w: int = 640,
                 rng: np.random.Generator | None = None,
                 density: float = 0.08, sway: float = 0.0,
                 sway_rng: np.random.Generator | None = None) -> np.ndarray:
    """Grass: many short, thin, nearly vertical blades of random shade.

    `sway` (radians) rotates every blade about its root by an independent
    N(0, sway^2) angle drawn from `sway_rng`: rendering the same `rng`
    seed with two different sway draws gives the same lawn moved by the
    wind (or brushed by the mower).
    """
    rng = rng or np.random.default_rng(0)
    img = _smooth_background(h, w, rng, 0.35, 0.55)
    n = int(density * h * w)
    us, vs = rng.uniform(-10, w + 10, n), rng.uniform(-10, h + 10, n)
    lengths = rng.uniform(5, 16, n)
    angles = np.pi / 2 + rng.normal(0, 0.35, n)
    shades = np.clip(rng.normal(0.45, 0.15, n), 0, 1)
    if sway > 0:
        angles = angles + (sway_rng or np.random.default_rng(1)).normal(
            0.0, sway, n)
    for u, v, L, a, s in zip(us, vs, lengths, angles, shades):
        p0 = (int(u), int(v))
        p1 = (int(u + L * np.cos(a)), int(v - L * np.sin(a)))
        cv2.line(img, p0, p1, float(s), 1, lineType=cv2.LINE_AA)
    img += rng.normal(0, 0.01, img.shape)
    return np.clip(cv2.GaussianBlur(img, (0, 0), 0.6), 0, 1)


def warp(img: np.ndarray, H: np.ndarray,
         shape: tuple[int, int] | None = None) -> np.ndarray:
    """Image of `img` under the homography H (pixel (u, v, 1) of the
    source maps to H (u, v, 1) in the result), bilinear interpolation."""
    h, w = shape or img.shape
    return cv2.warpPerspective(img.astype(np.float32), H, (w, h),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT).astype(float)


def apply_homography(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Map 2D points (N, 2) through H."""
    p = np.column_stack([pts, np.ones(len(pts))]) @ H.T
    return p[:, :2] / p[:, 2:3]


def plane_homography(K: np.ndarray, angle: float, axis: str = "y",
                     distance: float = 3.0) -> np.ndarray:
    """Homography between two views of the fronto-parallel plane Z = d.

    The first camera is K[I|0]; the second orbits about the point (0, 0, d)
    of the plane by `angle` radians about the given axis, keeping it at the
    image centre. For points X on the plane, X2 = (R + t n^T / d) X1, so
    H = K (R + t n^T / d) K^{-1}.
    """
    from . import lie

    e = {"x": 0, "y": 1, "z": 2}[axis]
    w = np.zeros(3)
    w[e] = angle
    R_wc2 = lie.so3_exp(w)
    centre = np.array([0.0, 0.0, distance])
    c2 = centre - R_wc2 @ centre  # orbit: same distance to the centre
    R = R_wc2.T
    t = -R @ c2
    n = np.array([0.0, 0.0, 1.0])
    H = K @ (R + np.outer(t, n) / distance) @ np.linalg.inv(K)
    return H / H[2, 2]


def illumination(img: np.ndarray, gain: float = 1.0, offset: float = 0.0,
                 gamma: float = 1.0, noise: float = 0.0,
                 rng: np.random.Generator | None = None) -> np.ndarray:
    """Photometric change: (gain * img + offset) ** gamma plus noise."""
    rng = rng or np.random.default_rng(0)
    out = np.clip(gain * img + offset, 0, 1) ** gamma
    if noise > 0:
        out = out + rng.normal(0, noise, img.shape)
    return np.clip(out, 0, 1)
