"""Corners, binary descriptors and matching (Chapter 5).

Everything is written from scratch with NumPy/SciPy so that each formula
of the chapter has a line of code; OpenCV's ORB (cv2.ORB_create) is the
production equivalent and is used in the notebook for comparison.

Conventions: images are float arrays img[v, u]; keypoints are (N, 2)
arrays of (u, v) = (column, row) pixel coordinates.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as nd


# --------------------------------------------------------------------------
# Structure tensor and corner responses
# --------------------------------------------------------------------------
def gradients(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Image derivatives (I_u, I_v) by Sobel filters, scaled to units of
    intensity per pixel."""
    Iu = nd.sobel(img, axis=1, mode="reflect") / 8.0
    Iv = nd.sobel(img, axis=0, mode="reflect") / 8.0
    return Iu, Iv


def structure_tensor(img: np.ndarray, sigma: float = 1.5
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Entries (A, B, C) of M = G_sigma * [[Iu^2, Iu Iv], [Iu Iv, Iv^2]]."""
    Iu, Iv = gradients(img)
    A = nd.gaussian_filter(Iu * Iu, sigma)
    B = nd.gaussian_filter(Iu * Iv, sigma)
    C = nd.gaussian_filter(Iv * Iv, sigma)
    return A, B, C


def eigenvalues(A, B, C) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvalues (l_min, l_max) of the 2x2 symmetric matrices
    [[A, B], [B, C]], elementwise."""
    mean = 0.5 * (A + C)
    rad = np.sqrt(0.25 * (A - C) ** 2 + B * B)
    return mean - rad, mean + rad


def harris(img: np.ndarray, sigma: float = 1.5, k: float = 0.04
           ) -> np.ndarray:
    """Harris response det(M) - k tr(M)^2."""
    A, B, C = structure_tensor(img, sigma)
    return A * C - B * B - k * (A + C) ** 2


def shi_tomasi(img: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """Shi-Tomasi response: the smaller eigenvalue of M."""
    return eigenvalues(*structure_tensor(img, sigma))[0]


def detect(response: np.ndarray, n: int = 500, min_distance: int = 5,
           border: int = 20, threshold: float = 0.0) -> np.ndarray:
    """The n strongest local maxima of a response map, at least
    `min_distance` pixels apart and `border` pixels from the image edge.
    Returns (n, 2) keypoints (u, v)."""
    size = 2 * min_distance + 1
    peaks = (response == nd.maximum_filter(response, size=size)) \
        & (response > threshold)
    peaks[:border], peaks[-border:] = False, False
    peaks[:, :border], peaks[:, -border:] = False, False
    v, u = np.nonzero(peaks)
    order = np.argsort(response[v, u])[::-1][:n]
    return np.column_stack([u[order], v[order]]).astype(float)


# --------------------------------------------------------------------------
# Oriented BRIEF
# --------------------------------------------------------------------------
def orientation(img: np.ndarray, kps: np.ndarray, radius: int = 15
                ) -> np.ndarray:
    """Intensity-centroid angle of each keypoint: atan2(m01, m10) with
    m_pq = sum over a disc of u^p v^q I(u, v), (u, v) relative to the
    keypoint. Angles are measured from the u axis towards the v axis."""
    du, dv = np.meshgrid(np.arange(-radius, radius + 1),
                         np.arange(-radius, radius + 1))
    disc = du ** 2 + dv ** 2 <= radius ** 2
    du, dv = du[disc].astype(float), dv[disc].astype(float)
    angles = np.empty(len(kps))
    for i, (u, v) in enumerate(kps):
        vals = nd.map_coordinates(img, [v + dv, u + du], order=1,
                                  mode="reflect")
        angles[i] = np.arctan2(np.sum(dv * vals), np.sum(du * vals))
    return angles


def brief_pattern(n_bits: int = 256, patch: int = 31,
                  seed: int = 12345) -> np.ndarray:
    """Test pairs (n_bits, 4) = (u1, v1, u2, v2) drawn i.i.d. from an
    isotropic Gaussian of std patch/5, clipped to the patch."""
    rng = np.random.default_rng(seed)
    half = patch // 2
    pts = rng.normal(0.0, patch / 5.0, size=(n_bits, 4))
    return np.clip(pts, -half, half)


def describe(img: np.ndarray, kps: np.ndarray, angles: np.ndarray | None,
             pattern: np.ndarray | None = None, smooth: float = 2.0
             ) -> np.ndarray:
    """Steered BRIEF: bit k is [I(p + R a_k) < I(p + R b_k)] on the
    smoothed image, R the rotation by the keypoint angle. Returns packed
    descriptors (N, n_bits / 8) of dtype uint8."""
    pattern = brief_pattern() if pattern is None else pattern
    S = nd.gaussian_filter(img, smooth)
    a, b = pattern[:, :2], pattern[:, 2:]
    angles = np.zeros(len(kps)) if angles is None else angles
    bits = np.empty((len(kps), len(pattern)), dtype=bool)
    for i, ((u, v), th) in enumerate(zip(kps, angles)):
        c, s = np.cos(th), np.sin(th)
        R = np.array([[c, -s], [s, c]])
        pa, pb = a @ R.T, b @ R.T
        Ia = nd.map_coordinates(S, [v + pa[:, 1], u + pa[:, 0]], order=1,
                                mode="reflect")
        Ib = nd.map_coordinates(S, [v + pb[:, 1], u + pb[:, 0]], order=1,
                                mode="reflect")
        bits[i] = Ia < Ib
    return np.packbits(bits, axis=1)


_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def hamming(d1: np.ndarray, d2: np.ndarray) -> np.ndarray:
    """Matrix of Hamming distances between packed descriptors."""
    x = np.bitwise_xor(d1[:, None, :], d2[None, :, :])
    return _POPCOUNT[x].sum(axis=2, dtype=np.int32)


def match(d1: np.ndarray, d2: np.ndarray, ratio: float = 0.8,
          mutual: bool = True, max_distance: int | None = None
          ) -> np.ndarray:
    """Nearest-neighbour matching with Lowe's ratio test and an optional
    mutual-consistency check. Returns (M, 2) index pairs (i, j)."""
    D = hamming(d1, d2)
    order = np.argsort(D, axis=1)
    best, second = order[:, 0], order[:, 1]
    rows = np.arange(len(d1))
    d_best, d_second = D[rows, best], D[rows, second]
    keep = d_best < ratio * d_second
    if mutual:
        keep &= np.argmin(D, axis=0)[best] == rows
    if max_distance is not None:
        keep &= d_best <= max_distance
    return np.column_stack([rows[keep], best[keep]])
