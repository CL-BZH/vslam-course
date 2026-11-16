"""A dense model of the garden for the mapping chapter (Chapter 11).

The landmark simulation of `vslam.sim` gives sparse points. To build maps
for mowing we need what a dense stereo matcher returns: a depth for (almost)
every pixel. This module describes the garden with a few solid primitives
(ground, axis-aligned boxes, vertical cylinders), ray-casts them, and adds
stereo noise in disparity.

Ground classes (z = 0): the lawn [0, 12] x [0, 8] except a paved patio in
one corner and a flower bed; a gravel border between the lawn and the
fence. The lawn surface seen by stereo is the top of the grass blades, a
random height in [0, grass_height].

Labels of a pixel: see LABELS.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .camera import StereoCamera

LABELS = ["sky", "lawn", "border", "patio", "fence", "house", "shed", "tree",
          "bed"]
GROUND = {"lawn", "border", "patio"}
L = {name: k for k, name in enumerate(LABELS)}


@dataclass
class Box:
    lo: tuple[float, float, float]
    hi: tuple[float, float, float]
    label: str


@dataclass
class Cylinder:
    c: tuple[float, float]
    r: float
    h: float
    label: str


def _default_boxes() -> list[Box]:
    t = 0.05
    return [
        Box((-1 - t, -1 - t, 0), (13 + t, -1 + t, 1.2), "fence"),
        Box((13 - t, -1 - t, 0), (13 + t, 9 + t, 1.2), "fence"),
        Box((-1 - t, 9 - t, 0), (13 + t, 9 + t, 1.2), "fence"),
        Box((-1 - t, -1 - t, 0), (-1 + t, 9 + t, 1.2), "fence"),
        Box((-2, 12, 0), (10, 15, 4.0), "house"),
        Box((14, 2, 0), (16, 5, 2.2), "shed"),
        Box((8.6, 4.45, 0), (10.2, 5.05, 0.3), "bed"),    # flower bed
    ]


def _default_trees() -> list[Cylinder]:
    outside = [(3, -3), (8, -2.5), (15, 7), (-3, 4), (6, 11)]
    return ([Cylinder(c, 0.25, 3.0, "tree") for c in outside]
            + [Cylinder((5.5, 3.25), 0.15, 2.5, "tree")])   # in the lawn


@dataclass
class Garden:
    boxes: list[Box] = field(default_factory=_default_boxes)
    trees: list[Cylinder] = field(default_factory=_default_trees)
    lawn: tuple[float, float, float, float] = (0.0, 12.0, 0.0, 8.0)
    patio: tuple[float, float, float, float] = (0.0, 3.0, 6.3, 8.0)
    grass_height: float = 0.08

    # ------------------------------------------------------------ ground
    def ground_label(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Label of the ground at (x, y): lawn, patio or border."""
        x0, x1, y0, y1 = self.lawn
        lab = np.full(x.shape, L["border"])
        inl = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
        lab[inl] = L["lawn"]
        px0, px1, py0, py1 = self.patio
        lab[inl & (x >= px0) & (x <= px1) & (y >= py0) & (y <= py1)] = \
            L["patio"]
        return lab

    def truth_grid(self, extent, res: float) -> np.ndarray:
        """Ground-truth class of each cell of a grid: 0 mowable lawn,
        1 obstacle (anything a mower cannot drive through), 2 drivable but
        not lawn (border, patio). `extent` = (x0, x1, y0, y1)."""
        x0, x1, y0, y1 = extent
        xs = x0 + res * (np.arange(int(round((x1 - x0) / res))) + 0.5)
        ys = y0 + res * (np.arange(int(round((y1 - y0) / res))) + 0.5)
        X, Y = np.meshgrid(xs, ys)          # rows = y, columns = x
        lab = self.ground_label(X, Y)
        out = np.where(lab == L["lawn"], 0, 2)
        h = res / 2                         # a cell is an obstacle if it
        for b in self.boxes:                # overlaps one (open sets)
            inside = ((X - h < b.hi[0]) & (X + h > b.lo[0])
                      & (Y - h < b.hi[1]) & (Y + h > b.lo[1]))
            out[inside] = 1
        for t in self.trees:                # nearest point of the cell
            dx = np.clip(t.c[0], X - h, X + h) - t.c[0]
            dy = np.clip(t.c[1], Y - h, Y + h) - t.c[1]
            out[np.hypot(dx, dy) < t.r] = 1
        return out

    # ------------------------------------------------------------ ray casting
    def raycast(self, o: np.ndarray, d: np.ndarray,
                rng: np.random.Generator | None = None
                ) -> tuple[np.ndarray, np.ndarray]:
        """Distance along unit rays d (N, 3) from o (3,) to the first
        surface, and its label. Rays that hit nothing within 60 m get
        inf and label "sky"."""
        n = len(d)
        best = np.full(n, np.inf)
        lab = np.zeros(n, dtype=int)
        # ground, then grass tops on the lawn
        with np.errstate(divide="ignore", invalid="ignore"):
            t = -o[2] / d[:, 2]
        t[~(t > 0)] = np.inf
        with np.errstate(invalid="ignore"):
            p = o + t[:, None] * d
        g = self.ground_label(p[:, 0], p[:, 1])
        if rng is not None and self.grass_height > 0:
            hit = np.isfinite(t) & (g == L["lawn"])
            h = rng.uniform(0, self.grass_height, n)
            t[hit] = (h[hit] - o[2]) / d[hit, 2]
        best, lab = np.where(t < best, t, best), np.where(t < best, g, lab)
        # boxes: slab test
        for b in self.boxes:
            lo, hi = np.array(b.lo), np.array(b.hi)
            with np.errstate(divide="ignore", invalid="ignore"):
                t1 = (lo - o) / d
                t2 = (hi - o) / d
            tmin = np.nanmax(np.minimum(t1, t2), axis=1)
            tmax = np.nanmin(np.maximum(t1, t2), axis=1)
            ok = (tmax >= tmin) & (tmin > 0) & (tmin < best)
            best[ok], lab[ok] = tmin[ok], L[b.label]
        # vertical cylinders
        for c in self.trees:
            ox, oy = o[0] - c.c[0], o[1] - c.c[1]
            a = d[:, 0] ** 2 + d[:, 1] ** 2
            bq = 2 * (ox * d[:, 0] + oy * d[:, 1])
            cq = ox ** 2 + oy ** 2 - c.r ** 2
            disc = bq ** 2 - 4 * a * cq
            with np.errstate(invalid="ignore", divide="ignore"):
                tc = (-bq - np.sqrt(disc)) / (2 * a)
            z = o[2] + tc * d[:, 2]
            ok = (disc > 0) & (tc > 0) & (z >= 0) & (z <= c.h) & (tc < best)
            best[ok], lab[ok] = tc[ok], L[c.label]
        far = best > 60.0
        best[far], lab[far] = np.inf, L["sky"]
        return best, lab


@dataclass
class DepthFrame:
    uv: np.ndarray          # (N, 2) pixel coordinates (left image)
    depth: np.ndarray       # (N,) measured depth Z (nan = no measurement)
    label: np.ndarray       # (N,) true label of the surface
    true_depth: np.ndarray  # (N,) noise-free depth (inf for sky)


def render_stereo(garden: Garden, stereo: StereoCamera, T_wc: np.ndarray,
                  rng: np.random.Generator, stride: int = 4,
                  sigma_d: float = 0.3, min_disp: float = 1.0,
                  dropout: float = 0.05) -> DepthFrame:
    """Depth of every `stride`-th pixel as a dense stereo matcher would
    return it: the true disparity f b / Z plus Gaussian noise of
    `sigma_d` pixels, a fraction `dropout` of failed matches, and no
    depth below `min_disp` pixels of disparity."""
    cam = stereo.cam
    us = np.arange(stride / 2, cam.width, stride)
    vs = np.arange(stride / 2, cam.height, stride)
    U, V = np.meshgrid(us, vs)
    uv = np.column_stack([U.ravel(), V.ravel()])
    rays_c = np.column_stack([(uv[:, 0] - cam.cx) / cam.fx,
                              (uv[:, 1] - cam.cy) / cam.fy,
                              np.ones(len(uv))])
    norm = np.linalg.norm(rays_c, axis=1)
    d_w = (rays_c / norm[:, None]) @ T_wc[:3, :3].T
    dist, lab = garden.raycast(T_wc[:3, 3], d_w, rng)
    Z = dist / norm                       # depth along the optical axis
    fb = cam.fx * stereo.baseline
    disp = fb / Z + rng.normal(0, sigma_d, len(Z))
    meas = np.where(disp > min_disp, fb / np.maximum(disp, 1e-9), np.nan)
    meas[rng.random(len(Z)) < dropout] = np.nan
    return DepthFrame(uv, meas, lab, Z)


def segment_grass(label: np.ndarray, rng: np.random.Generator,
                  p_detect: float = 0.9, p_false: float = 0.1
                  ) -> np.ndarray:
    """A simulated grass classifier: says 'grass' with probability
    `p_detect` on lawn pixels and `p_false` on everything else."""
    p = np.where(label == L["lawn"], p_detect, p_false)
    return rng.random(len(label)) < p
