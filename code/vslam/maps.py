"""Dense maps for mowing (Chapter 11).

From stereo depth and keyframe poses to the grid a coverage planner needs:

1. Stereo depth -> points in the camera frame, with their covariance
   (rank one, along the ray).
2. The ground plane of each keyframe, by RANSAC and a total-least-squares
   refit, and the height of every point above it.
3. Two log-odds layers on a fine grid: occupancy (obstacle vs. drivable
   ground) and lawn (grass vs. other ground), one vote per cell and
   keyframe.
4. A class per fine cell (mowable, obstacle, not lawn, unknown), then a
   coarse grid of blade-sized cells, exported in the text format of the
   Lawnmower coverage planners ('L' lawn, 'S' start, '+' blocked).

Keyframes keep their classified points in their own frame, so the grid can
be rebuilt whenever the pose graph moves the keyframes (`fuse`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .camera import StereoCamera

MOWABLE, OBSTACLE, NOT_LAWN, UNKNOWN = 0, 1, 2, 3


# --------------------------------------------------------------------------
# Stereo points
# --------------------------------------------------------------------------
def stereo_points(uv: np.ndarray, depth: np.ndarray, stereo: StereoCamera
                  ) -> np.ndarray:
    """Points q = Z (x, y, 1) in the left camera frame."""
    cam = stereo.cam
    x = (uv[:, 0] - cam.cx) / cam.fx
    y = (uv[:, 1] - cam.cy) / cam.fy
    return depth[:, None] * np.column_stack([x, y, np.ones(len(x))])


def point_covariance(q: np.ndarray, stereo: StereoCamera, sigma_d: float
                     ) -> np.ndarray:
    """First-order covariance of stereo points due to disparity noise:
    sigma_d^2 (Z^2 / f b)^2 r r^T with r = q / Z (rank one, along the
    ray). Shape (N, 3, 3)."""
    Z = q[:, 2]
    r = q / Z[:, None]
    s = (Z ** 2 / (stereo.cam.fx * stereo.baseline) * sigma_d) ** 2
    return s[:, None, None] * r[:, :, None] * r[:, None, :]


# --------------------------------------------------------------------------
# Ground plane
# --------------------------------------------------------------------------
@dataclass
class Plane:
    n: np.ndarray            # unit normal, pointing up (towards the camera)
    d: float                 # plane: n^T q = d

    def height(self, q: np.ndarray) -> np.ndarray:
        """Signed distance n^T q - d (positive above the ground)."""
        return q @ self.n - self.d


def fit_plane_tls(P: np.ndarray) -> Plane:
    """Total least squares plane: through the centroid, normal = the
    eigenvector of the scatter matrix with the smallest eigenvalue. The
    normal is oriented so that the origin (the camera) is above."""
    c = P.mean(axis=0)
    _, V = np.linalg.eigh((P - c).T @ (P - c))
    n = V[:, 0]
    d = float(n @ c)
    if d > 0:                # origin must have positive height: -d > 0
        n, d = -n, -d
    return Plane(n, d)


def nominal_ground(T_bc: np.ndarray) -> Plane:
    """The ground z = 0 of the body frame, seen from the camera: n^T q = d
    with n = R_bc^T e_z and d = -e_z^T t_bc."""
    n = T_bc[:3, :3].T @ np.array([0.0, 0.0, 1.0])
    return Plane(n, -float(T_bc[2, 3]))


def ground_plane(q: np.ndarray, prior: Plane, rng: np.random.Generator,
                 threshold: float = 0.05, iters: int = 100,
                 max_tilt_deg: float = 10.0, max_shift: float = 0.3,
                 max_range: float = 6.0) -> tuple[Plane, np.ndarray]:
    """RANSAC on three-point planes close to the prior (tilt and offset),
    then a TLS refit on the inliers. Returns the plane and the inlier mask
    over q."""
    near = np.nonzero(q[:, 2] < max_range)[0]
    Q = q[near]
    best, best_in = prior, np.abs(prior.height(Q)) < threshold
    cos_max = np.cos(np.radians(max_tilt_deg))
    for _ in range(iters):
        a, b, c = Q[rng.choice(len(Q), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        if n @ prior.n < 0:
            n = -n
        pl = Plane(n, float(n @ a))
        if n @ prior.n < cos_max or abs(pl.d - prior.d) > max_shift:
            continue
        inl = np.abs(pl.height(Q)) < threshold
        if inl.sum() > best_in.sum():
            best, best_in = pl, inl
    pl = fit_plane_tls(Q[best_in]) if best_in.sum() >= 3 else best
    mask = np.zeros(len(q), dtype=bool)
    mask[near] = np.abs(pl.height(Q)) < threshold
    return pl, mask


# --------------------------------------------------------------------------
# Keyframe observations
# --------------------------------------------------------------------------
@dataclass
class KeyframeCloud:
    q: np.ndarray            # (N, 3) points in the camera frame
    kind: np.ndarray         # (N,) 0 ground, 1 obstacle
    grass: np.ndarray        # (N,) bool, segmentation (ground points only)


def classify_points(q: np.ndarray, grass: np.ndarray, plane: Plane,
                    h_obstacle: float = 0.15, h_top: float = 1.5,
                    max_range: float = 4.0) -> KeyframeCloud:
    """Ground (|h| < h_obstacle) or obstacle (h_obstacle <= h <= h_top);
    points beyond `max_range`, above the mower (h > h_top) or below the
    ground are dropped."""
    h = plane.height(q)
    keep = (q[:, 2] < max_range) & (h > -h_obstacle) & (h <= h_top)
    kind = (h >= h_obstacle).astype(int)
    return KeyframeCloud(q[keep], kind[keep], grass[keep])


# --------------------------------------------------------------------------
# Grid
# --------------------------------------------------------------------------
def logit(p: float) -> float:
    return float(np.log(p / (1 - p)))


@dataclass
class GridSpec:
    x0: float = -2.0
    x1: float = 14.0
    y0: float = -2.0
    y1: float = 10.0
    res: float = 0.1

    @property
    def shape(self) -> tuple[int, int]:
        return (int(round((self.y1 - self.y0) / self.res)),
                int(round((self.x1 - self.x0) / self.res)))

    def cells(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Row (y) and column (x) of points, and a mask of those inside."""
        c = np.floor((xy[:, 0] - self.x0) / self.res).astype(int)
        r = np.floor((xy[:, 1] - self.y0) / self.res).astype(int)
        H, W = self.shape
        ok = (r >= 0) & (r < H) & (c >= 0) & (c < W)
        return r, c, ok


class LogOddsGrid:
    """Two binary Bayes filters per cell, in log-odds: P(occupied) and
    P(lawn). Each keyframe gives at most one occupancy vote and one lawn
    vote per cell (its points in a cell are not independent)."""

    def __init__(self, spec: GridSpec, p_hit: float = 0.8,
                 p_free: float = 0.3, p_grass: float = 0.75,
                 clamp: float = 6.0, min_obstacle_frac: float = 0.3
                 ) -> None:
        self.spec = spec
        self.occ = np.zeros(spec.shape)
        self.lawn = np.zeros(spec.shape)
        self.seen = np.zeros(spec.shape, dtype=int)
        self.l_hit, self.l_free = logit(p_hit), logit(p_free)
        self.l_grass = logit(p_grass)
        self.clamp, self.frac = clamp, min_obstacle_frac

    def update(self, T_wc: np.ndarray, kf: KeyframeCloud) -> None:
        p = kf.q @ T_wc[:3, :3].T + T_wc[:3, 3]
        r, c, ok = self.spec.cells(p[:, :2])
        H, W = self.spec.shape
        idx = r[ok] * W + c[ok]
        kind, grass = kf.kind[ok], kf.grass[ok]
        n = np.bincount(idx, minlength=H * W)
        n_obs = np.bincount(idx, weights=kind, minlength=H * W)
        ground = kind == 0
        n_g = np.bincount(idx[ground], minlength=H * W)
        n_gr = np.bincount(idx[ground], weights=grass[ground],
                           minlength=H * W)
        hit = n_obs >= np.maximum(1, self.frac * n)
        free = (n > 0) & ~hit
        occ, lawn = self.occ.ravel(), self.lawn.ravel()
        occ[hit] += self.l_hit
        occ[free] += self.l_free
        vote = n_g > 0
        lawn[vote] += np.where(n_gr[vote] > 0.5 * n_g[vote],
                               self.l_grass, -self.l_grass)
        np.clip(occ, -self.clamp, self.clamp, out=occ)
        np.clip(lawn, -self.clamp, self.clamp, out=lawn)
        self.seen.ravel()[n > 0] += 1

    def classes(self, p_occ: float = 0.7, p_free: float = 0.2,
                p_lawn: float = 0.8) -> np.ndarray:
        """MOWABLE: confidently free and lawn; OBSTACLE: confidently
        occupied; NOT_LAWN: confidently free and not lawn; UNKNOWN: the
        rest (never seen, or undecided)."""
        out = np.full(self.spec.shape, UNKNOWN)
        free = self.occ < logit(p_free)
        out[self.occ > logit(p_occ)] = OBSTACLE
        out[free & (self.lawn > logit(p_lawn))] = MOWABLE
        out[free & (self.lawn < -logit(p_lawn))] = NOT_LAWN
        return out


def fuse(spec: GridSpec, poses: list[np.ndarray],
         clouds: list[KeyframeCloud], **kw) -> LogOddsGrid:
    """Rebuild the grid from keyframe clouds and (possibly corrected)
    keyframe poses."""
    g = LogOddsGrid(spec, **kw)
    for T, kf in zip(poses, clouds):
        g.update(T, kf)
    return g


# --------------------------------------------------------------------------
# Planner grid
# --------------------------------------------------------------------------
def coarsen(cls: np.ndarray, factor: int, min_mowable: float = 0.9
            ) -> np.ndarray:
    """Blade-sized cells from `factor` x `factor` fine cells: a coarse cell
    is mowable (True) if at least `min_mowable` of its fine cells are
    mowable and none is an obstacle. Unknown counts as not mowable."""
    H, W = cls.shape
    H2, W2 = H // factor, W // factor
    b = cls[:H2 * factor, :W2 * factor].reshape(H2, factor, W2, factor)
    frac = (b == MOWABLE).mean(axis=(1, 3))
    obstacle = (b == OBSTACLE).any(axis=(1, 3))
    return (frac >= min_mowable) & ~obstacle


def to_lawn_text(mowable: np.ndarray, start: tuple[int, int] | None = None
                 ) -> str:
    """Text map for the Lawnmower planners: one character per cell, 'L'
    lawn, 'S' start, '+' blocked, with a border of '+'. `mowable` has row
    0 at the south (y0); the text has north at the top. `start` is (row,
    col) in `mowable`."""
    H, W = mowable.shape
    grid = np.full((H + 2, W + 2), "+")
    grid[1:-1, 1:-1] = np.where(mowable, "L", "+")
    if start is not None:
        grid[start[0] + 1, start[1] + 1] = "S"
    return "\n".join("".join(row) for row in grid[::-1]) + "\n"
