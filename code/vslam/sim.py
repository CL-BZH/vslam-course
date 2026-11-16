"""A simulated garden for the back-end chapters (Chapter 7 onwards).

The front-end of Chapter 5 is replaced by a geometric simulation of its
output: every frame, each visible landmark may be tracked, and a tracked
landmark yields a noisy rectified-stereo measurement (u_l, v, u_r) under
a TRACK id. Tracks end at random (sooner on grass, which the wind moves)
and a landmark seen again later gets a NEW track id, exactly like a real
tracker without loop closure. A small fraction of measurements are wrong
(random pixels), more on grass.

World frame: z up, the lawn is the rectangle [0, 12] x [0, 8] of the
plane z = 0. Mower body frame: x forward, y left, z up.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import lie
from .camera import PinholeCamera, StereoCamera


@dataclass
class Landmarks:
    points: np.ndarray        # (N, 3) world coordinates
    grass: np.ndarray         # (N,) bool: on the lawn (moves in the wind)
    kind: np.ndarray | None = None   # (N,) str: fence, house, shed, tree,
                                     # grass


def garden(rng: np.random.Generator | None = None,
           grass_density: float = 3.0) -> Landmarks:
    """Fences around the lawn, a house wall, a shed, trees, and sparse
    points on the grass itself."""
    rng = rng or np.random.default_rng(0)
    pts = []
    kinds = []

    def wall(p0, p1, h0, h1, density, kind):
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        length = np.linalg.norm(p1 - p0)
        n = int(density * length * (h1 - h0))
        s = rng.uniform(0, 1, n)
        xy = p0 + s[:, None] * (p1 - p0)
        pts.append(np.column_stack([xy, rng.uniform(h0, h1, n)]))
        kinds.append(np.full(n, kind))

    def trunk(c, r, h, density):
        n = int(density * 2 * np.pi * r * h)
        a = rng.uniform(0, 2 * np.pi, n)
        pts.append(np.column_stack([c[0] + r * np.cos(a), c[1] + r * np.sin(a),
                                    rng.uniform(0, h, n)]))
        kinds.append(np.full(n, "tree"))

    for p0, p1 in [((-1, -1), (13, -1)), ((13, -1), (13, 9)),
                   ((13, 9), (-1, 9)), ((-1, 9), (-1, -1))]:
        wall(p0, p1, 0.0, 1.2, 12.0, "fence")
    wall((-2, 12), (10, 12), 0.0, 4.0, 4.0, "house")
    wall((14, 2), (14, 5), 0.0, 2.2, 8.0, "shed")
    for c in [(3, -3), (8, -2.5), (15, 7), (-3, 4), (6, 11)]:
        trunk(c, 0.25, 3.0, 25.0)                         # trees
    structure = np.vstack(pts)
    n_g = int(grass_density * 12 * 8)
    grass = np.column_stack([rng.uniform(0, 12, n_g), rng.uniform(0, 8, n_g),
                             np.zeros(n_g)])
    points = np.vstack([structure, grass])
    flag = np.concatenate([np.zeros(len(structure), bool),
                           np.ones(n_g, bool)])
    kind = np.concatenate(kinds + [np.full(n_g, "grass")])
    return Landmarks(points, flag, kind)


def boustrophedon(rows: list[float] | None = None, x0: float = 1.0,
                  x1: float = 11.0, step: float = 0.05) -> list[np.ndarray]:
    """Body poses T_wb along back-and-forth lawn rows joined by
    semicircular U-turns, sampled every `step` metres."""
    rows = rows or [1.0, 2.5, 4.0, 5.5, 7.0]
    poses = []

    def add(x, y, yaw):
        poses.append(lie.make_T(lie.so3_exp(np.array([0.0, 0.0, yaw])),
                                np.array([x, y, 0.0])))

    for i, y in enumerate(rows):
        forward = i % 2 == 0
        xs = np.arange(x0, x1, step) if forward else np.arange(x1, x0, -step)
        for x in xs:
            add(x, y, 0.0 if forward else np.pi)
        if i + 1 < len(rows):
            r = 0.5 * (rows[i + 1] - y)
            cx = x1 if forward else x0
            n = int(np.pi * r / step)
            for a in np.linspace(0, np.pi, n, endpoint=False):
                if forward:   # turn left around (x1, y + r)
                    add(cx + r * np.sin(a), y + r - r * np.cos(a), a)
                else:         # turn right around (x0, y + r)
                    add(cx - r * np.sin(a), y + r - r * np.cos(a), np.pi - a)
    return poses


def camera_mount(height: float = 0.8, forward: float = 0.3,
                 pitch_deg: float = 15.0) -> np.ndarray:
    """T_bc of a forward-looking camera pitched down by `pitch_deg`."""
    p = np.radians(pitch_deg)
    z_c = np.array([np.cos(p), 0.0, -np.sin(p)])   # optical axis
    x_c = np.array([0.0, -1.0, 0.0])               # image right
    y_c = np.cross(z_c, x_c)                       # image down
    return lie.make_T(np.column_stack([x_c, y_c, z_c]),
                      np.array([forward, 0.0, height]))


@dataclass
class Frame:
    T_wc: np.ndarray                 # true left-camera pose
    track_ids: np.ndarray            # (M,) int
    z: np.ndarray                    # (M, 3) stereo measurements
    landmark: np.ndarray             # (M,) true landmark index (eval only)
    outlier: np.ndarray              # (M,) bool (eval only)


@dataclass
class GardenSim:
    """Generates stereo frames along a trajectory through a garden."""

    stereo: StereoCamera = field(default_factory=lambda: StereoCamera(
        PinholeCamera(458.7, 457.3, 367.2, 248.4, width=752, height=480),
        0.12))
    sigma: float = 0.5               # pixel noise
    max_depth: float = 25.0
    survive: float = 0.97            # per-frame track survival, structure
    survive_grass: float = 0.85      # ... on grass (wind)
    detect: float = 0.3              # per-frame probability to start a track
    outlier: float = 0.02            # wrong measurements, structure
    outlier_grass: float = 0.10      # ... on grass

    def run(self, world: Landmarks, poses_wb: list[np.ndarray],
            T_bc: np.ndarray, rng: np.random.Generator) -> list[Frame]:
        cam, b = self.stereo.cam, self.stereo.baseline
        n = len(world.points)
        track = np.full(n, -1)
        next_id = 0
        frames = []
        for T_wb in poses_wb:
            T_wc = T_wb @ T_bc
            q = (world.points - T_wc[:3, 3]) @ T_wc[:3, :3]
            Z = q[:, 2]
            vis = (Z > 0.3) & (Z < self.max_depth)
            u = np.full((n, 3), -1.0)
            u[vis, 0] = cam.fx * q[vis, 0] / Z[vis] + cam.cx
            u[vis, 1] = cam.fy * q[vis, 1] / Z[vis] + cam.cy
            u[vis, 2] = cam.fx * (q[vis, 0] - b) / Z[vis] + cam.cx
            vis &= ((u[:, 0] >= 0) & (u[:, 0] < cam.width) & (u[:, 2] >= 0)
                    & (u[:, 1] >= 0) & (u[:, 1] < cam.height))
            surv = np.where(world.grass, self.survive_grass, self.survive)
            alive = (track >= 0) & vis & (rng.random(n) < surv)
            track[~alive] = -1
            new = vis & (track < 0) & (rng.random(n) < self.detect)
            track[new] = np.arange(next_id, next_id + new.sum())
            next_id += int(new.sum())
            idx = np.nonzero(track >= 0)[0]
            z = u[idx] + rng.normal(0, self.sigma, (len(idx), 3))
            p_out = np.where(world.grass[idx], self.outlier_grass,
                             self.outlier)
            out = rng.random(len(idx)) < p_out
            if out.any():
                rnd = rng.uniform([0, 0], [cam.width, cam.height],
                                  (out.sum(), 2))
                disp = rng.uniform(1, 40, out.sum())
                z[out] = np.column_stack([rnd[:, 0], rnd[:, 1],
                                          rnd[:, 0] - disp])
            frames.append(Frame(T_wc, track[idx].copy(), z, idx, out))
        return frames


# --------------------------------------------------------------------------
# Appearance (Chapter 10)
# --------------------------------------------------------------------------
def landmark_descriptors(world: Landmarks, rng: np.random.Generator,
                         n_fence: int = 40, n_grass: int = 15,
                         personal: float = 0.04, n_bits: int = 256
                         ) -> np.ndarray:
    """A 256-bit base descriptor per landmark, packed (N, n_bits / 8).

    Unique random descriptors for the house, the shed and the trees. Fence
    and grass landmarks look alike: each copies one of a few prototypes
    (`n_fence` pickets, `n_grass` grass patterns) with only a fraction
    `personal` of its bits changed -- the perceptual aliasing of
    repetitive structure.
    """
    n = len(world.points)
    bits = rng.random((n, n_bits)) < 0.5
    for kind, n_proto in [("fence", n_fence), ("grass", n_grass)]:
        idx = np.nonzero(world.kind == kind)[0]
        protos = rng.random((n_proto, n_bits)) < 0.5
        own = protos[rng.integers(n_proto, size=len(idx))]
        flip = rng.random((len(idx), n_bits)) < personal
        bits[idx] = own ^ flip
    return np.packbits(bits, axis=1)


def observe_descriptors(base: np.ndarray, frame: Frame, world: Landmarks,
                        rng: np.random.Generator, flip: float = 0.08,
                        flip_grass: float = 0.15) -> np.ndarray:
    """Descriptors measured in one frame: the base descriptor of each
    observed landmark with a fraction of bits flipped (viewpoint, light,
    wind); wrong measurements get random descriptors."""
    d = np.unpackbits(base[frame.landmark], axis=1).astype(bool)
    p = np.where(world.grass[frame.landmark], flip_grass, flip)[:, None]
    d ^= rng.random(d.shape) < p
    if frame.outlier.any():
        d[frame.outlier] = rng.random((frame.outlier.sum(), d.shape[1])) \
            < 0.5
    return np.packbits(d, axis=1)
