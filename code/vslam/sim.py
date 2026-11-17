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


# --------------------------------------------------------------------------
# Continuous-time motion (Chapter 13)
# --------------------------------------------------------------------------
def _euler_zyx_rates(psi, th, ph, dpsi, dth, dph):
    """Rotation R = Rz(psi) Ry(th) Rx(ph) and its body angular rate."""
    cz, sz = np.cos(psi), np.sin(psi)
    cy, sy = np.cos(th), np.sin(th)
    cx, sx = np.cos(ph), np.sin(ph)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    w = np.array([dph - dpsi * sy,
                  dth * cx + dpsi * cy * sx,
                  -dth * sx + dpsi * cy * cx])
    return Rz @ Ry @ Rx, w


class GardenMotion:
    """The mowing pattern of the garden as a smooth body trajectory with
    closed-form angular rate and acceleration (for an exact IMU).

    The mower stands still for `still` seconds, accelerates smoothly to
    `speed`, drives the rows y = 1, 2.5, ..., 7 between x = 1 and x = 11,
    and turns between rows with a curvature that rises and falls as a
    raised cosine (no jump in lateral acceleration). The ground is bumpy
    (height `bump`) and the body pitches and rolls by about `sway_deg`.
    World frame z up; body frame x forward, y left, z up.
    """

    def __init__(self, rows=(1.0, 2.5, 4.0, 5.5, 7.0), x0: float = 1.0,
                 x1: float = 11.0, speed: float = 1.0, still: float = 2.0,
                 ramp: float = 1.0, bump: float = 0.01,
                 bump_len: float = 1.3, sway_deg: float = 1.0) -> None:
        self.speed, self.still, self.ramp = speed, still, ramp
        self.bump, self.kz = bump, 2 * np.pi / bump_len
        s = np.radians(sway_deg)
        self.th0, self.kth = s, 2 * np.pi / 1.7
        self.ph0, self.kph = s, 2 * np.pi / 2.3
        # the turn profile: psi(u) = pi u - sin(2 pi u) / 2 on u in [0, 1];
        # lateral offset per unit length c = int_0^1 sin(psi(u)) du
        u = np.linspace(0, 1, 20001)
        psi = np.pi * u - 0.5 * np.sin(2 * np.pi * u)
        c = float(np.sum(0.5 * (np.sin(psi[1:]) + np.sin(psi[:-1])))
                  * (u[1] - u[0]))
        self.turn_len = (rows[1] - rows[0]) / c
        # tables of int cos, int sin over the unit turn (Simpson-accurate
        # on this grid), interpolated with the exact derivatives
        du = u[1] - u[0]
        self._u = u
        self._ic = np.r_[0, np.cumsum(0.5 * (np.cos(psi[1:])
                                             + np.cos(psi[:-1])))] * du
        self._is = np.r_[0, np.cumsum(0.5 * (np.sin(psi[1:])
                                             + np.sin(psi[:-1])))] * du
        # segments: (kind, start arclength, length, start xy, heading0,
        # turn direction)
        self.segs = []
        s0 = 0.0
        for i, y in enumerate(rows):
            fwd = i % 2 == 0
            start = np.array([x0 if fwd else x1, y])
            head = 0.0 if fwd else np.pi
            self.segs.append(("row", s0, x1 - x0, start, head, 0))
            s0 += x1 - x0
            if i + 1 < len(rows):
                end = np.array([x1 if fwd else x0, y])
                self.segs.append(("turn", s0, self.turn_len, end, head,
                                  1 if fwd else -1))
                s0 += self.turn_len
        self.length = s0
        self.duration = (still + ramp / 2 + (s0) / speed)

    # ------------------------------------------------------------ s(t)
    def arclength(self, t: float) -> tuple[float, float, float]:
        """Distance s, speed and tangential acceleration at time t: still,
        then a smooth ramp (speed = V sin^2), then constant speed."""
        V, T0, Tr = self.speed, self.still, self.ramp
        if t <= T0:
            return 0.0, 0.0, 0.0
        tau = t - T0
        if tau < Tr:
            x = np.pi * tau / (2 * Tr)
            v = V * np.sin(x) ** 2
            a = V * np.sin(2 * x) * np.pi / (2 * Tr)
            s = V * (tau / 2 - Tr / (2 * np.pi) * np.sin(np.pi * tau / Tr))
            return s, v, a
        return V * Tr / 2 + V * (tau - Tr), V, 0.0

    # ------------------------------------------------------------ path
    def path(self, s: float):
        """Horizontal position, heading and curvature at arclength s."""
        s = min(max(s, 0.0), self.length)
        for kind, a, L, start, head, d in self.segs:
            if s <= a + L or (kind, a) == self.segs[-1][:2]:
                break
        ds = s - a
        if kind == "row":
            t = np.array([np.cos(head), np.sin(head)])
            return start + ds * t, head, 0.0
        u = ds / L
        psi_u = np.pi * u - 0.5 * np.sin(2 * np.pi * u)
        kap = d * (np.pi / L) * (1 - np.cos(2 * np.pi * u))
        # table value at the grid point below u, plus a trapezoid step to u
        # (the integrands are known exactly): error ~1e-9 m
        k = min(int(u / self._u[1]), len(self._u) - 2)
        uk = self._u[k]
        psik = np.pi * uk - 0.5 * np.sin(2 * np.pi * uk)
        ic = self._ic[k] + 0.5 * (np.cos(psik) + np.cos(psi_u)) * (u - uk)
        is_ = self._is[k] + 0.5 * (np.sin(psik) + np.sin(psi_u)) * (u - uk)
        # local frame of the turn: forward = initial heading, left = d
        f = np.array([np.cos(head), np.sin(head)])
        left = d * np.array([-np.sin(head), np.cos(head)])
        xy = start + L * (ic * f + is_ * left)
        return xy, head + d * psi_u, kap

    # ------------------------------------------------------------ state
    def state(self, t: float):
        """R, p, v, a (world) and w (body) at time t."""
        s, sp, ac = self.arclength(t)
        xy, psi, kap = self.path(s)
        tang = np.array([np.cos(psi), np.sin(psi)])
        nrm = np.array([-np.sin(psi), np.cos(psi)])
        z = self.bump * np.sin(self.kz * s)
        dz = self.bump * self.kz * np.cos(self.kz * s)
        ddz = -self.bump * self.kz ** 2 * np.sin(self.kz * s)
        p = np.r_[xy, z]
        v = np.r_[sp * tang, dz * sp]
        a = np.r_[ac * tang + sp ** 2 * kap * nrm,
                  ddz * sp ** 2 + dz * ac]
        th = self.th0 * np.sin(self.kth * s)
        ph = self.ph0 * np.sin(self.kph * s + 1.0) - self.ph0 * np.sin(1.0)
        dth = self.th0 * self.kth * np.cos(self.kth * s) * sp
        dph = self.ph0 * self.kph * np.cos(self.kph * s + 1.0) * sp
        R, w = _euler_zyx_rates(psi, th, ph, sp * kap, dth, dph)
        return R, p, v, a, w

    # the interface used by vslam.imu.simulate and friends
    def R(self, t):
        return self.state(t)[0]

    def p(self, t):
        return self.state(t)[1]

    def v(self, t):
        return self.state(t)[2]

    def acc(self, t):
        return self.state(t)[3]

    def omega(self, t):
        return self.state(t)[4]

    def T(self, t):
        R, p = self.state(t)[:2]
        return lie.make_T(R, p)
