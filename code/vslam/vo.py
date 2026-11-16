"""Keyframe-based stereo visual odometry (Chapter 7).

At a keyframe, every tracked feature is triangulated from its stereo
measurement and stored in a local map, in world coordinates, under its
track id. Every later frame is localised against that map: P3P inside
RANSAC on the left-image measurements, then motion-only bundle adjustment
(LM with a Cauchy kernel) on the full stereo measurements. A new keyframe
is created when too few map points are still tracked or the camera has
moved too far from the keyframe.

With `keyframe_every_frame=True` the same code becomes frame-to-frame VO,
which re-triangulates the map at every frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import ba, lie, optim, pnp, robust
from .camera import StereoCamera


@dataclass
class VOStats:
    keyframes: list[int] = field(default_factory=list)
    inliers: list[int] = field(default_factory=list)
    failures: int = 0


class StereoVO:
    def __init__(self, stereo: StereoCamera, sigma: float = 0.5,
                 min_tracked: float = 0.5, min_points: int = 30,
                 max_dist: float = 1.0, max_angle_deg: float = 15.0,
                 keyframe_every_frame: bool = False,
                 rng: np.random.Generator | None = None) -> None:
        self.st, self.sigma = stereo, sigma
        cam = stereo.cam
        self.K = cam.K
        self.min_tracked, self.min_points = min_tracked, min_points
        self.max_dist, self.max_angle = max_dist, np.radians(max_angle_deg)
        self.every = keyframe_every_frame
        self.rng = rng or np.random.default_rng(0)
        self.map: dict[int, np.ndarray] = {}
        self.T_wk = None
        self.stats = VOStats()

    # ------------------------------------------------------------------
    def _keyframe(self, T_wc: np.ndarray, ids: np.ndarray, z: np.ndarray,
                  frame: int) -> np.ndarray:
        """Triangulate the stereo measurements into a new local map.
        Returns the (possibly refined) pose of the new keyframe."""
        disp = z[:, 0] - z[:, 2]
        ok = disp > 1.0
        q = np.array([self.st.triangulate(m) for m in z[ok]]).reshape(-1, 3)
        ok_d = (q[:, 2] > 0.3) & (q[:, 2] < 20.0)
        P = q[ok_d] @ T_wc[:3, :3].T + T_wc[:3, 3]
        self.map = dict(zip(ids[ok][ok_d].tolist(), P))
        self.T_wk = T_wc.copy()
        self.stats.keyframes.append(frame)
        return T_wc

    def _localise(self, T_guess: np.ndarray, X: np.ndarray, z: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray] | None:
        cam = self.st.cam
        x = np.column_stack([(z[:, 0] - cam.cx) / cam.fx,
                             (z[:, 1] - cam.cy) / cam.fy])
        thr = 3.0 * self.sigma / cam.fx
        res = pnp.pnp_ransac(X, x, thr, rng=self.rng)
        if res.model is None or res.inliers.sum() < 6:
            return None
        T_wc = lie.se3_inv(res.model)
        pb = pnp.StereoPoseProblem(self.K, self.st.baseline, X, z,
                                   self.sigma)
        sol = optim.levenberg_marquardt(pb, T_wc, kernel=robust.Cauchy(2.0),
                                        max_iters=20)
        r, _ = pb.linearize(sol.x)
        inl = np.sum(r.reshape(-1, 3) ** 2, axis=1) < 3.0 * 9.0
        return sol.x, inl

    def _reference_count(self) -> int:
        """Number of map points the keyframe started with."""
        return len(self.map)

    # ------------------------------------------------------------------
    def run(self, frames, T_wc0: np.ndarray) -> list[np.ndarray]:
        """Process frames (objects with .track_ids and .z); the first pose
        is given (it fixes the gauge). Returns the estimated T_wc."""
        traj = [T_wc0.copy()]
        self._keyframe(T_wc0, frames[0].track_ids, frames[0].z, 0)
        T = T_wc0.copy()
        for k, f in enumerate(frames[1:], start=1):
            ids = f.track_ids
            in_map = np.array([i in self.map for i in ids.tolist()],
                              dtype=bool)
            out = None
            if in_map.sum() >= 6:
                X = np.array([self.map[i] for i in ids[in_map].tolist()])
                out = self._localise(T, X, f.z[in_map])
            if out is None:              # lost: keep the last pose
                self.stats.failures += 1
                self.stats.inliers.append(0)
                T = self._keyframe(T, ids, f.z, k)
                traj.append(T.copy())
                continue
            T, inl = out
            self.stats.inliers.append(int(inl.sum()))
            D = lie.se3_inv(self.T_wk) @ T
            if (self.every or inl.sum() < self.min_points
                    or inl.sum() < self.min_tracked * self._reference_count()
                    or np.linalg.norm(D[:3, 3]) > self.max_dist
                    or np.linalg.norm(lie.so3_log(D[:3, :3]))
                    > self.max_angle):
                T = self._keyframe(T, ids, f.z, k)
            traj.append(T.copy())
        return traj


class StereoVOBA(StereoVO):
    """Keyframe stereo VO with a persistent local map refined by a
    sliding-window bundle adjustment (Chapter 8).

    Map points survive from one keyframe to the next as long as their
    tracks do; a new keyframe only triangulates the NEW tracks. After each
    new keyframe, the last `window` keyframes and all the points they see
    are optimised together, with the oldest keyframe of the window held
    fixed (it fixes the gauge).
    """

    def __init__(self, stereo: StereoCamera, sigma: float = 0.5,
                 window: int = 6, ba_iters: int = 10,
                 min_tracked: float = 0.8, **kw) -> None:
        # keyframes must share most of their points for the window to be
        # well linked: renew earlier than the plain VO (0.8 vs 0.5)
        super().__init__(stereo, sigma, min_tracked=min_tracked, **kw)
        self.window, self.ba_iters = window, ba_iters
        self.points: dict[int, np.ndarray] = {}
        self.kfs: list[dict] = []

    def _keyframe(self, T_wc, ids, z, frame):
        new = np.array([i not in self.points for i in ids.tolist()],
                       dtype=bool)
        disp = z[:, 0] - z[:, 2]
        ok = new & (disp > 1.0)
        if ok.any():
            q = np.array([self.st.triangulate(m) for m in z[ok]])
            ok_d = (q[:, 2] > 0.3) & (q[:, 2] < 20.0)
            P = q[ok_d] @ T_wc[:3, :3].T + T_wc[:3, 3]
            self.points.update(zip(ids[ok][ok_d].tolist(), P))
        self.kfs.append({"T": T_wc.copy(), "ids": ids.copy(), "z": z.copy(),
                         "frame": frame})
        self.stats.keyframes.append(frame)
        if len(self.kfs) >= 2:
            self._local_ba()
        win = self.kfs[-self.window:]
        seen = set().union(*[set(kf["ids"].tolist()) for kf in win])
        self.points = {i: X for i, X in self.points.items() if i in seen}
        self.map = self.points
        self.T_wk = self.kfs[-1]["T"].copy()
        return self.T_wk

    def _reference_count(self) -> int:
        """Map points observed by the last keyframe (the map itself also
        holds points of older keyframes of the window)."""
        last = self.kfs[-1]["ids"].tolist() if self.kfs else []
        return sum(1 for i in last if i in self.points)

    def _local_ba(self) -> None:
        win = self.kfs[-self.window:]
        pid = {i: j for j, i in enumerate(sorted(set().union(
            *[set(kf["ids"].tolist()) for kf in win]) & set(self.points)))}
        if not pid:
            return
        ci, pi, zs = [], [], []
        for c, kf in enumerate(win):
            for i, m in zip(kf["ids"].tolist(), kf["z"]):
                if i in pid:
                    ci.append(c)
                    pi.append(pid[i])
                    zs.append(m)
        pts = np.array([self.points[i] for i in pid])
        pb = ba.BAProblem(self.K, [kf["T"] for kf in win], pts,
                          np.array(ci), np.array(pi), np.array(zs),
                          self.st.baseline, self.sigma, {0})
        res = ba.solve(pb, kernel=robust.Cauchy(2.0),
                       max_iters=self.ba_iters)
        for kf, T in zip(win, res.cams):
            kf["T"] = T
        for i, j in pid.items():
            self.points[i] = res.points[j]
