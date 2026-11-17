"""Sliding-window visual-inertial odometry (Chapter 13).

Unknowns: for each keyframe of the window, the body state
x = (R, v, p, b_g, b_a) (15 dof, perturbed as in vslam.imu:
R Exp(dphi), v + dv, p + R dp, b + db), and world landmarks X (3 dof).

Factors:
  * stereo reprojection of landmark X in keyframe k (camera pose
    T_wb T_bc), Huber kernel, whitened by the pixel noise;
  * the IMU factor of Chapter 12 between consecutive keyframes;
  * a bias random-walk factor between consecutive keyframes;
  * a prior on the states, from the initialisation and then from
    marginalisation.

The normal equations are solved with the Schur complement on the
landmarks (Chapter 8). When the window is full, the oldest keyframe is
marginalised together with every landmark it observes (and all their
observations), which leaves a dense prior on the remaining states only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import imu
from .camera import StereoCamera
from .lie import hat, so3_exp, so3_log
from .robust import Huber

NX = 15                      # dimension of a keyframe state


# --------------------------------------------------------------------------
# States
# --------------------------------------------------------------------------
@dataclass
class KFState:
    R: np.ndarray
    v: np.ndarray
    p: np.ndarray
    bg: np.ndarray
    ba: np.ndarray

    def copy(self) -> "KFState":
        return KFState(self.R.copy(), self.v.copy(), self.p.copy(),
                       self.bg.copy(), self.ba.copy())

    def nav(self) -> imu.NavState:
        return imu.NavState(self.R, self.v, self.p)


def boxplus(s: KFState, d: np.ndarray) -> KFState:
    return KFState(s.R @ so3_exp(d[0:3]), s.v + d[3:6],
                   s.p + s.R @ d[6:9], s.bg + d[9:12], s.ba + d[12:15])


def boxminus(s: KFState, s0: KFState) -> np.ndarray:
    """Local coordinates of s around s0 (inverse of boxplus)."""
    return np.concatenate([so3_log(s0.R.T @ s.R), s.v - s0.v,
                           s0.R.T @ (s.p - s0.p), s.bg - s0.bg,
                           s.ba - s0.ba])


# --------------------------------------------------------------------------
# Stereo reprojection with an IMU-centred state
# --------------------------------------------------------------------------
def stereo_residuals(R: np.ndarray, p: np.ndarray, X: np.ndarray,
                     z: np.ndarray, T_bc: np.ndarray, stereo: StereoCamera,
                     sigma: float):
    """Whitened residuals r (k, 3) of stereo measurements z (k, 3) of
    points X (k, 3) seen from body poses (R (k, 3, 3), p (k, 3)), their
    Jacobians A (k, 3, 15) w.r.t. the body state and B (k, 3, 3) w.r.t.
    the point, and the depths (Proposition 13.1)."""
    cam, b = stereo.cam, stereo.baseline
    Rbc, tbc = T_bc[:3, :3], T_bc[:3, 3]
    Rc = R @ Rbc                                       # (k, 3, 3)
    c = p + R @ tbc                                    # (k, 3)
    q = np.einsum("kji,kj->ki", Rc, X - c)             # Rc^T (X - c)
    x, y, Z = q.T
    iz = 1.0 / Z
    h = np.column_stack([cam.fx * x * iz + cam.cx, cam.fy * y * iz + cam.cy,
                         cam.fx * (x - b) * iz + cam.cx])
    Jq = np.zeros((len(q), 3, 3))
    Jq[:, 0, 0], Jq[:, 0, 2] = cam.fx * iz, -cam.fx * x * iz ** 2
    Jq[:, 1, 1], Jq[:, 1, 2] = cam.fy * iz, -cam.fy * y * iz ** 2
    Jq[:, 2, 0], Jq[:, 2, 2] = cam.fx * iz, -cam.fx * (x - b) * iz ** 2
    # dq/dphi = hat(q) Rbc^T + Rbc^T hat(t_bc); dq/ddp = -Rbc^T
    qh = np.zeros((len(q), 3, 3))
    qh[:, 0, 1], qh[:, 0, 2] = -q[:, 2], q[:, 1]
    qh[:, 1, 0], qh[:, 1, 2] = q[:, 2], -q[:, 0]
    qh[:, 2, 0], qh[:, 2, 1] = -q[:, 1], q[:, 0]
    dq_dphi = qh @ Rbc.T + Rbc.T @ hat(tbc)
    A = np.zeros((len(q), 3, NX))
    A[:, :, 0:3] = -(Jq @ dq_dphi) / sigma
    A[:, :, 6:9] = (Jq @ Rbc.T) / sigma
    B = -(Jq @ np.transpose(Rc, (0, 2, 1))) / sigma
    r = (z - h) / sigma
    return r, A, B, Z


def triangulate(z: np.ndarray, stereo: StereoCamera) -> np.ndarray:
    """Points in the left camera frame from stereo measurements."""
    cam = stereo.cam
    d = z[:, 0] - z[:, 2]
    Z = cam.fx * stereo.baseline / np.where(d > 0, d, np.nan)
    return np.column_stack([(z[:, 0] - cam.cx) * Z / cam.fx,
                            (z[:, 1] - cam.cy) * Z / cam.fy, Z])


# --------------------------------------------------------------------------
# The sliding window
# --------------------------------------------------------------------------
@dataclass
class Prior:
    """Gaussian prior on states, in information form around x_lin:
    cost(x) = 1/2 d^T H d + g^T d, d = x [-] x_lin, for the keyframes
    with global ids `ids`."""
    ids: list[int]
    H: np.ndarray
    g: np.ndarray
    x_lin: list[KFState]


@dataclass
class VIOStats:
    iters: list[int] = field(default_factory=list)
    n_points: list[int] = field(default_factory=list)
    n_obs: list[int] = field(default_factory=list)
    rejected: int = 0


class SlidingWindowVIO:
    """Keyframe-based sliding-window VIO with stereo measurements."""

    def __init__(self, stereo: StereoCamera, T_bc: np.ndarray,
                 params: imu.ImuParams, sigma_px: float = 0.5,
                 window: int = 8, huber: float = 2.0,
                 max_new_points: int = 120, iters: int = 6,
                 rng: np.random.Generator | None = None,
                 use_imu: bool = True) -> None:
        self.st, self.T_bc, self.prm = stereo, T_bc, params
        self.sigma, self.window = sigma_px, window
        self.kernel = Huber(huber)
        self.max_new, self.iters = max_new_points, iters
        self.rng = rng or np.random.default_rng(0)
        self.use_imu = use_imu
        self.kfs: list[KFState] = []
        self.ids: list[int] = []
        self.times: list[float] = []
        self.pre: list[imu.Preintegration] = []   # pre[k]: kf k -> k + 1
        self.obs: list[dict[int, np.ndarray]] = []
        self.points: dict[int, np.ndarray] = {}
        self.prior: Prior | None = None
        self.history: dict[int, KFState] = {}     # last estimate per kf
        self.stats = VIOStats()
        self._next_id = 0

    # ------------------------------------------------------------ setup
    def initialise(self, t: float, state: KFState, sigmas: np.ndarray,
                   track_ids: np.ndarray, z: np.ndarray) -> None:
        """First keyframe with a diagonal prior of standard deviations
        `sigmas` (15) on its state."""
        self._append(t, state, track_ids, z)
        H = np.diag(1.0 / np.asarray(sigmas, float) ** 2)
        self.prior = Prior([self.ids[0]], H, np.zeros(NX), [state.copy()])

    def _append(self, t, state, track_ids, z) -> None:
        self.kfs.append(state)
        self.ids.append(self._next_id)
        self._next_id += 1
        self.times.append(t)
        o = {int(i): zz for i, zz in zip(track_ids, z)}
        self.obs.append(o)
        self._new_points(len(self.kfs) - 1)

    def _new_points(self, k: int) -> None:
        o = self.obs[k]
        ids = np.array([i for i in o if i not in self.points], dtype=int)
        if len(ids) == 0:
            return
        z = np.array([o[i] for i in ids])
        q = triangulate(z, self.st)
        ok = np.isfinite(q[:, 2]) & (q[:, 2] > 0.3) & (q[:, 2] < 20.0)
        ids, q = ids[ok], q[ok]
        # the others of this keyframe are not used
        drop = set(o) - set(self.points) - set(ids.tolist())
        if len(ids) > self.max_new:
            keep = self.rng.choice(len(ids), self.max_new, replace=False)
            drop |= set(np.delete(ids, keep).tolist())
            ids, q = ids[keep], q[keep]
        for i in drop:
            o.pop(i, None)
        s = self.kfs[k]
        Rc = s.R @ self.T_bc[:3, :3]
        c = s.p + s.R @ self.T_bc[:3, 3]
        for i, qq in zip(ids, q):
            self.points[int(i)] = Rc @ qq + c

    # ------------------------------------------------------------ input
    def add_keyframe(self, t: float, gyro: np.ndarray, accel: np.ndarray,
                     track_ids: np.ndarray, z: np.ndarray) -> KFState:
        """IMU samples since the previous keyframe, and the stereo
        measurements of the new keyframe. Returns the estimate."""
        last = self.kfs[-1]
        pre = imu.Preintegration(last.bg.copy(), last.ba.copy(), self.prm)
        for w, a in zip(gyro, accel):
            pre.integrate(w, a, self.prm.dt)
        nav = pre.predict(last.nav())
        state = KFState(nav.R, nav.v, nav.p, last.bg.copy(), last.ba.copy())
        self.pre.append(pre)
        # observations of points already in the window are kept; new ones
        # are triangulated from the predicted pose
        self._append(t, state, track_ids, z)
        self.optimise()
        self._reject_outliers()
        for gid, s in zip(self.ids, self.kfs):
            self.history[gid] = s.copy()
        if len(self.kfs) > self.window:
            self.marginalise()
        return self.kfs[-1]

    # ------------------------------------------------------------ factors
    def _visual(self, kfs=None, points=None, which=None):
        """Stacked visual residuals and Jacobians. `which` restricts to a
        set of point ids."""
        kfs = self.kfs if kfs is None else kfs
        points = self.points if points is None else points
        kidx, lid, zs = [], [], []
        for k, o in enumerate(self.obs):
            for i, zz in o.items():
                if i in points and (which is None or i in which):
                    kidx.append(k)
                    lid.append(i)
                    zs.append(zz)
        if not kidx:
            return None
        kidx = np.array(kidx)
        lid = np.array(lid)
        R = np.array([kfs[k].R for k in kidx])
        p = np.array([kfs[k].p for k in kidx])
        X = np.array([points[i] for i in lid])
        r, A, B, Z = stereo_residuals(R, p, X, np.array(zs), self.T_bc,
                                      self.st, self.sigma)
        return kidx, lid, r, A, B, Z

    def _imu_factor(self, k: int, kfs):
        """Whitened IMU + bias residual (15) between k and k + 1, and its
        Jacobians (15 x 15) w.r.t. the two states."""
        pre = self.pre[k]
        si, sj = kfs[k], kfs[k + 1]
        r = pre.residual(si.nav(), sj.nav(), si.bg, si.ba)
        Ji9, Jj9, Jb = pre.jacobians(si.nav(), sj.nav(), si.bg, si.ba)
        L = np.linalg.cholesky(np.linalg.inv(pre.cov))       # info = L L^T
        Ji = np.zeros((15, NX))
        Jj = np.zeros((15, NX))
        Ji[:9, :9], Ji[:9, 9:] = L.T @ Ji9, L.T @ Jb
        Jj[:9, :9] = L.T @ Jj9
        T = pre.dt
        sb = np.r_[[self.prm.sigma_bg] * 3, [self.prm.sigma_ba] * 3] \
            * np.sqrt(T)
        Ji[9:, 9:] = -np.diag(1 / sb)
        Jj[9:, 9:] = np.diag(1 / sb)
        rr = np.r_[L.T @ r, np.r_[sj.bg - si.bg, sj.ba - si.ba] / sb]
        return rr, Ji, Jj

    def _prior_terms(self, kfs):
        """Prior cost, gradient and Hessian in window coordinates."""
        P = self.prior
        if P is None:
            return 0.0, None, None, []
        pos = [self.ids.index(g) for g in P.ids]
        d = np.concatenate([boxminus(kfs[k], x0)
                            for k, x0 in zip(pos, P.x_lin)])
        cost = 0.5 * d @ P.H @ d + P.g @ d
        return cost, P.g + P.H @ d, P.H, pos

    # ------------------------------------------------------------ cost
    def cost(self, kfs=None, points=None) -> float:
        kfs = self.kfs if kfs is None else kfs
        points = self.points if points is None else points
        F = 0.0
        vis = self._visual(kfs, points)
        if vis is not None:
            s = np.sum(vis[2] ** 2, axis=1)
            F += 0.5 * np.sum(self.kernel.rho(s))
        if self.use_imu:
            for k in range(len(kfs) - 1):
                r, _, _ = self._imu_factor(k, kfs)
                F += 0.5 * r @ r
        F += self._prior_terms(kfs)[0]
        return F

    # ------------------------------------------------------------ system
    def normal_equations(self, kfs=None, points=None, use_prior=True):
        """H_ss (15N x 15N), W (15N x 3P), V (P, 3, 3), g_s, g_p and the
        ordered point ids."""
        kfs = self.kfs if kfs is None else kfs
        points = self.points if points is None else points
        N = len(kfs)
        pid = sorted(points)
        col = {i: j for j, i in enumerate(pid)}
        P = len(pid)
        Hs = np.zeros((NX * N, NX * N))
        gs = np.zeros(NX * N)
        W = np.zeros((NX * N, 3 * P))
        V = np.zeros((P, 3, 3))
        gp = np.zeros((P, 3))
        vis = self._visual(kfs, points)
        if vis is not None:
            kidx, lid, r, A, B, _ = vis
            w = self.kernel.weight(np.sum(r ** 2, axis=1))
            cidx = np.array([col[i] for i in lid])
            Aw, Bw = A * w[:, None, None], B * w[:, None, None]
            AtA = np.einsum("kri,krj->kij", Aw, A)
            AtB = np.einsum("kri,krj->kij", Aw, B)
            BtB = np.einsum("kri,krj->kij", Bw, B)
            Atr = np.einsum("kri,kr->ki", Aw, r)
            Btr = np.einsum("kri,kr->ki", Bw, r)
            for k in range(N):
                m = kidx == k
                if m.any():
                    sl = slice(NX * k, NX * k + NX)
                    Hs[sl, sl] += AtA[m].sum(axis=0)
                    gs[sl] += Atr[m].sum(axis=0)
            np.add.at(V, cidx, BtB)
            np.add.at(gp, cidx, Btr)
            W4 = W.reshape(N, NX, P, 3)
            np.add.at(W4, (kidx, slice(None), cidx, slice(None)),
                      AtB)
            W = W4.reshape(NX * N, 3 * P)
        if self.use_imu:
            for k in range(N - 1):
                r, Ji, Jj = self._imu_factor(k, kfs)
                J = np.zeros((15, NX * N))
                J[:, NX * k:NX * k + NX] = Ji
                J[:, NX * (k + 1):NX * (k + 2)] = Jj
                Hs += J.T @ J
                gs += J.T @ r
        if use_prior and self.prior is not None:
            _, g0, H0, pos = self._prior_terms(kfs)
            idx = np.concatenate([np.arange(NX * k, NX * k + NX)
                                  for k in pos])
            Hs[np.ix_(idx, idx)] += H0
            gs[idx] += g0
        return Hs, W, V, gs, gp.ravel(), pid

    def solve_step(self, lam: float):
        """LM step: (H + lam diag) dx = -g, landmarks eliminated."""
        Hs, W, V, gs, gp, pid = self.normal_equations()
        P = len(pid)
        Hs_l = Hs + lam * np.diag(np.diag(Hs) + 1e-9)
        if P == 0:                     # no landmark (camera blind): IMU only
            return np.linalg.solve(Hs_l, -gs), np.zeros((0, 3)), pid, gs, gp
        Vl = V + lam * (V * np.eye(3)[None]) + 1e-9 * np.eye(3)[None]
        Vinv = np.linalg.inv(Vl)
        W3 = W.reshape(-1, P, 3)
        WV = np.einsum("spi,pij->spj", W3, Vinv)           # W V^-1
        S = Hs_l - WV.reshape(len(Hs), -1) @ W.T
        rhs = -gs + WV.reshape(len(Hs), -1) @ gp
        dx = np.linalg.solve(S, rhs)
        dp = -np.einsum("pij,pj->pi", Vinv,
                        gp.reshape(P, 3) + (W.T @ dx).reshape(P, 3))
        return dx, dp, pid, gs, gp

    def optimise(self) -> None:
        lam, F = 1e-4, self.cost()
        it = 0
        for it in range(self.iters):
            dx, dp, pid, gs, gp = self.solve_step(lam)
            kfs = [boxplus(s, dx[NX * k:NX * k + NX])
                   for k, s in enumerate(self.kfs)]
            pts = {i: self.points[i] + dp[j] for j, i in enumerate(pid)}
            F1 = self.cost(kfs, pts)
            if F1 < F:
                self.kfs, self.points = kfs, pts
                done = F - F1 < 1e-6 * F
                F, lam = F1, max(lam / 3, 1e-7)
                if done:
                    break
            else:
                lam *= 10
        self.stats.iters.append(it + 1)
        self.stats.n_points.append(len(self.points))
        self.stats.n_obs.append(sum(len(o) for o in self.obs))

    def _reject_outliers(self, chi2: float = 11.34) -> None:
        """Remove observations whose whitened squared error exceeds the
        99 % quantile of chi^2(3), and points left unobserved."""
        vis = self._visual()
        if vis is None:
            return
        kidx, lid, r, _, _, Z = vis
        bad = (np.sum(r ** 2, axis=1) > chi2) | (Z < 0.1)
        for k, i in zip(kidx[bad], lid[bad]):
            self.obs[k].pop(int(i), None)
        self.stats.rejected += int(bad.sum())
        seen = set().union(*[set(o) for o in self.obs])
        for i in list(self.points):
            if i not in seen:
                del self.points[i]

    # ------------------------------------------------------------ marg.
    def marginalise(self) -> None:
        """Marginalise the oldest keyframe and every landmark it observes
        (with all their observations) into a prior on the remaining
        states (Section 13.3, Lemma 1.8 in information form)."""
        N = len(self.kfs)
        gone = set(self.obs[0]) & set(self.points)
        # the factors to marginalise, linearised at the current estimate
        Hs = np.zeros((NX * N, NX * N))
        gs = np.zeros(NX * N)
        pid = sorted(gone)
        col = {i: j for j, i in enumerate(pid)}
        P = len(pid)
        W = np.zeros((NX * N, 3 * P))
        V = np.zeros((P, 3, 3))
        gp = np.zeros((P, 3))
        vis = self._visual(which=gone)
        if vis is not None:
            kidx, lid, r, A, B, _ = vis
            w = self.kernel.weight(np.sum(r ** 2, axis=1))
            cidx = np.array([col[i] for i in lid])
            Aw, Bw = A * w[:, None, None], B * w[:, None, None]
            for k in range(N):
                m = kidx == k
                if m.any():
                    sl = slice(NX * k, NX * k + NX)
                    Hs[sl, sl] += np.einsum("kri,krj->ij", Aw[m], A[m])
                    gs[sl] += np.einsum("kri,kr->i", Aw[m], r[m])
            np.add.at(V, cidx, np.einsum("kri,krj->kij", Bw, B))
            np.add.at(gp, cidx, np.einsum("kri,kr->ki", Bw, r))
            W4 = W.reshape(N, NX, P, 3)
            np.add.at(W4, (kidx, slice(None), cidx, slice(None)),
                      np.einsum("kri,krj->kij", Aw, B))
            W = W4.reshape(NX * N, 3 * P)
        if self.use_imu:
            r, Ji, Jj = self._imu_factor(0, self.kfs)
            J = np.zeros((15, NX * N))
            J[:, :NX], J[:, NX:2 * NX] = Ji, Jj
            Hs += J.T @ J
            gs += J.T @ r
        if self.prior is not None:
            _, g0, H0, pos = self._prior_terms(self.kfs)
            idx = np.concatenate([np.arange(NX * k, NX * k + NX)
                                  for k in pos])
            Hs[np.ix_(idx, idx)] += H0
            gs[idx] += g0
        # eliminate the landmarks, then the oldest state
        if P:
            Vinv = np.linalg.inv(V + 1e-9 * np.eye(3)[None])
            WV = np.einsum("spi,pij->spj", W.reshape(-1, P, 3), Vinv)
            Hs = Hs - WV.reshape(len(Hs), -1) @ W.T
            gs = gs - WV.reshape(len(Hs), -1) @ gp.ravel()
        m, keep = slice(0, NX), slice(NX, NX * N)
        Hmm = Hs[m, m] + 1e-9 * np.eye(NX)
        Hkm = Hs[keep, m]
        Hp = Hs[keep, keep] - Hkm @ np.linalg.solve(Hmm, Hkm.T)
        gp_ = gs[keep] - Hkm @ np.linalg.solve(Hmm, gs[m])
        Hp = 0.5 * (Hp + Hp.T)
        # drop what has been marginalised
        for i in gone:
            del self.points[i]
            for o in self.obs:
                o.pop(i, None)
        self.kfs.pop(0)
        self.ids.pop(0)
        self.times.pop(0)
        self.obs.pop(0)
        if self.pre:
            self.pre.pop(0)
        # states linked to nothing in the prior are left out of it
        nz = [k for k in range(N - 1)
              if np.abs(Hp[NX * k:NX * k + NX]).max() > 0]
        idx = np.concatenate([np.arange(NX * k, NX * k + NX) for k in nz])
        self.prior = Prior([self.ids[k] for k in nz], Hp[np.ix_(idx, idx)],
                           gp_[idx], [self.kfs[k].copy() for k in nz])


# --------------------------------------------------------------------------
# Null space and MSCKF
# --------------------------------------------------------------------------
def gauge_directions(kfs: list[KFState], points: dict[int, np.ndarray]
                     ) -> np.ndarray:
    """The four directions (columns) along which the VIO cost is
    invariant, in the local coordinates of (states, points): translations
    along x, y, z and the rotation about gravity (Theorem 13.3)."""
    N, pid = len(kfs), sorted(points)
    n = NX * N + 3 * len(pid)
    U = np.zeros((n, 4))
    ez = np.array([0.0, 0.0, 1.0])
    for k, s in enumerate(kfs):
        o = NX * k
        for a in range(3):                         # translations
            t = np.eye(3)[a]
            U[o + 6:o + 9, a] = s.R.T @ t
        U[o:o + 3, 3] = s.R.T @ ez                 # yaw
        U[o + 3:o + 6, 3] = np.cross(ez, s.v)
        U[o + 6:o + 9, 3] = s.R.T @ np.cross(ez, s.p)
    for j, i in enumerate(pid):
        o = NX * N + 3 * j
        U[o:o + 3, :3] = np.eye(3)
        U[o:o + 3, 3] = np.cross(ez, points[i])
    return U


def msckf_project(Hx: np.ndarray, Hf: np.ndarray, r: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Project a feature's linearised measurements r = Hx dx + Hf df + n
    onto the left null space of Hf (Proposition 13.5): returns
    (N^T Hx, N^T r) with N an orthonormal basis of that null space."""
    Q, _ = np.linalg.qr(Hf, mode="complete")
    N = Q[:, Hf.shape[1]:]
    return N.T @ Hx, N.T @ r


# --------------------------------------------------------------------------
# Initialisation
# --------------------------------------------------------------------------
def static_init(gyro: np.ndarray, accel: np.ndarray, yaw: float = 0.0
                ) -> tuple[np.ndarray, np.ndarray]:
    """Rotation and gyroscope bias from samples taken at rest: the mean
    specific force points up in the body frame (roll, pitch); the mean
    rate is the gyro bias. Yaw is unobservable and is set to `yaw`
    (Proposition 13.6)."""
    f = accel.mean(axis=0)
    up = f / np.linalg.norm(f)                 # R^T e_z
    # the rotation taking up to e_z (so that R^T e_z = up), then yaw
    # about the world vertical
    ez = np.array([0.0, 0.0, 1.0])
    axis = np.cross(up, ez)
    s, c = np.linalg.norm(axis), up @ ez
    Rt = np.eye(3) if s < 1e-12 else so3_exp(axis / s * np.arctan2(s, c))
    R = so3_exp(np.array([0.0, 0.0, yaw])) @ Rt
    return R, gyro.mean(axis=0)


def gyro_bias_from_rotations(Rs: list[np.ndarray],
                             pres: list[imu.Preintegration]) -> np.ndarray:
    """Least-squares gyroscope bias from known keyframe rotations:
    minimise sum |Log((dR Exp(J dbg))^T R_i^T R_j)|^2 to first order
    (Proposition 13.7)."""
    A, b = [], []
    for k, pre in enumerate(pres):
        e = so3_log(pre.dR.T @ Rs[k].T @ Rs[k + 1])
        A.append(pre.J_R_bg)
        b.append(e)
    dbg = np.linalg.lstsq(np.vstack(A), np.concatenate(b), rcond=None)[0]
    return pres[0].bg + dbg


def align_gravity_velocity(Rs: list[np.ndarray], ps: list[np.ndarray],
                           pres: list[imu.Preintegration]
                           ) -> tuple[np.ndarray, np.ndarray]:
    """Gravity (in the frame of the poses) and keyframe velocities from
    known poses and preintegrated measurements, by linear least squares
    on Theorem 12.7 (Proposition 13.7). Returns (g, v (n, 3))."""
    n = len(Rs)
    rows, rhs = [], []
    nv = 3 * n
    for k, pre in enumerate(pres):
        T = pre.dt
        # v_{k+1} - v_k - g T = R_k dv
        Av = np.zeros((3, nv + 3))
        Av[:, 3 * (k + 1):3 * (k + 2)] = np.eye(3)
        Av[:, 3 * k:3 * k + 3] = -np.eye(3)
        Av[:, nv:] = -T * np.eye(3)
        rows.append(Av)
        rhs.append(Rs[k] @ pre.dv)
        # p_{k+1} - p_k - v_k T - g T^2 / 2 = R_k dp
        Ap = np.zeros((3, nv + 3))
        Ap[:, 3 * k:3 * k + 3] = -T * np.eye(3)
        Ap[:, nv:] = -0.5 * T ** 2 * np.eye(3)
        rows.append(Ap)
        rhs.append(Rs[k] @ pre.dp - (ps[k + 1] - ps[k]))
    x = np.linalg.lstsq(np.vstack(rows), np.concatenate(rhs),
                        rcond=None)[0]
    return x[nv:], x[:nv].reshape(n, 3)
