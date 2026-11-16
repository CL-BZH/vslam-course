"""Bundle adjustment with the Schur complement (Chapter 8).

Unknowns: camera poses T_wc (right perturbation, 6 dof each) and 3D points
(3 dof each). Observations: for each (camera c, point p) pair, a monocular
pixel (u, v) or a rectified-stereo measurement (u_l, v, u_r). Cameras in
`fixed` are held constant, which fixes the gauge.

The normal equations have the block structure

    [ U   W ] [dc]     [g_c]
    [ W^T V ] [dp] = - [g_p],

with U and V block diagonal (6x6 per camera, 3x3 per point). The points
are eliminated first (the Schur complement):

    S dc = -g_c + W V^{-1} g_p,     S = U - W V^{-1} W^T,
    dp = -V^{-1} (g_p + W^T dc),

so the only matrix to factorise has the size of the cameras.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import lie
from .robust import Trivial


@dataclass
class BAProblem:
    K: np.ndarray
    cams: list[np.ndarray]           # T_wc
    points: np.ndarray               # (n, 3)
    cam_idx: np.ndarray              # (k,) camera of each observation
    pt_idx: np.ndarray               # (k,) point of each observation
    z: np.ndarray                    # (k, 2) or (k, 3)
    baseline: float | None = None    # set for stereo observations
    sigma: float = 1.0
    fixed: set = field(default_factory=set)

    @property
    def dim(self) -> int:
        return self.z.shape[1]

    # ------------------------------------------------------------------
    def residuals_jacobians(self, cams=None, points=None):
        """Whitened residuals r (k, d) and Jacobian blocks A (k, d, 6) with
        respect to the camera perturbation, B (k, d, 3) w.r.t. the point."""
        cams = self.cams if cams is None else cams
        points = self.points if points is None else points
        R = np.array([T[:3, :3] for T in cams])[self.cam_idx]
        c = np.array([T[:3, 3] for T in cams])[self.cam_idx]
        X = points[self.pt_idx]
        q = np.einsum("kji,kj->ki", R, X - c)          # R^T (X - c)
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], \
            self.K[1, 2]
        x, y, Z = q.T
        iz = 1.0 / Z
        d = self.dim
        h = np.empty((len(q), d))
        h[:, 0] = fx * x * iz + cx
        h[:, 1] = fy * y * iz + cy
        Jq = np.zeros((len(q), d, 3))
        Jq[:, 0, 0], Jq[:, 0, 2] = fx * iz, -fx * x * iz * iz
        Jq[:, 1, 1], Jq[:, 1, 2] = fy * iz, -fy * y * iz * iz
        if d == 3:
            b = self.baseline
            h[:, 2] = fx * (x - b) * iz + cx
            Jq[:, 2, 0], Jq[:, 2, 2] = fx * iz, -fx * (x - b) * iz * iz
        dq_dxi = np.zeros((len(q), 3, 6))
        dq_dxi[:, :, :3] = -np.eye(3)
        dq_dxi[:, 0, 4], dq_dxi[:, 0, 5] = -Z, y
        dq_dxi[:, 1, 3], dq_dxi[:, 1, 5] = Z, -x
        dq_dxi[:, 2, 3], dq_dxi[:, 2, 4] = -y, x
        r = (self.z - h) / self.sigma
        A = -(Jq @ dq_dxi) / self.sigma
        B = -(Jq @ np.transpose(R, (0, 2, 1))) / self.sigma
        return r, A, B, q[:, 2]

    def free_cameras(self) -> list[int]:
        return [i for i in range(len(self.cams)) if i not in self.fixed]

    # ------------------------------------------------------------------
    def normal_equations(self, r, A, B, w):
        """Blocks of the (robustly weighted) normal equations. Returns
        U (m, 6, 6), V (n, 3, 3), W per observation (k, 6, 3), g_c (m, 6),
        g_p (n, 3); rows of fixed cameras are zeroed."""
        m, n = len(self.cams), len(self.points)
        sw = np.sqrt(w)[:, None, None]
        A, B, rw = A * sw, B * sw, r * np.sqrt(w)[:, None]
        fixed = np.array([i in self.fixed for i in range(m)])
        A = A * ~fixed[self.cam_idx][:, None, None]
        U = np.zeros((m, 6, 6))
        V = np.zeros((n, 3, 3))
        np.add.at(U, self.cam_idx, np.einsum("kdi,kdj->kij", A, A))
        np.add.at(V, self.pt_idx, np.einsum("kdi,kdj->kij", B, B))
        Wk = np.einsum("kdi,kdj->kij", A, B)
        g_c = np.zeros((m, 6))
        g_p = np.zeros((n, 3))
        np.add.at(g_c, self.cam_idx, np.einsum("kdi,kd->ki", A, rw))
        np.add.at(g_p, self.pt_idx, np.einsum("kdi,kd->ki", B, rw))
        return U, V, Wk, g_c, g_p

    def schur(self, U, V, Wk, g_c, g_p):
        """Reduced camera system over the FREE cameras: S, b, and the
        per-point inverses Vinv. S dc = b."""
        free = self.free_cameras()
        pos = {c: i for i, c in enumerate(free)}
        nf = len(free)
        S = np.zeros((6 * nf, 6 * nf))
        b = np.zeros(6 * nf)
        Vinv = np.linalg.inv(V + 1e-12 * np.eye(3))
        for c in free:
            i = pos[c]
            S[6 * i:6 * i + 6, 6 * i:6 * i + 6] += U[c]
            b[6 * i:6 * i + 6] -= g_c[c]
        Y = np.einsum("kij,kjl->kil", Wk, Vinv[self.pt_idx])  # W V^-1
        cpos = np.array([pos.get(c, -1) for c in range(len(self.cams))])
        keep = np.nonzero(cpos[self.cam_idx] >= 0)[0]
        # b += sum_j W_ij V_j^{-1} g_p_j, observation by observation
        contrib = np.einsum("kij,kj->ki", Y[keep], g_p[self.pt_idx[keep]])
        np.add.at(b.reshape(-1, 6), cpos[self.cam_idx[keep]], contrib)
        # S -= sum over pairs (a, o) of observations of the same point
        a_idx, o_idx = _pairs_by_group(self.pt_idx[keep])
        a_idx, o_idx = keep[a_idx], keep[o_idx]
        blocks = np.einsum("kij,klj->kil", Y[a_idx], Wk[o_idx])
        S4 = S.reshape(nf, 6, nf, 6).transpose(0, 2, 1, 3).copy()
        np.subtract.at(S4, (cpos[self.cam_idx[a_idx]],
                            cpos[self.cam_idx[o_idx]]), blocks)
        S = S4.transpose(0, 2, 1, 3).reshape(6 * nf, 6 * nf)
        return S, b, Vinv, pos

    def back_substitute(self, dc_free, pos, Vinv, Wk, g_p):
        """Point increments dp_j = -V_j^{-1} (g_p_j + sum_i W_ij^T dc_i)."""
        m = len(self.cams)
        dc = np.zeros((m, 6))
        for c, i in pos.items():
            dc[c] = dc_free[6 * i:6 * i + 6]
        rhs = g_p.copy()
        np.add.at(rhs, self.pt_idx,
                  np.einsum("kij,ki->kj", Wk, dc[self.cam_idx]))
        dp = -np.einsum("nij,nj->ni", Vinv, rhs)
        return dc, dp

    def retract(self, dc, dp):
        cams = [T @ lie.se3_exp(d) if i not in self.fixed else T
                for i, (T, d) in enumerate(zip(self.cams, dc))]
        return cams, self.points + dp


def _pairs_by_group(groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """All ordered pairs (a, o) of positions in `groups` that share the
    same value, including a == o, without a Python loop over groups."""
    order = np.argsort(groups, kind="stable")
    g = groups[order]
    starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
    sizes = np.diff(np.r_[starts, len(g)])
    # for every element, the start and size of its group
    st = np.repeat(starts, sizes)
    sz = np.repeat(sizes, sizes)
    a = np.repeat(np.arange(len(g)), sz)            # each element sz times
    offs = np.arange(len(a)) - np.repeat(np.cumsum(sz) - sz, sz)
    o = np.repeat(st, sz) + offs
    return order[a], order[o]


@dataclass
class BAResult:
    cams: list[np.ndarray]
    points: np.ndarray
    costs: list[float]
    iterations: int
    weights: np.ndarray


def solve(pb: BAProblem, kernel=None, max_iters: int = 30,
          tau: float = 1e-4, tol: float = 1e-8) -> BAResult:
    """Levenberg-Marquardt (Marquardt scaling, Nielsen's rule) on the
    reduced camera system, with IRLS for the robust kernel. Updates and
    returns the cameras and points; `pb` itself is updated in place."""
    kernel = kernel or Trivial()

    def evaluate(cams, points):
        r, A, B, depth = pb.residuals_jacobians(cams, points)
        s = np.sum(r * r, axis=1)
        cost = 0.5 * float(np.sum(kernel.rho(s)))
        if np.any(depth <= 0):
            cost = np.inf
        return cost, r, A, B, kernel.weight(s)

    cost, r, A, B, w = evaluate(pb.cams, pb.points)
    lam, nu = tau, 2.0
    costs = [cost]
    it = 0
    for it in range(1, max_iters + 1):
        U, V, Wk, g_c, g_p = pb.normal_equations(r, A, B, w)
        Dc = np.einsum("mii->mi", U) + 1e-9
        Dp = np.einsum("nii->ni", V) + 1e-9
        Ul = U + lam * np.einsum("mi,ij->mij", Dc, np.eye(6))
        Vl = V + lam * np.einsum("ni,ij->nij", Dp, np.eye(3))
        S, b, Vinv, pos = pb.schur(Ul, Vl, Wk, g_c, g_p)
        dc_free = np.linalg.solve(S, b) if len(b) else np.zeros(0)
        dc, dp = pb.back_substitute(dc_free, pos, Vinv, Wk, g_p)
        free = np.array([i not in pb.fixed for i in range(len(pb.cams))])
        g = np.concatenate([(g_c * free[:, None]).ravel(), g_p.ravel()])
        delta = np.concatenate([dc.ravel(), dp.ravel()])
        D = np.concatenate([(Dc * free[:, None]).ravel(), Dp.ravel()])
        predicted = 0.5 * delta @ (lam * D * delta - g)
        cams_new, pts_new = pb.retract(dc, dp)
        cost_new, r_n, A_n, B_n, w_n = evaluate(cams_new, pts_new)
        gain = (cost - cost_new) / predicted if predicted > 0 else -1.0
        if gain > 0:
            small = cost - cost_new <= tol * cost
            pb.cams, pb.points = cams_new, pts_new
            cost, r, A, B, w = cost_new, r_n, A_n, B_n, w_n
            lam *= max(1.0 / 3.0, 1.0 - (2.0 * gain - 1.0) ** 3)
            nu = 2.0
            costs.append(cost)
            if small:
                break
        else:
            lam *= nu
            nu *= 2.0
            if lam > 1e10:
                break
    return BAResult(pb.cams, pb.points, costs, it, w)


def camera_covariance(pb: BAProblem) -> dict[int, np.ndarray]:
    """Marginal 6x6 covariance of each free camera at the current
    estimate: the diagonal blocks of S^{-1} (Laplace approximation, no
    robust weights)."""
    r, A, B, _ = pb.residuals_jacobians()
    U, V, Wk, g_c, g_p = pb.normal_equations(r, A, B, np.ones(len(r)))
    S, _, _, pos = pb.schur(U, V, Wk, g_c, g_p)
    Sinv = np.linalg.inv(S)
    return {c: Sinv[6 * i:6 * i + 6, 6 * i:6 * i + 6]
            for c, i in pos.items()}


def point_covariance(pb: BAProblem) -> np.ndarray:
    """Marginal 3x3 covariance of every point:
    V_j^{-1} + V_j^{-1} W_j^T S^{-1} W_j V_j^{-1}."""
    r, A, B, _ = pb.residuals_jacobians()
    U, V, Wk, g_c, g_p = pb.normal_equations(r, A, B, np.ones(len(r)))
    S, _, Vinv, pos = pb.schur(U, V, Wk, g_c, g_p)
    Sinv = np.linalg.inv(S)
    nf = len(pos)
    cov = Vinv.copy()
    for p in range(len(pb.points)):
        obs = np.nonzero(pb.pt_idx == p)[0]
        Wj = np.zeros((6 * nf, 3))       # stacked W_ij over free cameras
        for o in obs:
            c = pb.cam_idx[o]
            if c in pos:
                Wj[6 * pos[c]:6 * pos[c] + 6] += Wk[o]
        G = Wj @ Vinv[p]
        cov[p] = Vinv[p] + G.T @ Sinv @ G
    return cov


def dense_system(pb: BAProblem) -> tuple[np.ndarray, np.ndarray]:
    """The full normal matrix H = J^T J and gradient over (free cameras,
    points), for teaching and testing."""
    r, A, B, _ = pb.residuals_jacobians()
    free = pb.free_cameras()
    pos = {c: i for i, c in enumerate(free)}
    nc, n = 6 * len(free), 3 * len(pb.points)
    J = np.zeros((r.size, nc + n))
    d = pb.dim
    for k in range(len(r)):
        rows = slice(d * k, d * k + d)
        c = pb.cam_idx[k]
        if c in pos:
            J[rows, 6 * pos[c]:6 * pos[c] + 6] = A[k]
        p = pb.pt_idx[k]
        J[rows, nc + 3 * p:nc + 3 * p + 3] = B[k]
    return J.T @ J, J.T @ r.ravel()
