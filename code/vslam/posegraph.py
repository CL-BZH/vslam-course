"""Pose-graph optimisation (Chapter 9).

Nodes are poses T_i (SE(3)); an edge (i, j) carries a measurement Z_ij of
the relative pose T_i^{-1} T_j and its information matrix Omega_ij. The
residual is

    e_ij = Log(Z_ij^{-1} T_i^{-1} T_j)              (in R^6)

with Jacobians, for right perturbations of T_i and T_j,

    de/dxi_j =  J_r(e)^{-1},
    de/dxi_i = -J_r(e)^{-1} Ad(T_j^{-1} T_i).

`PoseGraphProblem` plugs into vslam.optim (Gauss-Newton or LM, with robust
kernels acting on whole edges).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import lie


@dataclass
class Edge:
    i: int
    j: int
    Z: np.ndarray            # measured T_i^{-1} T_j
    info: np.ndarray         # 6x6 information matrix
    loop: bool = False


def information(sigma_t: float, sigma_r: float) -> np.ndarray:
    """Diagonal information matrix for (rho, phi) standard deviations."""
    return np.diag([sigma_t ** -2] * 3 + [sigma_r ** -2] * 3)


def edge_residual(Ti: np.ndarray, Tj: np.ndarray, Z: np.ndarray
                  ) -> np.ndarray:
    return lie.se3_log(lie.se3_inv(Z) @ lie.se3_inv(Ti) @ Tj)


def edge_jacobians(Ti: np.ndarray, Tj: np.ndarray, e: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray]:
    Jinv = lie.se3_right_jacobian_inv(e)
    Ji = -Jinv @ lie.se3_adjoint(lie.se3_inv(Tj) @ Ti)
    return Ji, Jinv


@dataclass
class PoseGraphProblem:
    """A pose graph as a least-squares problem for vslam.optim. The state
    is the list of node poses; nodes in `fixed` are held constant (gauge);
    each edge is one robust block of 6 whitened residuals."""

    edges: list[Edge]
    n_nodes: int
    fixed: set = field(default_factory=lambda: {0})
    block_size: int = 6

    def __post_init__(self) -> None:
        self.free = [k for k in range(self.n_nodes) if k not in self.fixed]
        self.col = {k: 6 * c for c, k in enumerate(self.free)}
        # whitening: info = L L^T  =>  r = L^T e
        self.W = [np.linalg.cholesky(e.info).T for e in self.edges]

    def linearize(self, poses):
        m = len(self.edges)
        r = np.zeros(6 * m)
        J = np.zeros((6 * m, 6 * len(self.free)))
        for k, (ed, W) in enumerate(zip(self.edges, self.W)):
            Ti, Tj = poses[ed.i], poses[ed.j]
            e = edge_residual(Ti, Tj, ed.Z)
            Ji, Jj = edge_jacobians(Ti, Tj, e)
            rows = slice(6 * k, 6 * k + 6)
            r[rows] = W @ e
            if ed.i in self.col:
                J[rows, self.col[ed.i]:self.col[ed.i] + 6] += W @ Ji
            if ed.j in self.col:
                J[rows, self.col[ed.j]:self.col[ed.j] + 6] += W @ Jj
        return r, J

    def retract(self, poses, delta):
        out = list(poses)
        for k, c in self.col.items():
            out[k] = poses[k] @ lie.se3_exp(delta[c:c + 6])
        return out


# --------------------------------------------------------------------------
# Building graphs from odometry and from (simulated) place recognition
# --------------------------------------------------------------------------
def odometry_edges(poses: list[np.ndarray], sigma_t: float,
                   sigma_r: float) -> list[Edge]:
    """Edges between consecutive poses, measured from the poses themselves
    (e.g. the keyframes of a VO)."""
    info = information(sigma_t, sigma_r)
    return [Edge(k, k + 1, lie.se3_inv(poses[k]) @ poses[k + 1], info)
            for k in range(len(poses) - 1)]


def covisible_loops(landmark_sets: list[set], true_poses: list[np.ndarray],
                    rng: np.random.Generator, min_gap: int = 20,
                    min_shared: int = 30, sigma_t: float = 0.02,
                    sigma_r: float = 0.005, max_per_node: int = 1
                    ) -> list[Edge]:
    """Simulated loop closures: keyframes i < j with j - i >= min_gap that
    see at least `min_shared` common (structure) landmarks are assumed to
    be recognised (Chapter 10 does it from images), and their relative
    pose is measured with the given noise (Chapter 7's PnP does that)."""
    edges = []
    info = information(sigma_t, sigma_r)
    count = np.zeros(len(true_poses), dtype=int)
    for j in range(len(true_poses)):
        best = []
        for i in range(0, j - min_gap + 1):
            shared = len(landmark_sets[i] & landmark_sets[j])
            if shared >= min_shared:
                best.append((shared, i))
        for shared, i in sorted(best, reverse=True)[:max_per_node]:
            if count[i] >= max_per_node:
                continue
            true = lie.se3_inv(true_poses[i]) @ true_poses[j]
            noise = np.concatenate([rng.normal(0, sigma_t, 3),
                                    rng.normal(0, sigma_r, 3)])
            edges.append(Edge(i, j, true @ lie.se3_exp(noise), info,
                              loop=True))
            count[i] += 1
    return edges


# --------------------------------------------------------------------------
# Marginalisation
# --------------------------------------------------------------------------
def marginalize(H: np.ndarray, b: np.ndarray, keep: np.ndarray,
                drop: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Marginalise the variables `drop` out of the Gaussian with
    information H and information vector b (Lemma 1.8):
    H' = H_kk - H_kd H_dd^{-1} H_dk,  b' = b_k - H_kd H_dd^{-1} b_d."""
    Hkk = H[np.ix_(keep, keep)]
    Hkd = H[np.ix_(keep, drop)]
    Hdd = H[np.ix_(drop, drop)]
    X = np.linalg.solve(Hdd, np.column_stack([Hkd.T, b[drop]]))
    return Hkk - Hkd @ X[:, :-1], b[keep] - Hkd @ X[:, -1]
