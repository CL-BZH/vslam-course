"""Tests for vslam.posegraph (Chapter 9)."""

import numpy as np
import pytest

from vslam import lie, optim, posegraph as pg, robust

RNG = np.random.default_rng(41)


def square_path(n_side=10, step=0.5):
    """Poses around a square, coming back to the start."""
    poses, T = [], np.eye(4)
    U = lie.make_T(np.eye(3), np.array([step, 0.0, 0.0]))
    turn = lie.make_T(lie.so3_exp(np.array([0, 0, np.pi / 2])), np.zeros(3))
    for side in range(4):
        for _ in range(n_side):
            poses.append(T.copy())
            T = T @ U
        T = T @ turn
    return poses


def test_edge_jacobians():
    Ti = lie.se3_exp(RNG.normal(size=6))
    Tj = lie.se3_exp(RNG.normal(size=6))
    Z = lie.se3_inv(Ti) @ Tj @ lie.se3_exp(0.1 * RNG.normal(size=6))
    e = pg.edge_residual(Ti, Tj, Z)
    Ji, Jj = pg.edge_jacobians(Ti, Tj, e)
    eps = 1e-6
    for J, which in [(Ji, 0), (Jj, 1)]:
        num = []
        for d in eps * np.eye(6):
            a = [Ti, Tj]
            b = [Ti, Tj]
            a[which] = a[which] @ lie.se3_exp(d)
            b[which] = b[which] @ lie.se3_exp(-d)
            num.append((pg.edge_residual(*a, Z) - pg.edge_residual(*b, Z))
                       / (2 * eps))
        assert np.allclose(J, np.column_stack(num), atol=1e-6)


def test_noise_free_graph_is_recovered_exactly():
    truth = square_path()
    edges = pg.odometry_edges(truth, 0.01, 0.001)
    edges.append(pg.Edge(0, len(truth) - 1,
                         lie.se3_inv(truth[0]) @ truth[-1],
                         pg.information(0.01, 0.001), loop=True))
    start = [T @ lie.se3_exp(0.05 * RNG.normal(size=6)) if k else T
             for k, T in enumerate(truth)]
    pb = pg.PoseGraphProblem(edges, len(truth))
    res = optim.levenberg_marquardt(pb, start)
    for Te, Tt in zip(res.x, truth):
        assert np.linalg.norm(lie.se3_log(lie.se3_inv(Tt) @ Te)) < 1e-8


def test_gauge_without_fixed_node():
    truth = square_path(4)
    pb = pg.PoseGraphProblem(pg.odometry_edges(truth, 0.01, 0.001),
                             len(truth), fixed=set())
    _, J = pb.linearize(truth)
    ev = np.linalg.eigvalsh(J.T @ J)
    assert np.sum(ev < 1e-8 * ev[-1]) == 6


def drifting_odometry(truth, rng, yaw_bias=0.004):
    """Odometry estimate: true increments with a small heading bias and
    noise, integrated from the first pose."""
    est = [truth[0].copy()]
    for a, b in zip(truth[:-1], truth[1:]):
        inc = lie.se3_inv(a) @ b
        noise = np.concatenate([rng.normal(0, 0.005, 3),
                                rng.normal(0, 0.001, 3)])
        noise[5] += yaw_bias
        est.append(est[-1] @ inc @ lie.se3_exp(noise))
    return est


def test_loop_closure_removes_drift():
    truth = square_path()
    est = drifting_odometry(truth, np.random.default_rng(1))
    edges = pg.odometry_edges(est, 0.005, 0.002)
    edges.append(pg.Edge(0, len(truth) - 1,
                         lie.se3_inv(truth[0]) @ truth[-1],
                         pg.information(0.01, 0.002), loop=True))
    res = optim.levenberg_marquardt(pg.PoseGraphProblem(edges, len(truth)),
                                    est)
    err = lambda P: max(np.linalg.norm(a[:3, 3] - b[:3, 3])
                        for a, b in zip(P, truth))
    assert err(res.x) < 0.25 * err(est)


def test_robust_kernel_survives_a_false_loop():
    truth = square_path()
    est = drifting_odometry(truth, np.random.default_rng(2), yaw_bias=0.0)
    edges = pg.odometry_edges(est, 0.005, 0.002)
    info = pg.information(0.01, 0.002)
    edges.append(pg.Edge(0, len(truth) - 1,
                         lie.se3_inv(truth[0]) @ truth[-1], info, loop=True))
    # a wrong "recognition": node 5 matched with node 25 as if 30 cm apart
    edges.append(pg.Edge(5, 25, lie.make_T(np.eye(3), [0.3, 0, 0]), info,
                         loop=True))
    err = {}
    for name, k in [("l2", None), ("cauchy", robust.Cauchy(1.0))]:
        res = optim.levenberg_marquardt(
            pg.PoseGraphProblem(edges, len(truth)), est, kernel=k)
        err[name] = max(np.linalg.norm(a[:3, 3] - b[:3, 3])
                        for a, b in zip(res.x, truth))
    assert err["l2"] > 1.0 and err["cauchy"] < 0.2


def test_marginalisation_matches_the_joint_covariance():
    A = RNG.normal(size=(20, 8))
    H = A.T @ A + 0.1 * np.eye(8)
    b = RNG.normal(size=8)
    keep, drop = np.array([0, 1, 2, 5]), np.array([3, 4, 6, 7])
    Hm, bm = pg.marginalize(H, b, keep, drop)
    assert np.allclose(np.linalg.inv(Hm), np.linalg.inv(H)[np.ix_(keep,
                                                                  keep)])
    assert np.allclose(np.linalg.solve(Hm, bm),
                       np.linalg.solve(H, b)[keep])
