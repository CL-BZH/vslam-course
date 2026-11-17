"""Tests for Chapter 15: evaluation metrics and the localisation stack."""

import numpy as np
import pytest
from scipy.stats import chi2

from vslam import evaluation, fusion, imu, lie, sim, stack, vio

RNG = np.random.default_rng(15)


def test_ate_alignments():
    gt = np.cumsum(RNG.normal(0, 0.3, (100, 3)), axis=0)
    yaw, t = 0.4, np.array([2.0, -1.0, 0.3])
    est = (gt - t) @ lie.so3_exp(np.array([0, 0, yaw]))   # inverse motion
    assert evaluation.ate(est, gt, "none").mean() > 0.1
    assert evaluation.ate(est, gt, "4dof").max() < 1e-9
    assert evaluation.ate(est, gt, "se3").max() < 1e-9
    # a roll is not absorbed by the 4-DoF alignment
    est2 = gt @ lie.so3_exp(np.array([0.05, 0, 0])).T
    assert evaluation.ate(est2, gt, "4dof").mean() > \
        10 * evaluation.ate(est2, gt, "se3").mean()
    with pytest.raises(ValueError):
        evaluation.ate(est, gt, "bogus")


def test_rpe_distance_is_zero_for_a_rigidly_moved_trajectory():
    Tg = [lie.make_T(lie.so3_exp(np.array([0, 0, 0.1 * k])),
                     np.array([np.cos(0.1 * k), np.sin(0.1 * k), 0]) * k * 0.3)
          for k in range(60)]
    G = lie.make_T(lie.so3_exp(np.array([0.2, -0.1, 1.0])),
                   np.array([3.0, 1.0, -2.0]))
    Te = [G @ T for T in Tg]
    et, er = evaluation.rpe_distance(Te, Tg, 2.0)
    assert len(et) > 10 and et.max() < 1e-9 and er.max() < 1e-9


def test_nees_of_consistent_samples_is_chi_square():
    n, d = 4000, 3
    A = RNG.normal(size=(n, d, d))
    C = A @ np.transpose(A, (0, 2, 1)) + 0.1 * np.eye(d)
    L = np.linalg.cholesky(C)
    e = np.einsum("nij,nj->ni", L, RNG.normal(size=(n, d)))
    ne = evaluation.nees(e, C)
    lo, hi = evaluation.anees_bounds(n, d)
    assert lo < ne.mean() < hi
    assert abs(np.mean(ne > chi2.ppf(0.99, d)) - 0.01) < 0.006


def test_nis_gate():
    S = np.diag([0.01, 0.01, 0.04])
    q, ok = evaluation.nis_gate(np.array([0.05, 0.0, 0.0]), S)
    assert np.isclose(q, 0.25) and ok
    q, ok = evaluation.nis_gate(np.array([1.0, 0.0, 0.0]), S)
    assert not ok


def test_stack_rejects_an_outlier_and_reports_covariance():
    """Two seconds of driving with the full stack; a 2 m GNSS jump is
    rejected by the gate, good fixes are accepted, and the reported
    covariance is positive definite."""
    m = sim.GardenMotion()
    P = imu.ImuParams()
    rng = np.random.default_rng(0)
    data = imu.simulate(m, 5.0, P, rng, bg0=rng.normal(0, 0.005, 3))
    world = sim.garden(np.random.default_rng(0))
    S = sim.GardenSim()
    Tbc = sim.camera_mount()
    ts = np.arange(0, 5.0, 0.05)
    frames = S.run(world, [m.T(t) for t in ts], Tbc,
                   np.random.default_rng(1))
    lever = np.array([-0.4, 0.0, 1.4])
    s0 = vio.KFState(m.R(2.0), np.zeros(3), m.p(2.0), data.bg[400],
                     np.zeros(3))
    st = stack.LocalisationStack(S.stereo, Tbc, P, lever)
    sig = np.r_[[1e-2] * 2, [1e-6], [1e-3] * 3, [1e-6] * 3, [1e-3] * 3,
                [0.1] * 3]
    st.start(2.0, s0, sig, frames[40].track_ids, frames[40].z, 0.0)
    last = 400
    for i, k in enumerate(range(45, len(frames), 5)):
        t = ts[k]
        n = int(round(t / P.dt))
        ant = m.p(t) + m.R(t) @ lever
        zg = ant + rng.normal(0, 0.015, 3)
        if i == 6:
            zg = zg + np.array([2.0, 0.0, 0.0])           # multipath
        speed = (m.R(t).T @ m.v(t))[0]
        rep = st.step(t, data.gyro[last:n], data.accel[last:n],
                      frames[k].track_ids, frames[k].z, speed,
                      (zg, np.array([0.015, 0.015, 0.03])))
        last = n
        assert np.all(np.linalg.eigvalsh(rep.cov_p) > 0)
    dec = [r.gnss for r in st.reports]
    assert dec[6] == "rejected"
    assert dec.count("accepted") >= len(dec) - 2
    assert np.linalg.norm(st.reports[-1].state.p - m.p(st.reports[-1].t)) \
        < 0.03
