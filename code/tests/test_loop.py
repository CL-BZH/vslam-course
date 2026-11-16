"""Tests for Chapter 10: appearance simulation, bag of words, loops."""

import numpy as np
import pytest

from vslam import bow, features, lie, loop, sim

RNG = np.random.default_rng(51)


@pytest.fixture(scope="module")
def garden_run():
    world = sim.garden(np.random.default_rng(0))
    base = sim.landmark_descriptors(world, np.random.default_rng(1))
    S = sim.GardenSim()
    frames = S.run(world, sim.boustrophedon(), sim.camera_mount(),
                   np.random.default_rng(1))
    return world, base, S, frames


def test_descriptor_simulation_and_aliasing(garden_run):
    world, base, S, frames = garden_run
    rng = np.random.default_rng(2)
    f = frames[100]
    d1 = sim.observe_descriptors(base, f, world, rng)
    d2 = sim.observe_descriptors(base, f, world, rng)
    same = np.diag(features.hamming(d1, d2))[~f.outlier]
    assert np.median(same) < 50          # ~2 * 0.08 * 256 = 41
    fence = np.nonzero(world.kind == "fence")[0]
    house = np.nonzero(world.kind == "house")[0]
    Df = features.hamming(base[fence[:200]], base[fence[:200]])
    Dh = features.hamming(base[house[:100]], base[house[:100]])
    off = ~np.eye(len(Df), dtype=bool)
    # repetitive fence: many near-identical pairs; house: none
    assert np.mean(Df[off] < 40) > 0.01
    assert np.mean(Dh[~np.eye(len(Dh), dtype=bool)] < 40) == 0.0


def test_kmajority_decreases_cost():
    centres = RNG.integers(0, 256, size=(5, 32), dtype=np.uint8)
    bits = np.unpackbits(centres, axis=1)
    pts = []
    for c in bits:
        flip = RNG.random((60, 256)) < 0.1
        pts.append(np.packbits(c ^ flip, axis=1))
    desc = np.vstack(pts)
    _, costs = bow.kmajority(desc, 5, iters=6, rng=np.random.default_rng(0))
    assert np.all(np.diff(costs) <= 0)


def test_score_properties():
    k = 50
    v = RNG.random(k)
    v /= v.sum()
    w = RNG.random(k)
    w /= w.sum()
    assert np.isclose(bow.score(v, v), 1.0)
    assert np.isclose(bow.score(v, w), bow.score(w, v))
    a, b = np.zeros(k), np.zeros(k)
    a[:10], b[10:20] = 0.1, 0.1
    assert np.isclose(bow.score(a, b), 0.0)
    assert 0.0 <= bow.score(v, w) <= 1.0


def test_idf_of_an_ubiquitous_word_is_zero():
    words = RNG.integers(0, 256, size=(3, 32), dtype=np.uint8)
    voc = bow.Vocabulary(words)
    images = [np.vstack([words[0], words[i % 2 + 1]]) for i in range(10)]
    present = np.zeros(3)
    for d in images:
        present[np.unique(voc.quantise(d))] += 1
    voc.idf = np.log(len(images) / present)
    assert np.isclose(voc.idf[0], 0.0) and voc.idf[1] > 0


def test_verification_accepts_a_revisit_and_rejects_a_stranger(garden_run):
    world, base, S, frames = garden_run
    rng = np.random.default_rng(5)
    voc = bow.Vocabulary(base[:50].copy())
    for (i, j), expect in [((10, 500), True), ((10, 300), False)]:
        det = loop.LoopDetector(voc, S.stereo, rng=np.random.default_rng(0))
        for k in (i, j):
            det.add(frames[k].z, sim.observe_descriptors(base, frames[k],
                                                         world, rng))
        c = det.verify(loop.LoopCandidate(0, 1, 0.0, 0.0))
        if expect:
            G = lie.se3_inv(frames[i].T_wc) @ frames[j].T_wc
            assert c.Z is not None
            assert np.linalg.norm(c.Z[:3, 3] - G[:3, 3]) < 0.3
            assert c.cov is not None and np.all(np.linalg.eigvalsh(c.cov) > 0)
        else:
            assert c.Z is None
