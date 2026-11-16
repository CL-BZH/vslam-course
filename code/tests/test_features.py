"""Tests for the front-end of Chapter 5: corners, BRIEF, matching, LK,
RANSAC."""

import numpy as np
import pytest

from vslam import features as ft
from vslam import flow, image, ransac

RNG = np.random.default_rng(5)


def square_image():
    img = np.zeros((100, 100))
    img[30:70, 30:70] = 1.0
    return img


def test_structure_tensor_distinguishes_corner_edge_flat():
    lmin, lmax = ft.eigenvalues(*ft.structure_tensor(square_image(), 1.5))
    corner, edge, flat = (30, 30), (50, 30), (50, 50)  # (v, u)
    assert lmin[corner] > 10 * lmin[edge]
    assert lmax[edge] > 10 * lmin[edge]
    assert lmax[flat] < 1e-12


def test_detect_finds_the_four_corners():
    kps = ft.detect(ft.shi_tomasi(square_image()), n=4, border=5)
    expected = np.array([[29.5, 29.5], [69.5, 29.5], [29.5, 69.5],
                         [69.5, 69.5]])
    for e in expected:
        # smoothing by the structure tensor pulls the maxima ~1.5 px inside
        assert np.min(np.linalg.norm(kps - e, axis=1)) < 3.0


def test_orientation_is_covariant_with_rotation():
    # odd size: np.rot90 maps the centre pixel (100, 100) to itself
    img = image.garden_texture(201, 201, np.random.default_rng(1))
    kp = np.array([[100.0, 100.0]])
    a0 = ft.orientation(img, kp)[0]
    # np.rot90 maps pixel (u, v) to (v, W-1-u): a rotation by -90 deg in
    # the (u, v) frame, so angles decrease by pi/2
    a1 = ft.orientation(np.rot90(img), kp)[0]
    assert np.isclose(np.angle(np.exp(1j * (a1 - a0 + np.pi / 2))), 0,
                      atol=1e-6)


def test_steered_brief_is_rotation_invariant():
    img = image.garden_texture(201, 201, np.random.default_rng(2))
    kp = np.array([[100.0, 100.0]])
    rot = np.rot90(img)
    d0 = ft.describe(img, kp, ft.orientation(img, kp))
    d1 = ft.describe(rot, kp, ft.orientation(rot, kp))
    d_unsteered = ft.describe(rot, kp, None)
    assert ft.hamming(d0, d1)[0, 0] < 15
    assert ft.hamming(d0, d_unsteered)[0, 0] > 60


def test_hamming_matches_naive():
    d1 = RNG.integers(0, 256, size=(5, 32), dtype=np.uint8)
    d2 = RNG.integers(0, 256, size=(7, 32), dtype=np.uint8)
    naive = np.array([[np.unpackbits(a ^ b).sum() for b in d2] for a in d1])
    assert np.array_equal(ft.hamming(d1, d2), naive)


def test_matching_under_a_homography():
    K = np.array([[500.0, 0, 320], [0, 500, 240], [0, 0, 1]])
    H = image.plane_homography(K, np.radians(15))
    img0 = image.garden_texture(rng=np.random.default_rng(3))
    img1 = image.warp(img0, H)
    k0 = ft.detect(ft.shi_tomasi(img0), 400)
    k1 = ft.detect(ft.shi_tomasi(img1), 400)
    d0 = ft.describe(img0, k0, ft.orientation(img0, k0))
    d1 = ft.describe(img1, k1, ft.orientation(img1, k1))
    m = ft.match(d0, d1)
    err = np.linalg.norm(image.apply_homography(H, k0[m[:, 0]])
                         - k1[m[:, 1]], axis=1)
    assert len(m) > 100 and np.mean(err < 3.0) > 0.8


@pytest.mark.parametrize("shift", [(0.3, -0.7), (2.5, 1.25)])
def test_lucas_kanade_subpixel(shift):
    img0 = image.garden_texture(rng=np.random.default_rng(4))
    H = np.array([[1, 0, shift[0]], [0, 1, shift[1]], [0, 0, 1.0]])
    img1 = image.warp(img0, H)
    pts = ft.detect(ft.shi_tomasi(img0), 100, border=40)
    new, ok = flow.lucas_kanade(img0, img1, pts, levels=1)
    err = np.linalg.norm(new - pts - shift, axis=1)
    assert ok.mean() > 0.9 and np.median(err[ok]) < 0.05


def test_pyramid_is_needed_for_large_motion():
    img0 = image.garden_texture(rng=np.random.default_rng(6))
    shift = np.array([14.0, -9.0])
    img1 = image.warp(img0, np.array([[1, 0, shift[0]], [0, 1, shift[1]],
                                      [0, 0, 1.0]]))
    pts = ft.detect(ft.shi_tomasi(img0), 100, border=60)
    errs = {}
    for levels in (1, 4):
        new, ok = flow.lucas_kanade(img0, img1, pts, levels=levels)
        errs[levels] = np.median(np.linalg.norm(new - pts - shift, axis=1))
    assert errs[4] < 0.1 < errs[1]


def test_required_iterations():
    assert ransac.required_iterations(0.5, 4, 0.99) == 72
    assert ransac.required_iterations(1.0, 4) == 1
    assert ransac.required_iterations(0.9, 8, 0.99) == 9


def test_homography_ransac_with_half_outliers():
    K = np.array([[500.0, 0, 320], [0, 500, 240], [0, 0, 1]])
    H = image.plane_homography(K, np.radians(20))
    src = RNG.uniform([0, 0], [640, 480], size=(200, 2))
    dst = image.apply_homography(H, src) + RNG.normal(0, 0.5, (200, 2))
    out = RNG.random(200) < 0.5
    dst[out] = RNG.uniform([0, 0], [640, 480], size=(out.sum(), 2))
    res = ransac.homography_ransac(src, dst, 2.0,
                                   rng=np.random.default_rng(0))
    assert res.inliers[~out].mean() > 0.95
    assert res.inliers[out].mean() < 0.02
    grid = np.array([[u, v] for u in range(0, 641, 80)
                     for v in range(0, 481, 80)], dtype=float)
    transfer = image.apply_homography(res.model, grid) \
        - image.apply_homography(H, grid)
    assert np.abs(transfer).max() < 2.0  # 0.5 px noise, image corners
