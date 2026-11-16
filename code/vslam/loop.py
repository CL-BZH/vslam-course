"""Loop-closure detection and verification (Chapter 10).

For every new keyframe j:

1. Candidates: keyframes i <= j - min_gap (not the recent past), ranked by
   the bag-of-words score s(j, i), normalised by the score s(j, j-1)
   between j and its predecessor -- the score one can expect for "the same
   place" in this part of the garden.
2. Geometric verification of the best candidates: descriptor matching
   (ratio test, mutual check) between the features of j and those of i,
   five-point RANSAC on the bearings, then a two-keyframe stereo bundle
   adjustment. A loop is accepted with at least `min_inliers`
   inliers.
3. The relative pose T_i^{-1} T_j becomes a loop factor of the pose graph.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import ba, bow, features, lie, robust, twoview
from .camera import StereoCamera
from .posegraph import Edge


@dataclass
class KeyframeData:
    z: np.ndarray            # stereo measurements (m, 3)
    desc: np.ndarray         # descriptors (m, 32)
    bow: np.ndarray          # tf-idf vector
    points: np.ndarray       # stereo points in the camera frame (m, 3)
    valid: np.ndarray        # points with a usable depth


@dataclass
class LoopCandidate:
    i: int
    j: int
    score: float
    normalised: float
    inliers: int = 0
    Z: np.ndarray | None = None      # measured T_i^{-1} T_j
    cov: np.ndarray | None = None    # its 6x6 covariance (from the BA)
    pairs: np.ndarray | None = None  # inlier matches (feature in i, in j)


class LoopDetector:
    def __init__(self, vocab: bow.Vocabulary, stereo: StereoCamera,
                 min_gap: int = 20, min_normalised: float = 0.6,
                 n_candidates: int = 3, min_inliers: int = 25,
                 sigma: float = 0.5, use_idf: bool = True,
                 rng: np.random.Generator | None = None) -> None:
        self.voc, self.st = vocab, stereo
        self.min_gap, self.min_norm = min_gap, min_normalised
        self.n_cand, self.min_inliers = n_candidates, min_inliers
        self.sigma, self.use_idf = sigma, use_idf
        self.rng = rng or np.random.default_rng(0)
        self.kfs: list[KeyframeData] = []

    def add(self, z: np.ndarray, desc: np.ndarray) -> int:
        disp = z[:, 0] - z[:, 2]
        valid = disp > 1.0
        pts = np.zeros((len(z), 3))
        if valid.any():
            pts[valid] = np.array([self.st.triangulate(m) for m in z[valid]])
        valid &= (pts[:, 2] > 0.3) & (pts[:, 2] < 20.0)
        self.kfs.append(KeyframeData(z, desc,
                                     self.voc.bow(desc, self.use_idf),
                                     pts, valid))
        return len(self.kfs) - 1

    def candidates(self, j: int) -> list[LoopCandidate]:
        """Ranked candidates for keyframe j (no verification)."""
        if j < self.min_gap or j == 0:
            return []
        v = self.kfs[j].bow
        ref = max(bow.score(v, self.kfs[j - 1].bow), 1e-6)
        out = []
        for i in range(0, j - self.min_gap + 1):
            s = bow.score(v, self.kfs[i].bow)
            out.append(LoopCandidate(i, j, s, s / ref))
        out.sort(key=lambda c: c.normalised, reverse=True)
        return out

    def verify(self, c: LoopCandidate) -> LoopCandidate:
        """Descriptor matching, five-point RANSAC, then a two-keyframe
        bundle adjustment started from the five-point pose.

        Between neighbouring lawn rows the shared landmarks are the far
        ones (the fence across the lawn, the house), which one stereo
        keyframe triangulates badly (sigma_Z grows like Z^2, Chapter 3):
        seen from a viewpoint metres away, their reprojection errors reach
        tens of pixels, and PnP against them fails. The inliers are
        therefore selected with the five-point algorithm, which uses
        bearings only (Chapter 6); its rotation and direction of
        translation, with the scale taken from the stereo depths, give the
        starting pose; and the relative pose comes from a small BA in
        which the matched points are re-estimated from both keyframes'
        stereo measurements.
        """
        A, B = self.kfs[c.i], self.kfs[c.j]
        ia = np.nonzero(A.valid)[0]
        m = features.match(A.desc[ia], B.desc, ratio=0.8, mutual=True)
        if len(m) < max(8, self.min_inliers // 2):
            return c
        ia, ib = ia[m[:, 0]], m[:, 1]
        cam = self.st.cam

        def norm(z):
            return np.column_stack([(z[:, 0] - cam.cx) / cam.fx,
                                    (z[:, 1] - cam.cy) / cam.fy])

        xa, xb = norm(A.z[ia, :2]), norm(B.z[ib, :2])
        E, inl = twoview.five_point_ransac(xa, xb, 1.5 * self.sigma / cam.fx)
        if E is None or inl.sum() < self.min_inliers:
            c.inliers = int(inl.sum()) if E is not None else 0
            return c
        ia, ib = ia[inl], ib[inl]
        # metric starting pose: rotation and direction of translation from
        # E, the scale from the stereo points of both keyframes,
        # q_B = R q_A + s t  =>  s = median of t^T (q_B - R q_A)
        rel = twoview.recover_pose(E, xa[inl], xb[inl])
        both = B.valid[ib]
        if both.sum() < 5:
            return c
        d = (B.points[ib[both]] - A.points[ia[both]] @ rel.R.T) @ rel.t
        T_ba = lie.make_T(rel.R, np.median(d) * rel.t)
        X = A.points[ia]
        n = len(ia)
        pb = ba.BAProblem(cam.K, [np.eye(4), lie.se3_inv(T_ba)],
                          X.copy(), np.r_[np.zeros(n, int), np.ones(n, int)],
                          np.r_[np.arange(n), np.arange(n)],
                          np.vstack([A.z[ia], B.z[ib]]), self.st.baseline,
                          self.sigma, {0})
        sol = ba.solve(pb, kernel=robust.Cauchy(2.0), max_iters=30)
        r, _, _, depth = pb.residuals_jacobians()
        ok = np.sum(r * r, axis=1) < 27.0
        ok = ok[:n] & ok[n:] & (depth[:n] > 0) & (depth[n:] > 0)
        c.inliers = int(ok.sum())
        if c.inliers >= self.min_inliers:
            c.Z = sol.cams[1]
            # Laplace covariance of the relative pose, on the inliers
            keep = np.nonzero(ok)[0]
            c.pairs = np.column_stack([ia[keep], ib[keep]])
            pb2 = ba.BAProblem(cam.K, sol.cams, sol.points[keep],
                               np.r_[np.zeros(len(keep), int),
                                     np.ones(len(keep), int)],
                               np.r_[np.arange(len(keep)),
                                     np.arange(len(keep))],
                               np.vstack([A.z[ia[keep]], B.z[ib[keep]]]),
                               self.st.baseline, self.sigma, {0})
            c.cov = ba.camera_covariance(pb2)[1]
        return c

    def detect(self, j: int) -> LoopCandidate | None:
        """Best verified loop for keyframe j, if any."""
        for c in self.candidates(j)[:self.n_cand]:
            if c.normalised < self.min_norm:
                break
            c = self.verify(c)
            if c.Z is not None:
                return c
        return None


def loop_edges(found: list[LoopCandidate], inflate: float = 4.0
               ) -> list[Edge]:
    """Loop factors with the information of the verification BA, its
    covariance inflated by `inflate` (Chapter 9: covariances from a BA are
    optimistic)."""
    return [Edge(c.i, c.j, c.Z, np.linalg.inv(inflate * c.cov), loop=True)
            for c in found]
