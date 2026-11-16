"""Bag of binary words for place recognition (Chapter 10).

A vocabulary of k "words" is learnt offline by clustering binary
descriptors (k-majority: Lloyd's algorithm with the Hamming distance and
bitwise majority as the centre). An image becomes the histogram of the
words of its descriptors, weighted by tf-idf and normalised to unit L1
norm; two images are compared by the DBoW2 score

    s(v, w) = 1 - 1/2 |v - w|_1   in [0, 1].
"""

from __future__ import annotations

import numpy as np

from .features import hamming


def _assign(desc: np.ndarray, centres: np.ndarray, chunk: int = 2000
            ) -> tuple[np.ndarray, np.ndarray]:
    """Nearest centre (Hamming) of each descriptor, and its distance."""
    lab = np.empty(len(desc), dtype=int)
    dist = np.empty(len(desc), dtype=int)
    for a in range(0, len(desc), chunk):
        D = hamming(desc[a:a + chunk], centres)
        lab[a:a + chunk] = D.argmin(axis=1)
        dist[a:a + chunk] = D.min(axis=1)
    return lab, dist


def kmajority(desc: np.ndarray, k: int, iters: int = 10,
              rng: np.random.Generator | None = None
              ) -> tuple[np.ndarray, list[float]]:
    """Cluster packed binary descriptors into k centres. Returns the
    centres and the total Hamming cost after each iteration."""
    rng = rng or np.random.default_rng(0)
    centres = desc[rng.choice(len(desc), k, replace=False)].copy()
    costs = []
    for _ in range(iters):
        lab, dist = _assign(desc, centres)
        costs.append(float(dist.sum()))
        bits = np.unpackbits(desc, axis=1)
        for c in range(k):
            members = bits[lab == c]
            if len(members) == 0:          # re-seed an empty cluster
                centres[c] = desc[rng.integers(len(desc))]
            else:
                centres[c] = np.packbits(members.mean(axis=0) > 0.5)
    lab, dist = _assign(desc, centres)
    costs.append(float(dist.sum()))
    return centres, costs


class Vocabulary:
    """Flat vocabulary with idf weights learnt on training images."""

    def __init__(self, words: np.ndarray) -> None:
        self.words = words
        self.idf = np.ones(len(words))

    @classmethod
    def train(cls, images: list[np.ndarray], k: int = 500, iters: int = 10,
              rng: np.random.Generator | None = None) -> "Vocabulary":
        """`images` is a list of descriptor arrays, one per training
        image; the idf of word w is log(N / N_w), N_w the number of
        training images in which w occurs."""
        words, _ = kmajority(np.vstack(images), k, iters, rng)
        voc = cls(words)
        present = np.zeros(k)
        for d in images:
            present[np.unique(voc.quantise(d))] += 1
        voc.idf = np.log(len(images) / np.maximum(present, 1.0))
        return voc

    def quantise(self, desc: np.ndarray) -> np.ndarray:
        return _assign(desc, self.words)[0]

    def bow(self, desc: np.ndarray, use_idf: bool = True) -> np.ndarray:
        """tf-idf vector of an image, unit L1 norm (dense, length k)."""
        k = len(self.words)
        tf = np.bincount(self.quantise(desc), minlength=k) / max(len(desc), 1)
        v = tf * (self.idf if use_idf else 1.0)
        s = v.sum()
        return v / s if s > 0 else v


def score(v: np.ndarray, w: np.ndarray) -> float:
    """DBoW2 L1 score between two unit-L1 vectors: 1 - |v - w|_1 / 2."""
    return 1.0 - 0.5 * float(np.abs(v - w).sum())


def score_matrix(V: np.ndarray) -> np.ndarray:
    """Scores between all rows of V (n, k)."""
    n = len(V)
    S = np.empty((n, n))
    for i in range(n):
        S[i] = 1.0 - 0.5 * np.abs(V - V[i]).sum(axis=1)
    return S
