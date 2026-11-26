"""Distances between atoms in space, a block of rows at a time.

``np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)`` holds the difference of every pair before it
holds one distance: rows of ``a`` times rows of ``b`` times three numbers. For a ligand against its pocket
that is nothing; for a chain of a thousand residues against a complex it is gigabytes. The functions here
give the same numbers, pair for pair to the last bit (the arithmetic of a pair does not depend on the
block it is in), and never hold more than :data:`BLOCK` numbers of differences at once.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np

__all__ = ["BLOCK", "distance_matrix", "nearest_distance", "pairs_within"]

#: Numbers in one block of differences: 8 MB of floats (the square and the sum that follow are of that size too).
BLOCK = 1_000_000


def _points(xyz: object) -> np.ndarray:
    return np.asarray(xyz, dtype=float).reshape(-1, 3)


def _blocks(a: np.ndarray, b: np.ndarray) -> Iterator[tuple[int, np.ndarray]]:
    """``(first row, distances)`` from a block of rows of ``a`` to every row of ``b``."""
    rows = max(1, BLOCK // (3 * max(1, len(b))))
    for start in range(0, len(a), rows):
        yield start, np.linalg.norm(a[start:start + rows, None, :] - b[None, :, :], axis=2)


def distance_matrix(a: object, b: object) -> np.ndarray:
    """The distance from each point of ``a`` to each of ``b``: ``(len(a), len(b))``."""
    a, b = _points(a), _points(b)
    out = np.empty((len(a), len(b)), dtype=float)
    for start, d in _blocks(a, b):
        out[start:start + len(d)] = d
    return out


def nearest_distance(a: object, b: object) -> np.ndarray:
    """For each point of ``a`` the distance to the nearest point of ``b``; infinite where ``b`` has none."""
    a, b = _points(a), _points(b)
    out = np.full(len(a), np.inf, dtype=float)
    if len(b):
        for start, d in _blocks(a, b):
            out[start:start + len(d)] = d.min(axis=1)
    return out


def pairs_within(a: object, b: object, cutoff: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The pairs of a point of ``a`` and a point of ``b`` at most ``cutoff`` apart: the index in ``a``, the
    index in ``b`` and the distance, in the order of the first index and then of the second."""
    a, b = _points(a), _points(b)
    first: list[np.ndarray] = []
    second: list[np.ndarray] = []
    apart: list[np.ndarray] = []
    if len(b):
        for start, d in _blocks(a, b):
            i, j = np.nonzero(d <= cutoff)
            first.append(i + start)
            second.append(j)
            apart.append(d[i, j])
    if not first:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int), np.zeros(0, dtype=float)
    return np.concatenate(first), np.concatenate(second), np.concatenate(apart)
