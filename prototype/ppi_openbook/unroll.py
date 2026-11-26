"""The wall around a peptide, unrolled along the peptide's own path.

A SKETCH, not part of bindview: nothing here is tested or promised.

For a partner that lies inside the other, or whose path bends, no plane shows the interface. Here the
second partner (one chain) is taken as a path: the middles of four alpha carbons in a row, which is the
axis of a helix and the middle line of a strand, smoothed. Every place near it is then known by three
numbers: how far along the path it is, at which angle around the path, and how far from it. The map has
the first down the page and the second across it, so the wall around the peptide is slit open along the
path and laid flat, seen from the peptide. The angle is counted in a frame that is carried along the
path without twisting, and the slit is put where the wall has its widest gap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import book

STEP = 0.5            # ångström between two points of the path
BEYOND = 7.0          # the path goes on straight this far past its ends
REACH = 7.0           # the wall is what lies within this of the usual distance of the interface from the path
LONGEST = 60          # residues: a longer chain is taken to be folded, and not a path


@dataclass
class Frame:
    """An unrolled complex. Across the page the angle around the path (as a length at the radius
    ``radius``), up the page the way back along the path, towards the reader the way to the path."""

    xyz: np.ndarray                      # every atom in the frame; where ``wall`` is false it means nothing
    wall: np.ndarray                     # the atoms of the first partner that are drawn
    at: "dict[int, np.ndarray]"          # where every residue of the interface lies on the page
    bounds: "tuple[float, float, float, float]"
    radius: float
    width: float                         # once around, in the units of the page
    length: float                        # the length of the path between its first and last residue


def _unit(v: np.ndarray) -> np.ndarray:
    return v / max(1e-12, float(np.linalg.norm(v)))


def path_of(points: np.ndarray) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    """A smooth path through the points: places, how far along each is, and at each the direction of the
    path and two directions across it that do not twist."""
    n = len(points)
    window = 4 if n >= 6 else 2
    mid = np.array([points[k:k + window].mean(axis=0) for k in range(n - window + 1)])
    ends = [mid[0] - _unit(mid[1] - mid[0]) * (BEYOND + 1.5 * window), mid[-1] + _unit(mid[-1] - mid[-2]) * (BEYOND + 1.5 * window)]
    knots = np.vstack([ends[0], mid, ends[1]])
    along = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(knots, axis=0), axis=1))])
    s = np.arange(0.0, along[-1] + STEP, STEP)
    p = np.stack([np.interp(s, along, knots[:, k]) for k in range(3)], axis=1)
    for _ in range(3):                                   # round the corners of the polygon
        p[1:-1] = (p[:-2] + 2.0 * p[1:-1] + p[2:]) / 4.0
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
    t = np.gradient(p, axis=0)
    t /= np.linalg.norm(t, axis=1)[:, None]
    first = np.cross(t[0], [1.0, 0.0, 0.0])
    if np.linalg.norm(first) < 0.3:
        first = np.cross(t[0], [0.0, 1.0, 0.0])
    n1 = np.empty_like(p)
    n1[0] = _unit(first)
    for k in range(1, len(p)):                           # carried along: what is left of it across the new direction
        n1[k] = _unit(n1[k - 1] - (n1[k - 1] @ t[k]) * t[k])
    return p, s, t, n1, np.cross(t, n1)


def place(x: np.ndarray, path) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """For every point: how far along the path, at which angle around it, how far from it."""
    p, s, t, n1, n2 = path
    x = np.atleast_2d(x)
    k = np.empty(len(x), dtype=int)
    for i in range(0, len(x), 2000):
        k[i:i + 2000] = np.argmin(np.linalg.norm(x[i:i + 2000, None, :] - p[None, :, :], axis=2), axis=1)
    off = x - p[k]
    ds = np.einsum("ij,ij->i", off, t[k])
    r = off - ds[:, None] * t[k]
    return s[k] + ds, np.arctan2(np.einsum("ij,ij->i", r, n2[k]), np.einsum("ij,ij->i", r, n1[k])), np.linalg.norm(r, axis=1)


def chain_of(cx: book.Complex):
    """The second partner as a path: its residues in order, their alpha carbons, and the path through them."""
    order = sorted((r for r in cx.residues if r.side == 1), key=lambda r: (r.chain, r.number, r.icode))
    if len({r.chain for r in order}) != 1:
        raise ValueError("a complex is unrolled along one chain, and the second partner here has several")
    if len(order) < 4:
        raise ValueError(f"a chain of {len(order)} residues is too short to unroll anything along")
    if len(order) > LONGEST:
        raise ValueError(f"the second partner has {len(order)} residues: a complex is unrolled along a peptide "
                         f"(up to {LONGEST} residues), which is a path; a folded protein is not one")
    ca = np.array([cx.xyz[r.ca] if r.ca >= 0 else cx.xyz[r.atoms].mean(axis=0) for r in order])
    return order, ca, path_of(ca)


def frame(cx: book.Complex, margin: float = 6.0) -> Frame:
    """The complex unrolled along its second partner, which is one chain of four residues or more."""
    order, ca, path = chain_of(cx)
    s, angle, away = place(cx.xyz, path)
    a = cx.side == 0
    near = np.zeros(len(cx.xyz), dtype=bool)
    for r in cx.residues:
        near[r.near] = True
    radius = float(np.median(away[a & near]))
    s_ca, _, _ = place(ca, path)
    s0, s1 = float(s_ca.min()), float(s_ca.max())

    # The slit goes where the wall has its widest gap: the angles of the first partner's residues, sorted.
    mine = [r for r in cx.residues if r.side == 0 and len(r.near)]
    middles = np.array([cx.xyz[r.near].mean(axis=0) for r in mine])
    _, where, _ = place(middles, path)
    turned = np.sort(where)
    gaps = np.diff(np.concatenate([turned, [turned[0] + 2.0 * np.pi]]))
    widest = int(np.argmax(gaps))
    slit = turned[widest] + gaps[widest] / 2.0

    def across(theta: np.ndarray) -> np.ndarray:
        return (np.mod(theta - slit, 2.0 * np.pi) - np.pi) * radius

    xyz = np.stack([across(angle), -(s - s0), -away], axis=1)
    wall = a & (away <= radius + REACH) & (s >= s0 - BEYOND) & (s <= s1 + BEYOND)
    at = {}
    for r, m in zip(mine, middles):
        sm, th, _ = place(m, path)
        at[r.index] = np.array([float(across(th)[0]), -float(sm[0] - s0)])
    for r, own in zip(order, s_ca):
        if len(r.near):                                  # down the page by its own place in the chain
            _, th, _ = place(cx.xyz[r.near].mean(axis=0), path)
            at[r.index] = np.array([float(across(th)[0]), -float(own - s0)])
    xs, ys = [q[0] for q in at.values()], [q[1] for q in at.values()]
    half = np.pi * radius
    bounds = (max(min(xs) - margin, -half), min(max(xs) + margin, half), min(ys) - margin, max(ys) + margin)
    return Frame(xyz, wall, at, bounds, radius, 2.0 * half, s1 - s0)
