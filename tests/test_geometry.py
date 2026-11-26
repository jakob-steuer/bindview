"""Unit tests for the pure geometry kernels (no chemistry, no drawing)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from bindview import geometry as g


def test_point_segment_distance():
    assert g.point_segment_distance((0, 1), (-1, 0), (1, 0)) == pytest.approx(1.0)
    assert g.point_segment_distance((3, 0), (-1, 0), (1, 0)) == pytest.approx(2.0)      # beyond the end
    assert g.point_segment_distance((2, 2), (1, 1), (1, 1)) == pytest.approx(math.sqrt(2))  # degenerate segment


def test_segments_intersect_is_strict():
    assert g.segments_intersect((0, 0), (2, 2), (0, 2), (2, 0))
    assert not g.segments_intersect((0, 0), (1, 1), (2, 2), (3, 3))        # collinear, apart
    assert not g.segments_intersect((0, 0), (1, 0), (1, 0), (1, 1))        # sharing an end point is not a crossing
    assert not g.segments_intersect((0, 0), (1, 0), (0, 1), (1, 1))        # parallel


def test_segment_segment_distance():
    assert g.segment_segment_distance((0, 0), (1, 0), (0, 1), (1, 1)) == pytest.approx(1.0)
    assert g.segment_segment_distance((0, 0), (2, 2), (0, 2), (2, 0)) == 0.0
    assert g.segment_segment_distance((0, 0), (1, 0), (3, 0), (4, 0)) == pytest.approx(2.0)


def test_segment_box_entry():
    box = (-1.0, -1.0, 1.0, 1.0)
    assert g.segment_box_entry((10, 0), (0, 0), box) == pytest.approx((1.0, 0.0))
    assert g.segment_box_entry((0, -5), (0, 0), box) == pytest.approx((0.0, -1.0))
    hit = g.segment_box_entry((5, 5), (0, 0), box)
    assert hit == pytest.approx((1.0, 1.0))
    assert g.segment_box_entry((0.5, 0.5), (0, 0), box) == (0.5, 0.5)       # already inside


def test_free_direction_points_into_the_widest_gap():
    # Two bonds at 0 and 90 degrees: the free direction is their opposite bisector.
    d = g.free_direction((0, 0), [(1, 0), (0, 1)])
    assert d == pytest.approx((-math.sqrt(0.5), -math.sqrt(0.5)))
    # One bond: straight away from it.
    assert g.free_direction((0, 0), [(1, 0)]) == pytest.approx((-1.0, 0.0), abs=1e-9)
    mid, width = g.largest_angular_gap([0.0, math.pi / 2, math.pi])
    assert width == pytest.approx(math.pi)
    assert mid == pytest.approx(1.5 * math.pi)


@pytest.mark.parametrize("degenerate", ["none", "first", "second", "both", "mixed"])
def test_capsule_gaps_matches_scalar_reference(degenerate):
    """Discs are capsules of zero length; every combination must agree with the scalar code.

    (A kernel that mishandled a zero-length *second* segment would let interaction
    lines pass straight through disc-shaped bodies.)
    """
    rng = np.random.default_rng(7)
    a0, a1 = rng.uniform(-50, 50, (12, 2)), rng.uniform(-50, 50, (12, 2))
    b0, b1 = rng.uniform(-50, 50, (9, 2)), rng.uniform(-50, 50, (9, 2))
    if degenerate in ("first", "both"):
        a1 = a0.copy()
    if degenerate in ("second", "both"):
        b1 = b0.copy()
    if degenerate == "mixed":
        a1[::2] = a0[::2]
        b1[1::2] = b0[1::2]
    ra, rb = rng.uniform(0, 5, 12), rng.uniform(0, 5, 9)
    got = g.capsule_gaps(a0, a1, ra, b0, b1, rb)
    for i in range(12):
        for j in range(9):
            ref = g.segment_segment_distance(tuple(a0[i]), tuple(a1[i]), tuple(b0[j]), tuple(b1[j])) - ra[i] - rb[j]
            assert got[i, j] == pytest.approx(ref, abs=1e-9)


def test_line_through_a_disc_is_seen_from_either_side():
    line0, line1 = np.array([[-50.0, -140.0]]), np.array([[-26.0, -69.0]])
    disc = np.array([[-43.0, -115.0]])
    r_line, r_disc = np.zeros(1), np.array([16.5])
    one = g.capsule_gaps(line0, line1, r_line, disc, disc, r_disc)[0, 0]
    two = g.capsule_gaps(disc, disc, r_disc, line0, line1, r_line)[0, 0]
    assert one == pytest.approx(two) and one < -14.0
    assert g.segments_distance(line0, line1, disc, disc)[0] == pytest.approx(one + 16.5)


def test_points_segments_distance_matches_scalar_reference():
    rng = np.random.default_rng(3)
    pts, a, b = rng.uniform(-9, 9, (7, 2)), rng.uniform(-9, 9, (5, 2)), rng.uniform(-9, 9, (5, 2))
    got = g.points_segments_distance(pts, a, b)
    for i in range(7):
        for j in range(5):
            assert got[i, j] == pytest.approx(g.point_segment_distance(tuple(pts[i]), tuple(a[j]), tuple(b[j])))


def test_marching_squares_recovers_a_circle():
    xs = np.arange(-3.0, 3.0001, 0.05)
    X, Y = np.meshgrid(xs, xs)
    loops = g.marching_squares(np.hypot(X, Y), 2.0, -3.0, -3.0, 0.05)
    assert len(loops) == 1
    assert g.polyline_length(loops[0], closed=True) == pytest.approx(4 * math.pi, rel=2e-3)
    assert abs(g.polygon_area(loops[0])) == pytest.approx(4 * math.pi, rel=2e-3)
    assert np.hypot(loops[0][:, 0], loops[0][:, 1]) == pytest.approx(2.0, abs=2e-3)


def test_marching_squares_separates_two_blobs_and_is_deterministic():
    xs = np.arange(-6.0, 6.0001, 0.1)
    X, Y = np.meshgrid(xs, xs)
    field = np.minimum(np.hypot(X - 3, Y), np.hypot(X + 3, Y))
    loops = g.marching_squares(field, 1.0, -6.0, -6.0, 0.1)
    assert len(loops) == 2
    again = g.marching_squares(field, 1.0, -6.0, -6.0, 0.1)
    assert all(np.array_equal(p, q) for p, q in zip(loops, again))


def test_resample_closed_is_uniform():
    square = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]])
    pts = g.resample_closed(square, 0.5)
    step = np.linalg.norm(np.diff(np.vstack([pts, pts[:1]]), axis=0), axis=1)
    assert len(pts) == 32
    assert step.max() <= 0.5 + 1e-9


def test_spread_on_line_is_the_exact_minimiser():
    assert g.spread_on_line([0, 0, 0], [1, 1, 1]) == pytest.approx([-1.0, 0.0, 1.0])
    assert g.spread_on_line([0, 10], [1, 1]) == pytest.approx([0.0, 10.0])        # already apart: untouched
    # A heavier interval moves less.
    left, right = g.spread_on_line([0.0, 0.0], [2.0, 2.0], [3.0, 1.0])
    assert right - left == pytest.approx(2.0)
    assert abs(left) < abs(right)


@pytest.mark.parametrize("targets,widths", [
    ([1.0, 1.1, 1.2, 5.0], [1, 1, 1, 1]),
    ([9.8, 0.1, 0.2], [1, 1, 1]),
    ([0.0, 0.0, 0.0, 0.0], [2, 2, 2, 2]),
    ([3.0, 3.0, 3.1, 3.2, 3.3, 8.0], [1.5, 1.0, 1.0, 1.0, 1.5, 1.0]),
])
def test_spread_on_circle_removes_overlap_and_keeps_order(targets, widths):
    period = 10.0
    out = g.spread_on_circle(targets, widths, period)
    n = len(out)
    order = sorted(range(n), key=lambda i: (targets[i] % period, i))
    for k in range(n):
        i, j = order[k], order[(k + 1) % n]
        gap = (out[j] - out[i]) % period
        if n == 2 and k == 1:
            continue
        assert gap >= (widths[i] + widths[j]) / 2.0 - 1e-6, "neighbours overlap or swapped order"


def test_spread_on_circle_leaves_free_intervals_alone():
    out = g.spread_on_circle([1.0, 4.0, 7.0], [1, 1, 1], 10.0)
    assert out == pytest.approx([1.0, 4.0, 7.0])


def test_spread_on_circle_packs_evenly_when_it_cannot_fit():
    out = g.spread_on_circle([0.0, 0.0, 0.0, 0.0], [3, 3, 3, 3], 10.0)
    gaps = sorted((b - a) % 10.0 for a, b in zip(sorted(out), sorted(out)[1:] + sorted(out)[:1]))
    assert gaps == pytest.approx([2.5, 2.5, 2.5, 2.5])


def test_boxes():
    assert g.box_union([(0, 0, 1, 1), (2, -1, 3, 0.5)]) == (0, -1, 3, 1)
    assert g.boxes_overlap((0, 0, 1, 1), (0.5, 0.5, 2, 2))
    assert not g.boxes_overlap((0, 0, 1, 1), (1.0, 0, 2, 1))
    assert g.box_expand((0, 0, 1, 1), 1) == (-1, -1, 2, 2)
