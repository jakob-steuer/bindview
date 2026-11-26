"""The placement engine on small synthetic problems."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import diagram_at
from bindview.depict import depict_ligand
from bindview.geometry import capsule_gaps, points_segments_distance, segments_distance
from bindview.layout import Body, LayoutParams, LigandField, Port, layout_bodies
from bindview.layout.body import LabelSpec, march_entry
from bindview.layout.engine import _Engine
from bindview.ligand import build_ligand
from bindview.structure import Atom
from bindview.theme import Metrics


def _benzene():
    atoms = [Atom(k, f"C{k + 1}", "C", "BNZ", "L", 1, "", 1.39 * np.cos(np.pi / 3 * k),
                  1.39 * np.sin(np.pi / 3 * k), 0.0, True) for k in range(6)]
    m = Metrics()
    dep = depict_ligand(build_ligand(atoms, "c1ccccc1"), m)
    return dep, LigandField(dep, reach=10 * m.bond_length), m


def _disc(key, target, radius=18.0, **kw):
    return Body(key=key, kind="disc", shapes_p0=[(0, 0)], shapes_p1=[(0, 0)], shapes_r=[radius],
                ports=[Port((0, 0), target, ideal=radius + 14.0, drawn=False)], ligand_clearance=6.0, **kw)


def test_field_is_zero_on_the_skeleton_and_grows_outward():
    dep, fld, m = _benzene()
    on_bond = np.array([[(dep.atoms[0].x + dep.atoms[1].x) / 2, (dep.atoms[0].y + dep.atoms[1].y) / 2]])
    assert fld.sdf(on_bond)[0] < 0.5
    right = max(dep.atoms, key=lambda a: a.x)
    far = np.array([[right.x + 100.0, right.y]])
    # 100 units from the atom centre, less the ink of the (double) bond that ends there.
    assert 93.0 < fld.sdf(far)[0] < 100.0
    out = fld.outward(far)[0]
    assert out[0] > 0.95                              # the gradient points away from the ring


def test_field_counts_bond_crossings():
    dep, fld, _ = _benzene()
    cx = np.mean([a.x for a in dep.atoms])
    cy = np.mean([a.y for a in dep.atoms])
    through = fld.bond_crossings(np.array([[cx - 200.0, cy + 1.0]]), np.array([[cx + 200.0, cy + 1.0]]))
    assert through[0] == 2                            # in one side of the hexagon, out the other
    outside = fld.bond_crossings(np.array([[cx - 200.0, cy + 150.0]]), np.array([[cx + 200.0, cy + 150.0]]))
    assert outside[0] == 0


def test_many_bodies_wanting_the_same_atom_do_not_overlap():
    """The core problem in miniature: twelve residues that all want one atom."""
    dep, fld, m = _benzene()
    target = dep.pos(0)
    bodies = [_disc(i, target) for i in range(12)]
    res = layout_bodies(fld, bodies, LayoutParams(unit=m.bond_length))
    assert res.ok and len(res.poses) == 12 and res.overlaps == 0
    centres = np.array([[p.x, p.y] for p in res.poses.values()])
    d = np.linalg.norm(centres[:, None] - centres[None], axis=2) + np.eye(12) * 1e9
    assert d.min() >= 2 * 18.0 + LayoutParams().solid_gap - 1e-6
    assert (fld.sdf(centres) >= 18.0 + 6.0 - 1e-6).all()   # none of them sits on the ring


def test_bodies_land_next_to_their_own_atoms():
    dep, fld, m = _benzene()
    bodies = [_disc(i, dep.pos(i)) for i in range(6)]
    res = layout_bodies(fld, bodies, LayoutParams(unit=m.bond_length))
    assert res.ok
    for i in range(6):
        p = res.poses[i]
        own = np.hypot(p.x - dep.pos(i)[0], p.y - dep.pos(i)[1])
        others = min(np.hypot(p.x - dep.pos(j)[0], p.y - dep.pos(j)[1]) for j in range(6) if j != i)
        assert own < others
    assert res.max_stretch < 1.3


def test_layout_is_deterministic():
    dep, fld, m = _benzene()
    make = lambda: [_disc(i, dep.pos(i % 6)) for i in range(9)]
    a = layout_bodies(fld, make(), LayoutParams(unit=m.bond_length))
    b = layout_bodies(fld, make(), LayoutParams(unit=m.bond_length))
    assert a.poses == b.poses and a.cost == b.cost


def test_a_body_that_fits_nowhere_is_reported_unplaced_not_overlapped():
    dep, fld, m = _benzene()
    giant = _disc("giant", dep.pos(0), radius=5000.0)
    res = layout_bodies(fld, [giant, _disc("ok", dep.pos(3))], LayoutParams(unit=m.bond_length))
    assert res.unplaced == ["giant"] and "ok" in res.poses and not res.ok


def test_drawn_links_do_not_run_through_other_bodies():
    dep, fld, m = _benzene()

    def linked(key, target):
        return Body(key=key, kind="disc", shapes_p0=[(0, 0)], shapes_p1=[(0, 0)], shapes_r=[14.0],
                    ports=[Port((0, 0), target, ideal=60.0, min_length=40.0, drawn=True, port_radius=15.0,
                                target_radius=3.0, label_size=(20.0, 9.0))], ligand_clearance=8.0)
    bodies = [linked(i, dep.pos(i % 3)) for i in range(6)]
    res = layout_bodies(fld, bodies, LayoutParams(unit=m.bond_length))
    assert res.ok
    P = {k: np.array([[p.x, p.y]]) for k, p in res.poses.items()}
    for b in bodies:
        a_end, b_end, _ = b.link_segments(P[b.key], np.zeros(1), np.zeros(1, dtype=bool))
        for other in bodies:
            if other.key == b.key:
                continue
            s0, s1, r, _ = other.solids(P[other.key], np.zeros(1), np.zeros(1, dtype=bool),
                                        np.array([res.label_t[other.key]]))
            gap = capsule_gaps(a_end[0], b_end[0], np.zeros(1), s0[0], s1[0], r)
            assert gap.min() >= LayoutParams().line_gap - 1e-6


def test_upright_label_follows_the_body_without_rotating():
    body = Body(key="f", kind="fragment", shapes_p0=[(0, 0)], shapes_p1=[(30, 0)], shapes_r=[2.0],
                ports=[Port((30, 0), (100, 0), ideal=50.0)], rotatable=True,
                label=LabelSpec(anchor=(0, 0), direction=(-1, 0), width=40.0, height=10.0, gap=4.0))
    pos = np.zeros((2, 2))
    centre = body.label_centre(pos, np.array([0.0, np.pi / 2]), np.zeros(2, dtype=bool))
    assert centre[0] == pytest.approx((-24.0, 0.0))        # beside the anchor: half the width + gap
    assert centre[1] == pytest.approx((0.0, -9.0))         # below/above it: half the height + gap
    assert body.self_clear(pos, np.array([0.0, np.pi / 2]), np.zeros(2, dtype=bool)).all()


def test_march_entry_stops_a_link_at_the_ink():
    start = np.array([[20.0, 0.0], [0.0, 30.0], [50.0, 50.0]])
    end = np.zeros((3, 2))
    caps = np.zeros((3, 1, 2))
    hit = march_entry(start, end, caps, caps, np.array([5.0]))
    assert hit[0] == pytest.approx((5.0, 0.0), abs=1e-6)
    assert hit[1] == pytest.approx((0.0, 5.0), abs=1e-6)
    assert np.hypot(*hit[2]) == pytest.approx(5.0, abs=1e-6)
    miss = march_entry(np.array([[20.0, 20.0]]), np.array([[30.0, 20.0]]), caps[:1], caps[:1], np.array([5.0]))
    assert miss[0] == pytest.approx((30.0, 20.0))          # never enters: the far end is returned


# ---------------------------------------------------------------------------
# What the search leaves out changes nothing
# ---------------------------------------------------------------------------

def test_the_kernels_are_the_plain_formulas_to_the_last_bit():
    """They work on x and y apart, for speed; a sum of two products is the same either way."""
    def plain_points(points, a, b):
        ab = b - a
        l2 = np.maximum((ab * ab).sum(axis=1), 1e-12)
        t = np.clip(((points[:, None, :] - a[None]) * ab[None]).sum(axis=2) / l2[None], 0.0, 1.0)
        return np.linalg.norm(points[:, None, :] - (a[None] + ab[None] * t[:, :, None]), axis=2)

    def plain_segments(p1, q1, p2, q2):
        d1, d2, r = q1 - p1, q2 - p2, p1 - p2
        a, e, f = (d1 * d1).sum(-1), (d2 * d2).sum(-1), (d2 * r).sum(-1)
        c, b = (d1 * r).sum(-1), (d1 * d2).sum(-1)
        a_ok, e_ok = a > 1e-12, e > 1e-12
        safe_a, safe_e = np.where(a_ok, a, 1.0), np.where(e_ok, e, 1.0)
        denom = a * e - b * b
        s = np.where(denom > 1e-12, np.clip((b * f - c * e) / np.where(denom > 1e-12, denom, 1.0), 0.0, 1.0), 0.0)
        s = np.where(e_ok, s, np.clip(-c / safe_a, 0.0, 1.0))
        t = np.where(e_ok, (b * s + f) / safe_e, 0.0)
        s = np.where(t < 0.0, np.clip(-c / safe_a, 0.0, 1.0), s)
        s = np.where(t > 1.0, np.clip((b - c) / safe_a, 0.0, 1.0), s)
        t = np.clip(t, 0.0, 1.0)
        s = np.where(a_ok, s, 0.0)
        t = np.where(a_ok, t, np.where(e_ok, np.clip(f / safe_e, 0.0, 1.0), 0.0))
        return np.linalg.norm((p1 + d1 * s[..., None]) - (p2 + d2 * t[..., None]), axis=-1)

    rng = np.random.default_rng(3)
    for trial in range(40):
        p, a = rng.normal(0, 50, (rng.integers(1, 40), 2)), rng.normal(0, 50, (rng.integers(1, 40), 2))
        # (Some of the segments are points, and in every other trial all lie on a grid: parallel, touching, equal.)
        q = p + rng.normal(0, 20, p.shape) * (rng.random((len(p), 1)) > 0.3)
        b = a + rng.normal(0, 20, a.shape) * (rng.random((len(a), 1)) > 0.3)
        if trial % 2:
            p, q, a, b = (np.round(x / 10) * 10 for x in (p, q, a, b))
        assert np.array_equal(points_segments_distance(p, a, b), plain_points(p, a, b))
        assert np.array_equal(segments_distance(p[:, None], q[:, None], a[None], b[None]),
                              plain_segments(p[:, None], q[:, None], a[None], b[None]))


def _engine_of(case: str, style: str):
    """The engine as it stands when the residues of a figure have been placed."""
    held = []
    real = _Engine.run

    def run(self):
        held.append(self)
        return real(self)

    _Engine.run = run
    try:
        dia = diagram_at(case, None)
        dia.render(style=style)
    finally:
        _Engine.run = real
    return held[-1]


@pytest.mark.parametrize("case, style", [("ar_r1881", "detailed"), ("kit_imatinib", "detailed"),
                                         ("dhfr_methotrexate", "schematic")])
def test_the_search_finds_what_looking_at_everything_finds(case, style):
    engine = _engine_of(case, style)
    for i in list(engine.world):
        saved = engine._snapshot([i])
        engine._unset(i)
        batch = engine._batch(i)
        cost, ok = engine._collide(i, batch.geo)                     # every candidate, against everything
        total = np.where(ok, batch.base + cost, np.inf)
        hit = engine._best(i)
        assert (hit is None) == bool(np.isinf(total).all())
        if hit is not None:
            assert abs(hit[0] - total.min()) < 2e-9
            at = np.nonzero((batch.pos == hit[1]).all(axis=1) & (batch.ang == hit[2]) & (batch.mir == hit[3]))[0]
            assert len(at) and total[at[0]] == hit[0]                # ... with the cost that it has there, to the bit
            # Only what is cheaper than a bound is of interest: the same one below it, and none where there is none.
            assert engine._best(i, upper=hit[0] + 1e-6)[0] == hit[0] and engine._best(i, upper=hit[0] - 1e-6) is None
        # What it remembers of the candidates is so.
        known, tags = engine._known[(i, False)]
        for j in engine.world:
            if j != i and tags[j] == engine.tag[j]:
                bad, _ = engine._blocked(i, batch.geo, engine._others(i, [j]))
                assert (bad[known[j] == 2, 0]).all() and not bad[known[j] == 1, 0].any()
        engine._restore(saved, [i])


def test_little_of_what_is_tried_is_priced_in_full(monkeypatch):
    """What the speed of a crowded figure rests on: of the candidates that are looked at few are possible, and only
    those are priced against everything placed."""
    looked = priced = 0
    real_clear, real_crowding = _Engine._clear, _Engine._crowding

    def clear(self, i, batch, rows, ids, known):
        nonlocal looked
        looked += len(rows)
        return real_clear(self, i, batch, rows, ids, known)

    def crowding(self, i, g, others, gaps=None):
        nonlocal priced
        priced += len(g.pos) if gaps is None and len(g.pos) > 1 else 0
        return real_crowding(self, i, g, others, gaps)

    monkeypatch.setattr(_Engine, "_clear", clear)
    monkeypatch.setattr(_Engine, "_crowding", crowding)
    dia = diagram_at("kit_imatinib", None)                        # thirteen residues round imatinib
    dia.render(style="detailed", bond_length=30.0, on_unreadable="render")
    assert looked > 5_000 and priced < 0.2 * looked


def test_a_pose_that_comes_back_is_the_pose_it_was():
    dep, fld, m = _benzene()
    engine = _Engine(fld, [_disc(k, dep.pos(k)) for k in range(4)], LayoutParams(unit=m.bond_length))
    for i in range(4):
        assert engine._seat(i)
    before, tag, cost = engine._snapshot([0, 1, 2, 3]), dict(engine.tag), engine._current(2)
    engine._set(2, engine.pos[2] + 40.0, 0.0, False, engine.lt[2])
    assert engine.tag[2] != tag[2]
    engine._restore(before, [2])
    assert engine.tag == tag and engine._current(2) == cost
    engine._set(2, before[2][0], before[2][1], before[2][2], before[2][3])
    assert engine.tag == tag                                        # known by where it is, not by when it was put there


# ---------------------------------------------------------------------------
# A last bit decides nothing: what is cut to a whole number is rounded first
# ---------------------------------------------------------------------------

def test_the_grid_of_a_field_does_not_hang_on_a_last_bit():
    dep, fld, _ = _benzene()
    left = float(np.vstack([fld.seg_a, fld.seg_b])[:, 0].min())
    edge = float(np.floor(left)) - 300.0
    reach = left - edge                                           # the left edge of the grid on a whole number
    grids = set()
    for r in (np.nextafter(reach, 0.0), reach, np.nextafter(reach, np.inf)):
        f = LigandField(dep, reach=float(r))
        grids.add((f.x0, f.y0, f.nx, f.ny))
    assert len(grids) == 1 and next(iter(grids))[0] == edge


def test_the_samples_along_a_shape_do_not_hang_on_a_last_bit():
    def samples(length: float) -> int:
        body = Body(key=0, kind="fragment", shapes_p0=[(0.0, 0.0)], shapes_p1=[(length, 0.0)], shapes_r=[2.0], ports=[])
        return len(body._samples)
    # One every 7 units: two steps for a shape 14 long, and for one that is 14 long but for a last bit.
    assert samples(14.0) == samples(float(np.nextafter(14.0, 15.0))) == samples(float(np.nextafter(14.0, 13.0))) == 3
    assert (samples(14.1), samples(13.9), samples(0.0)) == (4, 3, 1)


def test_the_order_of_the_discs_does_not_hang_on_a_last_bit(monkeypatch):
    from types import SimpleNamespace

    dep, fld, m = _benzene()
    engine = _Engine(fld, [_disc(k, dep.pos(k)) for k in range(3)], LayoutParams(unit=m.bond_length))
    # What each disc costs in its places, cheapest first: the one with the fewest cheap places (below 1.5) goes first.
    costs = {0: [1.0, 1.2, 2.0], 1: [1.0, float(np.nextafter(1.5, 0.0)), 2.0], 2: [1.0, 1.5, 2.0]}
    monkeypatch.setattr(engine, "_batch", lambda i, wide=False: SimpleNamespace(base=np.array(costs[i])))
    assert engine._order() == [1, 2, 0]                           # 1.5 less a last bit is 1.5: one cheap place, like disc 2


def test_scores_a_last_bit_apart_are_equal_wherever_they_lie():
    from bindview.layout.engine import _classes, _least

    # Two scores a last bit apart that lie on either side of an edge of rounding to nine decimals.
    low = 1.0000000005
    for _ in range(8):
        low = float(np.nextafter(low, 0.0))
    while np.round(low, 9) == np.round(np.nextafter(low, 2.0), 9):
        low = float(np.nextafter(low, 2.0))
    high = float(np.nextafter(low, 2.0))
    assert int(np.argmin(np.round([high, low], 9))) == 1          # each rounded by itself: the second, by a last bit
    assert _least(np.array([high, low])) == 0                     # ... as one class: the first of the two
    assert _classes(np.array([high, low, 2.0])).tolist() == [low, low, 2.0]
    assert np.argsort(_classes(np.array([high, low, 0.5, low])), kind="stable").tolist() == [2, 0, 1, 3]
    # What lies between two scores joins them; what is further off than a billionth is another score.
    assert _classes(np.array([1.8e-9, 0.0, 0.9e-9, 1.0])).tolist() == [0.0, 0.0, 0.0, 1.0]
    assert _least(np.array([1.8e-9, 0.9e-9, 0.0, 1.0])) == 0 and _least(np.array([0.2, 0.1 + 2e-9, 0.1])) == 2
    assert _least(np.array([np.inf, 3.0, np.inf])) == 1 and _least(np.array([np.inf, np.inf])) == 0
    assert _least(np.array([5.0])) == 0 and _classes(np.array([5.0])).tolist() == [5.0]
    # The first of the least is the least of the classes, whatever the scores.
    rng = np.random.default_rng(7)
    for _ in range(200):
        scores = rng.choice(np.arange(0.0, 2.0, 0.125), 40) + rng.normal(0.0, 4e-10, 40)
        assert _least(scores) == int(np.argmin(_classes(scores)))


def test_a_direction_on_the_edge_of_a_sector_is_in_one_sector_whatever_its_last_bit():
    """The candidates of a fragment stand every 7.5 degrees, and the edges of the sectors lie at 22.5 + 45 k: every
    sixth candidate is on one. The sector decides which candidates are tried, and it was the last bit of the angle
    that said which sector that was."""
    from bindview.layout.engine import _sectors

    angles = np.radians(np.arange(0.0, 360.0, 7.5))
    for radius in (1.0, 37.3, 412.9):
        on = radius * np.column_stack([np.cos(angles), np.sin(angles)])
        want = ((np.arange(0.0, 360.0, 7.5) + 22.5) // 45.0).astype(int) % 8
        assert np.array_equal(_sectors(on), want)
        for dx in (-np.inf, 0.0, np.inf):                  # each coordinate one unit in the last place down, as it is, up
            for dy in (-np.inf, 0.0, np.inf):
                near = np.column_stack([on[:, 0] if dx == 0.0 else np.nextafter(on[:, 0], dx),
                                        on[:, 1] if dy == 0.0 else np.nextafter(on[:, 1], dy)])
                assert np.array_equal(_sectors(near), want)
