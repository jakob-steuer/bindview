"""The second layout of an interface: the residues of the spine where they lie, on the plane through them.

The geometry (``bindview.plane``) is tested on points whose arrangement is known. The figures are tested
on deposited structures in the repository: the two chains of thrombin (2UUF), HLA-A2 with its peptide
(3MRE), the antibody HyHEL-10 on lysozyme (3D9A) and the lambda repressor (1LMB), and on two models of the
AlphaFold database with their predicted aligned error. The numbers a figure states are computed a second
time here, from the positions in its graph and the coordinates in the structure, and have to be the same.
"""

from __future__ import annotations

import importlib.util
import json
import re
from itertools import combinations

import numpy as np
import pytest

from conftest import ROOT, chains_interface, model_of, needs_plip, pae_case, structure_of
from bindview import Interface, Participant
from bindview.cli import main as cli
from bindview.distance_table import interface_contacts
from bindview.errors import BindviewError
from bindview.interface import dump_interface_contacts
from bindview.pae import pae_tag
from bindview.plane import (APART, ERROR_MAX, KEPT_MIN, NEAR, SCALE_BAR, STRETCH_GAP, fidelity, overlapping,
                            passes_through, project, separate, stretches)
from bindview.scene import Line, Polyline, Text
from scenetools import groups, inside, walk, words


def thrombin(**options) -> Interface:
    """The light chain (A) and the heavy chain (B) of thrombin, contacts from the distance table."""
    return chains_interface("thrombin_hirudin", "A", "B", **options)


def percent(value: float) -> str:
    return f"{100.0 * value:.0f} %"


def boxes(res) -> dict[int, tuple[float, float]]:
    """Where the boxes of a figure stand, by position in the chain: from its graph."""
    return {int(n.attrs["position"]): (n.x, n.y) for n in res.graph.nodes
            if n.kind == "ligand_group" and "position" in n.attrs}


def drawn_at(res, n: int) -> np.ndarray:
    """The centres of the ``n`` boxes, in the order of the site, in page units."""
    at = boxes(res)
    return np.array([at[k + 1] for k in range(n)])


def centres(face) -> np.ndarray:
    """The centroid of the heavy atoms of every residue of the site, from the structure."""
    return np.array([np.mean([a.xyz for a in face.structure.residue_atoms(k) if a.element not in ("H", "D")], axis=0)
                     for k in face.site.residues])


def plane_normal(points: np.ndarray) -> np.ndarray:
    """The normal of the least-squares plane, not by the route the library takes: the eigenvector of the
    covariance with the smallest eigenvalue."""
    values, vectors = np.linalg.eigh(np.cov((points - points.mean(axis=0)).T))
    return vectors[:, int(np.argmin(values))]


def toward_partner(face) -> np.ndarray:
    shown = set(face.site.residues)
    centre = {k: np.mean([a.xyz for a in face.structure.residue_atoms(k) if a.element not in ("H", "D")], axis=0)
              for c in face._oriented for k in (c.residue_a, c.residue_b)}
    ways = {(c.residue_a, c.residue_b): centre[c.residue_b] - centre[c.residue_a]
            for c in face._oriented if c.residue_a in shown}
    return np.mean([v / np.linalg.norm(v) for v in ways.values()], axis=0)


def projected(face):
    """The site's residues projected onto their plane, measured here: the projection before any box is moved."""
    space = centres(face)
    return fidelity(space, project(space).xy)


# ---------------------------------------------------------------------------
# The geometry
# ---------------------------------------------------------------------------

def test_a_stretch_ends_where_more_than_three_residues_are_skipped():
    assert STRETCH_GAP == 3
    assert stretches([10, 11, 12]) == [(0, 2)]
    assert stretches([10, 14]) == [(0, 1)]                      # three between them: one face of a helix
    assert stretches([10, 15]) == [(0, 0), (1, 1)]              # four: two stretches
    assert stretches([26, 28, 31, 33, 50, 52, 53, 56, 97, 98, 99, 100]) == [(0, 3), (4, 7), (8, 11)]
    assert stretches([7]) == [(0, 0)] and stretches([]) == []


def test_points_on_a_plane_are_projected_without_any_loss():
    rng = np.random.default_rng(3)
    flat = np.column_stack([rng.uniform(-12, 12, 15), rng.uniform(-9, 9, 15), np.zeros(15)])
    turn = np.linalg.qr(rng.normal(size=(3, 3)))[0]             # the plane anywhere in space
    points = flat @ turn.T + np.array([40.0, -7.0, 3.0])
    proj = project(points, toward=turn[:, 2])
    assert proj.rms < 1e-9 and proj.tilt < 1e-3
    measured = fidelity(points, proj.xy)
    assert measured.error < 1e-9 and (measured.kept, measured.false, measured.lost) == (1.0, 0, 0)
    assert measured.pairs > 10 and measured.ok
    # The long axis stands upright, and the first point is nearer the top (y runs down a page) than the last.
    assert np.ptp(proj.xy[:, 1]) > np.ptp(proj.xy[:, 0]) and proj.xy[0, 1] < proj.xy[-1, 1] + 1e-9


def test_the_plane_is_seen_from_the_partner_s_side_and_not_mirrored():
    # Three points that run anticlockwise for somebody who looks down on them from +z.
    points = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 16.0, 0.0], [10.0, 16.0, 0.3]])

    def turning(xy: np.ndarray) -> float:
        (ax, ay), (bx, by), (cx, cy) = xy[0], xy[1], xy[2]
        return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)    # > 0: clockwise on a page, whose y runs down

    above = project(points, toward=np.array([0.0, 0.0, 1.0]))
    below = project(points, toward=np.array([0.0, 0.0, -1.0]))
    assert turning(above.xy) < 0.0 < turning(below.xy)          # anticlockwise from above, its mirror image from below
    assert above.normal[2] > 0.99 and below.normal[2] < -0.99
    with pytest.raises(ValueError, match="a plane needs 3 points, not 2"):
        project(points[:2])
    # Looked at from the side the direction is 80 degrees off the normal: that is reported, not hidden.
    assert 79.0 < project(points, toward=np.array([np.sin(np.radians(80)), 0.0, np.cos(np.radians(80))])).tilt < 81.0


def test_a_false_and_a_lost_neighbour_are_counted_and_a_near_miss_is_not():
    space = np.array([[0.0, 0.0, 0.0], [6.0, 0.0, 0.0], [0.0, 0.0, 12.0], [30.0, 0.0, 0.0], [30.0, 8.1, 0.0]])
    flat = space[:, :2].copy()                                   # straight down the z axis
    f = fidelity(space, flat)
    # Point 2 lies 12 A above point 0 and is drawn on top of it, and 6 A from point 1 (13.4 A away): two false ones.
    assert (f.false, f.lost) == (2, 0) and f.pairs == 3 and f.kept == pytest.approx(1.0 - 2.0 / 3.0)
    assert not f.ok
    apart = flat.copy()
    apart[1] = (11.0, 0.0)                                       # a neighbour drawn 11 A away: lost
    g = fidelity(space, apart)
    assert g.lost == 1 and g.false == 1
    # 8.1 A apart in space and drawn 7.9 A apart: neither a neighbour kept nor an error (between 8 and 10 A).
    near = np.array([[0.0, 0.0, 0.0], [8.1, 0.0, 0.0]])
    h = fidelity(near, np.array([[0.0, 0.0], [7.9, 0.0]]))
    assert (h.pairs, h.false, h.lost, h.kept) == (1, 0, 0, 1.0) and (NEAR, APART) == (8.0, 10.0)
    assert fidelity(np.zeros((1, 3)), np.zeros((1, 2))).kept == 1.0
    assert fidelity(np.array([[0, 0, 0], [20.0, 0, 0]]), np.array([[0, 0], [20.0, 0]])).pairs == 0


def test_the_distance_error_is_stress_at_the_figure_s_own_scale():
    space = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
    figure = space[:, :2] * 1.1                                  # everything drawn a tenth too far apart
    assert fidelity(space, figure).error == pytest.approx(0.1)
    assert fidelity(space, space[:, :2]).error == 0.0
    assert (KEPT_MIN, ERROR_MAX) == (0.90, 0.10)


def test_either_threshold_alone_refuses_a_figure():
    """Every neighbour relation kept and distances off by a fifth: refused. A flat grid with one point
    12 Å above its middle, seen from above: distances off by less than a tenth, and five neighbours
    drawn where there are none: refused too."""
    far = np.array([[0.0, 0.0, 0.0], [20.0, 0.0, 0.0], [0.0, 20.0, 0.0], [20.0, 20.0, 0.0]])
    stretched = fidelity(far, far[:, :2] * 1.2)
    assert (stretched.pairs, stretched.kept) == (0, 1.0) and stretched.error == pytest.approx(0.2)
    assert not stretched.ok
    grid = np.array([[6.0 * i, 6.0 * j, 0.0] for i in range(5) for j in range(5)] + [[12.0, 12.0, 12.0]])
    folded = fidelity(grid, grid[:, :2])
    # 40 pairs of the grid 6 Å apart; the point above is drawn on the middle one and beside its four neighbours.
    assert (folded.pairs, folded.false, folded.lost) == (45, 5, 0) and folded.kept == pytest.approx(40 / 45)
    assert folded.error <= ERROR_MAX and not folded.ok


def test_boxes_are_pushed_apart_by_the_least_it_takes_and_the_same_way_every_time():
    at = np.array([[0.0, 0.0], [10.0, 3.0], [200.0, 0.0]])
    out = separate(at, half_w=25.0, half_h=8.5, gap=5.0)
    assert np.allclose(out[2], at[2])                            # nothing near it: where it was
    dx, dy = abs(out[1, 0] - out[0, 0]), abs(out[1, 1] - out[0, 1])
    assert dy == pytest.approx(22.0) and dx == pytest.approx(10.0)      # up and down: 19 units, against 45 sideways
    assert np.allclose(out[:2].mean(axis=0), at[:2].mean(axis=0))       # half the way each
    assert np.array_equal(out, separate(at, 25.0, 8.5, 5.0))
    same = separate(np.zeros((3, 2)), 25.0, 8.5, 5.0)            # three boxes on one spot still come apart
    gaps = [(abs(same[i, 0] - same[j, 0]), abs(same[i, 1] - same[j, 1])) for i in range(3) for j in range(i)]
    assert all(x >= 55.0 - 1e-6 or y >= 22.0 - 1e-6 for x, y in gaps)
    assert overlapping(at, 25.0, 8.5) == 1 and overlapping(out, 25.0, 8.5, 5.0) == 0


def test_pushing_apart_ends_with_no_box_on_another_however_crowded():
    """Thirty boxes in a space that holds three: far denser than any figure, and they still come apart.
    Whether any are left on another is counted, not assumed; cut short, some are."""
    rng = np.random.default_rng(11)
    for n, side in ((30, 40.0), (26, 46.0), (21, 37.0), (12, 5.0)):
        crowd = rng.uniform(0.0, side, size=(n, 2))
        crowd[: n // 2] = crowd[0]                               # half of them on one spot
        assert overlapping(crowd, 26.0, 8.5) > n
        assert overlapping(separate(crowd, 26.0, 8.5), 26.0, 8.5, 5.0) == 0
    assert overlapping(separate(crowd, 26.0, 8.5, sweeps=3), 26.0, 8.5) > 0       # cut short, some are left


def test_a_line_that_runs_through_a_box():
    box = (0.0, 0.0, 50.0, 17.0)
    assert passes_through((-20.0, 8.0), (80.0, 8.0), box) and passes_through((25.0, -30.0), (25.0, 40.0), box)
    assert not passes_through((-20.0, 30.0), (80.0, 30.0), box) and not passes_through((60.0, -5.0), (90.0, 40.0), box)


def test_the_thresholds_draw_a_shallow_cap_and_refuse_a_hemisphere():
    """The calibration of the two thresholds on surfaces of known shape (``tests/data/plane-calibration.json``,
    made by ``scripts/calibrate_plane.py``), four of its rows run again."""
    spec = importlib.util.spec_from_file_location("calibrate_plane", ROOT / "scripts" / "calibrate_plane.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    stored = json.loads((ROOT / "tests" / "data" / "plane-calibration.json").read_text(encoding="utf-8"))
    assert stored["thresholds"] == {"neighbours_kept_min": KEPT_MIN, "distance_error_max": ERROR_MAX,
                                    "neighbour_within": NEAR, "apart_beyond": APART}
    rows = {(r["shape"], r["residues"], r["turn_deg"]): r for r in stored["rows"]}
    assert len(rows) == 2 * 5 * 10
    for key in (("cap", 20, 0), ("cap", 20, 90), ("cylinder", 30, 60), ("cap", 8, 120)):
        assert module.row(*key) == rows[key]
    assert rows[("cap", 20, 0)]["share_drawn"] == 1.0 and rows[("cap", 30, 60)]["share_drawn"] >= 0.95
    assert rows[("cap", 20, 90)]["share_drawn"] <= 0.15 and rows[("cap", 30, 120)]["share_drawn"] == 0.0


# ---------------------------------------------------------------------------
# The rule: column or plane
# ---------------------------------------------------------------------------

def test_several_stretches_are_drawn_where_they_lie_and_one_stretch_in_a_column():
    face = thrombin(layout="auto")
    runs = stretches([face._index[k] for k in face.site.residues])
    assert face.layout == "plane" and len(runs) > 1 and face._runs == runs
    res = face.render(style="schematic")
    assert res.status == "ok" and res.quality.valid and res.quality.faithful
    meta = res.graph.meta["interface"]
    assert (meta["layout"], meta["stretches"], meta["spine_order"]) == ("plane", len(runs), None)
    assert meta["layout_rule"] == ("on the plane through them: the residues of chain B at the interface that are "
                                   f"shown come in {len(runs)} stretches of its chain")
    assert meta["plane"]["drawn"] is True and res.data["interface"] == meta
    assert res.graph.to_dict()["version"] == "1.3"
    # The peptide of HLA-A2 is one stretch: a column, and the result says why.
    peptide = chains_interface("mhc_peptide", "A", "P", layout="auto")
    first, last = peptide.site.residues[0], peptide.site.residues[-1]
    assert peptide.spine.chains == ("P",) and peptide.layout == "column"
    said = peptide.render(style="schematic")
    assert said.graph.meta["interface"]["layout_rule"] == (
        "a column: the residues of chain P at the interface that are shown are one stretch of its chain "
        f"({first.number} to {last.number})")
    assert said.graph.to_dict()["version"] == "1.2" and "plane" not in said.graph.meta["interface"]
    assert "stand in a column" in words(said) and not said.warnings
    st = face.structure
    contacts = interface_contacts(st, ["A"], ["B"])
    assert Interface(st, ["A", "B"], contacts, layout="auto").layout == "plane"
    with pytest.raises(BindviewError, match="layout is 'auto', 'column' or 'plane'"):
        Interface(st, ["A", "B"], contacts, layout="map")


def test_the_rule_is_asked_for_and_the_default_is_the_column():
    """Left alone, an interface is a column, and its result says what the rule would have drawn."""
    st = structure_of("thrombin_hirudin")
    contacts = interface_contacts(st, ["A"], ["B"])
    plain = Interface(st, ["A", "B"], contacts)
    n = len(plain._runs)
    assert plain.layout_asked == "column" and plain.layout == "column" and n > 1
    assert plain.layout_rule == ('a column, the default; asked to choose (layout "auto"), the rule would draw these '
                                 f'residues on the plane through them: they come in {n} stretches of the chain')
    res = plain.render(style="schematic")
    meta = res.graph.meta["interface"]
    assert (meta["layout"], meta["layout_rule"], meta["stretches"]) == ("column", plain.layout_rule, n)
    assert "plane" not in meta and res.graph.to_dict()["version"] == "1.2" and not res.warnings
    assert "plane" not in words(res)
    assert Interface(st, ["A", "B"], contacts, layout="column").render(style="schematic").svg == res.svg
    # Where the rule would draw a column too, there is nothing to add: one stretch, two residues, a curved interface.
    assert chains_interface("mhc_peptide", "A", "P").layout_rule == "a column, the default"
    assert chains_interface("mhc_peptide", "A", "P", spine="A").layout_rule == "a column, the default"
    assert pae_case("afdb_apobec3d").layout_rule == "a column, the default"
    assert pae_case("afdb_tectb").layout_rule.startswith("a column, the default; asked to choose")


def test_either_layout_can_be_asked_for():
    column = thrombin(layout="column")
    assert column.layout == "column" and column.layout_rule.startswith("a column, the default")
    res = column.render(style="schematic")
    assert res.graph.meta["interface"]["layout"] == "column" and res.graph.to_dict()["version"] == "1.2"
    assert "stand in a column" in words(res) and "plane" not in res.graph.meta["interface"]
    plane = chains_interface("mhc_peptide", "A", "P", layout="plane")
    assert plane.layout == "plane" and plane.layout_rule == "on the plane through them: the plane was asked for"
    drawn = plane.render(style="schematic")
    assert drawn.status == "ok" and ("come in one stretch of its chain: each box stands where its residue's centre "
                                     "lies on the plane") in words(drawn)


def test_an_interface_that_is_not_near_one_plane_is_not_drawn_on_one():
    """The heavy chain of the antibody where it touches its light chain: a projection keeps too few
    neighbour relations, though its distances would pass. The column is drawn, in the order of the
    sequence, and the figure and the result say why with the numbers that were measured."""
    face = chains_interface("fab_lysozyme", "H", "L", layout="auto")
    bare = projected(face)
    assert face.layout == "column" and face._considered is not None and not face._considered.bare.ok
    assert bare.kept < KEPT_MIN and bare.error <= ERROR_MAX
    res = face.render(style="schematic")
    meta = res.graph.meta["interface"]
    n = len(face._runs)
    assert res.status == "ok" and (meta["layout"], meta["stretches"], meta["spine_order"]) == ("column", n, "sequence")
    plane = meta["plane"]
    assert plane["drawn"] is False and (plane["false_neighbours"], plane["lost_neighbours"]) == (bare.false, bare.lost)
    assert (plane["neighbours_kept"], plane["distance_error"]) == (round(bare.kept, 3), round(bare.error, 3))
    numbers = (f"a projection would keep {percent(bare.kept)} of neighbour relations and be off by {percent(bare.error)} "
               "in distances; at least 90 % and at most 10 % are required")
    said = (f"a column in the order of the sequence: the residues of chain H at the interface that are shown come in "
            f"{n} stretches of its chain, but they do not lie near enough to one plane to be drawn where they lie "
            f"({numbers})")
    assert meta["layout_rule"] == said and f"chain H is drawn as {said}" in res.warnings
    text = words(res)
    assert (f"These residues come in {n} stretches of the chain, which no column can set out as they lie in space. "
            f"They are not drawn where they lie either: they do not lie near enough to one plane ({numbers}). How "
            "the stretches lie to each other is not shown.") in text
    # ... and the column makes no promise about neighbours in space.
    assert "in the order of the sequence" in text and "neighbours in the column are neighbours in space" not in text
    # Asked for outright it is still not drawn.
    forced = chains_interface("fab_lysozyme", "H", "L", layout="plane")
    assert forced.layout == "column" and forced.layout_rule.startswith(
        "a column in the order of the sequence: the plane was asked for, but the residues do not lie near enough")
    # Whoever asks for the order along the interface by name still gets it.
    along = chains_interface("fab_lysozyme", "H", "L", layout="auto", order="interface")
    assert along.render(style="schematic").graph.meta["interface"]["spine_order"] == "interface"


def test_distances_alone_can_refuse_the_plane():
    """The residues of the heavy chain of thrombin within 8 Å of Asp14 of its light chain: every neighbour
    relation survives a projection and the distances do not. One threshold is enough to refuse."""
    face = thrombin(site="A:14", layout="auto")
    bare = projected(face)
    assert len(face._runs) > 1 and bare.kept >= KEPT_MIN and bare.error > ERROR_MAX
    assert face.layout == "column" and not face._considered.bare.ok
    plane = face.render(style="schematic").graph.meta["interface"]["plane"]
    assert plane["drawn"] is False and plane["distance_error"] == round(bare.error, 3)
    assert f"be off by {percent(bare.error)} in distances" in face.layout_rule


def test_fewer_than_three_residues_make_no_plane():
    face = thrombin(max_residues=2, layout="plane")
    assert len(face.site.residues) == 2 and face.layout == "column"
    assert face.layout_rule == "a column: 2 residues do not make a plane"
    # It was asked for and is not drawn: the result says so. Left to the rule, there is nothing to say.
    assert "chain B is drawn as a column: 2 residues do not make a plane" in face.render(style="schematic").warnings
    assert not thrombin(max_residues=2, layout="auto").render(style="schematic").warnings


# ---------------------------------------------------------------------------
# What a figure states, computed again
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def heavy():
    """The heavy chain of thrombin where it touches the light chain, on its plane."""
    face = thrombin(layout="auto")
    return face, face.render(style="schematic")


def test_the_numbers_in_the_figure_are_those_of_the_boxes_as_drawn(heavy):
    face, res = heavy
    plane = res.graph.meta["interface"]["plane"]
    space = centres(face)
    measured = fidelity(space, drawn_at(res, len(space)) / plane["px_per_angstrom"])
    assert measured.as_dict() == {k: plane[k] for k in measured.as_dict()}
    assert plane["thresholds"] == {"neighbours_kept_min": 0.9, "distance_error_max": 0.1, "neighbour_within": 8.0,
                                   "apart_beyond": 10.0}
    # The projection itself, before any box was moved, is there too.
    bare = projected(face)
    assert plane["projection"] == {k: v for k, v in bare.as_dict().items() if k in plane["projection"]}
    normal = plane_normal(space)
    rms = float(np.sqrt(np.mean(((space - space.mean(axis=0)) @ normal) ** 2)))
    assert plane["out_of_plane_rms"] == pytest.approx(rms, abs=0.006)
    text = words(res)
    assert (f"Measured on the boxes as drawn: {measured.pairs} pairs of boxes are neighbours (centres within 8 Å) in "
            f"the figure or in space;") in text
    assert (f"{percent(measured.kept)} of neighbour relations are kept. Distances between the centres of boxes differ "
            f"from those in space by {percent(measured.error)} (root mean square). The residues' centres lie "
            f"{plane['out_of_plane_rms']:.1f} Å (root mean square) off the plane; {plane['boxes_moved']} boxes were "
            f"moved off their places to keep boxes apart.") in text
    assert res.scene.description.endswith(
        f"The boxes stand where their residues lie on the plane through them: {percent(measured.kept)} of neighbour "
        f"relations kept, distances off by {percent(measured.error)}.")


def unmoved(face, res) -> list[int]:
    """The boxes nothing moved: to scale, exactly where the residues' centres fall on the plane through them."""
    space = centres(face)
    normal = plane_normal(space)
    rel = space - space.mean(axis=0)
    on_plane = rel - np.outer(rel @ normal, normal)              # still three coordinates, all in the plane
    scale = res.graph.meta["interface"]["plane"]["px_per_angstrom"]
    drawn = drawn_at(res, len(space))
    d_plane = np.linalg.norm(on_plane[:, None] - on_plane[None], axis=2) * scale
    d_drawn = np.linalg.norm(drawn[:, None] - drawn[None], axis=2)
    moved = res.graph.meta["interface"]["plane"]["boxes_moved"]
    same = np.abs(d_plane - d_drawn) < 2e-3                      # the graph holds positions to a thousandth
    kept = [i for i in range(len(space)) if same[i].sum() >= len(space) - moved]
    assert np.abs(d_plane - d_drawn)[np.ix_(kept, kept)].max() < 2e-3
    # One scale: no box that was moved went further than a box is wide.
    assert np.abs(d_plane - d_drawn).max() < 60.0
    return kept


def test_a_box_stands_where_its_residue_lies_on_the_least_squares_plane(heavy):
    """Projected a second time here, by another route: the boxes that were not moved are, to scale,
    exactly where the residues' centres fall on the plane through them."""
    face, res = heavy
    moved = res.graph.meta["interface"]["plane"]["boxes_moved"]
    assert moved > 0 and len(unmoved(face, res)) == len(face.site.residues) - moved
    # ... and the long axis of the residues is upright on the page.
    drawn = drawn_at(res, len(face.site.residues))
    assert np.ptp(drawn[:, 1]) > np.ptp(drawn[:, 0])


def test_the_figure_is_what_is_seen_from_the_partner_and_no_mirror_image(heavy):
    face, res = heavy
    space = centres(face)
    normal = plane_normal(space)
    toward = toward_partner(face)
    if normal @ toward < 0:
        normal = -normal
    drawn = drawn_at(res, len(space))
    triples = 0
    for i, j, k in combinations(unmoved(face, res), 3):
        in_space = np.cross(space[j] - space[i], space[k] - space[i]) @ normal
        if abs(in_space) < 10.0:                                 # three centres nearly in a line: no turn to tell
            continue
        on_page = ((drawn[j, 0] - drawn[i, 0]) * (drawn[k, 1] - drawn[i, 1])
                   - (drawn[j, 1] - drawn[i, 1]) * (drawn[k, 0] - drawn[i, 0]))
        assert (in_space > 0) == (on_page < 0)                   # anticlockwise for the viewer is clockwise in y-down
        triples += 1
    assert triples > 10
    tilt = res.graph.meta["interface"]["plane"]["partner_off_normal_deg"]
    assert tilt == pytest.approx(np.degrees(np.arccos(normal @ toward / np.linalg.norm(toward))), abs=0.06)
    assert tilt <= 45.0 and "seen from the side of chain A" in words(res)
    # Beta-2 microglobulin against the heavy chain of HLA-A2: more than 45 degrees off the line of sight, and the
    # figure does not claim to be seen from the partner's side.
    tilted = chains_interface("mhc_peptide", "B", "A", layout="auto").render(style="schematic")
    off = tilted.graph.meta["interface"]["plane"]["partner_off_normal_deg"]
    assert off > 45.0 and f"seen at right angles to that plane (chain A lies {off:.0f}° off that line of sight)" in words(
        tilted)


# ---------------------------------------------------------------------------
# What is drawn, and what is not
# ---------------------------------------------------------------------------

def test_lines_join_neighbours_in_the_chain_and_dots_what_is_skipped_within_a_stretch(heavy):
    face, res = heavy
    place = [face._index[k] for k in face.site.residues]
    solid = sum(1 for a, b in zip(place, place[1:]) if b - a == 1)
    within = sum(1 for a, b in zip(place, place[1:]) if 1 < b - a <= STRETCH_GAP + 1)
    between = sum(1 for a, b in zip(place, place[1:]) if b - a > STRETCH_GAP + 1)
    assert solid and within and between and solid + within + between == len(place) - 1
    backbone = groups(res, "backbone")[0]
    lines = [f.item for f in inside(res, backbone, Line)]
    dotted = [ln for ln in lines if ln.paint.dash]
    assert len(lines) - len(dotted) == solid and 0 < len(dotted) <= within
    assert not inside(res, backbone, Polyline)                   # no bracket, no stub: there is no column
    text = [f.item.content for f in walk(res.scene) if isinstance(f.item, Text)]
    assert "N" not in text and "C" not in text                   # ... and no terminus letter


def test_no_pocket_outline_and_a_scale_bar(heavy):
    face, res = heavy
    column = thrombin(layout="column").render(style="schematic")
    ribbon = [f for f in walk(column.scene) if isinstance(f.item, Polyline) and f.item.paint.width == 4.4]
    assert ribbon and not [f for f in walk(res.scene) if isinstance(f.item, Polyline) and f.item.paint.width == 4.4]
    legend = groups(res, "legend")[0]
    assert f"{SCALE_BAR:g} Å between the centres of two boxes" in [f.item.content for f in inside(res, legend, Text)]
    scale = res.graph.meta["interface"]["plane"]["px_per_angstrom"]
    bars = [f.item for f in inside(res, legend, Line) if abs(abs(f.item.x2 - f.item.x1) - SCALE_BAR * scale) < 1e-3]
    assert len(bars) == 1 and bars[0].y1 == bars[0].y2
    assert "between the centres of two boxes" not in words(column)


def test_a_plane_figure_does_not_speak_of_a_column_a_ligand_or_a_pocket(heavy):
    face, res = heavy
    said = [t.strip() for t in re.findall(r">([^<>]+)<", res.svg)] + list(res.warnings)
    assert len(said) > 200 and not [t for t in said if re.search(r"ligand|pocket", t, re.I)]
    assert not [t for t in said if "column" in t]
    titles = [g.title for g in groups(res, "unit")]
    assert len(titles) == len(face.site.residues)
    assert all(" of chain B (numbered as in the structure file)" in t for t in titles)
    assert "are not drawn where they lie but beside the boxes they touch" in words(res)
    # Where the model has a pLDDT, the numbers under the title speak of boxes, not of a column.
    tectb = pae_case("afdb_tectb", layout="auto").render(style="schematic")
    assert tectb.graph.meta["interface"]["layout"] == "plane"
    assert "boxes pLDDT" in words(tectb) and "column pLDDT" not in words(tectb)
    assert "of the residues around the boxes" in tectb.scene.description


def test_no_residue_is_drawn_twice_on_a_plane(heavy):
    face, res = heavy
    assert not [n for n in res.graph.nodes if n.attrs.get("drawings")]
    assert "Drawn more than once" not in words(res)
    assert res.quality.n_residues == len({c.residue_b for c in face._oriented})


def test_both_styles_and_both_themes(heavy):
    face, res = heavy
    dark = face.render(style="schematic", theme="dark")
    assert dark.status == "ok" and dark.svg != res.svg and dark.graph.to_json() == res.graph.to_json()
    # The detailed style, on an interface of a size it holds: the heavy chain of the antibody on lysozyme.
    detailed = chains_interface("fab_lysozyme", "H", "C", layout="auto").render(style="detailed")
    assert detailed.status == "ok" and detailed.graph.meta["interface"]["layout"] == "plane"
    assert f"{SCALE_BAR:g} Å between the centres of two boxes" in words(detailed)


# ---------------------------------------------------------------------------
# The graph, twice the same bytes, the command line
# ---------------------------------------------------------------------------

def test_the_same_bytes_twice_and_from_the_graph(heavy):
    face, res = heavy
    st = face.structure
    again = Interface(st, ["A", "B"], list(reversed(interface_contacts(st, ["A"], ["B"]))),
                      layout="auto").render(style="schematic")
    assert again.svg == res.svg and again.graph.to_json() == res.graph.to_json()
    from bindview.graph import Graph

    back = Graph.from_json(res.graph.to_json())
    redrawn = face.render(style="schematic", graph=back)
    assert redrawn.svg == res.svg


def test_boxes_moved_by_hand_are_measured_where_they_stand(heavy):
    """A graph can be edited. The numbers are of the boxes as they stand, and boxes dragged out of
    their arrangement make the figure unreadable like any other layout that fails its audit."""
    from bindview.graph import Graph

    face, res = heavy
    moved = Graph.from_json(res.graph.to_json())
    units = sorted((n for n in moved.nodes if n.kind == "ligand_group" and "position" in n.attrs),
                   key=lambda n: n.attrs["position"])
    for n in units[: len(units) // 2]:
        n.x += 420.0                                             # half of the boxes, far to the right
    out = face.render(style="schematic", graph=moved, on_unreadable="render")
    plane = out.graph.meta["interface"]["plane"]
    assert out.status == "unreadable" and not out.quality.faithful
    assert plane["distance_error"] > ERROR_MAX and plane["lost_neighbours"] > 0
    assert any("the boxes as they stand are not a faithful picture of where the residues lie" in w for w in out.warnings)
    # One box dragged onto another, where the numbers still pass both thresholds: the figure is refused all the same.
    space, scale = centres(face), res.graph.meta["interface"]["plane"]["px_per_angstrom"]
    at = drawn_at(res, len(space))

    def passes(i: int, j: int) -> bool:
        figure = at.copy()
        figure[i] = at[j] + (3.0, 2.0)
        return fidelity(space, figure / scale).ok

    i, j = next((i, j) for i in range(len(space)) for j in range(len(space)) if i != j and passes(i, j))
    onto = Graph.from_json(res.graph.to_json())
    place = {n.attrs["position"]: n for n in onto.nodes if n.kind == "ligand_group" and "position" in n.attrs}
    place[i + 1].x, place[i + 1].y = place[j + 1].x + 3.0, place[j + 1].y + 2.0
    covered = face.render(style="schematic", graph=onto, on_unreadable="render")
    said = covered.graph.meta["interface"]["plane"]
    assert said["boxes_covering_each_other"] == 1 and said["neighbours_kept"] >= KEPT_MIN
    assert said["distance_error"] <= ERROR_MAX
    assert covered.status == "unreadable" and not covered.quality.faithful
    assert any("1 pair of boxes lie on top of each other" in w for w in covered.warnings)
    assert "boxes_covering_each_other" not in plane and "boxes_covering_each_other" not in res.graph.meta["interface"]["plane"]
    # A graph that holds a column is not drawn as a plane.
    column = thrombin(layout="column", order="sequence").render(style="schematic")
    with pytest.raises(BindviewError, match="the graph holds a column"):
        face.render(style="schematic", graph=column.graph)
    # ... nor a column in one order as a column in another: its boxes are known by their place.
    assert column.graph.meta["interface"]["spine_order"] == "sequence"
    with pytest.raises(BindviewError, match="orders its column by the interface"):
        thrombin(layout="column", order="interface").render(style="schematic", graph=column.graph)


def test_command_line(heavy, tmp_path, capsys):
    face, res = heavy
    st = face.structure
    tsv = tmp_path / "contacts.tsv"
    tsv.write_text(dump_interface_contacts(interface_contacts(st, ["A"], ["B"])))
    out, graph = tmp_path / "out.svg", tmp_path / "graph.json"
    common = ["interface", str(model_of("thrombin_hirudin")), "--between", "A", "B", "--contacts", str(tsv),
              "-o", str(out), "--style", "schematic", "--graph", str(graph)]
    assert cli(common + ["--json", "--layout", "auto"]) == 0
    printed = capsys.readouterr()
    kept = res.graph.meta["interface"]["plane"]["neighbours_kept"]
    assert f"{SCALE_BAR:g} Å between the centres of two boxes" in out.read_text()
    assert '"layout": "plane"' in printed.out and f'"neighbours_kept": {kept}' in printed.out
    assert "layout: on the plane through them: the residues of chain B" in printed.err
    assert '"version": "1.3"' in graph.read_text()
    # Without the option: the column, and a line that says what the rule would have drawn.
    assert cli(common) == 0
    said = capsys.readouterr().err
    assert ('layout: a column, the default; asked to choose (layout "auto"), the rule would draw these residues on '
            'the plane through them') in said
    assert "between the centres of two boxes" not in out.read_text()
    assert '"layout": "column"' in graph.read_text() and '"version": "1.2"' in graph.read_text()


# ---------------------------------------------------------------------------
# With a predicted aligned error: three quantities, three places
# ---------------------------------------------------------------------------

def test_the_error_of_the_model_and_the_error_of_the_picture_are_said_apart():
    """pLDDT is in the strokes, PAE is a tag on a line, and how true the arrangement is stands in the
    footnote and the legend. None of the three borrows another's mark."""
    face = pae_case("afdb_tectb", layout="auto")
    assert face.layout == "plane"
    res = face.render(style="schematic")
    assert res.status == "ok"
    values = [c.pae for c in face.contacts]
    text = words(res)
    assert f"contact PAE {min(values):.1f}–{max(values):.1f} Å" in text and "Not pLDDT" in text
    kept = res.graph.meta["interface"]["plane"]["neighbours_kept"]
    assert f"{percent(kept)} of neighbour relations are kept" in text
    assert f"{SCALE_BAR:g} Å between the centres of two boxes" in text
    tags = [f.item for f in walk(res.scene) if isinstance(f.item, Text) and f.item.content.startswith("PAE ")
            and f.in_class("interaction") is not None]
    on = [pae_tag(c.attrs["pae"]) for c in res.graph.contacts]
    assert tags and sorted((t.content, t.bold) for t in tags) == sorted(t for t in on if t is not None)
    assert [g.data.get("confidence") for g in groups(res, "interaction")] and all(
        g.data.get("confidence") for g in groups(res, "interaction"))
    quiet = pae_case("afdb_fah", layout="auto").render(style="schematic")
    assert quiet.graph.meta["interface"]["layout"] == "plane" and "No contact here reaches 5 Å" in words(quiet)


# ---------------------------------------------------------------------------
# When the plane cannot be laid out
# ---------------------------------------------------------------------------

def test_a_plane_that_cannot_be_laid_out_gives_way_to_the_column_for_that_figure_only():
    face = thrombin(layout="auto")
    n = len(face._runs)
    plane = face.diagram
    real = plane.render
    # The plane of this interface lays out; make it fail the way a crowded one does, by giving the
    # figure with the boxes where they lie a page it cannot fit.
    plane.render = lambda style="detailed", **options: real(style=style, **{**options, "figure_width_mm": 25})
    small = face.render(style="schematic")
    meta = small.data["interface"]
    assert small.status == "ok" and meta["layout"] == "column" and meta["spine_order"] == "sequence"
    assert meta["layout_rule"] == ("a column in the order of the sequence: the residues of chain B at the interface "
                                   f"that are shown come in {n} stretches of its chain, but a figure with the boxes "
                                   "where they lie could not be laid out legibly")
    assert any(w.startswith("chain B is drawn as a column in the order of the sequence") for w in small.warnings)
    assert "a figure with the boxes where they lie could not be laid out legibly. How the stretches" in words(small)
    # The object is still what it was: with its own render back, the plane is drawn again.
    plane.render = real
    assert face.layout == "plane" and face.diagram is plane
    assert face.render(style="schematic").graph.meta["interface"]["layout"] == "plane"
    # ... and the graph of the column it drew can be handed back to it.
    assert face.render(style="schematic", graph=small.graph).svg == small.svg


def test_a_residue_without_a_backbone_has_no_box_and_so_there_is_no_plane():
    from bindview.structure import Structure

    st = structure_of("thrombin_hirudin")
    whole = thrombin()
    one = next(k for k in whole.site.residues if all("N" not in c.atoms_a for c in whole._oriented if c.residue_a == k))
    cut = Structure(atoms=[a for a in st.atoms if not (a.chain == one.chain and (a.resnum, a.icode) == (one.number, one.icode)
                                                       and a.name == "N")],
                    source="one nitrogen missing", plddt_scale=st.plddt_scale, qa_metrics=st.qa_metrics)
    # The plain table takes as a partner only residues with N, CA and C: the chain with the gap is its first side.
    face = Interface(cut, ["A", "B"], interface_contacts(cut, ["B"], ["A"]), layout="auto")
    assert face.spine.chains == ("B",) and one in face.site.residues and len(face._runs) > 1
    assert face.layout == "column" and face.layout_rule == (
        "a column in the order of the sequence: the residues of chain B at the interface that are shown come in "
        f"{len(face._runs)} stretches of its chain, but not every residue of them has a backbone of its own, so not "
        "every one can be given its place")
    res = face.render(style="schematic")
    assert res.status == "ok" and f"chain B is drawn as {face.layout_rule}" in res.warnings


def test_a_link_between_two_boxes_that_are_not_neighbours_is_a_straight_line_on_a_plane(heavy):
    """A disulfide inside the spine is a bracket beside a column. On a plane there is no column to stand
    beside: it is the straight line between the two boxes, and the layout treats it as ink."""
    from bindview.depict import DepBond
    from bindview.layout import LigandField
    from bindview.styles.polymer_draw import brackets, draw_polymer
    from bindview.theme import LIGHT as PALETTE, Metrics

    face, _ = heavy
    m = Metrics()
    dep = face.diagram.backend.depict(m)
    assert dep.arranged == "plane" and dep.plane is not None and not brackets(dep)
    before = LigandField(dep, reach=60.0)
    a, b = dep.units[2], dep.units[-3]
    dep.bonds.append(DepBond(a.anchor, b.anchor, 1, kind="crosslink"))
    after = LigandField(dep, reach=60.0)
    assert len(after.bond_a) == len(before.bond_a) + 1 and not brackets(dep)
    group = draw_polymer(dep, PALETTE, m)
    backbone = next(g for g in group.items if getattr(g, "cls", "") == "backbone")
    dashed = [it for it in backbone.items if isinstance(it, Line) and it.paint.dash == (3.2, 2.4)]
    assert len(dashed) == 1 and not [it for it in backbone.items if isinstance(it, Polyline)]
    (line,) = dashed

    def on(point, box) -> bool:
        return box[0] - 1e-6 <= point[0] <= box[2] + 1e-6 and box[1] - 1e-6 <= point[1] <= box[3] + 1e-6

    ends = ((line.x1, line.y1), (line.x2, line.y2))
    assert (on(ends[0], a.box) and on(ends[1], b.box)) or (on(ends[0], b.box) and on(ends[1], a.box))
    # In a column the same link is a bracket, and no straight line.
    column = thrombin(layout="column").diagram.backend.depict(m)
    column.bonds.append(DepBond(column.units[2].anchor, column.units[-3].anchor, 1, kind="crosslink"))
    assert len(brackets(column)) == 1


# ---------------------------------------------------------------------------
# Deposited structures
# ---------------------------------------------------------------------------

def deposited(slug: str, participants: list[str], layout: str = "auto", **options) -> Interface:
    """An interface of a deposited structure with contacts from the distance table, its layout left to
    the rule unless told otherwise."""
    st = structure_of(slug)
    parts = [Participant.parse(p) for p in participants]
    contacts = [c for a, b in zip(parts, parts[1:]) for c in interface_contacts(st, a.chains, b.chains)]
    return Interface(st, parts, contacts, layout=layout, **options)


def test_on_deposited_interfaces_the_layout_follows_the_two_thresholds():
    """For each side of several interfaces the stretches are those of the residues' places in the chain,
    the projection measured here is the one the rule judged, and the plane is drawn where it passes
    both thresholds and not elsewhere."""
    cases = [("thrombin_hirudin", ["A", "B"], None), ("thrombin_hirudin", ["A", "B"], "A"),
             ("mhc_peptide", ["B", "A"], None), ("mhc_peptide", ["B", "A"], "A"), ("mhc_peptide", ["A", "P"], "A"),
             ("fab_lysozyme", ["H", "C"], None), ("fab_lysozyme", ["H", "L"], None),
             ("repressor_dna", ["3", "4"], None), ("repressor_dna", ["3", "1+2"], None)]
    drawn = []
    for slug, between, spine in cases:
        face = deposited(slug, between, spine=spine)
        assert face._runs == stretches([face._index[k] for k in face.site.residues])
        judged = (face._arrangement or face._considered).bare
        again = projected(face)
        assert (judged.kept, judged.false, judged.lost, judged.pairs) == (again.kept, again.false, again.lost, again.pairs)
        assert judged.error == pytest.approx(again.error, abs=1e-9)
        assert judged.ok == (again.kept >= KEPT_MIN and again.error <= ERROR_MAX)
        assert face.layout == ("plane" if judged.ok else "column"), (slug, between, spine)
        drawn.append(judged.ok)
    assert any(drawn) and not all(drawn)


def test_a_groove_is_curved_and_is_not_projected():
    """The lining of the groove of HLA-A2 around its peptide: no plane is good for it."""
    face = deposited("mhc_peptide", ["A", "P"], spine="A")
    assert face.layout == "column" and len(face._runs) > 1
    bare = face._considered.bare
    assert bare.kept < KEPT_MIN and not bare.ok
    res = face.render(style="schematic")
    assert res.status == "ok" and "How the stretches lie to each other is not shown." in words(res)
    # The peptide itself is one stretch and is a column.
    peptide = deposited("mhc_peptide", ["A", "P"])
    assert peptide.spine.chains == ("P",) and peptide.layout == "column"
    assert peptide.layout_rule.startswith("a column: the residues of chain P") and "but" not in peptide.layout_rule
    # Asked for, it is drawn on its plane, seen from the side of the groove.
    forced = deposited("mhc_peptide", ["A", "P"], layout="plane")
    drawn = forced.render(style="schematic")
    assert drawn.status == "ok" and drawn.graph.meta["interface"]["layout_rule"] == (
        "on the plane through them: the plane was asked for")
    assert "seen from the side of chain A" in words(drawn)


def test_the_antibody_s_heavy_chain_is_drawn_by_its_three_loops():
    """3D9A, the HyHEL-10 Fab on hen egg lysozyme. The heavy chain's residues at the interface come in
    three stretches, its three CDR loops, and are drawn on a plane; so are those of the light chain, and
    the epitope with both chains of the antibody around it. The numbers are regression values."""
    heavy = deposited("fab_lysozyme", ["the heavy chain=H", "lysozyme=C"], spine="H")
    names = [f"{heavy.structure.residue_name(k).title()}{k.number}" for k in heavy.site.residues]
    # Three loops of the variable domain (the entry numbers the heavy chain from 301): H1, H2 and H3.
    assert names == ["Asp332", "Tyr333", "Tyr350", "Ser352", "Tyr353", "Ser354", "Ser356", "Trp398", "Asp399"]
    assert [(names[a], names[b]) for a, b in heavy._runs] == [("Asp332", "Tyr333"), ("Tyr350", "Ser356"),
                                                             ("Trp398", "Asp399")]
    bare = heavy._arrangement.bare
    assert heavy.layout == "plane" and bare.ok
    assert (bare.kept, bare.false, bare.lost, bare.pairs, round(bare.error, 3)) == (1.0, 0, 0, 20, 0.032)
    res = heavy.render(style="schematic")
    plane = res.graph.meta["interface"]["plane"]
    assert plane["drawn"] is True and (plane["neighbours_kept"], plane["distance_error"], plane["boxes_moved"]) == (
        1.0, 0.032, 2)
    assert (plane["out_of_plane_rms"], plane["partner_off_normal_deg"]) == (1.08, 8.1)
    assert res.status == "ok" and res.graph.to_dict()["version"] == "1.3" and len(heavy.contacts) == 19
    text = words(res)
    assert "come in 3 stretches of its chain: each box stands where its residue's centre lies on the plane" in text
    assert "seen from the side of lysozyme" in text and "100 % of neighbour relations are kept" in text
    # One lysine of the antigen between the first and the third loop, bridged to an aspartate of each.
    assert sorted((c.residue, c.ligand_residue) for c in res.graph.contacts if c.type == "saltbridge") == [
        ("R:C:LYS697", "Asp332"), ("R:C:LYS697", "Asp399")]
    # The three loops are three groups on the page: the lines of the chain join boxes within a loop only.
    backbone = groups(res, "backbone")[0]
    assert len([f for f in inside(res, backbone, Line) if not f.item.paint.dash]) == 4      # 332-333, 352-353-354, 398-399
    assert heavy.render(style="detailed").status == "ok"

    light = deposited("fab_lysozyme", ["the light chain=L", "lysozyme=C"], spine="L")
    assert [k.number for k in light.site.residues] == [31, 50, 53, 92, 96] and len(light._runs) == 3
    found = light._arrangement.bare
    assert light.layout == "plane" and (found.kept, found.pairs, round(found.error, 3)) == (1.0, 3, 0.008)
    assert light.render(style="schematic").status == "ok"

    # The other way round: the epitope as boxes, and the residues of both chains of the antibody around it.
    epitope = deposited("fab_lysozyme", ["lysozyme=C", "the antibody=H+L"])
    assert epitope.spine.chains == ("C",) and epitope.layout == "plane" and len(epitope._runs) == 5
    assert [k.number for k in epitope.site.residues] == [615, 621, 662, 663, 675, 693, 696, 697, 700, 701, 702, 703]
    seen = epitope._arrangement.bare
    assert (seen.kept, seen.false, seen.lost, seen.pairs, round(seen.error, 3)) == (1.0, 0, 0, 19, 0.075)
    drawn = epitope.render(style="schematic")
    said = drawn.graph.meta["interface"]["plane"]
    assert drawn.status == "ok" and (said["neighbours_kept"], said["distance_error"], said["boxes_moved"]) == (1.0, 0.082, 5)
    assert {n.attrs["chain"] for n in drawn.graph.nodes if n.kind == "residue"} == {"H", "L"}
    assert drawn.quality.n_residues == 14 and "seen from the side of the antibody" in words(drawn)


@needs_plip
def test_plip_reads_the_antibody_the_same_way():
    """With PLIP's contacts instead of the distance table the heavy chain has three more residues at the
    interface, in the same three loops, and is drawn on a plane all the same."""
    st = structure_of("fab_lysozyme")
    face = Interface(st, ["the heavy chain=H", "lysozyme=C"], spine="H", layout="auto")
    assert face.source == "detected" and face.layout == "plane"
    assert (len(face.site.residues), len(face._runs)) == (12, 3)
    bare = face._arrangement.bare
    assert (bare.kept, bare.false, bare.lost, bare.pairs, round(bare.error, 3)) == (1.0, 0, 0, 30, 0.069)
    assert face.render(style="schematic").status == "ok"


# ---------------------------------------------------------------------------
# Why this plane and not the one the contacts point along
# ---------------------------------------------------------------------------

def test_the_residues_own_plane_passes_wherever_the_plane_normal_to_the_contacts_does():
    """The plane could have been the one the contact set looks along: the direction between the two sides.
    Measured on the interfaces of the repository, side by side: the residues' own plane passes both
    thresholds wherever that one does, and also where it does not."""
    def along(points: np.ndarray, normal: np.ndarray) -> np.ndarray:
        n = normal / np.linalg.norm(normal)
        rel = points - points.mean(axis=0)
        flat = rel - np.outer(rel @ n, n)
        return flat @ np.linalg.svd(flat, full_matrices=False)[2][:2].T

    faces = [thrombin(), thrombin(spine="A"), chains_interface("mhc_peptide", "B", "A"),
             chains_interface("mhc_peptide", "B", "A", spine="A"), chains_interface("fab_lysozyme", "H", "C"),
             chains_interface("fab_lysozyme", "H", "C", spine="C"), chains_interface("fab_lysozyme", ["H", "L"], "C"),
             chains_interface("fab_lysozyme", "H", "L"), chains_interface("fab_lysozyme", "H", "L", spine="L"),
             chains_interface("repressor_dna", "3", "4"), chains_interface("repressor_dna", "3", "4", spine="3"),
             pae_case("afdb_fah"), pae_case("afdb_tectb")]
    better = 0
    for face in faces:
        space, toward = centres(face), toward_partner(face)
        own = fidelity(space, project(space, toward).xy)
        other = fidelity(space, along(space, toward))
        assert own.ok or not other.ok, face.title
        better += own.ok and not other.ok
    assert better > 0
