"""The semantic layer: the graph a figure is drawn from, its JSON, and what moving a node means.

The graph is three things at once -- the model a GUI drags, the response of a
web service, and a regression target that does not move when the SVG's
cosmetics do -- so these tests hold it to all three: it must round-trip, it
must redraw the identical figure, and its numbers must be the engine's own.
"""

from __future__ import annotations

import copy
import json
import math

import pytest

from conftest import (GOLDEN, TESTDATA, diagram_of, fresh_diagram, meta_of, model_of, on_golden_platform,
                      overlay_diagram, overlay_members, overlay_meta, render_of)
from bindview import Comparison, Diagram, Graph, load_interactions
from bindview.errors import GraphError
from bindview.graph import NODE_KINDS, SCHEMA, VERSION
from bindview.layout import SPRINGS, LayoutParams
from bindview.scene import Group
from scenetools import groups

#: The reference case: c-Kit with imatinib, with ring centroids, charged atoms and water bridges in its table.
REFERENCE = "kit_imatinib"
ROUND_TRIP = [REFERENCE, "dhfr_methotrexate", "ca2_celecoxib"]      # rings; charged groups; a small ligand
#: The graph of the reference case's detailed figure, as the test below digests it.
STORED = GOLDEN / f"{REFERENCE}.detailed.graph.json"


def comparison(key: str) -> Comparison:
    return Comparison([overlay_diagram(key, slug) for slug, _ in overlay_members(key)])


def residues_of(key: str) -> list[set[str]]:
    """The residues of each member of a set, as its committed table has them (``R:A:GLN570``)."""
    return [{f"R:{r.chain}:{r.residue_name}{r.residue_number}" for r in load_interactions(path / "interactions.tsv")}
            for _, path in overlay_members(key)]


# ---------------------------------------------------------------------------
# scene -> JSON -> scene
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("style", ("detailed", "schematic"))
@pytest.mark.parametrize("case", ROUND_TRIP)
def test_graph_round_trips_through_json_and_redraws_the_identical_figure(case, style):
    first = render_of(case, style)
    graph = first.graph
    assert graph is not None and graph.view == style
    text = graph.to_json()
    back = Graph.from_json(text)
    assert back == graph and back.to_json() == text                  # the object, and the bytes
    again = diagram_of(case).render(style=style, graph=back)         # no layout is run for this
    assert again.svg == first.svg                                    # byte for byte
    assert again.graph == graph and again.status == first.status
    assert again.quality.overlaps == first.quality.overlaps == 0 and again.quality.valid


def test_json_is_versioned_and_plain(tmp_path):
    graph = render_of(REFERENCE).graph
    data = json.loads(graph.to_json(tmp_path / "g.json"))
    assert (data["schema"], data["version"]) == (SCHEMA, VERSION) == ("bindview.graph", "1.0")
    assert list(data)[:5] == ["schema", "version", "view", "bond_length", "origin"]
    assert {n["kind"] for n in data["nodes"]} <= set(NODE_KINDS)
    assert Graph.from_json(tmp_path / "g.json") == graph
    # A reader keeps what it does not know, so a newer minor version survives a round trip ...
    data["version"] = "1.7"
    data["nodes"][0]["future_field"] = {"a": 1}
    kept = Graph.from_dict(data)
    assert kept.nodes[0].attrs["future_field"] == {"a": 1} and kept.to_dict()["nodes"][0]["future_field"] == {"a": 1}
    # ... and refuses what it cannot be trusted to understand.
    for bad, message in (({**data, "version": "2.0"}, "not supported"), ({**data, "schema": "other"}, "not a bindview"),
                         ({k: v for k, v in data.items() if k != "view"}, "lacks 'view'")):
        with pytest.raises(GraphError, match=message):
            Graph.from_dict(bad)
    broken = copy.deepcopy(data)
    broken["contacts"][0]["ligand"] = "L:NOPE"
    with pytest.raises(GraphError, match="unknown node 'L:NOPE'"):
        Graph.from_dict(broken)
    with pytest.raises(GraphError, match="not valid JSON"):
        Graph.from_json("{ this is not json")


@pytest.mark.skipif(not on_golden_platform() or not STORED.exists(),
                    reason="the stored graph is compared where the stored figures were made (tests/golden/VERSIONS.json)")
def test_golden_graph_is_a_regression_target_that_ignores_cosmetics():
    """Positions to a hundredth of a unit, identities and contacts exactly -- no colour, no stroke."""
    def digest(graph: Graph) -> dict:
        return {
            "view": graph.view, "bond_length": graph.bond_length,
            "nodes": {n.id: [n.kind, None if n.x is None else round(n.x, 2), None if n.y is None else round(n.y, 2),
                             round(n.angle, 4), n.mirror] for n in graph.nodes if n.kind != "legend"},
            "contacts": {c.id: [c.type, c.residue, c.ligand, c.drawn, c.distance] for c in graph.contacts},
        }
    got = digest(render_of(REFERENCE).graph)
    assert json.loads(json.dumps(got)) == json.loads(STORED.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# What a node carries
# ---------------------------------------------------------------------------

def test_nodes_carry_identity_anchor_spring_and_contacts():
    res = render_of(REFERENCE)
    g = res.graph
    rows = load_interactions(TESTDATA / REFERENCE / "interactions.tsv")
    ids = [n.id for n in g.nodes]
    assert len(ids) == len(set(ids)) and len({c.id for c in g.contacts}) == len(g.contacts) == len(rows)
    by_contact = {c.id: c for c in g.contacts}
    ligand = g.group("ligand")
    assert ligand.movable and set(ligand.members) == {n.id for n in g.nodes if n.group == "ligand"}
    residues = g.of_kind("residue")
    assert len(residues) == res.quality.n_residues == len({(r.chain, r.residue_number) for r in rows})
    for node in residues:
        own = [by_contact[cid] for cid in node.contacts]
        assert own and all(c.residue == node.id for c in own)
        assert set(node.anchors) == {c.ligand for c in own} <= set(ligand.members)        # anchored to the ligand
        assert node.spring == pytest.approx(sum(c.weight for c in own)) and node.spring > 0
        assert node.movable and node.group is None and node.x is not None
        assert {"resname", "chain", "number", "class", "form"} <= set(node.attrs)
        for c in own:                                                                    # and referenced back
            assert c.id in g.node(c.ligand).contacts
    # Every interaction attaches to an atom, a ring centroid or a group -- as a node of its own kind.
    (stack,) = [c for c in g.contacts if c.type == "pi_stacking"]
    (ring,) = [r.ligand_atoms for r in rows if r.type.value == "pi_stacking"]
    centroid = g.node(stack.ligand)
    assert centroid.kind == "ring_centroid" and len(centroid.members) == len(ring) and centroid.group == "ligand"
    assert all(g.node(c.ligand).kind == "ligand_atom" for c in g.contacts if c.type == "hbond")
    assert {"ligand_atom", "ring_centroid", "residue", "legend"} == {n.kind for n in g.nodes}
    # Page coordinates are the diagram's plus the origin: the atom map of the result, exactly.
    for name, xy in res.atoms.items():
        assert g.page(f"L:{name}") == xy
    assert g.size == (res.scene.width, res.scene.height)


def test_springs_are_the_engines_own():
    params = LayoutParams()
    detailed, schematic = render_of("ar_r1881").graph, render_of("ar_r1881", "schematic").graph
    assert detailed.springs == {**SPRINGS, "length": params.w_length, "short": params.w_short, "unit": 30.0}
    for g in (detailed, schematic):
        assert all(c.weight == SPRINGS["link"] for c in g.contacts if c.drawn)
        assert all(c.ideal > c.min_length >= 0.0 for c in g.contacts if c.drawn)
    # A residue that makes only line-less contacts is held by its sightlines together: like one link
    # when it is drawn as an eyelash, by the sightline constant when it is a bubble.
    lashes = [n for n in detailed.of_kind("residue") if n.attrs["form"] == "eyelash"]
    assert lashes and all(n.spring == pytest.approx(SPRINGS["eyelash"]) for n in lashes)
    plain = [n for n in schematic.of_kind("residue")
             if not any(schematic.contact(cid).drawn for cid in n.contacts)]
    assert plain and all(n.spring == pytest.approx(SPRINGS["sightline"]) for n in plain)
    # The ligand has no spring: it is the rigid frame everything else is tethered to.
    assert all(n.spring is None and not n.anchors for n in detailed.nodes if n.group == "ligand")


# ---------------------------------------------------------------------------
# Moving things
# ---------------------------------------------------------------------------

def test_dragging_the_ligand_moves_it_rigidly_and_a_residue_alone():
    dia = diagram_of(REFERENCE)
    base = render_of(REFERENCE).graph
    g = copy.deepcopy(base)
    before = {n.id: (n.x, n.y) for n in g.nodes}

    moved = g.move("L:C1", 12.5, -7.0)                       # any atom takes the whole depiction with it
    assert set(moved) == set(g.group("ligand").members)
    assert g.of_kind("ring_centroid") and all(n.id in moved for n in g.of_kind("ring_centroid"))
    for n in g.nodes:
        dx, dy = n.x - before[n.id][0], n.y - before[n.id][1]
        assert (dx, dy) == (pytest.approx(12.5), pytest.approx(-7.0)) if n.id in moved else (dx, dy) == (0.0, 0.0)
    atoms = [n for n in g.of_kind("ligand_atom")]
    was = {n.id: before[n.id] for n in atoms}
    for a in atoms[:6]:
        for b in atoms[-6:]:                                 # internal geometry is untouched
            assert math.dist((a.x, a.y), (b.x, b.y)) == pytest.approx(math.dist(was[a.id], was[b.id]), abs=1e-9)

    assert g.move("R:A:LEU595", 4.0, 3.0) == ["R:A:LEU595"]   # a residue moves alone
    redrawn = dia.render(graph=g, on_unreadable="render")
    assert redrawn.svg is not None and redrawn.graph.node("R:A:LEU595").x == g.node("R:A:LEU595").x
    # The redrawn ligand is where the graph put it, as one body.
    shift = {(round(redrawn.graph.node(n.id).x - before[n.id][0], 6), round(redrawn.graph.node(n.id).y - before[n.id][1], 6))
             for n in atoms}
    assert shift == {(12.5, -7.0)}
    assert redrawn.svg != render_of(REFERENCE).svg


def test_an_edited_layout_is_held_to_the_same_audit():
    dia = diagram_of(REFERENCE)
    g = copy.deepcopy(render_of(REFERENCE).graph)
    a, b = g.node("R:A:LEU595"), g.node("R:A:VAL603")
    g.move(a.id, b.x - a.x, b.y - a.y)                        # drop one residue onto another
    refused = dia.render(graph=g)
    assert refused.status == "unreadable" and refused.figure == "summary" and refused.quality.overlaps >= 1
    assert refused.graph is None                              # a summary has no node-level layout to hand out
    forced = dia.render(graph=g, on_unreadable="render")
    assert forced.status == "unreadable" and forced.figure == "diagram" and forced.quality.overlaps >= 1
    assert forced.graph.node(a.id).x == b.x


def test_the_legend_is_a_node_and_moves_by_an_offset():
    dia = diagram_of(REFERENCE)
    first = render_of(REFERENCE)
    g = copy.deepcopy(first.graph)
    legend = g.node("legend")
    assert legend.kind == "legend" and legend.movable and legend.attrs["offset"] == [0.0, 0.0]
    assert legend.attrs["width"] > 100 and g.move("legend", 10.0, 20.0) == ["legend"]
    moved = dia.render(graph=g)

    def at(res):
        (lg,) = [it for it in res.scene.items if isinstance(it, Group) and it.cls == "legend"]
        return lg.translate
    assert (at(moved)[0] - at(first)[0], at(moved)[1] - at(first)[1]) == (pytest.approx(10.0), pytest.approx(20.0))
    assert moved.graph.node("legend").attrs["offset"] == [10.0, 20.0]
    assert [n for n in moved.graph.nodes if n.kind != "legend"] == [n for n in first.graph.nodes if n.kind != "legend"]


def test_a_graph_is_refused_where_it_does_not_fit():
    g = render_of(REFERENCE).graph
    # A GraphError: an error of the package, which the command line says in words, and still a ValueError.
    with pytest.raises(GraphError, match="laid out for the 'detailed' style") as raised:
        diagram_of(REFERENCE).render(style="schematic", graph=g)
    assert isinstance(raised.value, ValueError)
    with pytest.raises(GraphError, match="does not hold this ligand"):
        diagram_of("ar_r1881").render(graph=g)
    fixed = copy.deepcopy(g)
    fixed.group("ligand").movable = False
    with pytest.raises(GraphError, match="not movable"):
        fixed.move("L:C1", 1.0, 1.0)
    with pytest.raises(GraphError, match="no node 'R:nope'"):
        g.node("R:nope")


# ---------------------------------------------------------------------------
# A number that is no position, no scale and no step is refused, in words
# ---------------------------------------------------------------------------

_open: dict = {}


def celecoxib():
    """An open case, its figure and the graph of that as plain data (to be copied, not changed)."""
    if not _open:
        dia = fresh_diagram("ca2_celecoxib")
        res = dia.render()
        _open.update(dia=dia, res=res, doc=json.loads(res.graph.to_json()))
    return _open["dia"], _open["res"], _open["doc"]


def with_node(doc: dict, kind: str, **changes) -> dict:
    out = copy.deepcopy(doc)
    next(n for n in out["nodes"] if n["kind"] == kind).update(changes)
    return out


NOT_A_GRAPH = [
    ("a residue at x NaN", lambda doc: with_node(doc, "residue", x=math.nan), r"x must be a finite number, not nan"),
    ("a residue at x Infinity", lambda doc: with_node(doc, "residue", x=math.inf), r"x must be a finite number, not inf"),
    ("an atom of the ligand at y NaN", lambda doc: with_node(doc, "ligand_atom", y=math.nan), r"y must be a finite number"),
    ("a residue turned by NaN", lambda doc: with_node(doc, "residue", angle=math.nan), r"angle must be a finite number"),
    ("a bond length of 0", lambda doc: {**doc, "bond_length": 0}, r"bond_length must be a positive number, not 0"),
    ("a negative bond length", lambda doc: {**doc, "bond_length": -30}, r"bond_length must be a positive number, not -30"),
    ("a bond length NaN", lambda doc: {**doc, "bond_length": math.nan}, r"bond_length must be a finite number"),
    ("a bond length in words", lambda doc: {**doc, "bond_length": "long"}, r"bond_length must be a number, not 'long'"),
    ("a legend moved by one number", lambda doc: with_node(doc, "legend", offset=[1]), r"offset of the legend must be two numbers, not \[1\]"),
    ("a legend moved by a word", lambda doc: with_node(doc, "legend", offset="ab"), r"offset of the legend must be two numbers, not 'ab'"),
    ("a legend moved by nothing", lambda doc: with_node(doc, "legend", offset=None), r"offset of the legend must be two numbers, not None"),
    ("a legend moved by NaN", lambda doc: with_node(doc, "legend", offset=[math.nan, 0]), r"offset of the legend must be two numbers"),
    ("an origin with a name", lambda doc: {**doc, "origin": {"x": 1}}, r"origin must be two numbers, not \{'x': 1\}"),
    ("an origin of one number", lambda doc: {**doc, "origin": [1]}, r"origin must be two numbers, not \[1\]"),
    ("a size in words", lambda doc: {**doc, "size": ["wide", 2]}, r"size must be two numbers"),
]


@pytest.mark.parametrize("what, change, said", NOT_A_GRAPH, ids=[row[0] for row in NOT_A_GRAPH])
def test_a_document_with_a_number_that_is_none_is_refused_when_it_is_read(what, change, said):
    _, _, doc = celecoxib()
    assert Graph.from_json(json.dumps(doc)).view == "detailed"          # (as it was written, it is read)
    with pytest.raises(GraphError, match=said):
        Graph.from_json(json.dumps(change(doc)))


def test_a_graph_that_was_given_such_a_number_by_hand_is_refused_when_it_is_drawn():
    dia, res, _ = celecoxib()
    for spoil, said in (
            (lambda g: setattr(g.of_kind("residue")[0], "x", math.nan), "x must be a finite number"),
            (lambda g: setattr(g, "bond_length", 0.0), "bond_length must be a positive number"),
            (lambda g: setattr(g, "bond_length", -30.0), "bond_length must be a positive number"),
            (lambda g: g.node("legend").attrs.update(offset=[1]), "offset of the legend must be two numbers")):
        graph = copy.deepcopy(res.graph)
        spoil(graph)
        for options in ({}, {"on_unreadable": "render"}):
            with pytest.raises(GraphError, match=said):
                dia.render(graph=graph, **options)


def test_a_step_or_a_turn_that_is_no_number_is_refused_and_moves_nothing():
    import numpy as np

    dia, res, _ = celecoxib()
    graph = copy.deepcopy(res.graph)
    turning = next(n for n in graph.of_kind("residue") if n.attrs.get("form") == "fragment")
    for dx, dy in ((math.nan, 0.0), (math.inf, 0.0), (0.0, -math.inf), ("3", 0), (None, 0), (True, 1.0)):
        for who in (turning.id, "ligand", "legend"):
            with pytest.raises(GraphError, match="move takes a step of two finite numbers"):
                graph.move(who, dx, dy)
    for degrees in (math.nan, math.inf, "90", None, False):
        for who in (turning.id, "ligand", "figure"):
            with pytest.raises(GraphError, match="rotate takes an angle in degrees, a finite number"):
                graph.rotate(who, degrees)
    assert graph == res.graph and dia.render(graph=graph).svg == res.svg
    # A number of NumPy's is a number.
    assert graph.move(turning.id, np.float64(2.0), np.int64(1)) == [turning.id]
    assert graph.rotate(turning.id, np.float32(90.0)) == [turning.id]
    assert (turning.x, turning.y) == (res.graph.node(turning.id).x + 2.0, res.graph.node(turning.id).y + 1.0)
    assert turning.angle == pytest.approx(res.graph.node(turning.id).angle + math.pi / 2.0)


def test_text_is_read_as_text_and_a_file_as_a_file(tmp_path):
    _, res, _ = celecoxib()
    text = res.graph.to_json()
    # JSON is JSON whatever it begins with ...
    for other in ("[1, 2]", "null", " 12 "):
        with pytest.raises(GraphError, match="a graph document must be a JSON object"):
            Graph.from_json(other)
    assert Graph.from_json("\ufeff" + text) == Graph.from_json("\n\t " + text) == res.graph
    # ... and what is neither JSON nor a file that is there is said to be neither.
    for neither in ("", "nonsense\x00", "layuot.json", str(tmp_path), text[:300]):
        with pytest.raises(GraphError, match="not valid JSON, and not the name of a file that is there") as told:
            Graph.from_json(neither)
        assert len(str(told.value)) < 300                         # (not the whole document again)
    path = tmp_path / "layout.json"
    path.write_text(text, encoding="utf-8")
    assert Graph.from_json(path) == Graph.from_json(str(path)) == res.graph
    path.write_bytes(b"\xef\xbb\xbf" + text.encode("utf-8"))        # as some editors save it
    assert Graph.from_json(path) == res.graph
    with pytest.raises(GraphError, match="cannot read .*missing.json"):
        Graph.from_json(tmp_path / "missing.json")
    path.write_text(text[:300], encoding="utf-8")
    with pytest.raises(GraphError, match="layout.json is not valid JSON"):
        Graph.from_json(str(path))
    path.write_bytes(b"\xff\xfe\x00{")
    with pytest.raises(GraphError, match="cannot read .*layout.json"):
        Graph.from_json(path)


# ---------------------------------------------------------------------------
# Every view hands out its graph
# ---------------------------------------------------------------------------

def test_figures_without_nodes_hand_out_no_graph(tmp_path):
    """The figure of a complex in which nothing was found (carbon monoxide, with a table that has only its header),
    and the table drawn in place of a figure that its page cannot hold (imatinib on 60 mm)."""
    empty = tmp_path / "interactions.tsv"
    empty.write_text((TESTDATA / "hb_carbon_monoxide" / "interactions.tsv").read_text().splitlines()[0] + "\n")
    m = meta_of("hb_carbon_monoxide")["ligand"]
    nothing = Diagram.from_structure(model_of("hb_carbon_monoxide"), m["residue_name_in_cif"], smiles=m["smiles"],
                                     interactions=load_interactions(empty)).render()
    assert nothing.figure == "empty" and nothing.graph is None
    summary = diagram_of(REFERENCE).render(figure_width_mm=60.0)
    assert summary.figure == "summary" and summary.graph is None


def test_difference_view_graph_has_two_rigid_ligands_on_one_residue_frame():
    """The two ligands of the androgen receptor (``ar_ligands``)."""
    cmp_ = comparison("ar_ligands")
    res = cmp_.render()
    g = res.graph
    assert g.view == "difference" and Graph.from_json(g.to_json()) == g
    assert [grp.id for grp in g.groups] == ["ligand:1", "ligand:2"] and all(grp.movable for grp in g.groups)
    names = [overlay_meta("ar_ligands", slug)["ligand"]["name"] for slug, _ in overlay_members("ar_ligands")]
    assert [p["label"] for p in g.panels] == names and g.panels[0]["origin"] == list(g.origin)
    assert g.panels[0]["origin"] != g.panels[1]["origin"] and g.size == (res.scene.width, res.scene.height)
    residues = g.of_kind("residue")
    assert len(residues) == res.quality.n_residues == len(cmp_.difference().residues())
    # Which of the two each residue touches, as their tables have it.
    one, two = residues_of("ar_ligands")

    def side(mine, other, ident):
        return "shared" if ident in mine and ident in other else "only" if ident in mine else "absent"
    assert {n.id: n.attrs["contact"] for n in residues} == {
        ident: {"ligand:1": side(one, two, ident), "ligand:2": side(two, one, ident)} for ident in one | two}
    assert one & two and one - two and two - one
    # One position per residue, used by both panels; contacts say which ligand they belong to.
    assert all(n.x is not None for n in residues)
    assert {c.attrs["ligand_group"] for c in g.contacts} == {"ligand:1", "ligand:2"}
    # (A residue and a kind of contact that one ligand has twice, as R1881 has Phe764's hydrophobic one, is one of them.)
    shared = {c.id.split("|")[0] + "|" + c.type for c in g.contacts
              if c.attrs["shared"] and c.attrs["ligand_group"] == "ligand:1"}
    assert shared == {f"{b.column.rsplit(':', 1)[0]}|{b.type.value}" for b in cmp_.difference().shared}
    # Moving an atom of one ligand moves that ligand and nothing else.
    moved = set(copy.deepcopy(g).move("L1:C1", 3.0, 0.0))
    assert moved == set(g.group("ligand:1").members) and not moved & set(g.group("ligand:2").members)


def test_consensus_graph_carries_occupancy_on_nodes_and_contacts():
    """Four steroids of the glucocorticoid receptor, one scaffold (``gr_steroids``); a residue's occupancy is the share
    of the members whose table has it."""
    cmp_ = comparison("gr_steroids")
    g = cmp_.render().graph
    members = residues_of("gr_steroids")
    assert g.view == "consensus" and g.ensemble == f"the {len(members)} ligands of the set"
    assert Graph.from_json(g.to_json()) == g and g.meta["frame"] == "core"
    assert g.meta["drawn_ligand"] in g.meta["ligands"]
    for n in g.of_kind("residue"):
        held = sum(n.id in mine for mine in members)
        assert n.occupancy == pytest.approx(held / len(members)), n.id
        assert n.attrs["via_core"] + n.attrs["via_substituent_only"] == held, n.id
    gln = g.node("R:A:GLN570")                                    # in every member's table, and by a hydrogen bond only
    assert gln.occupancy == 1.0
    assert [(g.contact(c).type, g.contact(c).occupancy, g.contact(c).ligand) for c in gln.contacts] == [
        ("hbond", 1.0, None)]
    assert gln.anchors and all(g.node(a).kind == "ligand_atom" for a in gln.anchors)
    # The same numbers as the consensus records: one path, not two.
    records = {(c.residue_label, c.type.value): c.occupancy for c in cmp_.consensus_contacts()}
    for c in g.contacts:
        label = g.node(c.residue).label
        assert c.occupancy == records[(label, c.type)] and not c.drawn
    # Without a common scaffold (``gr_diverse``) there is no ligand in the graph either.
    diverse = comparison("gr_diverse").render().graph
    assert diverse.meta["frame"] != "core" and not diverse.groups and not diverse.of_kind("ligand_atom")
    members = residues_of("gr_diverse")
    assert {n.id: n.occupancy for n in diverse.of_kind("residue")} == pytest.approx(
        {ident: sum(ident in mine for mine in members) / len(members) for ident in set().union(*members)})


# ---------------------------------------------------------------------------
# The other table a figure's content is handed out as: fingerprints, read back from CSV
# ---------------------------------------------------------------------------

def odd_bits():
    from bindview.fingerprint import Bit
    from bindview.interactions import InteractionType as T

    return [Bit("A", 7, "", "CA2", T.METAL), Bit("A", 27, "", "CA", T.METAL), Bit("A", 16, "", "H2U", T.HBOND),
            Bit("A", 40, "", "5MC", T.HBOND), Bit("A", 10, "", "2MG", T.PI_STACKING), Bit("A", 12, "B", "0AF", T.HBOND),
            Bit("B", -3, "A", "ASP", T.SALTBRIDGE), Bit("A", 60, "A", "GLN", T.HBOND), Bit("A", 570, "", "GLN", T.HBOND)]


def test_a_column_of_the_wide_table_reads_back_as_the_residue_it_was_written_for():
    from bindview.fingerprint import Bit

    bits = odd_bits()
    assert [Bit.from_column(b.column) for b in bits] == bits      # CA2 7 is not CA 27; H2U 16 is read at all
    assert len({b.column for b in bits}) == len(bits)
    # A name of letters is written next to its number, and such a column is read back as that residue.
    assert [b.column for b in bits[-3:]] == ["B:ASP-3A:saltbridge", "A:GLN60A:hbond", "A:GLN570:hbond"]
    # A name with a digit in it stands apart from the number.
    assert bits[0].column == "A:CA2:7::metal" and bits[5].column == "A:0AF:12:B:hbond"


@pytest.mark.parametrize("counts", [False, True])
def test_the_wide_table_round_trips_with_residues_that_have_digits_in_their_names(counts):
    from bindview.fingerprint import BitValue, Fingerprint, FingerprintTable

    bits = odd_bits()
    table = FingerprintTable([Fingerprint("one", {b: BitValue(2) for b in bits[:5]}),
                              Fingerprint("two", {b: BitValue(1) for b in bits[3:]})])
    wide = table.to_csv(form="wide", counts=counts)
    back = FingerprintTable.from_csv(wide)
    assert back.columns == table.columns and back.names == table.names
    assert back.matrix(counts) == table.matrix(counts) and back.to_csv(form="wide", counts=counts) == wide
    assert FingerprintTable.from_csv(table.to_csv()).columns == table.columns          # (and the long form)


def test_a_cell_of_the_wide_table_that_is_no_count_is_refused_in_words():
    from bindview.errors import InteractionError
    from bindview.fingerprint import FingerprintTable

    for cell in ("yes", "1.5", "-1", "x"):
        with pytest.raises(InteractionError, match=f"{cell!r} in column 'A:GLN570:hbond' of ligand 'lig' is not a count"):
            FingerprintTable.from_csv(f"ligand,A:GLN570:hbond\nlig,{cell}\n")
    # A count is one however the name of the residue was spelled; a row that ends early has none for the rest.
    table = FingerprintTable.from_csv("ligand,A:gln570:hbond,A:LEU3:hydrophobic\none,2, 1\ntwo,1\n")
    assert table.names == ["one", "two"] and table.matrix(counts=True) == [[1, 2], [0, 1]]
    assert [b.column for b in table.columns] == ["A:LEU3:hydrophobic", "A:GLN570:hbond"]


def test_a_fingerprint_table_is_read_from_text_or_from_a_file_whatever_the_file_is_called(tmp_path):
    from bindview.errors import InteractionError
    from bindview.fingerprint import FingerprintTable

    text = "ligand,A:GLN570:hbond\nlig,1\n"
    assert FingerprintTable.from_csv(text).names == FingerprintTable.from_csv("\ufeff" + text).names == ["lig"]
    assert FingerprintTable.from_csv("ligand,A:GLN570:hbond").names == []          # a table without a ligand yet
    for name in ("set.csv", "set.txt", "SET.CSV", "fingerprints"):
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        assert FingerprintTable.from_csv(path).names == FingerprintTable.from_csv(str(path)).names == ["lig"], name
    path.write_bytes(b"\xef\xbb\xbf" + text.encode("utf-8"))                       # as a spreadsheet saves it
    assert FingerprintTable.from_csv(path).names == ["lig"]
    with pytest.raises(InteractionError, match="cannot read .*missing.csv"):
        FingerprintTable.from_csv(tmp_path / "missing.csv")
    for name in ("missing.csv", "missing.txt"):
        with pytest.raises(InteractionError, match=f"not a fingerprint CSV: .*{name!r} is not the name of a file that is there"):
            FingerprintTable.from_csv(name)
    path.write_bytes(b"\xff\xfe\x00ligand")
    with pytest.raises(InteractionError, match="cannot read .*fingerprints"):
        FingerprintTable.from_csv(str(path))
