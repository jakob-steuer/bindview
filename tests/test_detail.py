"""How much of a residue is drawn: ``detail``.

A residue of a chain is a box, and a residue beside it is whatever its style makes of it. With
``detail`` a participant's residues that make a directional contact are drawn with atoms: at least the
part that makes the contact, and what the level adds (``"backbone"``, ``"sidechain"``, ``"atoms"``).

The first half of this file is geometry and needs no figure: what a level is, which residues get it and
where their atoms stand. The second half draws: the A-B interface of the TNF trimer (one pLDDT for a
residue), the heavy chain of an antibody on lysozyme (``fab_lysozyme``; with a pLDDT for every atom made
up for the test), the interfaces between the domains of two AlphaFold database models, bicalutamide in the
androgen receptor, hirudin's peptide on thrombin and the two hairpins of the riboswitch.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from bindview.fragments import DETAILS, build_fragment, check_detail, fragment_atoms
from bindview.polymer import PIECE_GAP, Piece, lay_piece, piece_sides, shown_atoms, stack_column
from bindview.residues import _TEMPLATES, BACKBONE, template_bonds
from bindview.theme import Metrics


# ---------------------------------------------------------------------------
# Which atoms of a residue a level draws
# ---------------------------------------------------------------------------

def test_the_levels_are_four_and_none_is_no_level():
    assert DETAILS == ("box", "backbone", "sidechain", "atoms")
    assert [check_detail(d) for d in DETAILS] == list(DETAILS) and check_detail(None) is None
    with pytest.raises(ValueError, match=r"detail is 'box', 'backbone', 'sidechain' or 'atoms' \(or None.*not 'all'"):
        check_detail("all")
    with pytest.raises(ValueError, match="the detail of chain B is"):
        check_detail("side chain", "the detail of chain B")


def test_a_level_adds_to_the_part_that_makes_the_contact_and_takes_nothing_away():
    # Serine through its hydroxyl: the side chain is the contact, so "sidechain" adds nothing.
    assert fragment_atoms("SER", ["OG"]) == ("CA", "CB", "OG") == fragment_atoms("SER", ["OG"], "sidechain")
    assert fragment_atoms("SER", ["OG"], "backbone") == ("N", "CA", "C", "O", "CB", "OG") == fragment_atoms("SER", ["OG"], "atoms")
    # Through its carbonyl: the backbone is the contact, and "sidechain" adds the side chain to it.
    assert fragment_atoms("SER", ["O"]) == ("CA", "C", "O")
    assert fragment_atoms("SER", ["O"], "sidechain") == ("CA", "C", "O", "CB", "OG")
    assert fragment_atoms("SER", ["O"], "backbone") == ("N", "CA", "C", "O")
    # Glycine has no side chain to add; a proline's ring closes on its nitrogen, which comes with it.
    assert fragment_atoms("GLY", ["N"], "sidechain") == ("N", "CA")
    assert fragment_atoms("PRO", ["O"], "sidechain") == ("N", "CA", "C", "O", "CB", "CG", "CD")
    # No template, or a contact through an atom the template does not have: no atoms at any level.
    assert fragment_atoms("MSE", ["SE"], "atoms") is None and fragment_atoms("SER", ["OXT"], "atoms") is None
    for name, t in _TEMPLATES.items():
        for atom in t.atoms:
            least = set(fragment_atoms(name, [atom]) or ())
            assert least and atom in least and "CA" in least
            for level in ("backbone", "sidechain", "atoms"):
                more = set(fragment_atoms(name, [atom], level) or ())
                assert least <= more <= set(t.atoms), (name, atom, level)
            assert set(fragment_atoms(name, [atom], "atoms") or ()) == set(t.atoms)
            assert set(BACKBONE) <= set(fragment_atoms(name, [atom], "backbone") or ())
            assert set(t.sidechain) <= set(fragment_atoms(name, [atom], "sidechain") or ())


def test_a_fragment_at_a_level_has_the_bonds_of_its_template():
    for name in ("ARG", "TYR", "TRP", "PRO", "GLY"):
        frag = build_fragment(name, ["O"], 20.0, "atoms")
        assert frag is not None and set(frag.atoms) == set(_TEMPLATES[name].atoms)
        assert set(frag.bonds) == set(template_bonds(name))
        lengths = [math.dist(frag.xy[a], frag.xy[b]) for a, b, _ in frag.bonds]
        assert max(abs(v - 20.0) for v in lengths) < 20.0 * 0.25


# ---------------------------------------------------------------------------
# Which side, and how far the column is stretched
# ---------------------------------------------------------------------------

def test_side_chains_alternate_and_residues_that_share_a_partner_stand_together():
    # No partner in common: right, left, right, left, from the top.
    assert piece_sides([3, 5, 8, 9], {}) == {3: 1, 5: -1, 8: 1, 9: -1}
    # 3 and 8 are held by one arginine: they stand on one side, and the alternation counts groups.
    partners = {3: {("A", 82, "")}, 8: {("A", 82, "")}, 5: {("A", 10, "")}}
    assert piece_sides([3, 5, 8, 9], partners) == {3: 1, 5: -1, 8: 1, 9: 1}
    # ... through a chain of partners as well, and whatever the order the partners are given in.
    chained = {1: {"x"}, 2: {"x", "y"}, 3: {"y"}, 4: {"z"}}
    assert piece_sides([1, 2, 3, 4], chained) == {1: 1, 2: 1, 3: 1, 4: -1}
    assert piece_sides([1, 2, 3, 4], {k: set(reversed(sorted(v))) for k, v in chained.items()}) == \
        piece_sides([1, 2, 3, 4], chained)
    assert piece_sides([], {}) == {}


def rect_piece(*rects) -> Piece:
    return Piece({}, tuple(rects))


def test_a_column_is_stretched_by_the_least_that_keeps_six_units_between_residues():
    assert PIECE_GAP == 6.0
    half_w, half_h, pitch = 30.0, 8.5, 22.0
    # No atoms beside any box: a column of boxes alone.
    assert stack_column(4, pitch, half_w, half_h, {}) == [(0.0, 0.0), (0.0, 22.0), (0.0, 44.0), (0.0, 66.0)]
    tall = (40.0, -20.0, 90.0, 20.0)                       # a side chain to the right, 40 units high, clear of the boxes
    left = (-90.0, -20.0, -40.0, 20.0)
    # Two neighbours with side chains on the same side: 40 high each, 6 between them.
    ys = [y for _, y in stack_column(2, pitch, half_w, half_h, {0: rect_piece(tall), 1: rect_piece(tall)})]
    assert ys == [0.0, 46.0]
    # On opposite sides they are in nobody's way: the boxes stay at their pitch.
    ys = [y for _, y in stack_column(2, pitch, half_w, half_h, {0: rect_piece(tall), 1: rect_piece(left)})]
    assert ys == [0.0, 22.0]
    # Three in a row, the outer two on the right: the third clears the first, not only the second.
    ys = [y for _, y in stack_column(3, pitch, half_w, half_h, {0: rect_piece(tall), 1: rect_piece(left), 2: rect_piece(tall)})]
    assert ys == [0.0, 22.0, 46.0]
    # Ink that reaches down beside the next box, within six units of it, moves that box: the room is
    # between a residue's atoms and another residue's box as well.
    close = (30.0, -20.0, 90.0, 20.0)
    ys = [y for _, y in stack_column(2, pitch, half_w, half_h, {0: rect_piece(close)})]
    assert ys == [0.0, 20.0 + 6.0 + 8.5]
    # Ink above a box (a backbone nitrogen on the line of the column) clears the box above it.
    above = (-4.0, -40.0, 4.0, -30.0)
    ys = [y for _, y in stack_column(2, pitch, half_w, half_h, {1: rect_piece(above)})]
    assert ys == [0.0, 8.5 + 6.0 + 40.0]
    # Never closer than the pitch, whatever stands beside the boxes.
    for pieces in ({0: rect_piece(tall)}, {1: rect_piece(left)}, {0: rect_piece(above), 2: rect_piece(above)}):
        ys = [y for _, y in stack_column(5, pitch, half_w, half_h, pieces)]
        assert all(b - a >= pitch - 1e-9 for a, b in zip(ys, ys[1:]))


def spine(detail="sidechain"):
    """The column of the TNF trimer's A-B interface as a chain backend, with a level."""
    from conftest import trimer_contacts, trimer_structure
    from bindview.interface import Interface

    return Interface(trimer_structure(), ["A", "B"], trimer_contacts("A", "B"), detail=detail).diagram.backend


def test_which_atoms_stand_beside_a_box():
    backend = spine()
    pep = backend.peptide
    by_number = {r.number: r for r in pep.residues}
    tyr, gly, arg = by_number["59"], by_number["148"], by_number["2"]
    local = lambda r: {i: n for i, n in r.atom_names}                          # noqa: E731
    oh = next(i for i, n in tyr.atom_names if n == "OH")
    shown = shown_atoms(pep, {tyr.position: {oh}})
    # The alpha carbon is the box: the whole side chain and nothing of the backbone.
    assert sorted(local(tyr)[i] for i in shown[tyr.position]) == sorted(("CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"))
    assert set(shown) == {tyr.position}
    # A residue without a directional contact stays a box at every level, and so does everything at "box".
    from dataclasses import replace
    assert shown_atoms(replace(pep, detail="atoms"), {}) == {}
    assert shown_atoms(replace(pep, detail="box"), {tyr.position: {oh}}) == {}
    assert shown_atoms(replace(pep, detail=None), {tyr.position: {oh}}) == {}
    # Glycine through its carbonyl oxygen: the carbonyl; at "atoms" the nitrogen too, never the alpha carbon.
    o = next(i for i, n in gly.atom_names if n == "O")
    assert sorted(local(gly)[i] for i in shown_atoms(pep, {gly.position: {o}})[gly.position]) == ["C", "O"]
    every = shown_atoms(replace(pep, detail="atoms"), {gly.position: {o}})[gly.position]
    assert sorted(local(gly)[i] for i in every) == ["C", "N", "O"]
    assert len(shown_atoms(replace(pep, detail="atoms"), {arg.position: {arg.n}})[arg.position]) == 10


def test_where_the_atoms_of_a_residue_stand():
    backend = spine()
    pep = backend.peptide
    m = Metrics()
    length = m.bond_length * m.protein_bond_scale
    half_w, half_h = 30.0, 8.5
    arg = next(r for r in pep.residues if r.number == "2")
    names = {i: n for i, n in arg.atom_names}
    every = tuple(i for i in arg.atoms if names[i] != "CA")
    right = lay_piece(arg, every, 1, half_w, half_h, m)
    left = lay_piece(arg, every, -1, half_w, half_h, m)
    at = {names[i]: p for i, p in right.xy.items()}
    side = [n for n in at if n not in ("N", "C", "O")]
    # The side chain stands to the right of the box, its middle on the level of the box ...
    assert all(at[n][0] > half_w for n in side)
    assert abs(sum(at[n][1] for n in side) / len(side)) < 1e-6
    # ... with the bonds of the template at their length, the first of them from the edge of the box.
    for a, b, _ in template_bonds("ARG"):
        if a in side and b in side:
            assert abs(math.dist(at[a], at[b]) - length) < 0.02 * length
    assert abs(math.dist(at["CB"], (half_w, 0.0)) - length) < 0.02 * length
    # The nitrogen on the line of the column above the box, the carbonyl carbon below, as the chain runs;
    # the oxygen beside the carbon, away from the side chain.
    assert at["N"] == (0.0, -half_h - length) and at["C"] == (0.0, half_h + length)
    assert at["O"][0] < 0.0 < at["O"][1] - at["C"][1] and abs(math.dist(at["O"], at["C"]) - length) < 1e-6
    # On the left it is the mirror image.
    for i, (x, y) in right.xy.items():
        assert left.xy[i] == pytest.approx((-x, y))
    # Three parts: the side chain from the edge of its box, the nitrogen, the carbonyl.
    assert [len(atoms) for atoms, _ in right.parts] == [7, 1, 2]
    assert [start for _, start in right.parts] == [(half_w, 0.0), None, None]
    # Its ink, for the room between residues, is measured finely: every atom, and every bond in two halves.
    # Five names (NE, NH1, NH2, N and O; a carbon is where bonds meet); seven bonds among the atoms and
    # three to the box (to CB, to N and to C), each in two halves.
    assert len(right.rects) == 5 + 2 * (7 + 3)
    assert len(right.bonds) == 7 and [point for point, _ in right.stems] == [(half_w, 0.0), (0.0, -half_h), (0.0, half_h)]
    # Beside the box there is only the first half of the bond to the side chain: nothing of an arginine's
    # side chain comes within six units of the box below it, 22 units down, and so that box is not moved.
    beside = [r for r in right.rects if half_w - 1.0 <= r[0] < half_w + PIECE_GAP]
    assert len(beside) == 1 and beside[0][3] + PIECE_GAP < 22.0 - half_h


# ---------------------------------------------------------------------------
# The figure
# ---------------------------------------------------------------------------

_drawn: dict = {}


def interface(detail=None, style="detailed", **options):
    """The A-B interface of the TNF trimer at a level, and its figure: an object of its own for every
    level and style (a figure that has to narrow keeps its site), drawn once."""
    from conftest import trimer_contacts, trimer_file, trimer_structure
    from bindview.interface import Interface

    key = (repr(detail), style, repr(sorted(options.items())))
    if key not in _drawn:
        face = Interface(trimer_structure(), ["A", "B"], trimer_contacts("A", "B"),
                         pae=[trimer_file("pae", k) for k in range(2)], detail=detail, **options)
        _drawn[key] = (face, face.render(style=style))
    return _drawn[key]


def lines(res) -> list:
    from scenetools import groups

    return groups(res, "interaction")


def boxes(res) -> list:
    from scenetools import groups

    return groups(res, "unit")


def words(res) -> str:
    from scenetools import words as said

    return said(res)


@pytest.mark.parametrize("style", ("schematic", "detailed"))
def test_without_a_level_a_figure_has_no_atoms_beside_its_boxes(style):
    """``detail=None`` is not a level: the figure is the interface's own without the option, and ``"box"``
    for the column draws the same figure."""
    from conftest import trimer

    face, res = interface(None, style)
    assert res.svg == trimer("A", "B").render(style=style).svg
    assert not any("beside" in b.data for b in boxes(res)) and "plddt-basis" not in res.svg
    assert "detail" not in res.graph.meta["interface"] and "drawn_atoms_at" not in res.confidence
    assert face.diagram.backend.shown == {} and all(u.beside == () for u in face.diagram.backend.depict(Metrics()).units)
    # "box" for the column is boxes too: the same figure, and no atom beside any box.
    boxed, as_boxes = interface({"chain B": "box"}, style)
    assert as_boxes.svg == res.svg and not any("beside" in b.data for b in boxes(as_boxes))
    assert as_boxes.graph.meta["interface"]["detail"] == {"chain A": None, "chain B": "box"}


@pytest.mark.parametrize("style", ("schematic", "detailed"))
def test_a_residue_of_the_column_with_a_directional_contact_gets_its_atoms(style):
    """Promotion by the kind of contact: thirteen of the seventeen residues of the column make an H-bond
    or a salt bridge and are drawn with atoms; the four that make hydrophobic contacts only stay boxes. The
    whole interface is drawn, and it is readable."""
    face, res = interface("sidechain", style)
    assert res.status == "ok" and face.site.whole and len(face.site.residues) == 17
    assert res.quality.valid and res.quality.readable and res.quality.min_font_pt >= 5.0
    with_atoms = {b.data["residue"] + b.title.split()[1] for b in boxes(res) if "beside" in b.data}
    assert len(boxes(res)) == 17 and len(with_atoms) == 13
    directional = {g.data["ligand-atoms"].split("/")[0] for g in lines(res) if g.data["type"] != "hydrophobic"}
    assert {b.title.split()[1] for b in boxes(res) if "beside" in b.data} == directional
    plain = sorted(b.title.split()[0] + b.title.split()[1] for b in boxes(res) if "beside" not in b.data)
    assert plain == ["Ile155", "Leu36", "Pro117", "Val13"]
    # Every directional line ends on an atom that is drawn, and names it; a hydrophobic one ends on a box.
    drawn = {name for b in boxes(res) if "beside" in b.data for name in b.data["beside"].split(",")}
    for g in lines(res):
        if g.data["type"] == "hydrophobic":
            assert "ligand-residue" in g.data
        else:
            assert "ligand-residue" not in g.data and set(g.data["ligand-atoms"].split(",")) <= drawn
    # The alpha carbon is the box, and a proline's nitrogen stays in it.
    assert not [n for n in drawn if n.endswith("/CA")] and "113/N" not in drawn and "113/CD" in drawn
    # The side chain of a tyrosine whole, although one atom of it makes the contact.
    tyr = next(b for b in boxes(res) if b.title.startswith("Tyr 59 "))
    assert sorted(n.split("/")[1] for n in tyr.data["beside"].split(",")) == sorted(
        ("CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"))
    # A glycine has no side chain to add: what makes its contact, the carbonyl.
    gly = next(b for b in boxes(res) if b.title.startswith("Gly 148 "))
    assert sorted(n.split("/")[1] for n in gly.data["beside"].split(",")) == ["C", "O"]


def test_the_figure_says_what_the_atoms_beside_a_box_are():
    face, res = interface("sidechain", "schematic")
    text = words(res)
    assert ("Where a residue of chain B makes a contact that has a direction (an H-bond, a salt bridge and the "
            "like), the atoms that make it are drawn beside its box, and with them the whole side chain. The box "
            "is the residue's alpha carbon. A line that ends on such an atom is the contact of that atom and gives "
            "its distance; a line that ends on a box stands for the pair of residues.") in text
    assert res.graph.meta["interface"]["detail"] == {"chain A": "sidechain", "chain B": "sidechain"}
    assert face.detail == {"chain A": "sidechain", "chain B": "sidechain"}
    # The bubbles of the schematic style have no atoms: a level that is not drawn is named, in the result.
    assert [w for w in res.warnings if w.startswith("the residues around chain B are drawn as bubbles, without the "
                                                    "atoms that detail 'sidechain' asks for")]
    assert not [w for w in interface("sidechain", "detailed")[1].warnings if "bubbles" in w]
    assert not [w for w in interface({"chain B": "sidechain"}, "schematic")[1].warnings if "bubbles" in w]


def test_no_line_of_another_residue_runs_across_atoms_that_are_drawn():
    """The ink of drawn atoms is in the way of every line that does not end on it, as a box is: no line of
    another residue runs across a side chain drawn beside the column."""
    from bindview.layout.field import LigandField

    face, res = interface("sidechain", "detailed")
    dep = face.diagram.backend.depict_from(Metrics().with_(bond_length=res.graph.bond_length), res.graph)
    assert sum(len(u.rects) for u in dep.units) > 150 and all(len(u.rects) == 0 for u in dep.units if not u.beside)
    # The audit of the figure counts such a line as it counts one across a box, and found none.
    assert res.quality.link_hits == 0 and res.quality.bond_crossings == 0
    # The rule itself, on the ink of this column: a line that runs across a side chain it does not end on
    # is in the way; one that ends on an atom of it, or at the centre of its ring, is not.
    field = LigandField(dep, reach=60.0)
    tyr = next(u for u in dep.units if u.name == "TYR" and u.number == "59")
    oh = next(i for i in tyr.beside if dep.atoms[i].name == "59/OH")
    ring = [i for i in tyr.beside if dep.atoms[i].name.split("/")[1] in ("CG", "CD1", "CD2", "CE1", "CE2", "CZ")]
    centre = np.array(dep.centroid(ring))
    xs = [r[0] for r in tyr.rects] + [r[2] for r in tyr.rects]
    ys = [r[1] for r in tyr.rects] + [r[3] for r in tyr.rects]
    above, below = np.array([[centre[0], min(ys) - 20.0]]), np.array([[centre[0], max(ys) + 20.0]])
    assert field.boxes_crossed(above, below, np.array([0.0, -500.0]))[0]                      # straight across the ring
    assert not field.boxes_crossed(above, centre[None, :], centre, reach=tyr.rects[0][2] - tyr.rects[0][0] + 30.0)[0]
    assert field.boxes_crossed(above, centre[None, :], centre)[0]                             # ... which is its own target's ink only by its reach
    tip = np.array(dep.pos(oh))
    away = tip + (tip - centre) / np.linalg.norm(tip - centre) * 40.0
    assert not field.boxes_crossed(away[None, :], tip[None, :], tip, reach=8.0)[0]            # from outside to the hydroxyl
    assert max(xs) > min(xs) and max(ys) > min(ys)


@pytest.mark.parametrize("style", ("schematic", "detailed"))
def test_a_figure_with_atoms_beside_its_boxes_is_redrawn_from_its_graph_as_it_was(style):
    from bindview.graph import Graph

    face, res = interface("sidechain", style)
    again = face.diagram.render(style=style, graph=Graph.from_json(res.graph.to_json()))
    assert again.svg == res.svg
    unit = next(n for n in res.graph.nodes if n.kind == "ligand_group" and n.attrs.get("resname") == "TYR"
                and n.attrs.get("number") == "59")
    assert sorted(a.split("/")[1] for a in unit.attrs["beside"]) == sorted(("CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"))
    atoms = {n.label for n in res.graph.nodes if n.kind == "ligand_atom"}
    assert set(unit.attrs["beside"]) <= atoms and "59/CA" not in atoms


def test_a_figure_at_a_level_does_not_depend_on_hash_randomisation():
    """Which side a side chain takes, which residues stand together and how much room they are given all
    come from sets of residues and atoms. Two interpreters with different hash seeds draw the same bytes:
    a column of a model with one pLDDT for a residue (the trimer), one with one for every atom (the antibody
    with a pLDDT made up for the test), and the residues around a ligand."""
    import os
    import subprocess
    import sys

    from conftest import ROOT

    code = (
        "import hashlib, sys\n"
        "sys.path.insert(0, 'tests')\n"
        "from conftest import diagram_at, invented_plddt, structure_of, trimer\n"
        "from bindview.distance_table import interface_contacts\n"
        "from bindview.interface import Interface\n"
        "st = invented_plddt(structure_of('fab_lysozyme'))\n"
        "made = [trimer('A', 'B', detail='sidechain').render(style='detailed').svg,\n"
        "        Interface(st, ['H', 'C'], interface_contacts(st, ['H'], ['C']), spine='H', layout='column',\n"
        "                  detail='sidechain').render(style='detailed').svg,\n"
        f"        diagram_at({LIGAND!r}, 'atoms').render(style='detailed').svg]\n"
        "print(hashlib.sha256(''.join(made).encode()).hexdigest())\n")
    digests = set()
    for seed in ("0", "1234"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, check=True)
        digests.add(out.stdout.strip().splitlines()[-1])
    assert len(digests) == 1


# ---------------------------------------------------------------------------
# What a drawn atom's confidence is, and that the figure says so
# ---------------------------------------------------------------------------

def antibody(detail=None, layout="column", structure=None, **options):
    """HyHEL-10's heavy chain (chain H, numbered from 301 in this entry) on lysozyme (chain C), the crystal
    ``fab_lysozyme``, as an interface with the heavy chain as the column; contacts from the plain table."""
    from conftest import structure_of
    from bindview.distance_table import interface_contacts
    from bindview.interface import Interface, Participant

    st = structure if structure is not None else structure_of("fab_lysozyme")
    parts = [Participant.parse(p) for p in ("the heavy chain=H", "lysozyme=C")]
    return Interface(st, parts, interface_contacts(st, ["H"], ["C"]), spine="H", layout=layout, detail=detail,
                     **options)


def test_a_model_with_one_value_for_a_residue_says_so():
    """Boltz writes one pLDDT on all atoms of a residue: 0 of 471 residues of the trimer have atoms that
    differ. The atoms that are drawn are at their residue's value, and the figure says that they are."""
    from bindview.confidence import ATOM_SPREAD, gives_atoms

    assert ATOM_SPREAD == 0.01 and not gives_atoms([[88.0, 88.0], [41.5]]) and gives_atoms([[88.0, 88.0], [41.5, 41.52]])
    face, res = interface("sidechain", "detailed")
    assert face.diagram.confidence is not None and face.diagram.confidence.per_atom is False
    assert ("The atoms that are drawn of a residue are all at their residue's pLDDT: this model gives one value "
            "for a residue, not one for each of its atoms.") in words(res)
    assert 'data-plddt-basis="residue"' in res.svg and res.graph.meta["plddt_basis"] == "residue"
    assert res.confidence["drawn_atoms_at"] == "their residue's pLDDT: the model gives one value for a residue"
    # ... so every atom beside a box is in the band of its box.
    band = {b.title.split()[1]: b.data.get("confidence") for b in boxes(res)}
    nodes = {n.label: n.confidence for n in res.graph.nodes if n.kind == "ligand_atom"}
    from bindview.confidence import band_of
    assert nodes and all(band_of(v) == band[name.split("/")[0]] for name, v in nodes.items())


def test_a_model_with_a_value_for_every_atom_draws_each_atom_at_its_own():
    """A model that gives every atom a pLDDT of its own: the antibody's crystal with one made up for the
    test (:func:`conftest.invented_plddt`). Where atoms are drawn at a level they are each at their own value,
    and the figure says that. A residue beside the column whose drawn atoms fall in more than one band is
    drawn in a part for each; without a level as well (``plddt_by_atom``), and with ``plddt_by_atom=False``
    a fragment is one look, and nothing is said of a basis."""
    from conftest import invented_plddt, structure_of
    from bindview.confidence import band_of
    from bindview.fragments import fragment_atoms
    from scenetools import groups

    st = invented_plddt(structure_of("fab_lysozyme"))
    plddt = {(a.chain, a.resnum, a.name): a.b_iso for a in st.atoms}
    face = antibody("sidechain", structure=st)
    res = face.render(style="detailed")
    assert res.status == "ok" and face.site.whole and face.diagram.confidence.per_atom is True
    assert ("The atoms that are drawn of a residue are each at their own pLDDT: the model gives one for every "
            "atom.") in words(res)
    assert 'data-plddt-basis="atom"' in res.svg and res.confidence["drawn_atoms_at"] == "each atom's own pLDDT"
    # In the column: every atom drawn beside a box at its own value.
    drawn = [n for n in res.graph.nodes if n.kind == "ligand_atom"]
    assert drawn and all(n.confidence == pytest.approx(plddt["H", int(n.label.split("/")[0]), n.label.split("/")[1]])
                         for n in drawn)
    # Beside it: the residues of lysozyme whose fragment spans more than one band, each band a part of its own.
    involved: dict = {}
    for c in face.contacts:
        involved.setdefault((c.number_b, c.resname_b), set()).update(c.atoms_b)
    spanning = set()
    for (number, name), atoms in involved.items():
        kept = fragment_atoms(name, sorted(atoms), "sidechain") or ()
        if len({band_of(plddt["C", number, a]) for a in kept if ("C", number, a) in plddt}) > 1:
            spanning.add(f"{name.title()}{number}")
    split = {g.title.split(":")[0] for g in groups(res, "residue")
             if len([f for f in g.items if getattr(f, "cls", "") == "fragment"]) > 1}
    assert split == spanning and split
    assert all(g.data["detail"] == "sidechain" for g in groups(res, "residue") if g.data["form"] == "fragment")
    plain = antibody(structure=st).render(style="detailed")
    assert 'data-plddt-basis="atom"' in plain.svg and "are each at their own pLDDT" in words(plain)
    one_look = antibody(structure=st).render(style="detailed", plddt_by_atom=False)
    assert "plddt-basis" not in one_look.svg and "are each at their own pLDDT" not in words(one_look)


def test_a_level_leaves_the_pae_of_a_contact_as_it_is():
    """The PAE of a contact is written on its line at a level as on the figure of boxes, and the key shows
    the same samples: the AlphaFold database's model of TECTB, cut between its two domains."""
    from conftest import pae_case
    from scenetools import key, tags

    level = pae_case("afdb_tectb", fresh=True, detail="sidechain").render(style="detailed")
    plain = pae_case("afdb_tectb", fresh=True).render(style="detailed")
    assert level.status == plain.status == "ok" and tags(level)
    assert {(t.content, t.bold) for t in tags(level)} == {(t.content, t.bold) for t in tags(plain)}
    assert [(k.content, k.bold) for k in key(level)] == [(k.content, k.bold) for k in key(plain)]


def test_a_structure_without_plddt_says_nothing_of_a_basis():
    from conftest import trimer_contacts, trimer_structure
    from bindview.interface import Interface

    bare = Interface(trimer_structure(), ["A", "B"], trimer_contacts("A", "B"), detail="sidechain")
    off = bare.render(style="detailed", confidence=False)
    assert "plddt-basis" not in off.svg and "pLDDT" not in words(off).split("Residue of chain B")[1][:400]


# ---------------------------------------------------------------------------
# Whose level it is, and what is refused
# ---------------------------------------------------------------------------

def test_a_participant_has_a_level_of_its_own_and_the_figure_one_beneath_it():
    from conftest import trimer_contacts, trimer_structure
    from bindview.errors import BindviewError
    from bindview.interface import Interface, Participant

    st, contacts = trimer_structure(), trimer_contacts("A", "B")
    # Its own level wins over the figure's; a participant that names none takes the figure's.
    face = Interface(st, [Participant("subunit A", ("A",)), Participant("subunit B", ("B",), detail="atoms")], contacts,
                     detail="sidechain")
    assert face.detail == {"subunit A": "sidechain", "subunit B": "atoms"}
    assert face.diagram.backend.peptide.detail == "atoms" and face.diagram.detail_by_chain == {"A": "sidechain"}
    # By name, for one side only: the other has none.
    face = Interface(st, ["A", "B"], contacts, detail={"chain A": "backbone"})
    assert face.detail == {"chain A": "backbone", "chain B": None} and face.diagram.backend.peptide.detail is None
    assert Participant.parse("receptor=A").detail is None
    with pytest.raises(BindviewError, match="detail names 'chain C', which is not a participant of this interface"):
        Interface(st, ["A", "B"], contacts, detail={"chain C": "atoms"})
    with pytest.raises(BindviewError, match=r"detail is 'box', 'backbone', 'sidechain' or 'atoms' .*not 'full'"):
        Interface(st, ["A", "B"], contacts, detail="full")
    with pytest.raises(BindviewError, match="the detail of x is"):
        Participant("x", ("A",), detail="all-atom")


def test_atoms_stand_beside_the_boxes_of_a_column_and_not_of_a_plane():
    """Asked for together with the plane, that is refused in words. Left to the rule, which would draw
    these residues on a plane, the column is drawn, and the figure and the result say that and why."""
    from conftest import pae_case
    from bindview.errors import BindviewError

    slug = "afdb_tectb"                                    # two stretches near one plane: the rule draws the plane
    plane = pae_case(slug, fresh=True, layout="auto")
    spine, other = plane.spine.name, plane.others[0].name
    assert plane.layout == "plane"
    with pytest.raises(BindviewError) as refused:
        pae_case(slug, fresh=True, layout="plane", detail="sidechain")
    assert str(refused.value).startswith(f"{spine} is asked for with the atoms of its residues (detail 'sidechain') "
                                         f"and with its boxes where they lie on a plane (layout 'plane'). Atoms are "
                                         f"drawn beside the boxes of a column only")
    assert str(refused.value).endswith("Ask for one of the two")
    face = pae_case(slug, fresh=True, layout="auto", detail="sidechain")
    assert face.layout == "column" and face.layout_asked == "auto"
    assert face.layout_rule.startswith("a column, because the atoms of its residues are drawn beside their boxes "
                                       "(detail 'sidechain') and boxes on a plane have no side for them; the rule "
                                       "alone would draw these residues on the plane through them")
    res = face.render(style="schematic")
    assert res.status == "ok" and any(b.data.get("beside") for b in boxes(res))
    assert [w for w in res.warnings if w.startswith(f"{spine} is drawn as a column, because the atoms")]
    assert res.graph.meta["interface"]["layout"] == "column"
    # Boxes alone go on the plane, whatever the residues around them are drawn as.
    assert pae_case(slug, fresh=True, layout="auto", detail={other: "atoms"}).layout == "plane"
    assert pae_case(slug, fresh=True, layout="plane", detail="box").layout == "plane"
    # An interface the rule draws as a column anyway is a column with atoms, and says nothing of a plane it lost.
    column = pae_case("afdb_apobec3d", fresh=True, layout="auto", detail="sidechain")
    assert column.layout == "column" and "because the atoms" not in column.layout_rule


def test_a_figure_that_does_not_fit_at_a_level_is_narrowed_or_refused_as_any_figure_is():
    """No level has a threshold of its own. With every atom drawn the interface of the trimer does not
    fit a page whole: it narrows to a site, and says so. On a smaller page nothing of it fits, and a
    table stands in with the reason."""
    face, res = interface("atoms", "schematic")
    assert res.status == "ok" and not face.site.whole and len(face.site.residues) == 10
    assert [w for w in res.warnings if w.startswith("the interface could not be drawn legibly with more residues: "
                                                    "narrowed to 10 residues of chain B")]
    assert res.quality.min_font_pt >= 5.0 and res.quality.valid
    small = face.render(style="schematic", figure_width_mm=70.0, figure_height_mm=90.0)
    assert small.status == "unreadable" and small.figure == "summary"
    assert [w for w in small.warnings if "is not readable" in w and "a per-residue summary was drawn instead" in w]


# ---------------------------------------------------------------------------
# The same option on a ligand diagram, on a peptide ligand, and on the command line
# ---------------------------------------------------------------------------

#: R-bicalutamide in the androgen receptor: a halogen bond to the carbonyl of Met742, H-bonds and
#: hydrophobic contacts, and a figure at every level.
LIGAND = "ar_bicalutamide"


def ligand_diagram(detail=None, case=LIGAND):
    from conftest import diagram_at

    return diagram_at(case, detail)


def fragments(res) -> dict[str, tuple[str, list[str]]]:
    """Every residue beside the ligand: the form it is drawn in, and the atoms that are named on it."""
    from scenetools import groups, walk

    out = {}
    for g in groups(res, "residue"):
        names = sorted({f.item.content for f in walk(res.scene) if g in f.path and f.in_class("fragment") is not None
                        and hasattr(f.item, "content")})
        out[g.data["residue"].split(":")[1]] = (g.data["form"], names)
    return out


def test_the_residues_around_a_ligand_take_the_same_levels():
    """Bicalutamide in the androgen receptor, with a pLDDT for every atom made up for the test. The residues
    that make a contact with a direction (in the table: an H-bond, a halogen bond, a water bridge) are
    fragments, those that make hydrophobic contacts only stay eyelashes, at every level. A level adds atoms
    to the fragments."""
    from conftest import TESTDATA, predicted_diagram
    from bindview import load_interactions

    drawn = {level: predicted_diagram(LIGAND, detail=level).render(style="detailed", atom_labels="names")
             for level in (None, "box", "backbone", "sidechain", "atoms")}
    assert all(r.status == "ok" for r in drawn.values())
    forms = {level: {k: v[0] for k, v in fragments(r).items()} for level, r in drawn.items()}
    assert forms[None] == forms["sidechain"] == forms["backbone"] == forms["atoms"]
    kinds: dict[str, set] = {}
    for row in load_interactions(TESTDATA / LIGAND / "interactions.tsv"):
        kinds.setdefault(f"{row.residue_name}{row.residue_number}", set()).add(row.type.value)
    directional = {k for k, v in kinds.items() if v != {"hydrophobic"}}
    assert {k for k, form in forms[None].items() if form == "fragment"} == directional
    assert {k for k, form in forms[None].items() if form == "eyelash"} == set(kinds) - directional
    assert {k for k, form in forms["box"].items() if form == "chip"} == directional
    named = {level: {k: v[1] for k, v in fragments(r).items() if v[0] == "fragment"} for level, r in drawn.items()}
    # Met742 holds a halogen through its carbonyl oxygen: the carbonyl; then its side chain or its nitrogen with it.
    assert named[None]["MET742"] == ["O"] and named["sidechain"]["MET742"] == ["O", "SD"]
    assert named["backbone"]["MET742"] == ["N", "O"] and named["atoms"]["MET742"] == ["N", "O", "SD"]
    for level in ("backbone", "sidechain", "atoms"):
        assert all(set(named[None][k]) <= set(named[level][k]) for k in named[None])
    # The basis of what a drawn atom stands at: said wherever atoms of a residue are drawn, which is every
    # level but "box" (``plddt_by_atom``); with ``plddt_by_atom=False`` only where a level is asked for.
    for level, res in drawn.items():
        said = "The atoms that are drawn of a residue are each at their own pLDDT" in words(res)
        assert said == (level != "box") == ("plddt-basis" in res.svg)
        assert res.graph.meta.get("detail") == (None if level is None else {"ligand": level, "protein": level})
    one_look = predicted_diagram(LIGAND).render(style="detailed", plddt_by_atom=False)
    assert "plddt-basis" not in one_look.svg and "each at their own pLDDT" not in words(one_look)
    # One level for the protein alone, by name; and nothing else is a side of a diagram.
    assert ligand_diagram({"protein": "atoms"}).render(style="detailed").svg == ligand_diagram("atoms").render(
        style="detailed").svg
    with pytest.raises(ValueError, match="detail is a level, or one for each of 'ligand' and 'protein'; not for 'pocket'"):
        ligand_diagram({"pocket": "atoms"})
    with pytest.raises(ValueError, match="not 'full'"):
        ligand_diagram("full")


def test_a_peptide_ligand_drawn_as_a_chain_is_the_column_of_an_interface_in_this():
    """``Peptide.detail`` is the one field both read: hirudin's peptide on thrombin, a ligand drawn as a
    chain. The atoms beside a box are named as the file names them, number and atom; those of a peptide
    written as one residue (the nonamer of ``mhc_peptide``) keep the ligand's own names."""
    from conftest import chain_diagram, one_residue_peptide
    from bindview import Diagram
    from bindview.distance_table import ligand_contacts

    plain, level = chain_diagram("thrombin_hirudin"), chain_diagram("thrombin_hirudin", detail={"ligand": "sidechain"})
    assert plain.backend.kind == level.backend.kind == "chain"
    assert plain.backend.peptide.detail is None and level.backend.peptide.detail == "sidechain"
    res, ref = level.render(style="schematic"), plain.render(style="schematic")
    assert ref.status == "ok" and not any("beside" in b.data for b in boxes(ref))
    with_atoms = [b for b in boxes(res) if "beside" in b.data]
    assert res.status == ref.status and with_atoms and len(boxes(res)) == len(boxes(ref))
    drawn = {n for b in with_atoms for n in b.data["beside"].split(",")}
    directional = [g for g in lines(res) if g.data["type"] != "hydrophobic"]
    assert directional and all("ligand-residue" not in g.data for g in directional
                               if set(g.data["ligand-atoms"].split(",")) <= drawn)
    assert any(set(g.data["ligand-atoms"].split(",")) <= drawn for g in directional)
    assert res.graph.meta["detail"] == {"ligand": "sidechain", "protein": None}
    numbers = {r.shown for r in level.backend.peptide.residues}
    assert all(n.split("/")[0] in numbers for n in drawn)                  # number/atom, as the file has them
    st, smiles = one_residue_peptide("mhc_peptide")
    perceived = Diagram.from_structure(st, "LIG", smiles=smiles, interactions=ligand_contacts(st, "LIG", smiles),
                                       detail={"ligand": "sidechain"}).render(style="schematic")
    own = {n for b in boxes(perceived) if "beside" in b.data for n in b.data["beside"].split(",")}
    names = set(perceived_names(st))
    assert perceived.status == "ok" and own and own <= names and not any("/" in n for n in own)


def perceived_names(structure) -> list[str]:
    """The atom names of the peptide written as one residue (chain Z)."""
    return [a.name for a in structure.atoms if a.chain == "Z"]


def test_command_line_takes_a_level(tmp_path, capsys):
    import json

    from conftest import TESTDATA, meta_of, trimer_contacts, trimer_file
    from bindview.cli import main as cli
    from bindview.interface import dump_interface_contacts

    tsv, out = tmp_path / "contacts.tsv", tmp_path / "out.svg"
    tsv.write_text(dump_interface_contacts(trimer_contacts("A", "B")), encoding="utf-8")
    base = ["interface", str(trimer_file("model")), "--between", "A", "B", "--contacts", str(tsv), "-o", str(out),
            "-q", "--style", "schematic", "--pae", str(trimer_file("pae")), "--pae", str(trimer_file("pae", 1))]
    assert cli(base + ["--detail", "sidechain", "--json"]) == 0
    said = capsys.readouterr().out
    assert '"chain B": "sidechain"' in said and out.read_text(encoding="utf-8") == interface("sidechain", "schematic")[1].svg
    assert cli(base) == 0 and out.read_text(encoding="utf-8") == interface(None, "schematic")[1].svg
    with pytest.raises(SystemExit):
        cli(base + ["--detail", "all"])
    capsys.readouterr()
    m = meta_of(LIGAND)
    graph = tmp_path / "graph.json"
    args = ["render", str(TESTDATA / LIGAND / "model.cif.gz"), "--ligand", m["ligand"]["residue_name_in_cif"],
            "--smiles", m["ligand"]["smiles"], "--interactions", str(TESTDATA / LIGAND / "interactions.tsv"),
            "-o", str(out), "-q", "--style", "detailed", "--graph", str(graph)]
    assert cli(args + ["--detail", "atoms"]) == 0
    assert json.loads(graph.read_text(encoding="utf-8"))["meta"]["detail"] == {"ligand": "atoms", "protein": "atoms"}


# ---------------------------------------------------------------------------
# An antibody on its antigen
# ---------------------------------------------------------------------------

def residues_with_atoms(res) -> list[str]:
    return [b.title.split()[0] + b.title.split()[1] for b in boxes(res) if "beside" in b.data]


def test_the_heavy_chain_of_an_antibody_on_lysozyme_at_a_level():
    """HyHEL-10's heavy chain on lysozyme (``fab_lysozyme``, 3D9A), whose residues at the interface come
    from three loops. At ``"sidechain"`` the column is drawn whole; its residues that make an H-bond or a
    salt bridge are drawn with atoms, and Trp398, with hydrophobic contacts only, stays a box. The column
    is 1.5 to 2.2 times as long as the column of boxes (a regression value of the layout). A crystal has no
    pLDDT: atoms are drawn, and nothing is said of what they are drawn at."""
    from conftest import structure_of
    from bindview.distance_table import interface_contacts

    contacts = interface_contacts(structure_of("fab_lysozyme"), ["H"], ["C"])
    face = antibody("sidechain")
    res, ref = face.render(style="detailed"), antibody(None).render(style="detailed")
    assert res.status == ref.status == "ok" and face.site.whole
    assert len(face.site.residues) == len({c.number_a for c in contacts})
    assert res.quality.valid and res.quality.min_font_pt >= 5.0 and res.quality.bond_crossings == 0
    directional = {f"{c.resname_a.title()}{c.number_a}" for c in contacts if c.type.value != "hydrophobic"}
    assert set(residues_with_atoms(res)) == directional and len(residues_with_atoms(res)) == len(directional)
    assert [b.title.split()[0] + b.title.split()[1] for b in boxes(res) if "beside" not in b.data] == ["Trp398"]

    def column(r) -> float:
        tops = [item.y for b in boxes(r) for item in b.items if hasattr(item, "h")]
        return max(tops) - min(tops)

    assert 1.5 < column(res) / column(ref) < 2.2
    # Lines end on the two oxygens of Asp701's carboxylate: one for each H-bond of the table, and none runs
    # through the name of the oxygen it does not end on, nor, with the backbone drawn, through the residue's.
    asp = sum(1 for c in contacts if c.type.value == "hbond" and (c.chain_b, c.number_b) == ("C", 701))
    assert len([g for g in lines(res) if g.data["residue"] == "C:ASP701" and g.data["type"] == "hbond"]) == asp > 1
    for figure in (res, antibody("backbone").render(style="detailed")):
        found = ink_on_ink(figure)
        assert figure.status == "ok" and figure.quality.link_hits == 0
        assert (found["names_struck"], found["atom_names_crossed"], found["labels_crossed"]) == ([], [], [])
    assert res.confidence is None and "plddt-basis" not in res.svg and "pLDDT" not in words(res)
    # Left to the rule, these three loops stand on a plane. With their atoms asked for they are a column,
    # and the figure says which of the two won and why.
    assert antibody(None, "auto").layout == "plane"
    auto = antibody("sidechain", "auto")
    assert auto.layout == "column" and auto.layout_rule.startswith("a column, because the atoms of its residues")
    assert "the rule alone would draw these residues on the plane through them" in auto.layout_rule
    assert labels_of_lines(res) and on_each_other(labels_of_lines(res)) == []


# ---------------------------------------------------------------------------
# The sizes of the trimer at each level, and what a level adds to a graph
# ---------------------------------------------------------------------------

def differ(st) -> tuple[int, int]:
    """In how many residues of a structure two heavy atoms have values of their own, and of how many."""
    from bindview.confidence import gives_atoms

    groups = [[a.b_iso for a in st.residue_atoms(key) if a.b_iso is not None and a.element not in ("H", "D")]
              for key in st.residues()]
    return sum(gives_atoms([g]) for g in groups), len(groups)


def test_the_sizes_of_the_trimer_at_each_level():
    """The A-B interface of the trimer as boxes, with side chains and with every atom. 17 residues in the
    column, 13 of them with atoms; 612 by 1003 units as boxes and 690 by 1172 with side chains, 1.17 times as
    tall; the smallest text 6.4 and 5.5 pt; with every atom the figure narrows to 10 of the 17. The sizes and
    the points are regression values of the layout."""
    (_, a), (face, b), (most, c) = (interface(d, "detailed") for d in (None, "sidechain", "atoms"))
    assert a.status == b.status == c.status == "ok" and face.site.whole and not most.site.whole
    assert all(r.quality.valid and r.quality.readable for r in (a, b, c))
    assert (len(face.site.residues), sum(1 for box in boxes(b) if "beside" in box.data)) == (17, 13)
    assert [f"{r.quality.width:.0f} × {r.quality.height:.0f}" for r in (a, b)] == ["612 × 1003", "690 × 1172"]
    assert [f"{r.quality.min_font_pt:.1f}" for r in (a, b)] == ["6.4", "5.5"]
    assert f"{b.quality.height / a.quality.height:.2f}" == "1.17"
    assert len(most.site.residues) == 10


def test_which_models_give_every_atom_a_plddt_of_its_own():
    """Whether a model gives every atom a pLDDT is counted, not assumed: in how many residues two heavy
    atoms differ by more than the 0.01 that a file's rounding leaves. The AlphaFold database's models give
    one value for a residue (0 of their 329, 386 and 419 residues), and so does the Boltz trimer (0 of 471);
    a pLDDT made up from a crystal's temperature factors differs within residues."""
    from conftest import TESTDATA, invented_plddt, structure_of, trimer_structure
    from bindview.confidence import ATOM_SPREAD
    from bindview.structure import read_structure

    assert ATOM_SPREAD == 0.01
    afdb = [differ(read_structure(TESTDATA / "interface_pae" / slug / "model.cif.gz"))
            for slug in ("afdb_tectb", "afdb_apobec3d", "afdb_fah")]
    assert afdb == [(0, 329), (0, 386), (0, 419)]
    assert {differ(trimer_structure(draw)) for draw in (0, 1)} == {(0, 471)}
    assert differ(invented_plddt(structure_of("fab_lysozyme")))[0] > 0


def test_what_a_level_adds_to_a_graph():
    """A figure at a level writes three things into a graph that no other figure does: ``beside`` on a
    residue of the column, ``detail`` and ``plddt_basis``. A graph that no level was asked for has none of
    them, and the version of a document says that a level adds only optional keys."""
    from conftest import predicted_diagram

    at_a_level = interface("sidechain", "detailed")[1].graph
    around = predicted_diagram(LIGAND, detail="atoms").render(style="detailed").graph
    assert sum("beside" in node for node in at_a_level.to_dict()["nodes"]) == 13
    assert at_a_level.meta["interface"]["detail"] == {"chain A": "sidechain", "chain B": "sidechain"}
    assert at_a_level.meta["plddt_basis"] == "residue" and "detail" not in at_a_level.meta
    assert around.meta["detail"] == {"ligand": "atoms", "protein": "atoms"} and around.meta["plddt_basis"] == "atom"
    plain = interface(None, "detailed")[1].graph.to_dict()
    assert "plddt_basis" not in plain["meta"] and "detail" not in plain["meta"]["interface"]
    assert not any("beside" in node for node in plain["nodes"])
    assert at_a_level.to_dict()["version"] == plain["version"] == "1.2"


def test_a_line_through_a_tyrosines_ring_to_its_hydroxyl_is_in_the_way():
    """The ink of a residue's atoms is measured atom by atom and half a bond at a time. So for each tyrosine
    drawn with atoms in the trimer's column, a line to its hydroxyl that comes from inside the ring runs
    through the ring's own ink and is in the way; one that comes from outside is not."""
    from bindview.layout.field import LigandField

    owner, res = interface("sidechain", "detailed")
    dep = owner.diagram.backend.depict_from(Metrics().with_(bond_length=res.graph.bond_length), res.graph)
    field = LigandField(dep, reach=60.0)
    found = []
    for tyr in (u for u in dep.units if u.name == "TYR" and u.beside):
        oh = next(i for i in tyr.beside if dep.atoms[i].name.endswith("/OH"))
        ring = [i for i in tyr.beside if dep.atoms[i].name.split("/")[1] in ("CG", "CD1", "CD2", "CE1", "CE2", "CZ")]
        centre, tip = np.array(dep.centroid(ring)), np.array(dep.pos(oh))
        along = (tip - centre) / np.linalg.norm(tip - centre)
        inside, outside = centre - along * 3.0, tip + along * 40.0
        found.append((bool(field.boxes_crossed(inside[None, :], tip[None, :], tip, reach=8.0)[0]),
                      bool(field.boxes_crossed(outside[None, :], tip[None, :], tip, reach=8.0)[0])))
    tyrosines = sum(1 for b in boxes(res) if "beside" in b.data and b.title.startswith("Tyr "))
    assert found and found == [(True, False)] * tyrosines


# ---------------------------------------------------------------------------
# What the audit does not count: two labels on top of each other
# ---------------------------------------------------------------------------

def labels_of_lines(res) -> list[tuple[int, str, tuple[float, float, float, float]]]:
    """Every label on a contact line (a distance, a PAE tag, a mark): which line it is on, what it says,
    and the box of its text, as wide as the library's own text metrics make it."""
    from bindview.scene import Text
    from bindview.textmetrics import CAP_HEIGHT, text_width
    from scenetools import inside

    found = []
    for k, group in enumerate(lines(res)):
        for f in inside(res, group, Text):
            t = f.item
            w, h = text_width(t.content, t.size, t.bold), CAP_HEIGHT * t.size
            x0 = t.x - w / 2.0 if t.anchor == "middle" else t.x if t.anchor == "start" else t.x - w
            found.append((k, t.content, (x0, t.y - h, x0 + w, t.y)))
    return found


def on_each_other(found) -> list[tuple[str, str]]:
    """Pairs of labels of two different lines whose boxes overlap by more than half a unit both ways."""
    return [(a[1], b[1]) for i, a in enumerate(found) for b in found[i + 1:] if a[0] != b[0]
            and min(a[2][2], b[2][2]) - max(a[2][0], b[2][0]) > 0.5
            and min(a[2][3], b[2][3]) - max(a[2][1], b[2][1]) > 0.5]


def test_no_two_labels_of_different_lines_stand_on_each_other_at_a_level():
    """Lines that end on atoms run closer together than lines that end on boxes, and each has its
    distance. The audit of a figure has no count of two labels overlapping, so they are counted here, on
    the figures at a level: none. The count is tried first on labels that are put on each other."""
    from conftest import pae_case

    assert on_each_other([(0, "2.9", (0.0, 0.0, 12.0, 6.0)), (1, "3.1", (5.0, 2.0, 17.0, 8.0))]) == [("2.9", "3.1")]
    assert on_each_other([(0, "2.9", (0.0, 0.0, 12.0, 6.0)), (0, "PAE 7", (5.0, 2.0, 17.0, 8.0))]) == []   # of one line
    assert on_each_other([(0, "2.9", (0.0, 0.0, 12.0, 6.0)), (1, "3.1", (12.2, 0.0, 24.0, 6.0))]) == []    # side by side
    figures = [interface(level, style)[1] for level in ("sidechain", "atoms") for style in ("detailed", "schematic")]
    figures += [pae_case(slug, fresh=True, detail=level).render(style="detailed")
                for slug, level in (("afdb_tectb", "sidechain"), ("afdb_apobec3d", "backbone"))]
    figures.append(ligand_diagram("atoms").render(style="detailed"))
    for res in figures:
        found = labels_of_lines(res)
        assert res.status == "ok" and found and on_each_other(found) == []
    assert len(labels_of_lines(figures[0])) == 21                # the distances and the PAE tags of the trimer


# ---------------------------------------------------------------------------
# A residue's lines and its own names
# ---------------------------------------------------------------------------

def ink_on_ink(res, shrink: float = 0.75) -> dict[str, list]:
    """Ink on ink between a residue's own lines and its own text, counted on the scene of a figure.

    ``names_struck``: a line through the name of the residue it comes from (the name is the largest text
    of the residue's group, bold or not). ``atom_names_crossed``: a line through the name of an atom of a
    residue beside the column or the ligand that it does not end on. ``labels_crossed``: a line through
    the label of another line. Text boxes are as wide as the library's own text metrics make them, and
    are made smaller by ``shrink`` on every side, so that touching does not count."""
    from bindview.scene import Line, Text
    from bindview.textmetrics import CAP_HEIGHT, text_width
    from scenetools import groups, inside

    def box_of(t) -> tuple[float, float, float, float]:
        w, h = text_width(t.content, t.size, t.bold), CAP_HEIGHT * t.size
        x0 = t.x - w / 2.0 if t.anchor == "middle" else t.x if t.anchor == "start" else t.x - w
        return (x0, t.y - h, x0 + w, t.y)

    def through(p, q, box) -> bool:
        x0, y0, x1, y1 = box[0] + shrink, box[1] + shrink, box[2] - shrink, box[3] - shrink
        t0, t1 = 0.0, 1.0
        for start, d, lo, hi in ((p[0], q[0] - p[0], x0, x1), (p[1], q[1] - p[1], y0, y1)):
            if abs(d) < 1e-12:
                if start < lo or start > hi:
                    return False
                continue
            ta, tb = sorted(((lo - start) / d, (hi - start) / d))
            t0, t1 = max(t0, ta), min(t1, tb)
        return x1 > x0 and y1 > y0 and t0 < t1

    names: dict = {}
    atoms = []
    for g in groups(res, "residue"):
        texts = [f.item for f in inside(res, g, Text)]
        for t in texts:
            if t.size == max(u.size for u in texts):
                names[g.data.get("residue")] = (t.content, box_of(t))
            else:
                atoms.append((g.data.get("residue"), t.content, box_of(t)))
    drawn_lines, labels = [], []
    for k, g in enumerate(lines(res)):
        ends = (g.data.get("residue"), set((g.data.get("protein-atoms") or "").split(",")), g.data.get("type"))
        drawn_lines += [(k, ends, (f.item.x1, f.item.y1), (f.item.x2, f.item.y2)) for f in inside(res, g, Line)]
        labels += [(k, f.item.content, box_of(f.item)) for f in inside(res, g, Text)]
    return {
        "names": sorted(v[0] for v in names.values()),
        "names_struck": sorted({(names[e[0]][0], e[2]) for _, e, p, q in drawn_lines
                                if e[0] in names and through(p, q, names[e[0]][1])}),
        "atom_names_crossed": sorted({(resid, name) for resid, name, box in atoms for _, e, p, q in drawn_lines
                                      if not (e[0] == resid and name in e[1]) and through(p, q, box)}),
        "labels_crossed": sorted({text for k, text, box in labels for k2, _, p, q in drawn_lines
                                  if k2 != k and through(p, q, box)}),
    }


def test_the_rule_about_the_names_of_a_lines_own_residue_on_a_body_made_by_hand():
    """What is forbidden for a residue that is drawn at a level, without a figure: a line through the name
    of an atom of its own residue that it does not end on, a line that ends on an atom through the
    residue's name, and a label on another line of the same residue. And what is done with a line that
    leaves from where the name stands: it ends on the name."""
    from bindview.layout.body import Body, LabelSpec, Port
    from bindview.layout.engine import _labels_on_siblings, _through_names
    from bindview.styles.detailed import _NAME_GAP, _from_the_name

    def body_of(*ports) -> "Body":
        return Body(key="r", kind="fragment", shapes_p0=np.zeros((0, 2)), shapes_p1=np.zeros((0, 2)),
                    shapes_r=np.zeros(0), ports=list(ports),
                    label=LabelSpec(anchor=(0.0, 0.0), direction=(0.0, 1.0), width=30.0, height=8.0),
                    upright_pts=np.array([(0.0, 0.0), (40.0, 0.0)]), upright_wh=np.array([(10.0, 6.0), (12.0, 6.0)]))

    here = (np.zeros((1, 2)), np.zeros(1), np.zeros(1, dtype=bool))
    to_an_atom = Port(local=(0.0, 0.0), target=(100.0, 0.0), ideal=50.0, clear_of=(1,), clear_of_name=True)
    of_the_residue = Port(local=(0.0, 0.0), target=(100.0, 0.0), ideal=50.0, clear_of=(0, 1))
    plain = Port(local=(0.0, 0.0), target=(100.0, 0.0), ideal=50.0)
    assert (plain.clear_of, plain.clear_of_name) == ((), False)                  # what every other body has
    body = body_of(to_an_atom)

    def through(port, a, b) -> bool:
        return bool(_through_names(body, port, np.array([a], dtype=float), np.array([b], dtype=float), *here)[0])

    # The name of the other atom stands at (40, 0), the residue's own name 7 below its alpha carbon.
    assert through(to_an_atom, (8.0, 0.0), (100.0, 0.0)) and not through(to_an_atom, (8.0, 0.0), (100.0, -45.0))
    assert through(to_an_atom, (-30.0, 7.0), (30.0, 7.0)) and not through(to_an_atom, (-30.0, 20.0), (30.0, 20.0))
    # A line of the residue as a whole keeps off every atom name, and may leave through the residue's.
    assert through(of_the_residue, (8.0, 0.0), (100.0, 0.0)) and through(of_the_residue, (-20.0, -1.0), (-6.0, -1.0))
    assert not through(of_the_residue, (-30.0, 7.0), (30.0, 7.0))
    assert not through(plain, (8.0, 0.0), (100.0, 0.0)) and not through(plain, (-30.0, 7.0), (30.0, 7.0))

    # Two lines of one residue, each with a label 12 by 6 at its middle: one on the other's line, or not.
    first = Port(local=(0.0, 0.0), target=(100.0, 0.0), ideal=50.0, label_size=(12.0, 6.0), clear_of_name=True)
    second = Port(local=(0.0, 0.0), target=(100.0, 0.0), ideal=50.0, label_size=(12.0, 6.0), clear_of_name=True)
    pair = body_of(first, second)
    middle = np.full((1, 2), 0.5)
    a = np.array([[(0.0, 0.0), (50.0, -40.0)]])
    crossing = np.array([[(100.0, 0.0), (50.0, 40.0)]])            # the second runs through the middle of the first
    beside = np.array([[(100.0, 0.0), (90.0, -40.0)]])
    assert _labels_on_siblings(pair, a, crossing, middle) == 2      # each label stands on the other's line
    assert _labels_on_siblings(pair, a, beside, middle) == 0

    # A line from the alpha carbon that would run through the residue's name starts beyond the name.
    name = (10.0, -4.0, 50.0, 4.0)
    start, at = _from_the_name((0.0, 0.0), (100.0, 0.0), 0.8, name)
    assert start == pytest.approx((50.0 + _NAME_GAP, 0.0))
    assert start[0] + at * (100.0 - start[0]) == pytest.approx(80.0)              # its label stays where it was
    assert _from_the_name((0.0, 0.0), (100.0, 0.0), 0.3, name)[1] == 0.0          # a label on the part that is gone
    assert _from_the_name((0.0, 20.0), (100.0, 20.0), 0.5, name) == ((0.0, 20.0), 0.5)      # past the name
    assert _from_the_name((0.0, 0.0), (30.0, 0.0), 0.5, name) == ((0.0, 0.0), 0.5)          # ends in it
    assert _from_the_name((0.0, 5.5), (100.0, 5.5), 0.5, name) == ((0.0, 5.5), 0.5)         # past the text, in its margin


def test_at_a_level_no_line_runs_through_a_name_or_a_label_of_its_own_residue():
    """A residue at a level has atoms that none of its lines ends on, and its lines come from the atoms
    beside the column, from more than one side. None of them runs through the name of its residue, through
    the name of another atom of its residue or through the label of another line, and the audit counts what
    the layout forbids. A line of the residue as a whole starts at its name."""
    from conftest import pae_case, predicted_diagram
    from bindview.styles.detailed import _NAME_GAP
    from scenetools import groups

    figures = {"the trimer with side chains": interface("sidechain", "detailed")[1],
               "the trimer with every atom": interface("atoms", "detailed")[1],
               "TECTB with side chains": pae_case("afdb_tectb", fresh=True, detail="sidechain").render(style="detailed"),
               "APOBEC3D with its backbone": pae_case("afdb_apobec3d", fresh=True, detail="backbone").render(style="detailed"),
               "bicalutamide with every atom": predicted_diagram(LIGAND, detail="atoms").render(style="detailed")}
    for what, res in figures.items():
        found = ink_on_ink(res)
        assert res.status == "ok" and res.quality.valid and res.quality.link_hits == 0, what
        assert found["names"] and len(found["names"]) == len(groups(res, "residue")), what
        assert (found["names_struck"], found["atom_names_crossed"], found["labels_crossed"]) == ([], [], []), what
    # The three contacts of Val123 with the residue as a whole: each line starts at the name, not beyond it.
    from bindview.scene import Line, Text
    from scenetools import inside

    res = figures["the trimer with side chains"]
    (val,) = [g for g in groups(res, "residue") if g.data["residue"] == "A:VAL123"]
    name = max((f.item for f in inside(res, val, Text)), key=lambda t: t.size)
    assert name.content == "Val123"
    whole = [g for g in lines(res) if g.data["residue"] == "A:VAL123" and g.data["type"] == "hydrophobic"]
    assert len(whole) == 3
    for g in whole:
        start = min(((f.item.x1, f.item.y1) for f in inside(res, g, Line)), key=lambda p: math.hypot(p[0] - name.x, p[1] - name.y))
        assert math.hypot(start[0] - name.x, start[1] - name.y) < 40.0             # at the name: it is 37 wide
        assert abs(start[1] - name.y) <= 12.0 + _NAME_GAP


def test_a_nucleotide_of_a_column_stands_with_its_base_beside_and_its_phosphate_above():
    """The box of a nucleotide is its C4'; the base, and what is drawn of the sugar but C4' and O3', stand to
    the side; C5', O5' and P above the box on the column's line, the phosphate's two free oxygens beside P;
    O3' below. A level on a chain of RNA draws."""
    from conftest import TESTDATA
    from bindview import Interface, read_structure
    from bindview.distance_table import interface_contacts

    model = read_structure(TESTDATA / "nucleic" / "riboswitch_bisguanidine" / "model.pdb")
    face = Interface(model, ["A", "B"], interface_contacts(model, ["A"], ["B"], stacking=True), detail="sidechain")
    pep = face.diagram.backend.peptide
    g10 = next(r for r in pep.residues if r.number == "10")
    assert g10.kind == "nucleotide"
    names = {i: n for i, n in g10.atom_names}
    index = {n: i for i, n in names.items()}
    m, half_w, half_h = Metrics(), 16.0, 8.0
    length = m.bond_length * m.protein_bond_scale

    # A contact of the base: the base with C1', hung on the box by C1', on the side asked for.
    (base,) = shown_atoms(pep, {g10.position: {index["N1"]}}).values()
    drawn = {names[i] for i in base}
    assert "C1'" in drawn and "N9" in drawn and "C4'" not in drawn and "P" not in drawn
    for side in (1, -1):
        piece = lay_piece(g10, base, side, half_w, half_h, m)
        assert all(side * x > half_w for x, _ in piece.xy.values())
        assert [names[i] for _, i in piece.stems] == ["C1'"]
        assert piece.stems[0][0] == (side * half_w, 0.0)

    # A contact of the phosphate: the phosphate with O5' and C5', above the box, where the chain comes in.
    (phosphate,) = shown_atoms(pep, {g10.position: {index[next(n for n in ("OP1", "O1P") if n in index)]}}).values()
    piece = lay_piece(g10, phosphate, 1, half_w, half_h, m)
    at = {names[i]: piece.xy[i] for i in phosphate}
    canonical = {"O1P": "OP1", "O2P": "OP2"}
    at = {canonical.get(n, n): xy for n, xy in at.items()}
    assert at["C5'"] == pytest.approx((0.0, -half_h - length))
    assert at["O5'"] == pytest.approx((0.0, -half_h - 2 * length))
    assert at["P"] == pytest.approx((0.0, -half_h - 3 * length))
    assert at["OP1"][0] < 0.0 < at["OP2"][0] and at["OP1"][1] < at["P"][1] and at["OP2"][1] < at["P"][1]
    # (At "sidechain" the level adds the base, and the sugar between base and phosphate comes with them.)
    stems = {names[i]: start for start, i in piece.stems}
    assert stems["C5'"] == (0.0, -half_h) and stems["C3'"] == stems["O4'"] == (half_w, 0.0)

    # At "atoms", the whole nucleotide: the ring closes on the box (two stems, from C3' and O4').
    face_all = Interface(model, ["A", "B"], interface_contacts(model, ["A"], ["B"], stacking=True), detail="atoms")
    pep_all = face_all.diagram.backend.peptide
    g10_all = next(r for r in pep_all.residues if r.number == "10")
    (every,) = shown_atoms(pep_all, {g10_all.position: {index["N1"]}}).values()
    piece = lay_piece(g10_all, every, -1, half_w, half_h, m)
    assert sorted(names[i] for _, i in piece.stems) == ["C3'", "C5'", "O3'", "O4'"]
    assert piece.xy[index["O3'"]] == pytest.approx((0.0, half_h + length))

    # The figure is drawn, readable or not: no level on a chain of nucleotides fails.
    figure = face.render(on_unreadable="render")
    assert figure.svg.count("data-beside") >= 4
