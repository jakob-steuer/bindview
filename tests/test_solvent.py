"""Predicted solvent sites: the geometry, the guard, and above all the labelling.

A figure with a predicted water that could be mistaken for one showing an
observed water would be the worst thing this tool could produce. So most of
this file is about what a predicted site must never be: on by default, drawn
like an observed water, shown where real waters exist, or counted as a contact.
"""

from __future__ import annotations

import gzip
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from conftest import TESTDATA, fetched, meta_of, model_of, structure_of, without_waters
from bindview import Diagram, Graph, Interaction, load_interactions
from bindview.errors import LigandNotFoundError
from bindview.cli import main as cli
from bindview.scene import Line, Rect, Text
from bindview.solvent import (VALIDATION, SolventParams, _open_sides, candidate_points, has_waters,
                            predict_solvent_sites, water_oxygens)
from bindview.structure import Atom, ResidueKey, Structure
from bindview.styles.base import LINK_STYLES, PREDICTED_LABEL, predicted_style
from bindview.interactions import InteractionType
from scenetools import groups, inside, walk

T = InteractionType
#: R1881 in the androgen receptor: one site where a water would bridge its keto oxygen and Met745.
CASE = "ar_r1881"


def dry_table() -> list:
    """The case's table without its water bridges, whose waters a model without waters does not have."""
    return [r for r in load_interactions(TESTDATA / CASE / "interactions.tsv") if r.type is not T.WATER_BRIDGE]


@lru_cache(maxsize=1)
def dry() -> Diagram:
    """The case as a predicted model holds it: the crystal without its waters, and the table without the water
    bridges. Built once: do not change it."""
    ligand = meta_of(CASE)["ligand"]
    return Diagram.from_structure(without_waters(structure_of(CASE)), ligand["residue_name_in_cif"], ligand["smiles"],
                                  dry_table(), name=ligand["name"])


def words(res) -> str:
    return " ".join(f.item.content for f in walk(res.scene) if isinstance(f.item, Text))


def toy(cage: bool = True, water: bool = False) -> tuple[Structure, ResidueKey]:
    """A ligand oxygen and a serine oxygen 5.6 A apart, facing each other, in a tube of carbon."""
    rows = [("O1", "O", "LIG", "L", 1, (0.0, 0.0, 0.0), True), ("C1", "C", "LIG", "L", 1, (-1.23, 0.0, 0.0), True),
            ("OG", "O", "SER", "A", 10, (5.6, 0.0, 0.0), False), ("CB", "C", "SER", "A", 10, (6.83, 0.0, 0.0), False)]
    if cage:
        n = 0
        for x in (-1.0, 0.5, 2.0, 3.5, 5.0, 6.5):
            for k in range(10):
                ang = 2.0 * math.pi * k / 10.0
                n += 1
                rows.append((f"C{n}", "C", "LEU", "A", 100 + n, (x, 4.3 * math.cos(ang), 4.3 * math.sin(ang)), False))
    if water:
        rows.append(("O", "O", "HOH", "W", 1, (2.8, 0.0, 0.0), True))
    atoms = [Atom(index=i, name=nm, element=el, resname=res, chain=ch, resnum=num, icode="", x=p[0], y=p[1], z=p[2],
                  hetero=het) for i, (nm, el, res, ch, num, p, het) in enumerate(rows)]
    return Structure(atoms=atoms, source="toy"), ResidueKey("L", 1)


# ---------------------------------------------------------------------------
# The geometry
# ---------------------------------------------------------------------------

def test_a_site_is_found_between_two_facing_oxygens_in_an_enclosed_gap():
    st, key = toy()
    (site,) = predict_solvent_sites(st, key)
    assert math.dist(site.xyz, (2.8, 0.0, 0.0)) < 0.5 and site.bridging and site.score == 2
    assert site.enclosure >= SolventParams().min_enclosure
    ((chain, number, icode, resname, atom, d_pro),) = site.protein
    ((lig_atom, d_lig),) = site.ligand
    assert (chain, number, resname, atom, lig_atom) == ("A", 10, "SER", "OG", "O1")
    assert 2.6 - 1e-6 <= d_pro <= 3.3 and 2.6 - 1e-6 <= d_lig <= 3.3   # hydrogen-bond distance to both
    assert predict_solvent_sites(st, key) == [site]                    # deterministic
    assert site.as_dict()["predicted"] is True
    # The same two oxygens in open solvent: bulk water is not a site.
    bare, key2 = toy(cage=False)
    assert predict_solvent_sites(bare, key2) == []
    # A ligand with nothing polar has nothing to bridge from.
    apolar = Structure(atoms=[a if a.name != "O1" else Atom(a.index, "C9", "C", a.resname, a.chain, a.resnum, "",
                                                            a.x, a.y, a.z, a.hetero) for a in st.atoms], source="x")
    assert predict_solvent_sites(apolar, key) == []


def test_partners_must_point_at_the_site():
    xyz = np.array([[0.0, 0, 0], [-1.23, 0, 0], [5.6, 0, 0], [6.83, 0, 0]])
    side, bonded = _open_sides(xyz, np.array([0, 2]))
    assert list(bonded) == [1, 1]
    assert side[0] == pytest.approx([1.0, 0.0, 0.0]) and side[1] == pytest.approx([-1.0, 0.0, 0.0])
    # Turn the serine round, so that its oxygen's open side faces away: the gap has one partner left.
    st, key = toy()
    turned = Structure(atoms=[a if a.name != "CB" else Atom(a.index, "CB", "C", a.resname, a.chain, a.resnum, "",
                                                            4.9, 1.0, 0.0, a.hetero) for a in st.atoms], source="t")
    assert not [s for s in predict_solvent_sites(turned, key) if s.bridging and s.score >= 2]


def test_waters_in_the_structure_are_never_predicted_over():
    wet, key = toy(water=True)
    assert has_waters(wet) and water_oxygens(wet).shape == (1, 3)
    # The predictor itself ignores waters as atoms (that is what validation needs) ...
    (site,) = predict_solvent_sites(wet, key)
    assert math.dist(site.xyz, tuple(water_oxygens(wet)[0])) < 0.5      # ... and here it lands on the one held out
    bare, _ = toy()
    assert not has_waters(bare) and not has_waters(dry().structure) and has_waters(structure_of(CASE))


# ---------------------------------------------------------------------------
# Off unless asked for
# ---------------------------------------------------------------------------

def test_nothing_is_predicted_unless_asked():
    res = dry().render()
    assert "solvent_sites" not in res.data and not res.graph.of_kind("solvent_site")
    assert "pred." not in words(res) and "Predicted" not in words(res)
    assert not [g for g in groups(res, "interaction") if g.data.get("predicted")]
    assert not [w for w in res.warnings if "water" in w]


# ---------------------------------------------------------------------------
# Marked as predicted wherever it appears
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("style", ("detailed", "schematic"))
def test_a_predicted_site_cannot_be_mistaken_for_an_observed_water(style):
    dia = dry()
    res = dia.render(style=style, solvent_sites=True)
    assert res.status == "ok" and res.quality.valid
    (bridge,) = [g for g in groups(res, "interaction") if g.data.get("predicted")]
    assert bridge.data["type"] == "water_bridge" and bridge.data["residue"] == "A:MET745"
    assert bridge.data["ligand-atoms"] == "O83" and bridge.title.startswith("Predicted water site, not observed")
    # On the figure: the word, on the line; a glow under it, soft from the outside in (or a filled box with a
    # broken border round it, ``solvent_mark="filled"``, or the border alone, ``"outline"``); a dotted line of its own kind.
    labels = [f.item.content for f in inside(res, bridge, Text)]
    assert labels == [PREDICTED_LABEL] and PREDICTED_LABEL.startswith("pred.")
    glow = [f.item for f in inside(res, bridge, Rect)]
    assert len(glow) == 4 and all(r.paint.fill is not None and r.paint.stroke is None for r in glow)
    assert [r.w for r in glow] == sorted((r.w for r in glow), reverse=True)
    assert [r.paint.opacity for r in glow] == sorted(r.paint.opacity for r in glow)
    boxed = dia.render(style=style, solvent_sites=True, solvent_mark="filled")
    (was,) = [g for g in groups(boxed, "interaction") if g.data.get("predicted")]
    (box, border) = [f.item for f in inside(boxed, was, Rect)]
    assert box.paint.fill is not None and box.paint.stroke is None
    assert border.paint.dash and border.paint.fill is None
    assert (box.x, box.y, box.w, box.h) == (border.x, border.y, border.w, border.h)
    assert (glow[-1].x, glow[-1].y, glow[-1].w, glow[-1].h) == (box.x, box.y, box.w, box.h)      # in the same place
    thin = dia.render(style=style, solvent_sites=True, solvent_mark="outline")
    (same,) = [g for g in groups(thin, "interaction") if g.data.get("predicted")]
    (outline,) = [f.item for f in inside(thin, same, Rect)]
    assert outline.paint.dash and outline.paint.fill is None and outline.paint.width < border.paint.width
    lines = [f.item.paint for f in inside(res, bridge, Line)]
    observed = LINK_STYLES[T.WATER_BRIDGE]
    assert lines and all(p.dash == predicted_style(observed).dash != observed.dash for p in lines)
    assert all(p.cap == "round" for p in lines)
    # In the legend and under the figure: predicted, not observed, and how far it has been checked.
    text = words(res)
    assert "Predicted water site (not observed)" in text and "Water bridge" not in text
    assert "geometric guesses for a model that has no solvent, not observations" in text and VALIDATION in text
    # In the result: said in words, and listed as data.
    assert any("predicted water site" in w and "not observations" in w for w in res.warnings)
    (listed,) = res.data["solvent_sites"]
    assert listed["predicted"] is True and listed["score"] >= 2
    assert {p["resname"] for p in listed["protein"]} >= {"GLN", "ARG", "MET"}
    # Everything that was observed is drawn exactly as without the option.
    plain = dia.render(style=style)
    def observed_links(result):
        return sorted((g.data["residue"], g.data["type"], g.data["ligand-atoms"], g.data.get("distance", ""))
                      for g in groups(result, "interaction") if not g.data.get("predicted"))
    assert observed_links(res) == observed_links(plain)


def test_a_site_between_partners_already_in_contact_is_not_drawn_as_a_bridge():
    """The site at the keto oxygen O83 has Gln711 and Arg752 among its partners, which hydrogen-bond O83 already
    (the table says so): the bridge is drawn to Met745, which does not."""
    dia = dry()
    (site,) = dia.solvent_sites()
    assert site.ligand[0][0] == "O83"
    assert {(p[3], p[1]) for p in site.protein} >= {("GLN", 711), ("ARG", 752), ("MET", 745)}
    res = dia.render(solvent_sites=True)
    assert [g.data["residue"] for g in groups(res, "interaction") if g.data.get("predicted")] == ["A:MET745"]
    direct = [(g.data["residue"], g.data["ligand-atoms"]) for g in groups(res, "interaction")
              if not g.data.get("predicted") and g.data["type"] == "hbond"]
    assert {("A:GLN711", "O83"), ("A:ARG752", "O83")} <= set(direct) and ("A:MET745", "O83") not in direct


def test_the_graph_holds_a_predicted_site_as_a_node_of_its_own_kind():
    dia = dry()
    res = dia.render(solvent_sites=True)
    g = res.graph
    (node,) = g.of_kind("solvent_site")
    assert node.attrs["predicted"] is True and not node.movable and node.label == "predicted water"
    assert node.anchors == ("L:O83", "R:A:MET745") and 0.0 <= node.attrs["enclosure"] <= 1.0
    assert len(node.attrs["xyz"]) == 3 and node.attrs["partners"]["ligand"][0]["atom"] == "O83"
    (contact,) = [c for c in g.contacts if c.attrs.get("predicted")]
    assert contact.type == "water_bridge" and contact.attrs["via"] == node.id and node.contacts == (contact.id,)
    assert not [c for c in g.contacts if c.attrs.get("predicted") and c.distance is not None]      # nothing was measured
    back = Graph.from_json(g.to_json())
    assert back == g and dia.render(solvent_sites=True, graph=back).svg == res.svg


def test_predicted_sites_are_not_contacts():
    dia = dry()
    before = dia.fingerprint()
    dia.render(solvent_sites=True)
    assert dia.fingerprint().bits == before.bits and dia.n_interactions == len(dry_table())
    assert not [b for b in dia.fingerprint().bits if b.type is T.WATER_BRIDGE]
    assert Interaction("water_bridge", 1, "SER").predicted is False
    assert Interaction.from_mapping({"type": "water_bridge", "residue_number": 1, "residue_name": "SER",
                                     "predicted": "true"}).predicted is True


def test_no_prediction_on_an_empty_figure_or_beside_observed_waters():
    base = dry()
    nothing = Diagram.from_structure(base.structure, base.ligand_key, base.ligand.smiles, [])
    empty = nothing.render(solvent_sites=True)
    assert empty.status == "empty" and "pred." not in words(empty)      # "nothing was found" stays that
    # A structure with waters of its own: observed ones take precedence, and the result says so.
    st = base.structure
    wet = Structure(atoms=list(st.atoms) + [Atom(index=len(st.atoms), name="O", element="O", resname="HOH", chain="W",
                                                 resnum=1, icode="", x=0.0, y=0.0, z=0.0, hetero=True)], source="wet")
    dia = Diagram(base.ligand, base.interactions, structure=wet, ligand_key=base.ligand_key, confidence=base.confidence)
    assert dia.solvent_sites() == []
    res = dia.render(solvent_sites=True)
    assert not [g for g in groups(res, "interaction") if g.data.get("predicted")] and res.data["solvent_sites"] == []
    assert any("1 water molecule of its own" in w and "observed waters take precedence" in w for w in res.warnings)


def test_command_line_flag(tmp_path, capsys):
    """The case's file without its water molecules, and its table without its water bridges."""
    with gzip.open(model_of(CASE), "rt", encoding="utf-8") as handle:
        lines = [line for line in handle if not (line.startswith("HETATM") and " HOH " in line)]
    (tmp_path / "dry.cif").write_text("".join(lines), encoding="utf-8")
    table = (TESTDATA / CASE / "interactions.tsv").read_text().splitlines()
    (tmp_path / "dry.tsv").write_text("\n".join(line for line in table if not line.startswith("water_bridge")) + "\n")
    out = tmp_path / "s.svg"
    ligand = meta_of(CASE)["ligand"]
    args = ["render", str(tmp_path / "dry.cif"), "--ligand", ligand["residue_name_in_cif"], "--smiles", ligand["smiles"],
            "--interactions", str(tmp_path / "dry.tsv"), "-o", str(out), "-q"]
    assert cli(args) == 0 and "pred." not in out.read_text()
    assert cli(args + ["--solvent-sites", "--json"]) == 0
    assert "pred. H" in out.read_text() and "Predicted water site (not observed)" in out.read_text()
    import json

    printed = json.loads(capsys.readouterr().out)
    assert len(printed["solvent_sites"]) == 1 and printed["solvent_sites"][0]["predicted"] is True
    assert load_interactions(TESTDATA / CASE / "interactions.tsv")[0].predicted is False


# ---------------------------------------------------------------------------
# The hold-out validation: the counting is tested here, on made-up structures. The numbers measured on crystal
# structures are stored in tests/data/solvent-validation*.json.
# ---------------------------------------------------------------------------

def _validation():
    import sys
    from pathlib import Path

    scripts = str(Path(__file__).resolve().parents[1] / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import validate_solvent

    return validate_solvent


def plus(st: Structure, rows) -> Structure:
    """The structure with more atoms: ``(name, element, resname, chain, number, xyz, hetero)`` each."""
    atoms = [(a.name, a.element, a.resname, a.chain, a.resnum, a.xyz, a.hetero) for a in st.atoms] + list(rows)
    return Structure(atoms=[Atom(index=i, name=nm, element=el, resname=res, chain=ch, resnum=num, icode="", x=q[0],
                                 y=q[1], z=q[2], hetero=het) for i, (nm, el, res, ch, num, q, het) in enumerate(atoms)],
                     source="toy")


#: Four more carbons on the toy ligand, so that it counts as a ligand without being named.
BODY = [("C2", "C", "LIG", "L", 1, (-1.9, 1.3, 0.0), True), ("C3", "C", "LIG", "L", 1, (-3.4, 1.3, 0.0), True),
        ("C4", "C", "LIG", "L", 1, (-4.1, 0.0, 0.0), True), ("C5", "C", "LIG", "L", 1, (-3.4, -1.3, 0.0), True)]
BRIDGING_WATER = ("O", "O", "HOH", "W", 1, (2.8, 0.0, 0.0), True)
FAR_WATER = ("O", "O", "HOH", "W", 2, (30.0, 0.0, 0.0), True)


def pdb_text(st: Structure, resolution: float | None = None) -> str:
    lines = [f"REMARK   2 RESOLUTION.    {resolution:.2f} ANGSTROMS."] if resolution is not None else []
    for a in st.atoms:
        lines.append(f"{'HETATM' if a.hetero else 'ATOM  '}{a.index + 1:5d} {a.name:<4s} {a.resname:>3s} {a.chain}"
                     f"{a.resnum:4d}    {a.x:8.3f}{a.y:8.3f}{a.z:8.3f}{1.0:6.2f}{20.0:6.2f}          {a.element:>2s}")
    return "\n".join(lines) + "\nEND\n"


def test_candidate_points_are_the_places_with_room_for_a_water():
    st, key = toy()
    points = candidate_points(st, key)
    assert len(points) and candidate_points(st, ResidueKey("A", 10)).shape[1] == 3
    oxygens = np.array([a.xyz for a in st.atoms if a.element == "O"])
    carbons = np.array([a.xyz for a in st.atoms if a.element == "C"])
    assert np.linalg.norm(points[:, None, :] - oxygens[None, :, :], axis=2).min() >= 2.6 - 1e-6
    assert np.linalg.norm(points[:, None, :] - carbons[None, :, :], axis=2).min() >= 1.7 + 1.4 - 1e-6
    assert np.linalg.norm(points - np.array([0.0, 0.0, 0.0]), axis=1).max() <= 3.5 + 1e-6
    # Every predicted site lies among them (it is the middle of a few of them).
    (site,) = predict_solvent_sites(st, key)
    assert np.linalg.norm(points - np.array(site.xyz), axis=1).min() <= 0.5
    # A ligand without N or O has no such place at all.
    bare = plus(st, [("C9", "C", "UNK", "X", 1, (2.8, 0.0, 0.0), True)])
    assert candidate_points(bare, ResidueKey("X", 1)).shape == (0, 3)


def test_validation_counts_matches_the_way_the_protocol_says():
    v = _validation()
    truth = np.array([[0.0, 0, 0], [10.0, 0, 0], [20.0, 0, 0]])
    predicted = np.array([[1.2, 0, 0], [10.0, 1.8, 0], [50.0, 0, 0]])
    assert v.matches(predicted, truth, 1.5) == (1, 1)            # one water recovered, by one site
    assert v.matches(predicted, truth, 2.0) == (2, 2) and v.matches(predicted, truth, 1.0) == (0, 0)
    assert v.matches(np.zeros((0, 3)), truth, 1.5) == (0, 0) and v.matches(predicted, np.zeros((0, 3)), 1.5) == (0, 0)
    # A site between two waters accounts for one of them, not for both ...
    assert v.matches(np.array([[1.4, 0.0, 0.0]]), np.array([[0.0, 0, 0], [2.8, 0, 0]]), 1.5) == (1, 1)
    # ... and the pairing is the largest there is: the first water does not keep the only site the second could have.
    assert v.matches(np.array([[1.2, 0.0, 0.0], [-1.2, 0.0, 0.0]]), np.array([[0.0, 0, 0], [2.4, 0, 0]]), 1.5) == (2, 2)

    def row(ligand, bridging, predicted, recovered, invented, reachable, by_chance, invented_by_chance):
        out = {"structure": "a", "ligand": ligand, "bridging_waters": bridging, "predicted": predicted,
               "chance_sites": float(predicted)}
        for cut in v.CUTS:
            out.update({f"recovered_{cut}": recovered, f"invented_{cut}": invented, f"reachable_{cut}": reachable,
                        f"chance_recovered_{cut}": by_chance, f"chance_invented_{cut}": invented_by_chance})
        return out

    total = v.summarise([row("A:X:1", 4, 5, 2, 2, 4, 0.4, 4.0), row("A:Y:2", 4, 5, 2, 2, 2, 0.4, 4.0)])
    assert total["structures"] == 1 and total["ligands"] == 2 and total["bridging_waters"] == 8
    assert total["recall_1.5"] == 0.5 and total["invented_share_1.5"] == 0.4
    assert total["reachable_share_1.5"] == 0.75 and total["chance_recall_1.5"] == 0.1
    assert total["chance_invented_share_1.5"] == 0.8
    assert v.summarise([]) == {"structures": 0, "ligands": 0}


def test_validation_takes_every_free_ligand_once_and_nothing_else(tmp_path):
    v = _validation()
    st, key = toy()
    whole = plus(st, BODY + [
        # a second copy of the ligand, far away
        *[(nm, el, "LIG", "M", 1, (q[0] + 60.0, q[1], q[2]), True) for nm, el, _res, _ch, _n, q, _h in
          [("O1", "O", "", "", 0, (0.0, 0.0, 0.0), True), ("C1", "C", "", "", 0, (-1.23, 0.0, 0.0), True)] + BODY],
        # glycerol: an additive
        *[(f"C{k}", "C", "GOL", "G", 1, (80.0 + 1.5 * k, 0.0, 0.0), True) for k in range(6)],
        # a modified residue: six atoms, bonded to the serine
        ("N", "N", "MSE", "A", 11, (7.9, 0.9, 0.0), True),
        *[(f"C{k}", "C", "MSE", "A", 11, (9.0 + 1.5 * k, 1.0, 0.0), True) for k in range(5)],
        # a fragment too small to be a ligand
        *[(f"C{k}", "C", "FRG", "F", 1, (100.0 + 1.5 * k, 0.0, 0.0), True) for k in range(3)],
        # a zinc beside the ligand's oxygen is not a covalent bond
        ("ZN", "ZN", "ZN", "Z", 1, (0.0, 2.0, 0.0), True),
        BRIDGING_WATER])
    assert v.covalently_attached(whole, ResidueKey("A", 11)) and not v.covalently_attached(whole, key)
    ligands, passed, settled = v.ligands_of(whole, tmp_path)        # nothing names a ligand: the rule decides
    assert ligands == [key] and settled == "rule"                   # the first copy, and only the ligand
    assert {d["compound"]: d["reason"] for d in passed} == {
        "GOL": "additive or ion", "MSE": "covalently attached to another residue",
        "FRG": "fewer than 6 heavy atoms", "ZN": "additive or ion"}
    # The list of the validation set names the ligand of a structure, and then nothing else is looked at ...
    assert v.ligands_of(whole, tmp_path, named="LIG") == ([key], [], "list of the set")
    assert v.ligands_of(whole, tmp_path, named="GOL") == ([ResidueKey("G", 1)], [], "list of the set")
    with pytest.raises(LigandNotFoundError, match="no residue named 'XYZ'"):
        v.ligands_of(whole, tmp_path, named="XYZ")                  # ... and a ligand that is not there is an error
    # ... unless a meta.json beside the structure says otherwise.
    (tmp_path / "meta.json").write_text('{"ligand": "M:LIG:1"}')
    assert v.ligands_of(whole, tmp_path, named="LIG") == ([ResidueKey("M", 1)], [], "meta.json")


def test_a_named_ligand_is_the_compound_and_not_a_residue_of_the_same_name(tmp_path):
    """1x8x: the ligand is free tyrosine, and the first tyrosine in the file belongs to the protein."""
    v = _validation()
    base, _ = toy()
    in_chain = [("N", "N", "TYR", "A", 37, (12.0, 9.0, 0.0), False), ("OH", "O", "TYR", "A", 37, (14.0, 9.0, 0.0), False)]
    free = [("N", "N", "TYR", "A", 952, (12.0, -9.0, 0.0), True), ("OH", "O", "TYR", "A", 952, (14.0, -9.0, 0.0), True)]
    both = plus(base, in_chain + free)
    assert both.residue_name(both.residues()[-2]) == "TYR" and not both.residue_atoms(both.residues()[-2])[0].hetero
    ligands, _, settled = v.ligands_of(both, tmp_path, "TYR")
    assert [str(k) for k in ligands] == ["A:952"] and settled == "list of the set"
    with pytest.raises(ValueError, match="only as a residue of a chain"):
        v.ligands_of(plus(base, in_chain), tmp_path, "TYR")


def test_the_validation_set_is_a_fixed_list():
    v = _validation()
    named = v.named_ligands(v.SET)
    assert len(named) == 85 and named["1t46"] == "STI" and named["1g9v"] == "RQ3" and named["2bsm"] == "BSM"
    assert all(len(k) == 4 and k == k.lower() and 1 <= len(name) <= 3 for k, name in named.items())
    assert list(named) == sorted(named)                             # so the halves are what the file says they are
    assert v.named_ligands(v.SET.with_name("no-such-list.tsv")) == {}


def test_validation_run_on_made_up_structures(tmp_path):
    import json

    v = _validation()
    st, key = toy()
    body = plus(st, BODY)
    cases = {"1aaa": (plus(body, [BRIDGING_WATER, FAR_WATER]), 1.50, None),      # its water is found
             "1bbb": (body, 1.90, None),                                         # no waters: cannot be evaluated
             "1ccc": (plus(body, [FAR_WATER]), 2.60, None),                      # the site has no water: invented
             "1ddd": (plus(st, [BRIDGING_WATER]), 1.80, "L:LIG:1")}              # a two-atom ligand, named in meta.json
    for name, (structure, resolution, ligand) in cases.items():
        folder = tmp_path / name
        folder.mkdir()
        (folder / f"{name}.pdb").write_text(pdb_text(structure, resolution))
        if ligand:
            (folder / "meta.json").write_text(json.dumps({"ligand": ligand}))
    result = v.run(tmp_path, SolventParams(), named={"1ccc": "LIG"})
    assert result["files"] == 4 and result["tuned"] is False
    assert [d["ligand_from"] for d in result["structures"]] == ["rule", "list of the set", "meta.json"]
    assert [(d["structure"], d["half"]) for d in result["structures"]] == [
        ("1aaa", "development"), ("1ccc", "development"), ("1ddd", "held_out")]
    # The half is the place in the sorted list of files, whatever becomes of the file.
    assert [(d["structure"], d["half"]) for d in result["not_evaluated"]] == [("1bbb", "held_out")]
    assert "no waters" in result["not_evaluated"][0]["error"]
    dev, held = result["development"], result["held_out"]
    assert (dev["structures"], dev["ligands"], dev["bridging_waters"], dev["predicted"]) == (2, 2, 1, 2)
    assert (dev["recovered_1.5"], dev["invented_1.5"], dev["reachable_1.5"]) == (1, 1, 1)
    assert dev["recall_1.5"] == 1.0 and dev["invented_share_1.5"] == 0.5
    assert (held["structures"], held["bridging_waters"], held["recovered_1.5"], held["invented_1.5"]) == (1, 1, 1, 0)
    assert result["held_out_high_resolution"]["structures"] == 1 and result["all"]["ligands"] == 3
    assert [r["resolution"] for r in result["per_ligand"]] == [1.5, 2.6, 1.8]
    # Chance: the same number of sites, from the places with room for a water. Reproducible, and a share.
    assert 0.0 <= dev["chance_recall_1.5"] <= 1.0 and 0.0 <= dev["chance_invented_share_1.5"] <= 1.0
    assert dev["chance_sites"] == 2.0 and v.run(tmp_path, SolventParams(), named={"1ccc": "LIG"}) == result
    # The wording, with these numbers.
    assert v.footnote(result) == ("In a hold-out test on 1 crystal structures they recovered 100 % of the bridging "
                                  "waters, and 0 % of the predicted sites had no observed water within 1.5 Å.")
    assert v.headline(result).startswith("Held-out half (1 structures, 1 ligands), parameters not tuned: 1 of 1 "
                                         "bridging waters recovered within 1.5 Å (100 %; ")
    assert "tuned on the other half" in v.headline(v.run(tmp_path, SolventParams(), tuned=True))
    # For a structure of the set, the half is its place in the set, so a missing file moves no other structure.
    moved = v.run(tmp_path, SolventParams(), named={"0zzz": "LIG", "1aaa": "LIG", "1ccc": "LIG"})
    assert {d["structure"]: d["half"] for d in moved["structures"]} == {
        "1aaa": "held_out", "1ccc": "development", "1ddd": "held_out"}
    assert v.table(result).count("\n") == 5 and "**held out (even positions)**" in v.table(result)


def test_a_second_set_comes_with_a_manifest_and_is_reported_by_target(tmp_path):
    """Compressed files in one folder, a manifest naming file, ligand and gene: read as they are, never pooled."""
    import gzip
    import json

    v = _validation()
    base, _ = toy()
    lig = plus(base, BODY + [BRIDGING_WATER, FAR_WATER])
    stuck = plus(lig, [("SG", "S", "CYS", "A", 9, (-5.6, 0.0, 0.0), False)])      # 1.5 A from C4 of the ligand
    data = tmp_path / "waters"
    data.mkdir()
    for name, st in (("AAA1_1ABC", lig), ("AAA1_2ABC", stuck), ("BBB2_3ABC", lig)):
        (data / f"{name}.pdb.gz").write_bytes(gzip.compress(pdb_text(st, 1.5).encode()))
    manifest = data / "manifest.json"
    manifest.write_text(json.dumps({"entries": [{"file": "AAA1_1ABC.pdb", "ligand": "LIG", "gene": "AAA1"},
                                                {"file": "AAA1_2ABC.pdb", "ligand": "LIG", "gene": "AAA1"},
                                                {"file": "BBB2_3ABC.pdb", "ligand": "LIG", "gene": "BBB2",
                                                 "ligand_is_cofactor": True}],
                                    "dropped": [{"file": "CCC3_4ABC.pdb"}]}), encoding="utf-8")
    assert v.is_structure(data / "AAA1_1ABC.pdb.gz") and not v.is_structure(manifest)
    assert v.name_of(data / "AAA1_1ABC.pdb.gz") == "AAA1_1ABC" and v.name_of(Path("x/1abc/1abc.cif.gz")) == "1abc"
    named, groups = v.named_ligands(manifest), v.named_groups(manifest)
    assert named == {"aaa1_1abc": "LIG", "aaa1_2abc": "LIG", "bbb2_3abc": "LIG"}
    assert groups == {"aaa1_1abc": "AAA1", "aaa1_2abc": "AAA1", "bbb2_3abc": "BBB2"}

    result = v.run(data, SolventParams(), named=named, groups=groups)
    assert not result["not_evaluated"] and result["all"]["structures"] == 3
    assert [s["group"] for s in result["structures"]] == ["AAA1", "AAA1", "BBB2"]
    assert all(len(s["sha256"]) == 64 for s in result["structures"])
    assert list(result["by_group"]) == ["AAA1", "BBB2"]
    assert result["by_group"]["AAA1"]["structures"] == 2 and result["by_group"]["BBB2"]["structures"] == 1
    assert sum(g["bridging_waters"] for g in result["by_group"].values()) == result["all"]["bridging_waters"] == 3
    # a ligand bonded to the protein is kept, as the set names it, and can be left out of a row of its own
    assert [r["attached"] for r in result["per_ligand"]] == [False, True, False]
    assert all(r["complete"] for r in result["per_ligand"])
    assert result["free_and_complete"]["structures"] == 2
    text = v.table(result) + "\n" + v.group_table(result)
    assert "all, ligand free and complete" in text and "| AAA1 | 2 |" in text and "| BBB2 | 1 |" in text
    assert "attached" not in result["all"] and "complete" not in result["all"]
    # by site, all of a set is sorted only where nothing is tuned on it; otherwise its development half alone
    assert "all_sites" not in result and set(result["development_sites"]) == {"by_enclosure", "by_partners", "by_rank"}
    whole = v.run(data, SolventParams(), named=named, groups=groups, nothing_tuned_here=True)
    assert sum(c["sites"] for c in whole["all_sites"]["by_rank"].values()) == whole["all"]["predicted"]
    assert {k: x for k, x in whole.items() if k != "all_sites"} == result
    # by resolution: every structure is in one band, and the bands add up to the whole
    assert [b["structures"] for b in result["by_resolution"].values()] == [3, 0, 0]
    assert "| up to 1.5 Å | 3 |" in v.resolution_table(result) and "| above 2.0 Å | 0 |" in v.resolution_table(result)
    # a ligand that the set flags as a cofactor stays in, and there is a row without it
    assert v.named_cofactors(manifest) == {"bbb2_3abc"} and "not_a_cofactor" not in result
    drugs = v.run(data, SolventParams(), named=named, groups=groups, cofactors=v.named_cofactors(manifest))
    assert [r["cofactor"] for r in drugs["per_ligand"]] == [False, False, True]
    assert drugs["not_a_cofactor"]["structures"] == 2 and drugs["all"] == result["all"]
    assert "all, ligand not a cofactor" in v.table(drugs) and "all, ligand not a cofactor" not in v.table(result)


def test_the_heavy_atoms_of_a_compound_are_read_from_its_formula():
    v = _validation()
    cif = "data_x\nloop_\n_chem_comp.id\n_chem_comp.formula\nSTI 'C29 H31 N7 O'\nHOH 'H2 O'\nZN Zn\nCL 'Cl -1'\n"
    assert v.heavy_atoms_by_formula(cif) == {"STI": 37, "HOH": 1, "ZN": 1, "CL": 1}
    assert v.heavy_atoms_by_formula("HEADER    a file in the old format\n") == {}


ROOT = Path(__file__).resolve().parents[1]
#: The two sets the hold-out was run on: folder under testdata, the set, and the file with the numbers.
SETS = {"astex": ("scripts/solvent_validation_set.tsv", "solvent-validation.json"),
        "waters": ("testdata/waters/manifest.json", "solvent-validation.targets.json")}


def _measured(name: str = "solvent-validation.json"):
    import json

    return json.loads((ROOT / "tests" / "data" / name).read_text(encoding="utf-8"))


def test_the_footnote_and_the_parameters_are_those_of_the_measurement():
    """The footnote every figure with a predicted site carries is the one the stored hold-out gives, and the
    numbers stored describe the predictor as it is: these parameters, never tuned, every structure of the set."""
    import dataclasses

    v = _validation()
    result = _measured()
    assert VALIDATION == v.footnote(result)
    assert result["parameters"] == dataclasses.asdict(SolventParams()) and result["tuned"] is False
    assert not result["not_evaluated"] and result["all"]["structures"] == len(v.named_ligands(v.SET)) == 85
    assert result["free_and_complete"] == result["all"]
    second = _measured(SETS["waters"][1])
    assert second["parameters"] == result["parameters"] and second["tuned"] is False and not second["not_evaluated"]


def test_the_burial_warning_on_the_crystal_structures_of_the_hold_out():
    """The stored hold-out was measured with the cut of the warning the code has: each ligand is warned of
    exactly where its buried fraction is below it, and the warned are listed from the least buried."""
    from bindview.exposure import BURIAL_WARNING_BELOW

    for result in (_measured(), _measured(SETS["waters"][1])):
        assert result["burial"]["cut"] == BURIAL_WARNING_BELOW
        for row in result["per_ligand"]:
            assert row["burial_warning"] == (row["buried_fraction"] < BURIAL_WARNING_BELOW)
        assert [w["structure"] for w in result["burial"]["warned"]] == [
            r["structure"] for r in sorted(result["per_ligand"], key=lambda r: (r["buried_fraction"], r["structure"]))
            if r["burial_warning"]]


def _again(v, folder: str, tmp_path, every: int):
    """Run the hold-out again on every ``every``-th structure of a set: (rows as measured, rows now)."""
    import json

    set_file, name = SETS[folder]
    data, result = ROOT / "testdata" / folder, _measured(name)
    files = sorted(p for p in data.rglob("*") if p.is_file() and v.is_structure(p))
    assert files, f"testdata/{folder} holds no structure"
    measured = {s["structure"]: s["sha256"] for s in result["structures"]}
    for path in files[::every]:
        (tmp_path / path.name).write_bytes(path.read_bytes())
    again = v.run(tmp_path, SolventParams(), named=v.named_ligands(ROOT / set_file),
                  groups=v.named_groups(ROOT / set_file), cofactors=v.named_cofactors(ROOT / set_file))
    assert not again["not_evaluated"]
    same = {s["structure"] for s in again["structures"] if measured.get(s["structure"]) == s["sha256"]}
    assert same == {s["structure"] for s in again["structures"]}, f"testdata/{folder} holds files that were not measured"
    want = {r["structure"]: r for r in result["per_ligand"]}
    rows = json.loads(json.dumps(again["per_ligand"]))
    return [want[r["structure"]] for r in rows], rows


@pytest.mark.parametrize("folder, every", [pytest.param("astex", 7, marks=fetched("astex")),
                                           pytest.param("waters", 5, marks=fetched("waters"))])
def test_the_hold_out_comes_out_the_same_where_the_structures_are_present(tmp_path, folder, every):
    measured, now = _again(_validation(), folder, tmp_path, every)
    assert len(now) >= 3 and now == measured


@pytest.mark.slow
@pytest.mark.parametrize("folder", [pytest.param("astex", marks=fetched("astex")),
                                    pytest.param("waters", marks=fetched("waters"))])
def test_the_whole_hold_out_comes_out_the_same(tmp_path, folder):
    measured, now = _again(_validation(), folder, tmp_path, 1)
    assert now == measured and len(now) == len(_measured(SETS[folder][1])["per_ligand"])
