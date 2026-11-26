"""Fingerprints, alignment, and the difference and consensus views of a set of ligands.

The sets are ligands of one pocket, each from a crystal structure of its own (``testdata/overlay``): four steroids of
the glucocorticoid receptor on one scaffold (``gr_steroids``), a steroid and two non-steroidal ligands of that pocket
(``gr_diverse``), and two unrelated ligands of the androgen receptor (``ar_ligands``). What a figure says of the
contacts is checked against the members' contact tables, read here as text.
"""

from __future__ import annotations

import csv
import math
import re
from collections import Counter

import numpy as np
import pytest

from conftest import (SETS, TESTDATA, diagram_of, meta_of, model_of, overlay_diagram, overlay_members, overlay_meta,
                      predicted_diagram, structure_of)
from bindview import Comparison, Diagram, Fingerprint, FingerprintTable, InteractionType, load_interactions
from bindview.cli import main as cli
from bindview.compare import _SUBSTITUENT, TYPE_CODES
from bindview.confidence import MARK
from bindview.fingerprint import Bit
from bindview.scene import Circle, Group, Polyline, Text
from bindview.structure import Structure
from scenetools import groups, inside, words

T = InteractionType
#: Two steroids of ``gr_steroids``: dexamethasone, and mometasone furoate with its furoate ester.
STEROID_PAIR = ("gr_dexamethasone", "gr_mometasone")
_cmp: dict[str, Comparison] = {}


def comparison(key: str) -> Comparison:
    if key not in _cmp:
        if key == "steroid_pair":
            _cmp[key] = Comparison([overlay_diagram("gr_steroids", s) for s in STEROID_PAIR])
        else:
            _cmp[key] = Comparison([overlay_diagram(key, slug) for slug, _ in overlay_members(key)])
    return _cmp[key]


def listed(slug: str) -> list[dict]:
    """The rows of a member's contact table (``interactions.tsv``), read as text."""
    with open(TESTDATA / slug / "interactions.tsv", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def bits_of(slug: str) -> set[str]:
    """A member's contacts as columns of a fingerprint (``A:GLN570:hbond``), from its table."""
    return {f"{r['chain']}:{r['residue_name']}{r['residue_number']}:{r['interaction_type']}" for r in listed(slug)}


def kinds_of(slug: str) -> dict[int, set[str]]:
    """The kinds of contact a member's table lists for each residue number."""
    out: dict[int, set[str]] = {}
    for r in listed(slug):
        out.setdefault(int(r["residue_number"]), set()).add(r["interaction_type"])
    return out


def all_groups(res, name: str) -> list[Group]:
    """Every group of a class, including those that hold no ink (a contact that is data only)."""
    out = []

    def rec(items):
        for it in items:
            if isinstance(it, Group):
                if it.cls == name or it.cls.startswith(name + " "):
                    out.append(it)
                rec(it.items)
    rec(res.scene.items)
    return out


def node_centres(res, panel) -> dict[str, tuple[float, float]]:
    """Centre of every residue node of a panel, relative to the panel's own origin."""
    ox, oy = panel.translate or (0.0, 0.0)
    out = {}
    for g in groups(res, "residue"):
        found = [f for f in inside(res, g, Circle) if any(p is panel for p in f.path)]
        if found:
            b = found[0].box()
            block = [p for p in found[0].path if p.cls == "block"][0].translate
            out[g.data["residue"]] = (round((b[0] + b[2]) / 2.0 - ox - block[0], 3),
                                      round((b[1] + b[3]) / 2.0 - oy - block[1], 3))
    return out


# ---------------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key", SETS)
def test_the_consensus_is_what_the_tables_of_the_members_say(key):
    """Each residue is counted once for every ligand whose table lists a contact with it, and each kind of contact
    once for every ligand that makes it; the rows are ordered by the number of ligands, then by residue, whatever
    order the ligands come in. The figure shows those counts."""
    cmp_ = comparison(key)
    rows: dict[tuple[str, int], dict] = {}
    for slug, _ in overlay_members(key):
        name = overlay_meta(key, slug)["ligand"]["name"]
        for r in listed(slug):
            row = rows.setdefault((r["chain"], int(r["residue_number"])),
                                  {"label": r["residue_name"].capitalize() + r["residue_number"], "ligands": [],
                                   "kinds": Counter()})
            if name not in row["ligands"]:
                row["ligands"].append(name)
            row["kinds"][(r["interaction_type"], name)] = 1
    want = sorted(((w["label"], len(w["ligands"]), tuple(w["ligands"]),
                    sorted(Counter(kind for kind, _ in w["kinds"]).items())) for w in rows.values()),
                  key=lambda w: (-w[1], int(re.search(r"\d+", w[0]).group())))
    got = cmp_.consensus()
    assert [(r.label, r.n_ligands, r.ligands, sorted((t.value, n) for t, n in r.types)) for r in got] == want
    assert all(r.fraction == r.n_ligands / cmp_.n for r in got)
    res = cmp_.render("consensus")
    nodes = {g.data["residue"]: g for g in groups(res, "residue")}
    assert len(nodes) == len(got)
    for row in got:
        node = nodes[f"{row.residue[0]}:{row.label.upper()}"]
        assert (node.data["n-ligands"], node.data["n-total"]) == (str(row.n_ligands), str(cmp_.n))
        assert any(f.item.content.startswith(f"{row.n_ligands}/{cmp_.n}") for f in inside(res, node, Text))
    assert [(e["residue"], e["ligands"]) for e in res.data["consensus"]] == [(r.label, list(r.ligands)) for r in got]


def test_fingerprint_columns_have_one_order_however_the_records_arrive():
    dia = overlay_diagram("ar_ligands", "ar_bicalutamide")
    fp = dia.fingerprint("bicalutamide")
    shuffled = Diagram(dia.ligand, list(reversed(dia.interactions)), structure=dia.structure,
                       ligand_key=dia.ligand_key, confidence=dia.confidence).fingerprint("bicalutamide")
    assert shuffled.columns == fp.columns and shuffled.bits == fp.bits
    keys = [b.sort_key for b in fp.columns]
    assert keys == sorted(keys) and len(set(fp.columns)) == len(fp)
    # By chain, residue number, then interaction type in the order of the enum: Leu704 has a hydrophobic contact and
    # a hydrogen bond with bicalutamide, and Leu701 a hydrophobic contact with R1881 alone (their tables).
    columns = comparison("ar_ligands").table.columns
    leu = [b for b in columns if b.number == 704]
    assert [b.type for b in leu] == [T.HYDROPHOBIC, T.HBOND]
    (other,) = [b for b in columns if b.number == 701]
    assert fp.vector([other, leu[0]]) == [0, 1] and len(fp.vector()) == len(fp) == sum(fp.vector())
    assert Bit.from_column(leu[1].column) == leu[1] and leu[1].column == "A:LEU704:hbond"
    assert Bit.from_column("B:ASP-3A:saltbridge") == Bit("B", -3, "A", "ASP", T.SALTBRIDGE)


@pytest.mark.parametrize("key", SETS)
def test_fingerprints_round_trip_through_csv(key, tmp_path):
    table = comparison(key).table
    long_text = table.to_csv(tmp_path / "long.csv")
    back = FingerprintTable.from_csv(tmp_path / "long.csv")
    assert back.names == table.names and [f.bits for f in back.fingerprints] == [f.bits for f in table.fingerprints]
    assert back.to_csv() == long_text and FingerprintTable.from_csv(long_text).to_csv() == long_text
    wide = table.to_csv(form="wide")
    again = FingerprintTable.from_csv(wide)
    assert again.columns == table.columns and again.matrix() == table.matrix() and again.to_csv(form="wide") == wide
    assert wide.splitlines()[0].split(",")[1:] == [b.column for b in table.columns]
    one = table.fingerprints[0]
    assert Fingerprint.from_csv(one.to_csv()).bits == one.bits
    # Evidence travels with presence but is never folded into it: the atoms, and a pLDDT where there is one (here
    # made up for the test from the temperature factors of the first member).
    first = next(iter(one.bits.values()))
    assert first.count >= 1 and first.ligand_atoms and first.plddt is None
    rated = predicted_diagram(overlay_members(key)[0][0]).fingerprint()
    assert all(v.plddt is not None for v in rated.bits.values())
    assert Fingerprint.from_csv(rated.to_csv()).bits == rated.bits and set(rated.bits) == set(one.bits)


# ---------------------------------------------------------------------------
# Alignment: on a common scaffold when there is one, refused when there is not
# ---------------------------------------------------------------------------

def test_a_common_scaffold_is_found_for_the_congeneric_series_only():
    steroids = comparison("gr_steroids").core
    # At least the steroid nucleus, seventeen carbons in four fused rings, which every member has.
    assert steroids.substantial and steroids.complete and steroids.n_atoms >= 17
    assert steroids.fraction >= steroids.min_fraction
    for key in ("ar_ligands", "gr_diverse"):
        core = comparison(key).core
        assert not core.substantial and core.complete and core.fraction < core.min_fraction
        assert "No common scaffold" in core.describe() and "No substructure alignment was made" in core.describe()
        made = comparison(key).alignment()
        assert made["common_atoms"] == core.n_atoms and made["aligned"] is False


def test_the_shared_core_is_on_identical_coordinates_in_every_ligand():
    """The common substructure is drawn in the same orientation in every ligand of the set, so that the differences
    read as geometry and not as a caption; the member drawn is the one with the fewest atoms outside it."""
    cmp_ = comparison("gr_steroids")
    rep = cmp_.representative
    outside = [cmp_.diagrams[k].ligand.n_atoms - len(cmp_.core.atoms[k]) for k in range(cmp_.n)]
    assert rep == outside.index(min(outside)) and cmp_.names[rep] == "dexamethasone"
    coords, mode = cmp_._coords(rep)
    assert mode == "core"
    reference = [coords[rep][n] for n in cmp_.core.atoms[rep]]
    for k in range(cmp_.n):
        assert [coords[k][n] for n in cmp_.core.atoms[k]] == reference           # exactly, not approximately
        assert set(coords[k]) == set(cmp_.diagrams[k].ligand.names)
    # ... and that is what the figure draws: one thumbnail per ligand, on those coordinates.
    res = cmp_.render()
    assert res.data["thumbnails"] is True
    assert [g.data["ligand"] for g in groups(res, "thumbnail")] == cmp_.names


# ---------------------------------------------------------------------------
# Difference view
# ---------------------------------------------------------------------------

def test_difference_view_of_two_unrelated_ligands_claims_no_scaffold():
    """Two ligands that share no scaffold: the view must not claim one."""
    cmp_ = comparison("ar_ligands")
    res = cmp_.render()
    assert res.status == "ok" and res.figure == "difference" and res.style == "difference"
    assert res.quality.valid and res.quality.readable and res.quality.overlaps == 0 and res.quality.link_hits == 0
    assert res.data["alignment"]["aligned"] is False and res.data["frame"] == "pose"
    text = words(res)
    assert "No common scaffold" in text and "No substructure alignment was made" in text
    assert f"({100 * cmp_.core.fraction:.0f} %)" in text
    assert not groups(res, "marks")                         # nothing is highlighted as "the part that differs"


def test_shared_and_unique_contacts_are_classified_and_told_apart():
    """Contacts made by both ligands, by the first only and by the second only are told apart at a glance, by the
    weight and kind of a residue's outline and not by colour; they are what the two tables list."""
    cmp_ = comparison("steroid_pair")
    diff = cmp_.difference()
    assert diff.names == ("dexamethasone", "mometasone furoate")
    a, b = (bits_of(s) for s in STEROID_PAIR)
    assert {x.column for x in diff.shared} == a & b and a & b
    assert {x.column for x in diff.only_a} == a - b and {x.column for x in diff.only_b} == b - a
    kinds_a, kinds_b = (kinds_of(s) for s in STEROID_PAIR)
    residues = diff.residues()
    assert {number for (_, number, _) in residues} == set(kinds_a) | set(kinds_b)
    for (_, number, _), cls in residues.items():
        assert cls == ("shared" if number in kinds_a and number in kinds_b else "a" if number in kinds_a else "b")
    assert not set(diff.shared) & set(diff.only_a) and not set(diff.only_a) & set(diff.only_b)
    assert len(diff.shared) + len(diff.only_a) == len(cmp_.fingerprints[0])
    assert len(diff.shared) + len(diff.only_b) == len(cmp_.fingerprints[1])
    assert diff.tanimoto == pytest.approx(cmp_.fingerprints[0].tanimoto(cmp_.fingerprints[1]))

    res = cmp_.render()
    assert res.data["difference"]["shared"] == [b.column for b in diff.shared]
    panels = groups(res, "panel")
    assert [p.data["ligand"] for p in panels] == list(diff.names)
    looks = {"shared": set(), "only": set(), "absent": set()}
    for k, panel in enumerate(panels):
        role = "a" if k == 0 else "b"
        for g in groups(res, "residue"):
            circles = [f.item for f in inside(res, g, Circle) if any(p is panel for p in f.path)]
            if not circles:
                continue
            chain, rest = g.data["residue"].split(":")
            cls = residues[(chain, int(re.search(r"\d+", rest).group()), "")]
            want = "shared" if cls == "shared" else ("only" if cls == role else "absent")
            assert g.data["contact"] == want
            outline = [c.paint for c in circles if c.paint.stroke][0]
            filled = any(c.paint.fill for c in circles)
            looks[want].add((filled, round(outline.width, 1), bool(outline.dash)))
    # Three states, three outlines -- weight and kind of line, not colour.
    assert looks["shared"] == {(True, 0.9, False)} and looks["only"] == {(True, 2.5, False)}
    assert looks["absent"] == {(False, 0.8, True)}
    # A kind of contact that one ligand adds at a shared residue is written on the node (in the tables: Leu563 has a
    # hydrogen bond with both, and a hydrophobic contact with mometasone furoate alone).
    notes = [f.item.content for g in groups(res, "residue") for f in inside(res, g, Text)
             if f.item.content.startswith("+")]
    added = ["+" + TYPE_CODES[T(kind)] for number in set(kinds_a) & set(kinds_b)
             for kind in kinds_a[number] ^ kinds_b[number]]
    assert sorted(notes) == sorted(added) == ["+hyd"]
    # Every interaction, drawn as a line or not, says whether it is shared.
    marks = [g.data["contact"] for g in all_groups(res, "interaction") + all_groups(res, "contact")]
    assert set(marks) == {"shared", "only"} and marks.count("shared") >= 2 * len(diff.shared)
    assert len(marks) >= len(cmp_.fingerprints[0]) + len(cmp_.fingerprints[1])


@pytest.mark.parametrize("key", ["ar_ligands", "steroid_pair"])
def test_the_residue_frame_is_identical_in_both_panels(key):
    """Every residue stands in the same place in both panels, and none is drawn twice in a panel."""
    cmp_ = comparison(key)
    res = cmp_.render()
    assert res.status == "ok"
    a, b = groups(res, "panel")
    ca, cb = node_centres(res, a), node_centres(res, b)
    union = set(cmp_.fingerprints[0].residues()) | set(cmp_.fingerprints[1].residues())
    assert len(ca) == len(cb) == len(union) and ca == cb
    # Nodes keep off each other.
    pts = list(ca.values())
    assert all(math.dist(p, q) >= 30.0 for i, p in enumerate(pts) for q in pts[i + 1:])


def test_aligned_pair_shows_what_differs_and_attributes_the_ester_contact():
    """Of two steroids on a common substructure, what lies outside it is marked in each. Ile629 is contacted by
    mometasone furoate alone, through atoms of its furoate ester, outside the common substructure."""
    cmp_ = comparison("steroid_pair")
    res = cmp_.render()
    assert res.data["frame"] == "core" and res.data["alignment"]["aligned"] is True
    differ = [k for k in range(cmp_.n) if set(cmp_.diagrams[k].ligand.names) - cmp_.core_atoms(k)]
    assert "Aligned on a common substructure" in words(res) and len(groups(res, "marks")) == len(differ) == 2
    ile = [g for g in groups(res, "residue") if g.data["residue"] == "A:ILE629"]
    assert [g.data["contact"] for g in ile] == ["absent", "only"]          # dexamethasone's panel, then the furoate's
    (contact,) = [g for g in all_groups(res, "contact") if g.data["residue"] == "A:ILE629"]
    atoms = set(contact.data["ligand-atoms"].split(","))
    assert atoms and not atoms & cmp_.core_atoms(1)                        # made by atoms outside the common core
    lig = cmp_.diagrams[1].ligand
    (furan,) = [ring for ring in lig.mol.GetRingInfo().AtomRings()
                if len(ring) == 5 and any(lig.mol.GetAtomWithIdx(i).GetSymbol() == "O" for i in ring)
                and all(lig.mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring)]
    assert atoms <= {lig.names[i] for i in furan}


def test_a_residue_that_one_ligand_contacts_is_named_in_each_panel_as_its_structure_names_it():
    """Bicalutamide's receptor is the W741L mutant: residue 741 is a leucine in its structure and a tryptophan in
    R1881's. Only bicalutamide contacts it; the panel of R1881 keeps a place for it and names it as R1881's
    structure does."""
    cmp_ = comparison("ar_ligands")
    own = {}
    for slug, _ in overlay_members("ar_ligands"):
        (name,) = {a.resname for a in structure_of(slug).atoms if (a.chain, a.resnum, a.hetero) == ("A", 741, False)}
        own[meta_of(slug)["ligand"]["name"]] = name
    assert own == {"metribolone (R1881)": "TRP", "bicalutamide": "LEU"}
    res = cmp_.render("difference")
    named = {}
    for panel in groups(res, "panel"):
        for g in groups(res, "residue"):
            texts = [f.item.content for f in inside(res, g, Text) if any(p is panel for p in f.path)]
            if g.data["residue-number"] == "741" and texts:
                named[panel.data["ligand"]] = texts[0]
    assert named == own


# ---------------------------------------------------------------------------
# Consensus view
# ---------------------------------------------------------------------------

def _ring(res, residue: str) -> tuple[float, float]:
    """The share of a full turn that the conservation arc around a consensus node runs, and its heaviest stroke."""
    (g,) = [g for g in groups(res, "residue") if g.data["residue"] == residue]
    arcs = [f.item for f in inside(res, g, Polyline)]
    (track,) = [f.item for f in inside(res, g, Circle) if f.item.paint.stroke == "panel_border"]
    length = sum(math.dist(p, q) for a in arcs for p, q in zip(a.points, a.points[1:]))
    return length / (2.0 * math.pi * track.r), max(a.paint.width for a in arcs)


def test_consensus_of_the_congeneric_series():
    """Contacts that every ligand of the set makes are drawn over those that one makes: a longer and heavier ring,
    a larger disc, a bold name; and no residue is drawn twice."""
    cmp_ = comparison("gr_steroids")
    res = cmp_.render()
    assert res.status == "ok" and res.figure == "consensus" and res.quality.valid and res.quality.readable
    assert res.data["drawn_ligand"] == cmp_.names[cmp_.representative] == "dexamethasone"
    assert res.data["frame"] == "core"
    nodes = groups(res, "residue")
    names = [g.data["residue"] for g in nodes]
    contacted = {number for slug, _ in overlay_members("gr_steroids") for number in kinds_of(slug)}
    assert len(names) == len(set(names)) == len(cmp_.consensus()) == len(contacted)
    by = {g.data["residue"]: g for g in nodes}
    # (In the tables: every steroid has its hydrogen bonds to Gln570 and Arg611; three of the four contact Asn564,
    # and only dexamethasone Tyr735.)
    assert by["A:GLN570"].data["n-ligands"] == by["A:ARG611"].data["n-ligands"] == "4"
    assert by["A:ASN564"].data["n-ligands"] == "3" and by["A:TYR735"].data["n-ligands"] == "1"
    # Dominance is drawn: a ring as long as the share and heavier, a larger disc, a bold name.
    full, rare = _ring(res, "A:GLN570"), _ring(res, "A:TYR735")
    assert full[0] == pytest.approx(1.0, abs=0.01) and rare[0] == pytest.approx(0.25, abs=0.01) and full[1] >= rare[1]

    def disc(g):
        return max(f.item.r for f in inside(res, g, Circle) if f.item.paint.fill)

    def name(g):
        return [f.item for f in inside(res, g, Text)][0]
    assert disc(by["A:GLN570"]) > disc(by["A:TYR735"])
    assert name(by["A:GLN570"]).bold and not name(by["A:TYR735"]).bold
    text = words(res)
    assert "4/4" in text and "1/4" in text and "No per-ligand lines are drawn" in text
    assert "Drawn: dexamethasone" in text                      # which ligand is shown is stated in the figure


def test_a_substituent_only_contact_is_attributed_to_the_substituent():
    """Asn564 is contacted by three of the four steroids, each through its C21 hydroxyl. Mometasone furoate has a
    chlorine there, so that oxygen is not part of the common substructure: the contact is attributed to atoms
    outside it, and a residue contacted through the core is not."""
    cmp_ = comparison("gr_steroids")
    via = cmp_.contact_via()
    asn, gln = ("A", 564, ""), ("A", 570, "")
    makers = [overlay_meta("gr_steroids", s)["ligand"]["name"] for s, _ in overlay_members("gr_steroids")
              if 564 in kinds_of(s)]
    assert len(makers) == 3 and via[asn] == (0, 3)
    assert via[gln] == (4, 0)                                                   # the 3-keto oxygen, in every one
    row = next(r for r in cmp_.consensus() if r.residue == asn)
    assert list(row.ligands) == makers
    for k, fp in enumerate(cmp_.fingerprints):
        for bit, contact in fp.bits.items():
            if bit.residue == asn:
                assert contact.ligand_atoms and not set(contact.ligand_atoms) & cmp_.core_atoms(k)
    res = cmp_.render()
    for node in groups(res, "residue"):
        number = int(node.data["residue-number"])
        marked = any(f.item.content.endswith(_SUBSTITUENT) for f in inside(res, node, Text))
        assert marked == (via[("A", number, "")][0] == 0)
    assert "outside the common substructure" in words(res)
    entry = next(r for r in res.data["consensus"] if r["residue"] == "Asn564")
    assert entry["via_core"] == 0 and entry["via_substituent_only"] == 3


def test_consensus_of_unrelated_ligands_refuses_alignment_and_draws_no_lines():
    """A set without a common substructure worth aligning on says so instead of making a meaningless alignment;
    no ligand is drawn, nor any per-ligand line, and the figure carries counts. The pockets superpose, so the frame
    is their footprint."""
    cmp_ = comparison("gr_diverse")
    res = cmp_.render()
    assert res.status == "ok" and res.figure == "consensus" and res.quality.valid and res.quality.readable
    assert res.data["alignment"]["aligned"] is False and res.data["drawn_ligand"] is None
    assert cmp_.representative is None
    text = words(res)
    assert "No common scaffold" in text and "No substructure alignment was made" in text
    assert "No ligand is drawn" in text and not groups(res, "ligand") and not groups(res, "thumbnail")
    assert not all_groups(res, "interaction") and "No per-ligand lines are drawn" in text
    assert f"{cmp_.n}/{cmp_.n}" in text and f"1/{cmp_.n}" in text
    assert res.data["alignment"]["superposed"] is True and res.data["frame"] == "envelope"


def _folded(key: str, slug: str) -> Diagram:
    """A member of a set with its pocket folded otherwise, made for the test: every second residue within 12 A of the
    ligand that its table does not name moved 8 A away. Its contacts stay what its table says."""
    from dataclasses import replace

    folder = dict(overlay_members(key))[slug]
    m = overlay_meta(key, slug)
    rows = load_interactions(folder / "interactions.tsv")
    named = {r.residue_number for r in rows}
    base = overlay_diagram(key, slug)
    near = set(base.pocket_residues(12.0))
    st = base.structure
    atoms = [replace(a, x=a.x + 8.0) if a.residue in near and a.resnum % 2 and a.resnum not in named else a
             for a in st.atoms]
    moved = Structure(atoms=atoms, source=st.source, chem_comp_smiles=st.chem_comp_smiles, plddt_scale=st.plddt_scale,
                      qa_metrics=st.qa_metrics)
    return Diagram.from_structure(moved, ligand=m["ligand"]["residue_name_in_cif"], smiles=m["ligand"]["smiles"],
                                  interactions=rows, name=m["ligand"]["name"])


def test_unrelated_ligands_whose_pockets_do_not_superpose_are_set_out_in_sequence_order():
    """Nor is a pocket invented: where a member's pocket does not superpose on the others', the frame of a set
    without a scaffold is the sequence, and the figure says so."""
    members = [overlay_diagram("gr_diverse", slug) for slug, _ in overlay_members("gr_diverse")]
    members[1] = _folded("gr_diverse", overlay_members("gr_diverse")[1][0])
    cmp_ = Comparison(members)
    assert cmp_.transforms[1] is None and cmp_.pocket_rmsd[1] > 3.0              # the fit was made, and is too poor
    res = cmp_.render()
    assert res.status == "ok" and res.figure == "consensus" and res.quality.valid
    assert res.data["alignment"]["superposed"] is False and res.data["frame"] == "sequence"
    assert groups(res, "sequence-frame") and not groups(res, "envelope") and not all_groups(res, "interaction")
    text = words(res)
    assert "no spatial frame is claimed" in text and "sequence order" in text
    xs = {g.data["residue"]: f.box() for g in groups(res, "residue") for f in inside(res, g, Circle)[:1]}
    order = sorted(xs, key=lambda r: (round((xs[r][1] + xs[r][3]) / 2.0), xs[r][0]))    # by line, then left to right
    assert order == sorted(xs, key=lambda r: int(re.search(r"\d+", r).group()))        # reading order = sequence


def test_unrelated_ligands_in_one_rigid_pocket_get_a_pose_footprint():
    """Where the structures do superpose, the frame of an unaligned set is the pocket itself."""
    cmp_ = comparison("ar_ligands")
    assert all(t is not None for t in cmp_.transforms) and max(cmp_.pocket_rmsd) < 3.0
    res = cmp_.render(view="consensus")
    assert res.status == "ok" and res.data["frame"] == "envelope" and res.quality.valid
    assert groups(res, "envelope") and not groups(res, "ligand") and not all_groups(res, "interaction")
    text = words(res)
    assert "footprint of the 2 superposed poses" in text and "2/2" in text and "1/2" in text
    assert len(groups(res, "residue")) == len(cmp_.consensus())


def test_consensus_draws_weak_evidence_weakly():
    """A contact on a residue with a low pLDDT looks weaker. The pLDDT is made up for the test from the temperature
    factors of the crystal structures: a ring is heavy where ligands contact the residue confidently, and its
    count is marked where none does better than low."""
    cmp_ = Comparison([predicted_diagram(slug) for slug, _ in overlay_members("gr_steroids")])
    res = cmp_.render()
    rows = {r["residue"]: r for r in res.data["consensus"]}
    for row in rows.values():
        assert sum(row["by_confidence"]) == row["n_ligands"] and row["n_confident"] == row["by_confidence"][0]
    sure = [label for label, row in rows.items() if row["n_confident"]]
    unsure = [label for label, row in rows.items() if not row["n_confident"]]
    assert sure and unsure
    heaviest = {label: _ring(res, f"A:{label.upper()}")[1] for label in rows}
    assert max(heaviest[label] for label in unsure) < min(heaviest[label] for label in sure)
    for g in groups(res, "residue"):
        row = next(r for label, r in rows.items() if f"A:{label.upper()}" == g.data["residue"])
        count = next(f.item.content for f in inside(res, g, Text) if f.item.content.startswith(f"{row['n_ligands']}/"))
        assert count.endswith(MARK) == (row["by_confidence"][0] + row["by_confidence"][1] == 0)
    assert "pocket pLDDT" in words(res) and res.confidence and len(res.confidence) == cmp_.n


# ---------------------------------------------------------------------------
# Honest degradation, determinism, the command line
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key", ["ar_ligands", "gr_diverse"])
def test_a_comparison_that_cannot_be_read_becomes_a_stated_table(key):
    cmp_ = comparison(key)
    res = cmp_.render(figure_width_mm=70.0)
    assert res.status == "unreadable" and res.figure == "summary" and res.svg is not None
    text = words(res)
    assert "not readable at 70 mm" in text and "listed per residue instead" in text
    assert any("not readable" in w and "summary" in w for w in res.warnings)
    assert not groups(res, "panel") and groups(res, "summary-table")
    for row in cmp_.consensus():
        assert row.label in text
    assert cmp_.render(figure_width_mm=70.0, on_unreadable="none").svg is None
    forced = cmp_.render(figure_width_mm=70.0, on_unreadable="render")
    assert forced.status == "unreadable" and forced.figure == ("difference" if cmp_.n == 2 else "consensus")


@pytest.mark.parametrize("view", ["difference", "consensus"])
def test_a_text_size_that_no_page_gives_is_said_so_in_a_comparison(view):
    """The notes of a comparison are 6.37 pt at full size, and no ``Metrics`` changes them: a
    ``min_font_pt`` above that is met on no page, and the result does not send the caller after one."""
    res = comparison("ar_ligands").render(view=view, min_font_pt=8)
    said = res.warnings[-1]
    assert res.status == "unreadable"
    assert "its smallest text is 6.37 pt at full size" in said and "no page gives it the 8 pt" in said
    assert "Ask for 6.3 pt or less: the text of this view has one size" in said
    assert "metrics" not in said and "mm wide" not in said and "cannot be set in one frame" not in said


def test_structures_of_different_proteins_are_not_superposed_and_a_flag_is_carried():
    """Two ligands in two proteins: no common frame is made, and each panel carries what was handed in about its
    model (an ipTM, a verdict). The structures have a pLDDT made up for the test; the ipTM values are handed in."""
    cmp_ = Comparison([predicted_diagram("ar_r1881", model_confidence={"iptm": 0.41}),
                       predicted_diagram("kit_imatinib", model_confidence={"iptm": 0.54}, trusted=False,
                                         flag="grade C")])
    assert cmp_.transforms[1] is None
    res = cmp_.render()
    assert res.status == "degraded" and res.data["frame"] == "own"
    assert any("could not be superposed" in w for w in res.warnings)
    (banner,) = groups(res, "model-flag")
    assert "grade C" in " ".join(f.item.content for f in inside(res, banner, Text))
    assert "ipTM 0.41" in words(res) and "ipTM 0.54" in words(res)       # each panel carries its model's evidence


def test_comparison_views_are_deterministic():
    def fresh(key, slugs):
        folder = dict(overlay_members(key))
        dias = []
        for slug in slugs:
            meta = overlay_meta(key, slug)
            rows = load_interactions(folder[slug] / "interactions.tsv")
            dias.append(Diagram.from_structure(folder[slug] / "model.cif.gz", meta["ligand"]["residue_name_in_cif"],
                                               meta["ligand"]["smiles"], rows, name=meta["ligand"]["name"]))
        return Comparison(dias)
    pair = [s for s, _ in overlay_members("ar_ligands")]
    assert fresh("ar_ligands", pair).render().svg == fresh("ar_ligands", pair).render().svg \
        == comparison("ar_ligands").render().svg
    four = [s for s, _ in overlay_members("gr_steroids")]
    one, two = fresh("gr_steroids", four), fresh("gr_steroids", four)
    assert one.render(theme="dark").svg == two.render(theme="dark").svg
    assert one.table.to_csv() == two.table.to_csv()
    assert one.render(theme="light").svg != one.render(theme="dark").svg


def test_command_line_compare_and_fingerprint(tmp_path, capsys):
    a, b = TESTDATA / "ar_r1881", TESTDATA / "ar_bicalutamide"
    out, fp = tmp_path / "diff.svg", tmp_path / "fp.csv"
    assert cli(["compare", str(a), str(b), "-o", str(out), "--fingerprints", str(fp), "-q"]) == 0
    assert out.read_text().startswith("<svg") and "No common scaffold" in out.read_text()
    assert FingerprintTable.from_csv(fp).names == ["metribolone (R1881)", "bicalutamide"]
    capsys.readouterr()
    assert cli(["fingerprint", str(a), str(b), "--form", "wide", "-q"]) == 0
    wide = capsys.readouterr().out
    # The first column is the contact with the lowest residue number in the two tables: R1881's Leu701.
    assert wide.splitlines()[0].startswith("ligand,A:LEU701:hydrophobic") and len(wide.splitlines()) == 3
    assert cli(["compare", str(a), "-o", str(out), "-q"]) == 2            # one ligand is not a comparison


# ---------------------------------------------------------------------------
# The scaffold a search is seeded with: RDKit's, and one made here where RDKit gives none
# ---------------------------------------------------------------------------

def _set_ligands():
    return [overlay_diagram(key, slug).ligand for key in SETS for slug, _ in overlay_members(key)]


def _as_made(mol):
    """A molecule as far as a substructure search, a SMILES and a reader of hydrogens can tell it apart."""
    return ([(a.GetSymbol(), a.GetIsAromatic(), str(a.GetChiralTag()), a.GetNumExplicitHs(), a.GetNoImplicit(),
              a.GetTotalNumHs(), a.GetProp("atomName")) for a in mol.GetAtoms()],
            [(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), str(b.GetBondType())) for b in mol.GetBonds()],
            [tuple(ring) for ring in mol.GetRingInfo().AtomRings()])


def test_the_scaffold_made_here_is_rdkits_atom_for_atom():
    from rdkit.Chem.Scaffolds import MurckoScaffold

    from bindview import align

    ligands = _set_ligands()
    try:
        MurckoScaffold.GetScaffoldForMol(ligands[0].mol)
    except RuntimeError as broken:                       # RDKit 2026.9.2: "bad any cast", for every molecule
        pytest.skip(f"this RDKit gives no scaffold to compare with ({str(broken).strip().splitlines()[0]})")
    assert len(ligands) == sum(len(overlay_members(key)) for key in SETS)
    for ligand in ligands:
        ours, theirs = align._framework(ligand.mol), MurckoScaffold.GetScaffoldForMol(ligand.mol)
        assert _as_made(ours) == _as_made(theirs)
        assert 0 < ours.GetNumAtoms() < ligand.mol.GetNumAtoms()             # every one has a ring, and more than rings


def test_a_comparison_does_not_fall_with_rdkits_scaffold(monkeypatch):
    from bindview import align

    pair, steroids, diverse = ([overlay_diagram(key, slug) for slug, _ in overlay_members(key)]
                               for key in ("ar_ligands", "gr_steroids", "gr_diverse"))
    cores = [align.find_common_core([d.ligand for d in members]) for members in (pair, steroids, diverse)]
    drawn = Comparison(pair).render().svg

    def broken(mol):
        raise RuntimeError("bad any cast")                # what Chem.MurckoDecompose says in RDKit 2026.9.2

    monkeypatch.setattr(align.MurckoScaffold, "GetScaffoldForMol", broken)
    assert [align.find_common_core([d.ligand for d in members]) for members in (pair, steroids, diverse)] == cores
    assert cores[1].substantial and not cores[2].substantial                  # a series is one, a diverse set is not
    assert Comparison(pair).render().svg == drawn
    # A molecule without a ring has no scaffold, here as there.
    from rdkit import Chem

    assert align._framework(Chem.MolFromSmiles("CCC(=O)O")).GetNumAtoms() == 0
    assert Chem.MolToSmiles(align._framework(Chem.MolFromSmiles("CC1CCC(=O)CC1Cc1ccccc1"))) == "O=C1CCCC(Cc2ccccc2)C1"


def _clashing(diagram: Diagram) -> Diagram:
    """The same complex without its template, and with three more atoms of the ligand on one of its ring oxygens,
    as a model can have them: only the connectivity is known, and the oxygen has five bonds."""
    from dataclasses import replace

    from bindview.ligand import build_ligand

    mol, names = diagram.ligand.mol, diagram.ligand.names
    oxygen = next(a for a in mol.GetAtoms() if a.GetSymbol() == "O" and a.IsInRing())
    near = {n.GetIdx() for n in oxygen.GetNeighbors()}
    others = [a.GetIdx() for a in mol.GetAtoms() if a.GetIdx() not in near | {oxygen.GetIdx()} and a.GetDegree() > 1]
    centre = diagram.ligand.coords3d[oxygen.GetIdx()]
    moved = dict(zip(others[-3:], centre + 1.35 * np.array([(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)])))
    atoms = {a.name: a for a in diagram.structure.residue_atoms(diagram.ligand_key)}
    made = []
    for k, name in enumerate(names):
        x, y, z = moved.get(k, diagram.ligand.coords3d[k])
        made.append(replace(atoms[name], x=float(x), y=float(y), z=float(z)))
    return Diagram(build_ligand(made, None), diagram.interactions, structure=diagram.structure,
                   ligand_key=diagram.ligand_key, name="with a clash")


def test_a_ligand_that_rdkit_gives_no_scaffold_of_does_not_take_the_comparison_with_it():
    """Of a ligand without a template only the connectivity is known, and nothing has checked its valences."""
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    from bindview import align
    from bindview.chem import quiet_rdkit

    good, odd = diagram_of("gr_dexamethasone"), _clashing(diagram_of("gr_budesonide"))      # an acetal oxygen
    assert odd.ligand.bond_source == "connectivity"
    assert max(a.GetDegree() for a in odd.ligand.mol.GetAtoms() if a.GetSymbol() == "O") == 5
    with quiet_rdkit(), pytest.raises((ValueError, RuntimeError)) as refused:
        MurckoScaffold.GetScaffoldForMol(odd.ligand.mol)
    if "bad any cast" not in str(refused.value):          # (RDKit 2026.9.2 gives a scaffold of no molecule at all)
        assert isinstance(refused.value, Chem.AtomValenceException)
    assert 0 < align._scaffold(odd.ligand.mol).GetNumAtoms() < odd.ligand.n_atoms
    for members in ([good, odd], [odd, good], [good, odd, diagram_of("gr_budesonide")]):
        made = Comparison(members)
        assert made.core.complete and len(made.core.atoms) == len(members)
        for view in ("auto", "consensus"):
            res = made.render(view)
            assert res.svg is not None and res.status in ("ok", "degraded")


def test_a_consensus_with_a_residue_that_found_no_place_says_so(monkeypatch):
    """A node of a consensus has a contact for each kind that the set makes, and a tether for each place it belongs to.
    A residue that found no place in the layout has no tether: the figure is a summary that says so, and drawn
    regardless the residue is a node of the graph without a place."""
    cmp = comparison("gr_steroids")
    lay_out = Comparison._lay_out
    lost: list[str] = []

    def crowded(self, m, bodies, obstacle, extra=()):
        layout, field = lay_out(self, m, bodies, obstacle, extra)
        body = next(b for b in bodies if b.n_ports != len(b.payload.row.contacts))
        layout.poses.pop(body.key, None)
        layout.unplaced.append(body.key)
        lost.append(body.payload.ident)
        return layout, field

    monkeypatch.setattr(Comparison, "_lay_out", crowded)
    cmp._cache.clear()
    try:
        res = cmp.render("consensus")
        assert lost and res.status == "unreadable" and res.figure == "summary"
        assert "1 residue(s) could not be placed" in " ".join(res.warnings)
        drawn = cmp.render("consensus", on_unreadable="render")
        assert drawn.figure == "consensus" and "R:" + lost[-1] in {n.id for n in drawn.graph.nodes}
        assert drawn.graph.find("R:" + lost[-1]).x is None
    finally:
        cmp._cache.clear()


# ---------------------------------------------------------------------------
# The time limit of the substructure search
# ---------------------------------------------------------------------------

#: Azithromycin: a ring of fifteen atoms and two sugars with nineteen bonds leaving them, one of them a double bond.
_MACROLIDE = ("CC[C@@H]1[C@@]([C@@H]([C@H](N(C[C@@H](C[C@@]([C@@H]([C@H]([C@@H]([C@H](C(=O)O1)C)O[C@H]2C[C@@]([C@H]"
              "([C@@H](O2)C)O)(C)OC)C)O[C@H]3[C@@H]([C@H](C[C@H](O3)C)N(C)C)O)(C)O)C)C)C)O)(C)O")
_SEARCH_WITH_A_LIMIT = """
import json, sys, time
import numpy as np
from rdkit import Chem
from bindview import align
from bindview.ligand import Ligand

align._TIMEOUT = 2
drawn = Chem.MolFromSmiles(sys.argv[1])
bare = Chem.RWMol(drawn)                         # the same atoms as a file without hydrogens gives them: all bonds single
for bond in bare.GetBonds():
    bond.SetBondType(Chem.BondType.SINGLE)
for atom in bare.GetAtoms():
    atom.SetNoImplicit(True)
    atom.SetNumExplicitHs(0)
    atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
bare = bare.GetMol()
bare.UpdatePropertyCache(strict=False)
Chem.GetSymmSSSR(bare)
names = [f"{a.GetSymbol()}{a.GetIdx() + 1}" for a in drawn.GetAtoms()]
pair = [Ligand(mol=mol, names=names, coords3d=np.zeros((len(names), 3)), bond_source=source)
        for mol, source in ((drawn, "template"), (bare, "connectivity"))]
start = time.process_time()
core = align.find_common_core(pair)
print(json.dumps({"seconds": time.process_time() - start, "atoms": core.n_atoms, "of": min(core.sizes),
                  "complete": core.complete, "aligned": core.substantial}))
"""


def test_the_substructure_search_is_back_at_its_time_limit():
    """A ligand against itself as a file without bond orders gives it. The rings of the two are the seed of the search,
    and RDKit tries every subset of the bonds that leave a seed before it reads its clock again: with nineteen of
    them that takes minutes, and the search has to come back at its limit all the same. Its time is the CPU time
    of the search, which a busy machine does not lengthen. (In a process of its own, so that a search that does
    not come back fails here and does not stop the tests.)"""
    import json
    import os
    import subprocess
    import sys

    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
    try:
        done = subprocess.run([sys.executable, "-c", _SEARCH_WITH_A_LIMIT, _MACROLIDE], capture_output=True, text=True,
                              timeout=300, env=env)
    except subprocess.TimeoutExpired:                    # (the process has to start, too, on a machine that is busy)
        pytest.fail("a search with a limit of 2 s was not back after 300 s")
    assert done.returncode == 0, done.stderr
    found = json.loads(done.stdout.strip().splitlines()[-1])
    assert found["seconds"] < 60                          # three searches of 2 s at most, ten times over
    # The search that was stopped is the seeded one; the unseeded one that follows ends by itself, with all but the
    # oxygen of the double bond.
    assert (found["atoms"], found["of"], found["complete"], found["aligned"]) == (51, 52, True, True)


def test_a_core_that_is_too_small_is_refused_for_its_size_and_not_for_its_share():
    """Two ligands of two atoms share both, which is all of the smaller one: what is missing is atoms, not a share."""
    twice = Comparison([diagram_of("hb_carbon_monoxide"), diagram_of("hb_carbon_monoxide")])
    core = twice.core
    assert (core.n_atoms, min(core.sizes), core.fraction, core.substantial) == (2, 2, 1.0, False)
    said = core.describe()
    assert said.startswith("No common scaffold: ") and "(100 %)" in said
    assert f"fewer than the {core.min_atoms} atoms needed to align on" in said
    assert "below the" not in said
    # Where the share is what is missing, that is the reason.
    apart = Comparison([overlay_diagram("ar_ligands", slug) for slug, _ in overlay_members("ar_ligands")]).core
    assert not apart.substantial and apart.fraction < apart.min_fraction
    assert f"below the {100 * apart.min_fraction:.0f} % needed to align on" in apart.describe()


# ---------------------------------------------------------------------------
# What a comparison says of members that do not fit the others
# ---------------------------------------------------------------------------

def _steroids() -> list[Diagram]:
    return [overlay_diagram("gr_steroids", slug) for slug, _ in overlay_members("gr_steroids")]


def _from_connectivity(slug: str) -> Diagram:
    """A case without its SMILES. Its file is an entry of the PDB, which has neither a SMILES nor hydrogens: only
    which atoms are bonded is known, and every bond is drawn and compared as a single one."""
    m = meta_of(slug)
    return Diagram.from_structure(model_of(slug), ligand=m["ligand"]["residue_name_in_cif"], smiles=None,
                                  interactions=load_interactions(TESTDATA / slug / "interactions.tsv"),
                                  name=m["ligand"]["name"])


def test_a_member_without_bond_orders_is_named_where_the_scaffold_is_decided():
    """One steroid of four without a SMILES: its double bonds match nothing, the common substructure loses a ring
    and, at a stricter share, is lost. The member is named in the figure, in the data and in a warning."""
    whole = Comparison(_steroids())
    assert whole.core.n_atoms == 24 and whole.alignment()["without_bond_orders"] == []
    assert "bond orders" not in words(whole.render()) + " ".join(whole.render().warnings)
    slug = overlay_members("gr_steroids")[1][0]
    members = _steroids()
    members[1] = _from_connectivity(slug)
    for share, frame, atoms in ((0.6, "core", 18), (0.75, "envelope", 18)):
        mixed = Comparison(members, min_fraction=share)
        name = mixed.names[1]
        assert mixed.diagrams[1].ligand.bond_source == "connectivity" and name == meta_of(slug)["ligand"]["name"]
        assert mixed.core.n_atoms == atoms and mixed.core.substantial == (frame == "core")
        assert mixed.alignment()["without_bond_orders"] == [name]
        assert name in mixed.alignment()["statement"] and "no bond orders" in mixed.alignment()["statement"]
        res = mixed.render()
        assert (res.status, res.data["frame"]) == ("degraded", frame)
        said = [s for s in re.split(r"(?<=\.) ", words(res)) if "no bond orders" in s]
        assert len(said) == 1 and said[0].startswith(f"{name} has no bond orders") and "common substructure" in said[0]
        told = [w for w in res.warnings if "no bond orders" in w]
        assert len(told) == 1 and told[0].startswith(f"{name}: ") and "common substructure" in told[0]
    two = Comparison([members[1], _from_connectivity(overlay_members("gr_steroids")[2][0])] + _steroids()[:1])
    assert two.alignment()["without_bond_orders"] == two.names[:2]
    assert f"{two.names[0]} and {two.names[1]} have no bond orders" in words(two.render())


def test_a_consensus_without_a_scaffold_puts_each_residue_where_it_is_contacted_whatever_the_table_calls_its_chain():
    """A table may leave the chain empty where the residue number says it all. The structure resolves the chain, and
    the residues of the figure are keyed on it: so must be the places where they are contacted."""
    from bindview import read_structure

    def drawn(blank: bool):
        members = []
        for slug, _ in overlay_members("ar_ligands"):
            m = meta_of(slug)
            model = read_structure(model_of(slug))
            rows = load_interactions(TESTDATA / slug / "interactions.tsv")
            assert all(r.chain == "A" for r in rows)
            if blank:                                    # (but for a number that the peptide of chain B carries too)
                twice = {key.number for key in model.residues() if key.chain != "A"}
                rows = [r if r.residue_number in twice else r.with_(chain="") for r in rows]
                assert sum(r.chain == "" for r in rows) >= len(rows) - 1
            members.append(Diagram.from_structure(model, ligand=m["ligand"]["residue_name_in_cif"],
                                                  smiles=m["ligand"]["smiles"], interactions=rows,
                                                  name=m["ligand"]["name"]))
        return Comparison(members).render("consensus")

    def places(res) -> dict:
        return {n.id: (tuple(n.attrs["anchor_point"]), n.x, n.y) for n in res.graph.nodes if n.kind == "residue"}

    given, blank = drawn(False), drawn(True)
    assert given.data["frame"] == blank.data["frame"] == "envelope" and "stands where it is contacted" in words(blank)
    # Two residues are contacted at one place (the same atoms): equal to the last digits on one platform, to a
    # millionth on another.
    anchors = {(round(x, 6), round(y, 6)) for (x, y), _, _ in places(given).values()}
    assert len(places(given)) == 14 and len(anchors) == 13
    assert places(blank) == places(given)
    assert blank.svg == given.svg


def _changed(slug: str, by: int = 0, rename: "dict[tuple[str, int], str] | None" = None) -> Diagram:
    """A case with the residues of its protein numbered ``by`` higher, in the model and in the table alike, and with
    another name for the residues of ``rename`` (by chain and new number)."""
    from dataclasses import replace

    from bindview import read_structure

    rename = rename or {}
    m = meta_of(slug)
    model = read_structure(model_of(slug))
    atoms = [a if a.hetero else replace(a, resnum=a.resnum + by, resname=rename.get((a.chain, a.resnum + by), a.resname))
             for a in model.atoms]
    rows = [r.with_(residue_number=r.residue_number + by,
                    residue_name=rename.get((r.chain, r.residue_number + by), r.residue_name))
            for r in load_interactions(TESTDATA / slug / "interactions.tsv")]
    return Diagram.from_structure(replace(model, atoms=atoms), ligand=m["ligand"]["residue_name_in_cif"],
                                  smiles=m["ligand"]["smiles"], interactions=rows, name=m["ligand"]["name"])


def test_a_member_that_numbers_its_residues_otherwise_is_named_on_a_common_scaffold_too():
    """Residues are compared by chain and number. Two steroids on one scaffold, the second numbered a thousand higher:
    nothing is shared, and the figure says why, in a warning and in its text, and is not ok."""
    slugs = [slug for slug, _ in overlay_members("gr_steroids")]
    first, moved = overlay_diagram("gr_steroids", slugs[0]), _changed(slugs[1], by=1000)
    pair = Comparison([first, moved])
    assert pair.core.substantial and pair.pocket_rmsd == [0.0, None] and pair.transforms[1] is None
    res = pair.render()
    assert res.data["frame"] == "core" and not pair.difference().shared
    assert res.status == "degraded"
    (told,) = [w for w in res.warnings if "residue numbering" in w]
    assert told.startswith(f"{pair.names[1]}: ") and pair.names[0] in told and "counted as different residues" in told
    (said,) = [s for s in re.split(r"(?<=\.) ", words(res)) if "does not number" in s]
    assert said.startswith(f"{pair.names[1]} does not number the residues of the pocket as {pair.names[0]} does")
    three = Comparison([first, moved, _changed(slugs[2], by=1000)])
    res = three.render("consensus")
    assert (res.status, res.data["frame"]) == ("degraded", "core")
    (told,) = [w for w in res.warnings if "residue numbering" in w]
    assert told.startswith(f"{three.names[1]}, {three.names[2]}: ")
    assert f"{three.names[1]} and {three.names[2]} do not number the residues of the pocket" in words(res)
    # Without a scaffold either: a ligand of one protein and a ligand of another.
    apart = Comparison([diagram_of("hb_carbon_monoxide"), diagram_of("kit_imatinib")])
    assert not apart.core.substantial and apart.pocket_rmsd == [0.0, None]
    for view, frame in (("difference", "own"), ("consensus", "sequence")):
        res = apart.render(view)
        assert (res.status, res.data["frame"]) == ("degraded", frame)
        assert sum("residue numbering" in w for w in res.warnings) == 1
        assert "do not share a residue numbering" in words(res)
    # And where the numbering is one, nothing of it is said.
    res = Comparison([first, overlay_diagram("gr_steroids", slugs[1])]).render()
    assert res.status == "ok" and "number" not in words(res) + " ".join(res.warnings)


def test_a_residue_with_two_names_is_one_residue_in_the_figure_and_in_its_data():
    """A point mutant among the structures: one residue number, two names. A comparison goes by chain and number,
    so it is one residue: the bits, ``Difference.residues()``, the counts of the header and the nodes say the same."""
    slugs = [slug for slug, _ in overlay_members("gr_steroids")]
    first = overlay_diagram("gr_steroids", slugs[0])
    plain = Comparison([first, overlay_diagram("gr_steroids", slugs[1])]).difference()
    bit = plain.shared[0]                                 # a contact of the same kind with both ligands
    other = "TRP" if bit.resname != "TRP" else "PHE"
    pair = Comparison([first, _changed(slugs[1], rename={(bit.chain, bit.number): other})])
    assert {b.resname for fp in pair.fingerprints for b in fp.bits if b.residue == bit.residue} == {bit.resname, other}
    diff = pair.difference()
    assert [b.column for b in diff.shared] == [b.column for b in plain.shared]          # under the first one's name
    assert [(b.residue, b.type) for b in diff.only_a] == [(b.residue, b.type) for b in plain.only_a]
    assert [(b.residue, b.type) for b in diff.only_b] == [(b.residue, b.type) for b in plain.only_b]
    assert diff.tanimoto == plain.tanimoto and diff.residues() == plain.residues()
    assert diff.residues()[bit.residue] == "shared"

    res = pair.render()
    text = words(res)
    for k, only in enumerate((diff.only_a, diff.only_b)):
        n = len(diff.shared) + len(only)
        assert f"{n} contacts (residue × type): {len(diff.shared)} shared, {len(only)} only this ligand" in text
    assert res.data["difference"]["shared"] == [b.column for b in diff.shared]
    assert res.data["difference"]["residues"][f"{bit.chain}:{bit.number}"] == "shared"
    panels = groups(res, "panel")
    named = []
    for panel in panels:
        (node,) = [g for g in groups(res, "residue") if g.data["residue-number"] == str(bit.number)
                   and any(any(p is panel for p in f.path) for f in inside(res, g))]
        assert node.data["contact"] == "shared"
        named.append([f.item.content for f in inside(res, node, Text)][0])
    assert named == [bit.resname, other]                  # each panel names the residue as its own structure does
    marks = [g.data["contact"] for g in all_groups(res, "interaction") + all_groups(res, "contact")
             if g.data["residue-number"] == str(bit.number) and g.data["type"] == bit.type.value]
    assert marks == ["shared", "shared"]
    (told,) = [w for w in res.warnings if "two names" in w]
    assert f"{bit.chain}:{bit.number}" in told and bit.resname in told and other in told
    assert pair.names[0] in told and pair.names[1] in told and "one residue" in told
    # The consensus view counts it once as well, and says so.
    rows = [r for r in pair.consensus() if r.residue == bit.residue]
    assert len(rows) == 1 and rows[0].n_ligands == 2
    assert sum("two names" in w for w in pair.render("consensus").warnings) == 1
    # A set in which every residue has one name says nothing of names.
    assert not [w for w in Comparison(_steroids()).render().warnings if "two names" in w]


def test_what_a_caller_does_to_a_result_does_not_reach_the_next_one():
    """The views of a comparison are kept, and a result is what the caller works on: its data are its own."""
    import copy

    for key, view in (("ar_ligands", "difference"), ("gr_steroids", "consensus")):
        cmp_ = Comparison([overlay_diagram(key, slug) for slug, _ in overlay_members(key)])
        first = cmp_.render(view)
        kept = copy.deepcopy(first.data)
        first.data["alignment"]["aligned"] = "changed by the caller"
        first.data["alignment"]["pocket_rmsd"].clear()
        (first.data["difference"]["shared"] if view == "difference" else first.data["consensus"]).clear()
        assert first.data != kept
        again = cmp_.render(view)
        assert again.data == kept and again.svg == first.svg


def _flat(smiles: str, source: str = "template"):
    """A ligand from a SMILES, as the search for a common core takes it: without a structure behind it."""
    from rdkit import Chem

    from bindview.ligand import Ligand

    mol = Chem.MolFromSmiles(smiles)
    if source == "connectivity":                            # what is known without a SMILES and without hydrogens
        bare = Chem.RWMol()
        for a in mol.GetAtoms():
            bare.AddAtom(Chem.Atom(a.GetAtomicNum()))
        for b in mol.GetBonds():
            bare.AddBond(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), Chem.BondType.SINGLE)
        mol = bare.GetMol()
        mol.UpdatePropertyCache(strict=False)
        Chem.GetSymmSSSR(mol)
    names = [f"{a.GetSymbol()}{a.GetIdx() + 1}" for a in mol.GetAtoms()]
    return Ligand(mol=mol, names=names, coords3d=np.zeros((len(names), 3)), bond_source=source)


@pytest.mark.parametrize("one, other, shared", [
    ("c1ccccc1", "C1CCCCC1", 0),                            # benzene and cyclohexane: no bond alike
    ("Cc1ccccc1", "CC1CCCCC1", 2),                          # the methyl and the atom it is on: no bond of the ring
    ("CCc1ccccc1", "OCc1ccccc1", 7),                        # the phenyl ring and the CH2 on it
    ("C1=CC=CC=C1", "c1ccccc1", 6),                         # a ring written with double bonds is the same ring
    ("c1ccc2[nH]ccc2c1", "C1=CC=C2C(=C1)NC=C2", 9),
    ("CC(=O)O", "CC(O)O", 3),                               # a double bond is no single bond: the C, C and one O remain
])
def test_an_aromatic_ring_is_no_saturated_ring_to_the_search(one, other, shared):
    from bindview.align import find_common_core

    core = find_common_core([_flat(one), _flat(other)])
    assert core.n_atoms == shared and core.complete


def test_a_ligand_without_bond_orders_is_matched_by_its_bare_graph():
    """Where one ligand of a set is known by its connectivity alone, an aromatic ring of another is found in it, as
    a ring of the same atoms: its bonds are all single ones to the search."""
    from bindview.align import find_common_core

    bare = _flat("CCc1ccccc1", "connectivity")
    assert find_common_core([_flat("CCc1ccccc1"), bare]).n_atoms == 8
    assert find_common_core([_flat("CCc1ccccc1"), _flat("OCc1ccccc1")]).n_atoms == 7
    assert find_common_core([_flat("CCc1ccccc1"), _flat("CCC1CCCCC1"), bare]).n_atoms == 8
