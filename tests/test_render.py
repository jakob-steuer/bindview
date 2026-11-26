"""Acceptance tests on what is actually drawn.

The scene graph is inspected in page coordinates, so these tests check the
picture rather than the intentions behind it: where a link ends, whether two
labels touch, whether a hydrophobic contact got a line it must not have.
"""

from __future__ import annotations

import copy
import gzip
import math
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from conftest import (AP5A, BIOEMU, CASES, ROOT, STYLES, TESTDATA, diagram_of, fresh_diagram, invented_plddt, meta_of,
                      model_of, one_residue_peptide, render_of, structure_of)
from bindview import Diagram, Interaction, load_interactions
from bindview.errors import BindviewError, DetectionUnavailableError, LigandNotFoundError, RenderError
from bindview.scene import Circle, Group, Line, Path, Polyline, Text
from scenetools import groups, inside, interactions, overlap, segment_hits_box, texts_with_owner, walk, words

#: The reference case: c-Kit with imatinib (PDB 1T46), whose table has hydrogen bonds, ring stacking, a ring over a
#: charged group, salt bridges, water bridges and hydrophobic contacts.
REFERENCE = "kit_imatinib"


def dist(p, q):
    return math.hypot(p[0] - q[0], p[1] - q[1])


def table(case: str) -> list[Interaction]:
    """The rows of a case's committed table (``interactions.tsv``, written by PLIP)."""
    return load_interactions(TESTDATA / case / "interactions.tsv")


def who(row: Interaction) -> str:
    """A row's residue as a figure names it in its data (``A:GLN711``)."""
    return f"{row.chain}:{row.residue_name}{row.residue_number}"


def target_point(result, group):
    pts = [result.atoms[n] for n in group.data["ligand-atoms"].split(",")]
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


def link_points(result, group):
    """Every end point of the drawn link: its line pieces and its arrowhead."""
    pts = [p for f in inside(result, group, Line) for p in f.segment()]
    for f in inside(result, group):
        if type(f.item).__name__ == "Polygon":
            pts.extend((x + f.dx, y + f.dy) for x, y in f.item.points)
    return pts


def ligand_end(result, group):
    """The end of the drawn link that is nearest its ligand target."""
    tgt = target_point(result, group)
    return min(link_points(result, group), key=lambda p: dist(p, tgt)), tgt


def nearest_atom(result, point):
    return min(result.atoms, key=lambda n: dist(result.atoms[n], point))


def ring_systems(mol) -> list[set[int]]:
    """The atoms of each ring system of a molecule: rings that share an atom are one system."""
    systems: list[set[int]] = []
    for ring in map(set, mol.GetRingInfo().AtomRings()):
        for s in [s for s in systems if s & ring]:
            systems.remove(s)
            ring |= s
        systems.append(ring)
    return systems


def box_of(result, ligand, idx):
    pts = [result.atoms[ligand.names[i]] for i in idx]
    return (min(p[0] for p in pts), min(p[1] for p in pts), max(p[0] for p in pts), max(p[1] for p in pts))


def label_boxes(result) -> dict[str, tuple]:
    """The box of the text written on each atom of the ligand that has some (its runs joined)."""
    boxes: dict[str, tuple] = {}
    for f in walk(result.scene):
        atom = f.in_class("atom")
        if atom is not None and isinstance(f.item, Text):
            b, old = f.box(), boxes.get(atom.data["atom"])
            boxes[atom.data["atom"]] = b if old is None else (min(old[0], b[0]), min(old[1], b[1]),
                                                              max(old[2], b[2]), max(old[3], b[3]))
    return boxes


_kept: dict = {}


def ap5a():
    """AP5A in the first of the twenty BioEmu models of adenylate kinase, with the plain distance table: a long ligand
    of two halves joined by a chain of five phosphates."""
    if "ap5a" not in _kept:
        from bindview import read_models
        from bindview.distance_table import ligand_contacts

        model = read_models(BIOEMU)[0]
        dia = Diagram.from_structure(model, "AP5", smiles=AP5A, interactions=ligand_contacts(model, "AP5", AP5A))
        _kept["ap5a"] = (dia, dia.render())
    return _kept["ap5a"]


def zinc() -> Diagram:
    """The zinc of carbonic anhydrase II as the ligand: one atom, no SMILES, and the plain distance table."""
    if "zinc" not in _kept:
        _kept["zinc"] = Diagram.from_structure(model_of("ca2_celecoxib"), ligand="ZN", interactions="distances")
    return _kept["zinc"]


def haem() -> Diagram:
    """The haem of haemoglobin (2DN3, chain A) as the ligand: 43 atoms round an iron, built from the connectivity of
    the file (no SMILES), with the plain distance table."""
    if "haem" not in _kept:
        _kept["haem"] = Diagram.from_structure(model_of("hb_carbon_monoxide"), ligand="A:142", smiles=None,
                                               interactions="distances")
    return _kept["haem"]


def crowded():
    """The nonamer of HLA-A2 (3MRE) written as one hetero residue with its SMILES, with the plain distance table, and its
    detailed figure with ``effort="quick"``: too crowded to be drawn, so a table is drawn instead."""
    if "crowded" not in _kept:
        from bindview.distance_table import ligand_contacts

        model, smiles = one_residue_peptide("mhc_peptide")
        dia = Diagram.from_structure(model, "LIG", smiles=smiles, interactions=ligand_contacts(model, "LIG", smiles))
        _kept["crowded"] = (dia, dia.render(effort="quick"))
    return _kept["crowded"]


def residues_near(slug: str, dia: Diagram, cutoff: float = 4.5) -> list[tuple[str, int]]:
    """``(name, number)`` of every residue with a heavy atom within ``cutoff`` Å of a heavy atom of the diagram's
    ligand, in the order of the chain: worked out here from the atoms of the model."""
    key = (dia.ligand_key.chain, dia.ligand_key.number, dia.ligand_key.icode)
    heavy = [a for a in structure_of(slug).atoms if a.element not in ("H", "D")]
    mine = np.array([(a.x, a.y, a.z) for a in heavy if (a.chain, a.resnum, a.icode) == key])
    found = {(a.chain, a.resnum, a.icode, a.resname) for a in heavy if (a.chain, a.resnum, a.icode) != key
             and float(np.linalg.norm(mine - (a.x, a.y, a.z), axis=1).min()) <= cutoff}
    return [(name, number) for _, number, _, name in sorted(found)]


# ---------------------------------------------------------------------------
# Status: every case ends in the state it should
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("style", STYLES)
@pytest.mark.parametrize("case", CASES)
def test_status(case, style):
    """Every single case of ``testdata/cases.json`` is drawn whole, in both styles."""
    res = render_of(case, style)
    assert res.status == "ok"
    assert res.svg is not None and res.svg.startswith("<svg ")
    ET.fromstring(res.svg)                                    # well-formed XML


@pytest.mark.parametrize("style", STYLES)
@pytest.mark.parametrize("case", CASES)
def test_layout_is_valid(case, style):
    q = render_of(case, style).quality
    assert q.overlaps == 0 and q.unplaced == 0 and q.link_hits == 0 and q.valid
    assert q.readable and q.min_font_pt >= 5.0
    assert q.n_residues == len({r.record.residue_id for r in diagram_of(case).interactions})


# ---------------------------------------------------------------------------
# Hard requirement 2: nothing overlaps
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("style", STYLES)
@pytest.mark.parametrize("case", CASES)
def test_no_text_overlaps_other_text(case, style):
    texts = texts_with_owner(render_of(case, style))
    clashes = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            if texts[i][1] != texts[j][1] and overlap(texts[i][0], texts[j][0]):
                clashes.append((texts[i][2].item.content, texts[j][2].item.content))
    assert not clashes


def _residue_label_boxes(result):
    """Box of each residue's name: the text in the detailed style, the square
    inscribed in the bubble's disc in the schematic style."""
    out = {}
    for g in groups(result, "residue"):
        if result.style == "schematic":
            cx, cy, r = _bubble(result, g)
            h = r / math.sqrt(2.0)
            out[g.data["residue"]] = (cx - h, cy - h, cx + h, cy + h)
        else:
            # The name is the residue's own text; atom names live in its fragment sub-group.
            (label,) = [f for f in inside(result, g, Text) if f.in_class("fragment") is None]
            out[g.data["residue"]] = label.box()
    return out


def _bubble(result, group):
    """Centre and disc radius of a schematic bubble (its teardrop reaches 1.3 radii)."""
    path = inside(result, group, Path)[0]          # the fill; a low-confidence bubble adds an outline
    b = path.box()
    return (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0, (b[2] - b[0]) / 2.0 / 1.3


@pytest.mark.parametrize("style", STYLES)
@pytest.mark.parametrize("case", CASES)
def test_residue_labels_keep_off_each_other_and_off_the_ligand(case, style):
    res = render_of(case, style)
    boxes = _residue_label_boxes(res)
    names = sorted(boxes)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not overlap(boxes[a], boxes[b]), f"{a} overlaps {b}"
    (ligand,) = groups(res, "ligand")
    bonds = [f.segment() for f in inside(res, ligand, Line)]
    labels = [f.box() for f in inside(res, ligand, Text)]
    for name, box in boxes.items():
        assert not any(segment_hits_box(p, q, box, pad=0.75) for p, q in bonds), f"{name} lies on a ligand bond"
        assert not any(overlap(box, lab) for lab in labels), f"{name} lies on a ligand atom label"


@pytest.mark.parametrize("case", CASES)
def test_schematic_bubbles_do_not_touch(case):
    res = render_of(case, "schematic")
    discs = [_bubble(res, g) for g in groups(res, "residue")]
    for i, a in enumerate(discs):
        for b in discs[i + 1:]:
            assert dist(a[:2], b[:2]) >= a[2] + b[2] + 3.0


@pytest.mark.parametrize("style", STYLES)
@pytest.mark.parametrize("case", CASES)
def test_links_do_not_run_through_other_residues(case, style):
    res = render_of(case, style)
    boxes = _residue_label_boxes(res)
    for g in groups(res, "interaction"):
        for f in inside(res, g, Line):
            p, q = f.segment()
            for name, box in boxes.items():
                if name != g.data["residue"]:
                    assert not segment_hits_box(p, q, box, pad=1.0), f"{g.data['residue']} link crosses {name}"


@pytest.mark.parametrize("style", STYLES)
@pytest.mark.parametrize("case", CASES)
def test_residue_drawings_never_cross_the_ligand_skeleton(case, style):
    """Fragment bonds, eyelash arcs and bubbles keep clear of every ligand bond.

    (Interaction *links* may cross a bond when the atom they go to is on the far
    side of one; that is counted in ``quality.bond_crossings``, not forbidden.)
    """
    from bindview.geometry import point_segment_distance, segments_intersect

    res = render_of(case, style)
    (ligand,) = groups(res, "ligand")
    bonds = [f.segment() for f in inside(res, ligand, Line)]
    for g in groups(res, "residue"):
        segs = [f.segment() for f in inside(res, g, Line)]
        for f in inside(res, g, Polyline):
            pts = [(x + f.dx, y + f.dy) for x, y in f.item.points]
            segs.extend(zip(pts, pts[1:]))
        for p, q in segs:
            assert not any(segments_intersect(p, q, a, b) for a, b in bonds), g.data["residue"]
        if style == "schematic":
            cx, cy, r = _bubble(res, g)
            for a, b in bonds:
                assert point_segment_distance((cx, cy), a, b) > r + 5.0, g.data["residue"]


@pytest.mark.parametrize("case", CASES)
def test_every_residue_of_the_table_is_drawn_in_the_form_its_contacts_ask_for(case):
    """In the detailed style every residue of the table is drawn once: as an eyelash where all its contacts are
    hydrophobic, with its atoms where any is not (the steroid R1881 in the androgen receptor is the densest packing)."""
    kinds: dict[str, set] = {}
    for row in table(case):
        kinds.setdefault(who(row), set()).add(row.type.value)
    forms = [(g.data["residue"], g.data["form"]) for g in groups(render_of(case), "residue")]
    assert sorted(forms) == sorted((r, "eyelash" if k == {"hydrophobic"} else "fragment") for r, k in kinds.items())


# ---------------------------------------------------------------------------
# Hard requirement 1, on the page: links end where the chemistry says
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("style", STYLES)
def test_a_halogen_bond_ends_on_its_halogen(style):
    """Bicalutamide's one halogen bond in its table: a fluorine of its CF3 group and the backbone oxygen of Met742."""
    res = render_of("ar_bicalutamide", style)
    (row,) = [r for r in table("ar_bicalutamide") if r.type.value == "halogen"]
    (g,) = interactions(res, "halogen")
    assert g.data["ligand-atoms"] == row.ligand_atom and g.data["residue"] == who(row)
    end, tgt = ligand_end(res, g)
    assert dist(end, tgt) < 13.0                      # stops at the F label, a few units from its centre
    assert nearest_atom(res, end) == row.ligand_atom  # and not at the carbon or at another fluorine of the group


@pytest.mark.parametrize("style", STYLES)
def test_an_hbond_points_away_from_its_donor(style):
    """Imatinib's four hydrogen bonds: where the protein donates the arrowhead is at the ligand's atom, where the
    ligand donates it is not."""
    res = render_of(REFERENCE, style)
    rows = [r for r in table(REFERENCE) if r.type.value == "hbond"]
    assert {r.ligand_is_donor for r in rows} == {True, False}
    for row in rows:
        (g,) = [g for g in interactions(res, "hbond", row.residue_name + str(row.residue_number))
                if g.data["ligand-atoms"] == row.ligand_atom]
        end, _ = ligand_end(res, g)
        assert nearest_atom(res, end) == row.ligand_atom
        (head,) = [f for f in inside(res, g) if type(f.item).__name__ == "Polygon"]
        tip = (head.item.points[0][0] + head.dx, head.item.points[0][1] + head.dy)
        assert (dist(tip, end) < 1e-6) is (row.ligand_is_donor is False), who(row)


@pytest.mark.parametrize("style", STYLES)
@pytest.mark.parametrize("case", [REFERENCE, "dhfr_methotrexate"])
def test_every_hbond_ends_on_its_own_atom(case, style):
    res = render_of(case, style)
    hb = interactions(res, "hbond")
    labels = label_boxes(res)
    for g in hb:
        end, tgt = ligand_end(res, g)
        name = g.data["ligand-atoms"]
        assert nearest_atom(res, end) == name, g.data
        box = labels[name]                                 # every atom of these that makes an H-bond is a labelled N or O
        assert box[0] - 6 <= end[0] <= box[2] + 6 and box[1] - 6 <= end[1] <= box[3] + 6, "stops at the atom's label"
    got = sorted((g.data["residue"].split(":")[1], g.data["ligand-atoms"]) for g in hb)
    rows = [(r.residue_name + str(r.residue_number), r.ligand_atom) for r in table(case) if r.type.value == "hbond"]
    if style == "detailed":
        assert got == sorted(rows)                         # one link per record: protein atom to ligand atom
    else:
        assert got == sorted(set(rows))                    # a bubble is the whole residue: one line to each atom


def test_two_hbonds_on_one_oxygen_stay_legible():
    """R1881's keto oxygen takes two hydrogen bonds in its table, from Gln711 and from Arg752."""
    res = render_of("ar_r1881")
    rows = [r for r in table("ar_r1881") if r.type.value == "hbond"]
    (oxygen,) = [a for a in {r.ligand_atom for r in rows} if sum(r.ligand_atom == a for r in rows) == 2]
    pair = [r for r in rows if r.ligand_atom == oxygen]
    links = [g for r in pair for g in interactions(res, "hbond", r.residue_name + str(r.residue_number))]
    assert len(links) == 2 and all(g.data["ligand-atoms"] == oxygen for g in links)
    o = res.atoms[oxygen]
    ends = []
    for g in links:
        end, tgt = ligand_end(res, g)
        assert nearest_atom(res, end) == oxygen
        far = max((p for f in inside(res, g, Line) for p in f.segment()), key=lambda p: dist(p, o))
        ends.append(math.atan2(far[1] - o[1], far[0] - o[0]))
    spread = abs((ends[0] - ends[1] + math.pi) % (2 * math.pi) - math.pi)
    assert spread > math.radians(25), "the two links fan out instead of lying on top of each other"
    labels = [f for g in links for f in inside(res, g, Text)]
    assert sorted(f.item.content for f in labels) == sorted(f"{r.distance:.2f}" for r in pair)
    assert not overlap(labels[0].box(), labels[1].box())


def test_a_long_ligand_keeps_its_parts_apart():
    """AP5A: its two adenines, its two riboses and its chain of phosphates do not sit on top of each other."""
    dia, res = ap5a()
    lig = dia.ligand
    systems = ring_systems(lig.mol)
    parts = {f"adenine {k}": s for k, s in enumerate(s for s in systems if len(s) == 9)}
    parts |= {f"ribose {k}": s for k, s in enumerate(s for s in systems if len(s) == 5)}
    parts["phosphates"] = {i for i in range(lig.n_atoms) if lig.element(i) == "P"}
    assert len(parts) == 5
    names = sorted(parts)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not overlap(box_of(res, lig, parts[a]), box_of(res, lig, parts[b]), slack=0.0), f"{a} overlaps {b}"


@pytest.mark.parametrize("style", STYLES)
def test_pi_links_end_on_ring_centroids(style):
    """Imatinib's ring stacking and its ring over a lysine end at the centre of the ring, which is marked; a ring of
    the protein over a charged nitrogen of the ligand ends at that nitrogen."""
    res = render_of(REFERENCE, style)
    pis = interactions(res, "pi_stacking") + interactions(res, "pi_cation")
    rows = [r for r in table(REFERENCE) if r.type.value in ("pi_stacking", "pi_cation")]
    assert sorted((g.data["type"], g.data["residue"]) for g in pis) == sorted((r.type.value, who(r)) for r in rows)
    for g in pis:
        ring = g.data["ligand-atoms"].split(",")
        (row,) = [r for r in rows if who(r) == g.data["residue"] and r.type.value == g.data["type"]]
        assert set(ring) == set(row.ligand_atoms)
        end, centroid = ligand_end(res, g)
        if len(ring) == 1:
            assert nearest_atom(res, end) == ring[0]
            continue
        assert len(ring) == 6 and dist(end, centroid) < 5.0
        assert all(dist(end, res.atoms[a]) > 15.0 for a in ring), "the link must not end on a ring atom"
        dots = [f for f in inside(res, g, Circle) if dist((f.item.cx + f.dx, f.item.cy + f.dy), centroid) < 0.5]
        assert len(dots) == 1, "the centroid is marked"


def test_methotrexate_rings_distinguishable_and_carboxylates_charged():
    res = render_of("dhfr_methotrexate")
    lig = diagram_of("dhfr_methotrexate").ligand
    systems = ring_systems(lig.mol)
    pteridine, benzene = max(systems, key=len), min(systems, key=len)
    assert (len(pteridine), len(benzene)) == (10, 6)

    def centre(idx):
        pts = [res.atoms[lig.names[i]] for i in idx]
        return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))
    assert dist(centre(pteridine), centre(benzene)) > 3.0 * res.quality.bond_length
    # The two carboxylates are drawn with a minus sign on an oxygen each.
    minus = [f for f in walk(res.scene) if isinstance(f.item, Text) and f.in_class("atom") is not None
             and any(s.text == "−" for s in f.item.spans)]
    assert sorted(f.in_class("atom").data["atom"] for f in minus) == sorted(lig.names[i] for i in lig.ionized)
    assert len(minus) == 2


def test_salt_bridges_are_drawn_distinctly_from_hbonds():
    """AP5A in adenylate kinase: six salt bridges, each to a phosphate, and eleven hydrogen bonds in its table."""
    dia, res = ap5a()
    rows = [i.record.type.value for i in dia.interactions]
    sb, hb = interactions(res, "saltbridge"), interactions(res, "hbond")
    assert len(sb) == rows.count("saltbridge") > 1 and len(hb) == rows.count("hbond") > 1

    def look(group_list):
        return {(f.item.paint.stroke, f.item.paint.dash) for g in group_list for f in inside(res, g, Line)}
    sb_look, hb_look = look(sb), look(hb)
    assert len(sb_look) == 1 and len(hb_look) == 1
    (sb_stroke, sb_dash), (hb_stroke, hb_dash) = next(iter(sb_look)), next(iter(hb_look))
    assert sb_stroke != hb_stroke and sb_dash != hb_dash      # colour *and* dash differ: survives greyscale
    # Each one lands on the phosphate it belongs to, not somewhere along the chain.
    lig = dia.ligand
    for g in sb:
        end, _ = ligand_end(res, g)
        group_atoms = g.data["ligand-atoms"].split(",")
        assert nearest_atom(res, end) in group_atoms
        assert sum(lig.element(lig.index_of(a)) == "P" for a in group_atoms) == 1


def test_a_chain_of_phosphates_and_the_halves_it_joins_are_laid_out_cleanly():
    dia, res = ap5a()
    lig = dia.ligand
    bl = res.quality.bond_length
    ps = [res.atoms[name] for name in ("PA", "PB", "PG", "PD", "PE")]      # the chain, as the component names it
    steps = [dist(ps[k], ps[k + 1]) for k in range(4)]
    assert min(steps) > 1.4 * bl, "consecutive phosphorus atoms are well apart"
    assert dist(ps[0], ps[-1]) > 0.8 * sum(steps), "the chain is stretched out, not folded back on itself"
    adenines = [s for s in ring_systems(lig.mol) if len(s) == 9]
    assert len(adenines) == 2
    a, b = box_of(res, lig, adenines[0]), box_of(res, lig, adenines[1])
    assert not overlap(a, b, slack=0.0)
    mid = ps[2]
    da = dist(((a[0] + a[2]) / 2, (a[1] + a[3]) / 2), mid)
    db = dist(((b[0] + b[2]) / 2, (b[1] + b[3]) / 2), mid)
    assert abs(da - db) / max(da, db) < 0.2, "the two halves sit at about the same distance from the middle"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("style", STYLES)
def test_single_atom_ligand_renders(style):
    res = zinc().render(style=style)
    assert res.status == "ok" and list(res.atoms) == ["ZN"]
    assert math.isfinite(res.quality.width) and res.quality.width > 50 and res.quality.height > 50
    links = interactions(res, "metal")
    # One line for each atom that holds the zinc, in either style: each is of a residue of its own.
    held = [i for i in zinc().interactions if i.record.type.value == "metal"]
    assert len(links) == len(held) == 3
    assert sum(g.title.count("Metal coordination") for g in links) == 3
    zn = res.atoms["ZN"]
    angles = []
    for g in links:
        end, tgt = ligand_end(res, g)
        assert dist(end, zn) < 22.0                        # stops at the Zn label
        far = max((p for f in inside(res, g, Line) for p in f.segment()), key=lambda p: dist(p, zn))
        angles.append(math.atan2(far[1] - zn[1], far[0] - zn[0]) % (2 * math.pi))
    angles.sort()
    gaps = [(b - a) % (2 * math.pi) for a, b in zip(angles, angles[1:] + angles[:1])]
    assert max(gaps) < math.radians(200), "the partners surround the ion rather than queue on one side"


def test_metal_coordination_is_drawn_distinctly_from_an_hbond():
    metal = zinc().render()
    hbond = render_of(REFERENCE)
    m_look = {(f.item.paint.stroke, f.item.paint.dash) for g in interactions(metal, "metal")
              for f in inside(metal, g, Line)}
    h_look = {(f.item.paint.stroke, f.item.paint.dash) for g in interactions(hbond, "hbond")
              for f in inside(hbond, g, Line)}
    assert m_look == {("metal", ())}                       # one solid stroke
    assert all(stroke != "metal" and dash for stroke, dash in h_look)


def test_metal_partners_are_drawn_as_the_atoms_that_coordinate():
    """The three atoms that hold the zinc of carbonic anhydrase II, as the entry's own ``_struct_conn`` records
    (``metalc1`` to ``metalc3``) name them."""
    got = sorted((g.data["residue"].split(":")[1], g.data["protein-atoms"]) for g in interactions(zinc().render(), "metal"))
    assert got == [("HIS119", "ND1"), ("HIS94", "NE2"), ("HIS96", "NE2")]


@pytest.fixture(scope="module")
def nothing_found(tmp_path_factory) -> Diagram:
    """Carbon monoxide in haemoglobin, with a table that has its header and no row: a complex that was analysed and
    in which nothing was found."""
    path = tmp_path_factory.mktemp("empty") / "interactions.tsv"
    path.write_text((TESTDATA / "hb_carbon_monoxide" / "interactions.tsv").read_text().splitlines()[0] + "\n")
    m = meta_of("hb_carbon_monoxide")["ligand"]
    return Diagram.from_structure(model_of("hb_carbon_monoxide"), ligand=m["residue_name_in_cif"], smiles=m["smiles"],
                                  interactions=load_interactions(path), name=m["name"])


@pytest.mark.parametrize("style", STYLES)
def test_empty_complex_is_a_result_that_says_so(style, nothing_found):
    res = nothing_found.render(style=style)
    assert res.status == "empty" and res.is_empty and not res.ok and res.figure == "empty"
    assert res.quality is None and res.warnings == []
    said = words(res)
    assert "No interactions in the supplied set" in said
    assert "analysed and nothing was found" in said
    # What is nearby, so nobody thinks nobody looked.
    near = residues_near("hb_carbon_monoxide", nothing_found)
    assert near and f"{len(near)} residues lie within 4.5 Å of the ligand" in said
    assert all(f"{name.capitalize()}{number}" in said for name, number in near)
    assert not groups(res, "legend") and not groups(res, "interaction") and not groups(res, "residue")
    assert groups(res, "ligand"), "the ligand itself is still drawn"


def test_proximal_residues_are_those_with_a_heavy_atom_within_the_cutoff(nothing_found):
    """Carbon monoxide; the zinc; the haem: what the diagram calls its pocket is what the atoms of the model say."""
    for slug, dia in (("hb_carbon_monoxide", nothing_found), ("ca2_celecoxib", zinc()), ("hb_carbon_monoxide", haem())):
        got = [(re.match(r"[A-Za-z]+", label).group().upper(), int(re.search(r"-?\d+$", label).group()))
               for label in dia.proximal_residues()]
        assert got == residues_near(slug, dia) and got


def test_empty_is_not_an_error_and_errors_are_not_empty():
    with pytest.raises(LigandNotFoundError):
        Diagram.from_structure(model_of("hb_carbon_monoxide"), "NOPE", interactions=[])
    try:
        import plip  # noqa: F401
    except ImportError:
        with pytest.raises(DetectionUnavailableError, match="pass your own interactions"):
            Diagram.from_structure(model_of("hb_carbon_monoxide"), "A:143", interactions=None)


def test_oversized_complex_gets_a_stated_fallback_not_a_hairball():
    dia, res = crowded()
    assert res.status == "unreadable" and res.figure == "summary" and not res.ok
    assert res.quality is not None and res.quality.unplaced > 0          # residues that found no place
    assert any("not readable" in w and "could not be placed" in w and "summary" in w for w in res.warnings)
    said = words(res)
    assert "Residue-level summary" in said and "not readable at 180 mm" in said
    # What it is, counted here: the nine residues of the peptide and their heavy atoms, the residues and rows of its table.
    from bindview.exposure import WATER_NAMES

    atoms = sum(1 for a in structure_of("mhc_peptide").atoms if a.chain == meta_of("mhc_peptide")["peptide_chain"]
                and a.element not in ("H", "D") and a.resname.upper() not in WATER_NAMES)
    rows = [i.record for i in dia.interactions]
    residues = {(r.chain, r.residue_number, r.insertion_code) for r in rows}
    assert f"9 ligand residues ({atoms} atoms), {len(residues)} residues, {len(rows)} interactions" in said
    assert not groups(res, "ligand"), "no atom-level drawing is passed off as the answer"
    texts = [f.item.content for f in walk(res.scene) if isinstance(f.item, Text)]
    labels = {r.record.residue_label for r in dia.interactions}
    assert labels <= set(texts) and len(labels) == len(residues)
    # The summary itself is readable at the figure size it was asked for.
    assert res.scene.width <= 180 / 25.4 * 96 + 1e-6


def test_summary_fallback_is_compact():
    _, res = crowded()
    assert res.scene.height < res.scene.width * 1.2, "set in side-by-side blocks, not one tall column"


def test_fallback_can_be_refused_or_overridden():
    """Imatinib on a page 60 mm wide, which it does not fit: no figure at all, or the diagram drawn anyway and said to
    be unreadable."""
    dia = diagram_of(REFERENCE)
    none = dia.render(figure_width_mm=60.0, on_unreadable="none")
    assert none.status == "unreadable" and none.svg is None and none.figure == "none"
    forced = dia.render(figure_width_mm=60.0, on_unreadable="render")
    assert forced.status == "unreadable" and forced.figure == "diagram" and forced.svg
    assert any("rendered anyway on request" in w for w in forced.warnings)


def test_readability_follows_the_requested_figure_size():
    dia = diagram_of(REFERENCE)
    small = dia.render(figure_width_mm=60.0)
    assert small.status == "unreadable" and small.figure == "summary"
    lenient = dia.render(figure_width_mm=60.0, min_font_pt=1.0)
    assert lenient.status == "ok"


def test_a_text_size_that_no_page_gives_is_said_to_be_that(capsys, tmp_path):
    """A figure is made smaller to fit a page, never larger. A ``min_font_pt`` above its smallest text at
    full size is therefore met on no page: the result says that, and names no width to try."""
    from bindview import Metrics
    from bindview.cli import main

    dia = diagram_of(REFERENCE)
    natural = render_of(REFERENCE).quality.min_font_pt                # its smallest text, on a page that fits it
    assert 5.0 < natural < 8.0
    is_, most = f"{math.floor(natural * 100) / 100:g}", f"{math.floor(natural * 10) / 10:g}"
    for options in ({}, {"on_unreadable": "render"}, {"figure_width_mm": 400.0, "figure_height_mm": 400.0}):
        res = dia.render(min_font_pt=8, **options)
        said = res.warnings[-1]
        assert res.status == "unreadable" and not res.quality.readable
        assert f"its smallest text is {is_} pt at full size" in said and "no page gives it the 8 pt" in said
        assert f"Ask for {most} pt or less, or hand in larger fonts (metrics=)" in said
        assert "mm wide" not in said and "needs a figure" not in said
        assert math.isinf(res.quality.required_width_mm)
    # What it says to do, does it: the size it names, and larger fonts.
    assert dia.render(min_font_pt=float(most)).status == "ok"
    larger = Metrics().with_(distance_font=11, protein_atom_font=11, sub_font=11, charge_font=11, legend_font=11,
                             caption_font=11)
    assert dia.render(min_font_pt=8, metrics=larger).status == "ok"
    # A page that is too small is still said to be that, with the width that would do ...
    small = dia.render(figure_width_mm=45, figure_height_mm=45)
    needed = re.search(r"needs a figure at least (\d+) mm wide for 5 pt text \(requested: 45 mm\)", small.warnings[-1])
    assert needed and int(needed.group(1)) == round(small.quality.required_width_mm) > 45
    # ... and what goes out as JSON stays JSON: no width is null there, not a number that is none.
    ligand = meta_of(REFERENCE)["ligand"]["residue_name_in_cif"]
    code = main(["render", str(model_of(REFERENCE)), "--ligand", ligand, "--smiles-from", "meta.json", "-q", "--json",
                 "--min-font-pt", "8", "--on-unreadable", "none", "-o", str(tmp_path / "figure.svg")])
    printed = capsys.readouterr().out
    assert code == 3 and "Infinity" not in printed
    import json

    assert json.loads(printed)["quality"]["required_width_mm"] is None


def test_the_size_a_figure_is_said_to_have_is_never_above_the_one_it_has():
    """The sizes in that reason are cut off, not rounded. Text of 6.375 pt, rounded, "is 6.4 pt": said to
    a caller who asked for 6.4 pt, next to "no page gives it the 6.4 pt"."""
    from bindview.styles.base import RenderOptions, beyond_any_page

    said = beyond_any_page(6.375, RenderOptions(min_font_pt=6.4), fixed=True)
    assert said is not None
    assert "is 6.37 pt at full size" in said and "the 6.4 pt" in said and "Ask for 6.3 pt or less" in said
    for asked in (6.3, 6.37, 6.375, 5.0):
        assert beyond_any_page(6.375, RenderOptions(min_font_pt=asked)) is None
    whole = beyond_any_page(7.5, RenderOptions(min_font_pt=9))
    assert whole is not None and "is 7.5 pt at full size" in whole and "Ask for 7.5 pt or less, or hand in" in whole


def test_the_page_a_figure_is_said_to_need_is_one_that_was_not_asked_for():
    """A figure that needs 90.3 mm, said of a page of 90 mm, is said to need 90.3 mm and to print at 4.98 pt: rounded,
    it would need the page it has and print at the minimum. One that a low page makes small is sent after a height."""
    from bindview.styles.base import RenderOptions, too_small_on_page

    assert too_small_on_page(4.983, 90.3, 600.0, 400.0, RenderOptions(figure_width_mm=90)) == (
        "at 90 mm the smallest text would print at 4.98 pt (minimum 5 pt); it needs a figure about 90.3 mm wide")
    assert too_small_on_page(2.5, 90.0, 600.0, 400.0, RenderOptions(figure_width_mm=45)) == (
        "at 45 mm the smallest text would print at 2.5 pt (minimum 5 pt); it needs a figure about 90 mm wide")
    # A wide page that is low: the height is what is named (90 mm wide is 60 mm high for a figure of 600 by 400).
    assert too_small_on_page(3.3, 90.0, 600.0, 400.0, RenderOptions(figure_width_mm=180, figure_height_mm=40)) == (
        "on a page 40 mm high the smallest text would print at 3.3 pt (minimum 5 pt); it needs a figure about 60 mm high")
    # The same words, of a figure: imatinib on pages too narrow and too low for it.
    dia = diagram_of(REFERENCE)
    narrow = dia.render(figure_width_mm=45, figure_height_mm=45, on_unreadable="render")
    assert re.search(r"at 45 mm the smallest text would print at [0-9.]+ pt \(minimum 5 pt\); "
                     r"it needs a figure about [0-9.]+ mm wide; rendered anyway on request$", narrow.warnings[-1])
    low = dia.render(figure_width_mm=180, figure_height_mm=40, on_unreadable="render").warnings[-1]
    assert re.search(r"on a page 40 mm high the smallest text would print at [0-9.]+ pt \(minimum 5 pt\); "
                     r"it needs a figure about [0-9.]+ mm high", low) and "mm wide" not in low
    alone = dia.render(figure_width_mm=180, figure_height_mm=20).warnings[-1]
    assert re.search(r"needs a figure at least [0-9.]+ mm high for 5 pt text \(requested: 20 mm\)", alone)
    assert "mm wide" not in alone


@pytest.mark.parametrize("style", STYLES)
def test_a_metal_of_the_ligands_own_is_drawn_with_it_and_a_contact_ends_on_it_only_from_a_residue(style):
    """The haem: built from connectivity, so the figure says that it is degraded and why. The iron's bonds to the four
    nitrogens of its ring are bonds of the drawing, and the only contacts that end on the iron are those its table
    has (the histidine below it)."""
    dia = haem()
    res = dia.render(style=style)
    assert res.status == "degraded" and any("only the connectivity is known" in w for w in res.warnings)
    fe = res.atoms["FE"]
    (ligand,) = groups(res, "ligand")
    to_iron = [f for f in inside(res, ligand, Line) if min(dist(p, fe) for p in f.segment()) < 20.0]
    assert len(to_iron) == 4, "the four nitrogens of the porphyrin"
    on_iron = sorted((g.data["type"], g.data["residue"]) for g in groups(res, "interaction") + groups(res, "contact")
                     if "FE" in g.data["ligand-atoms"].split(","))
    rows = sorted((i.record.type.value, who(i.record)) for i in dia.interactions
                  if "FE" in (i.record.ligand_atom, *i.record.ligand_atoms))
    assert on_iron == rows and rows == [("metal", "A:HIS87")]


# ---------------------------------------------------------------------------
# Flags, options, themes
# ---------------------------------------------------------------------------

def reference_with(rows: list[Interaction]) -> Diagram:
    m = meta_of(REFERENCE)["ligand"]
    return Diagram.from_structure(model_of(REFERENCE), m["residue_name_in_cif"], smiles=m["smiles"], interactions=rows)


@pytest.mark.parametrize("style", STYLES)
def test_contacts_beyond_the_contact_set_are_visibly_different(style):
    """The supplied tables flag no row, so the flag is exercised on a copy of the reference case: Phe811, whose one
    row is its ring stacking."""
    res = reference_with([r.with_(beyond_contact_set=True) if r.residue_number == 811 else r
                          for r in table(REFERENCE)]).render(style=style)
    base = render_of(REFERENCE, style)
    (g,) = interactions(res, "pi_stacking")
    assert g.data.get("beyond") == "true" and g.data["residue"] == "A:PHE811"
    assert all(f.item.paint.opacity < 1.0 for f in inside(res, g, Line))
    assert "beyond" not in interactions(base, "pi_stacking")[0].data
    texts = [f.item.content for f in walk(res.scene) if isinstance(f.item, Text)]
    assert any("†" in w and "811" in w for w in texts), "the residue is daggered"
    assert any("outside the supplied contact set" in w for w in texts), "and the legend says what the dagger means"
    assert not any("†" in f.item.content for f in walk(base.scene) if isinstance(f.item, Text))


@pytest.mark.parametrize("style", STYLES)
def test_water_bridge_shows_the_water_on_the_link(style):
    """Imatinib's two water bridges, both from Lys623 through one water."""
    res = render_of(REFERENCE, style)
    rows = [r for r in table(REFERENCE) if r.type.value == "water_bridge"]
    bridges = interactions(res, "water_bridge")
    assert sorted((g.data["residue"], g.data["ligand-atoms"]) for g in bridges) == sorted((who(r), r.ligand_atom)
                                                                                           for r in rows)

    def look(grp):
        return {(f.item.paint.stroke, f.item.paint.dash) for f in inside(res, grp, Line)}
    for g in bridges:
        assert [f.item.content for f in inside(res, g, Text)] == ["H₂O"]
        end, tgt = ligand_end(res, g)
        assert nearest_atom(res, end) == g.data["ligand-atoms"]
        assert all(look(g).isdisjoint(look(hb)) for hb in interactions(res, "hbond")), "drawn otherwise than an H-bond"
    assert "Water bridge" in [f.item.content for f in walk(res.scene) if isinstance(f.item, Text)]


def test_a_record_that_cannot_be_attached_degrades_the_result_and_is_not_drawn():
    rows = table(REFERENCE)
    res = reference_with(rows + [Interaction("hbond", 673, "CYS", chain="A", protein_atom="O", ligand_atom="N99",
                                             distance=3.0)]).render()
    assert res.status == "degraded" and not res.ok and res.figure == "diagram"
    assert any("N99" in w and "not drawn" in w for w in res.warnings)
    assert len(interactions(res, "hbond")) == sum(r.type.value == "hbond" for r in rows)
    assert "1 reported interaction could not be attached" in words(res)


def test_smiles_is_taken_from_the_structure_file_when_not_given(tmp_path):
    """A file that carries the SMILES of its ligand (``_chem_comp.pdbx_smiles``, as a predicted model writes it): the
    celecoxib entry with that one column given to its ligand."""
    m = meta_of("ca2_celecoxib")["ligand"]
    text = gzip.open(model_of("ca2_celecoxib"), "rt", encoding="utf-8").read()
    told = re.sub(r"loop_\n_chem_comp\.id \n.*?\n#", f'loop_\n_chem_comp.id\n_chem_comp.pdbx_smiles\n{m["component"]} '
                  f'"{m["smiles"]}"\n#', text, count=1, flags=re.S)
    assert told.count("_chem_comp.pdbx_smiles") == 1
    (tmp_path / "model.cif").write_text(told, encoding="utf-8")
    without = Diagram.from_structure(tmp_path / "model.cif", m["residue_name_in_cif"], name=m["name"],
                                     interactions=table("ca2_celecoxib")).render()
    assert without.status == "ok" and without.svg == render_of("ca2_celecoxib").svg


def test_atom_map_covers_every_ligand_atom_inside_the_page():
    res = render_of(REFERENCE)
    assert sorted(res.atoms) == sorted(diagram_of(REFERENCE).ligand.names)
    assert all(0 <= x <= res.scene.width and 0 <= y <= res.scene.height for x, y in res.atoms.values())


def test_options_change_what_is_drawn():
    dia = diagram_of(REFERENCE)
    plain = dia.render(legend=False, caption=False)
    assert not groups(plain, "legend") and not groups(plain, "footnotes")
    titled = dia.render(title="Imatinib", subtitle="c-Kit")
    texts = [f.item.content for f in walk(titled.scene) if isinstance(f.item, Text)]
    assert "Imatinib" in texts and "c-Kit" in texts

    def distance_labels(res):
        return sorted(f.item.content for g in groups(res, "interaction") for f in inside(res, g, Text))
    rows = table(REFERENCE)
    water = ["H₂O"] * sum(r.type.value == "water_bridge" for r in rows)
    # By default the distance of every hydrogen bond, halogen bond and metal coordination; a water bridge has its water.
    assert distance_labels(render_of(REFERENCE)) == sorted(
        [f"{r.distance:.2f}" for r in rows if r.type.value in ("hbond", "halogen", "metal")] + water)
    assert distance_labels(dia.render(distance_labels="none")) == water
    assert distance_labels(dia.render(distance_labels="all")) == sorted(
        [f"{r.distance:.2f}" for r in rows if r.type.value not in ("hydrophobic", "water_bridge")] + water)
    assert distance_labels(render_of(REFERENCE, "schematic")) == water                 # the schematic style prints none
    # A model whose every pLDDT is below 50 (made up here): with its confidence drawn, the same four distances are
    # there, each carrying the mark.
    m = meta_of("dhfr_methotrexate")["ligand"]
    doubtful = Diagram.from_structure(invented_plddt(structure_of("dhfr_methotrexate"), high=45.0),
                                      ligand=m["residue_name_in_cif"], smiles=m["smiles"],
                                      interactions=table("dhfr_methotrexate"))
    plain = [t for t in distance_labels(doubtful.render(confidence=False)) if t[:1].isdigit()]
    assert len(plain) == sum(r.type.value == "hbond" for r in table("dhfr_methotrexate")) == 4
    marked = [t for t in distance_labels(doubtful.render()) if t[:1].isdigit()]
    assert all(t.endswith("?") for t in marked) and [t.rstrip(" ?") for t in marked] == plain


def test_unknown_style_and_theme_are_rejected():
    dia = diagram_of(REFERENCE)
    with pytest.raises(ValueError, match="unknown style"):
        dia.render(style="cubist")
    with pytest.raises(ValueError, match="unknown theme"):
        dia.render(theme="sepia")
    with pytest.raises(TypeError):
        dia.render(colour="blue")


def test_themes_recolour_the_same_drawing():
    dia = diagram_of(REFERENCE)
    light, dark = dia.render(theme="light"), dia.render(theme="dark")
    assert light.scene == dark.scene                          # one layout, two palettes
    assert light.svg != dark.svg
    strip = lambda s: re.sub(r'#[0-9a-fA-F]{6}', "#", s)
    assert strip(light.svg) == strip(dark.svg), "only colours differ"
    assert 'fill="#ffffff"' in light.svg and 'fill="#0f172a"' in dark.svg


def test_auto_theme_embeds_both_palettes_as_css_variables():
    res = diagram_of(REFERENCE).render(theme="auto")
    assert res.theme == "auto"
    assert "prefers-color-scheme:dark" in res.svg and "svg.bindview[data-theme=dark]" in res.svg
    assert "var(--lf-hbond,#1a9641)" in res.svg and "--lf-hbond:#4ade80" in res.svg
    ET.fromstring(res.svg)


def test_custom_palette():
    from bindview import LIGHT
    res = diagram_of(REFERENCE).render(theme=LIGHT.with_(name="brand", hbond="#123456"))
    assert 'stroke="#123456"' in res.svg and res.theme == "brand"


def test_background_can_be_left_transparent():
    assert 'class="background"' not in diagram_of(REFERENCE).render(background=False).svg


def test_svg_carries_addressable_metadata():
    """Bicalutamide's halogen bond, as its table has it, in the attributes and the tooltip of its group."""
    svg = render_of("ar_bicalutamide").svg
    (row,) = [r for r in table("ar_bicalutamide") if r.type.value == "halogen"]
    root = ET.fromstring(svg)
    ns = "{http://www.w3.org/2000/svg}"
    hal = [g for g in root.iter(ns + "g") if g.get("class") == "interaction halogen"]
    assert len(hal) == 1
    assert hal[0].get("data-ligand-atoms") == row.ligand_atom and hal[0].get("data-protein-atoms") == row.protein_atom
    assert hal[0].get("data-distance") == f"{row.distance:.2f}" and hal[0].get("data-residue") == who(row)
    name = f"{row.residue_name.capitalize()}{row.residue_number}"
    assert f"Halogen bond: {name} {row.protein_atom} – {row.ligand_atom}, {row.distance:.2f} Å" in hal[0].find(ns + "title").text
    assert [g.get("data-atom") for g in root.iter(ns + "g") if g.get("class") == "atom"].count(row.ligand_atom) == 1


# ---------------------------------------------------------------------------
# Hard requirement 3: deterministic
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", CASES)
def test_rebuilding_from_scratch_gives_identical_bytes(case):
    fresh = fresh_diagram(case)
    for style in STYLES:
        assert fresh.render(style=style).svg == render_of(case, style).svg


def test_output_does_not_depend_on_hash_randomisation():
    """Run the reference case in two interpreters with different hash seeds."""
    code = (
        "import json, hashlib\n"
        "from bindview import Diagram, load_interactions\n"
        f"d = 'testdata/{REFERENCE}'\n"
        "m = json.load(open(d + '/meta.json'))['ligand']\n"
        "dia = Diagram.from_structure(d + '/model.cif.gz', m['residue_name_in_cif'], smiles=m['smiles'],\n"
        "                             interactions=load_interactions(d + '/interactions.tsv'))\n"
        "print(hashlib.sha256((dia.render().svg + dia.render(style='schematic').svg).encode()).hexdigest())\n")
    digests = set()
    for seed in ("0", "1234"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, check=True)
        digests.add(out.stdout.strip().splitlines()[-1])
    assert len(digests) == 1


def test_no_network_and_no_writes_to_the_structure(tmp_path, monkeypatch):
    import shutil
    import socket

    cif = tmp_path / "model.cif.gz"
    shutil.copy(model_of("ca2_celecoxib"), cif)
    before = (cif.stat().st_mtime_ns, cif.read_bytes())

    def refuse(*a, **k):
        raise AssertionError("network access during rendering")
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    m = meta_of("ca2_celecoxib")["ligand"]
    dia = Diagram.from_structure(cif, m["residue_name_in_cif"], smiles=m["smiles"], interactions=table("ca2_celecoxib"))
    out = tmp_path / "out.svg"
    res = dia.render_svg(out)
    assert res.path == out and out.read_text() == res.svg
    assert (cif.stat().st_mtime_ns, cif.read_bytes()) == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["model.cif.gz", "out.svg"]


# ---------------------------------------------------------------------------
# What is kept of a figure is the figure of the diagram as it is now
# ---------------------------------------------------------------------------

#: What can be done to a diagram between two figures of it.
CHANGES = {
    "another name": lambda dia: setattr(dia, "name", "Another name"),
    "a contact less": lambda dia: dia.interactions.pop(),
    "a contact with another distance": lambda dia: setattr(
        dia.interactions[0], "record", dia.interactions[0].record.with_(distance=1.23)),
    "a remark": lambda dia: dia.remarks.append("A remark of the caller."),
    "no confidence": lambda dia: setattr(dia, "confidence", None),
    "what the occupancy is a share of": lambda dia: setattr(dia, "ensemble", "ten models"),
}


def flagged(case: str = "ca2_celecoxib") -> Diagram:
    """An open case as a model that its caller does not trust: a diagram with a confidence to take away."""
    return fresh_diagram(case, trusted=False, flag="grade C")


@pytest.mark.parametrize("change", sorted(CHANGES))
def test_a_diagram_that_was_changed_is_drawn_again(change):
    dia, own = flagged(), flagged()
    before = dia.render()
    assert before.svg == own.render().svg
    for one in (dia, own):
        if change == "what the occupancy is a share of":
            one.interactions[0].record = one.interactions[0].record.with_(occupancy=0.4)
        CHANGES[change](one)
    after, fresh = dia.render(), own.render()
    assert after.svg != before.svg
    assert after.svg == fresh.svg and after.quality == fresh.quality and after.notes == fresh.notes


def test_a_copy_that_was_given_other_contacts_is_not_the_figure_of_the_original():
    dia = flagged()
    whole = dia.render()
    part = copy.copy(dia)                                         # (and nothing emptied by hand)
    part.interactions = dia.interactions[:2]
    own = flagged()
    own.interactions = own.interactions[:2]
    assert part.render().quality.n_interactions == 2 and part.render().svg == own.render().svg
    assert dia.render().svg == whole.svg and dia.render().quality.n_interactions == len(table("ca2_celecoxib"))


def test_a_diagram_that_was_left_as_it_is_is_laid_out_once(monkeypatch):
    from bindview.layout import engine

    made = []
    run = engine._Engine.run
    monkeypatch.setattr(engine._Engine, "run", lambda self: made.append(1) or run(self))
    dia = flagged()
    first = dia.render()
    assert len(made) == 1
    assert dia.render().svg == first.svg and copy.copy(dia).render().svg == first.svg and len(made) == 1
    dia.name = "Another name"                                     # the name moves nothing: drawn again, on the layout there is
    assert dia.render().svg != first.svg and len(made) == 1


def test_a_result_that_its_caller_changes_changes_no_later_figure():
    dia = flagged()
    first = dia.render()
    svg, notes, n_atoms, n_nodes = first.svg, list(first.notes), len(first.atoms), len(first.graph.nodes)
    first.scene.title = "Changed by the caller"
    first.scene.notes.append("a note of the caller")
    for item in first.scene.items:
        if isinstance(item, Group):
            item.data["changed"] = "yes"
            item.items.clear()
    first.scene.items.pop()
    first.notes.append("a note of the caller")
    first.atoms.clear()
    first.warnings.append("a warning of the caller")
    first.graph.nodes.clear()
    first.data["changed"] = "yes"
    again = dia.render()
    assert again.scene is not first.scene and again.scene.title != "Changed by the caller"
    assert again.svg == svg and again.notes == notes and again.scene.notes == notes
    assert len(again.atoms) == n_atoms and len(again.graph.nodes) == n_nodes
    assert "a warning of the caller" not in again.warnings and "changed" not in again.data
    # ... and it is the whole figure that is handed out: what is read out of its scene is what the file has
    assert sum(isinstance(f.item, Text) for f in walk(again.scene)) == again.svg.count("<text ")


def hairpin(detail=None) -> Diagram:
    """One hairpin of the riboswitch as the ligand, a chain of eighteen nucleotides, with the plain distance table."""
    from bindview import read_structure
    from bindview.distance_table import contact_table

    model = read_structure(TESTDATA / "nucleic" / "riboswitch_bisguanidine" / "model.pdb")
    return Diagram.from_structure(model, "chain:B", interactions=contact_table(model, "B"), detail=detail)


def test_a_chain_with_atoms_beside_its_boxes_is_the_same_file_the_second_time():
    """On a page that the chain's boxes alone would fit and the boxes with their atoms do not."""
    dia = hairpin(detail="atoms")
    first = dia.render(figure_height_mm=120)
    second = dia.render(figure_height_mm=120)
    assert first.status == second.status == "unreadable" and first.figure == second.figure == "summary"
    assert first.svg == second.svg and first.warnings == second.warnings and first.quality == second.quality
    # The first is refused for the same reason as the second: the ligand with its atoms, before anything is laid out.
    assert first.quality.estimated and "the ligand alone, at the smallest bond length" in first.warnings[-1]


def test_a_figure_drawn_although_it_is_unreadable_answers_the_pointer_like_any_other():
    from bindview.interactive import OVER

    dia = fresh_diagram("ca2_celecoxib")
    readable = dia.render(interactive=True)
    forced = dia.render(interactive=True, on_unreadable="render", figure_width_mm=40)
    assert readable.status == "ok" and (forced.status, forced.figure) == ("unreadable", "diagram")
    # The close-ups and the places to point at on the ligand's atoms, and what can take the focus.
    for mark in (f'class="{OVER}"', 'class="hit', "tabindex="):
        assert forced.svg.count(mark) == readable.svg.count(mark) > 0, mark
    ET.fromstring(forced.svg)


# ---------------------------------------------------------------------------
# What a result says of itself
# ---------------------------------------------------------------------------

def test_ok_is_described_as_what_it_is():
    """``ok`` is the status ``ok`` and no other (a degraded figure is not ok, above): its description names none."""
    from bindview import RenderResult

    said = RenderResult.ok.__doc__
    assert "``ok``" in said and not any(f"``{other}``" in said for other in ("degraded", "empty", "unreadable"))


def test_an_estimate_has_the_height_of_the_ligand_as_well_as_its_width():
    res = fresh_diagram("kit_imatinib").render(figure_width_mm=20, on_unreadable="none")
    q = res.quality
    assert res.status == "unreadable" and q.estimated
    assert q.ligand_width > 0 and q.ligand_height > 0
    # (The estimate is the ligand with the same margin on every side.)
    assert q.height - q.ligand_height == pytest.approx(q.width - q.ligand_width)


def test_what_a_graph_overrules_is_said():
    dia = fresh_diagram("ca2_celecoxib")
    first = dia.render()
    scale = first.graph.bond_length
    quiet = dia.render(graph=first.graph, bond_length=scale)
    assert quiet.svg == first.svg and quiet.warnings == first.warnings
    other = dia.render(graph=first.graph, bond_length=scale - 8.0)
    assert other.svg == first.svg and other.quality.bond_length == scale
    assert [w for w in other.warnings if w not in first.warnings] == [
        f"bond_length={scale - 8.0:g} was asked for with a graph, which has its own scale: the figure is drawn at {scale:g}"]
    placed = dia.render(graph=first.graph, coords="rdkit")
    assert placed.svg == first.svg
    assert [w for w in placed.warnings if w not in first.warnings] == [
        "coords='rdkit' was asked for with a graph, which has the atoms of the ligand where they are drawn: it "
        "changes nothing"]


@pytest.mark.parametrize("colors", ["standard", "contrast"])
def test_the_theme_of_a_result_is_the_theme_whatever_the_colour_scheme(colors):
    dia = fresh_diagram("ca2_celecoxib")
    assert [dia.render(theme=theme, colors=colors).theme for theme in ("light", "dark", "auto", "Dark")] == \
        ["light", "dark", "auto", "dark"]


# ---------------------------------------------------------------------------
# A page or a scale that is no size is refused before anything is drawn for it
# ---------------------------------------------------------------------------

NOT_A_SIZE = [
    (dict(bond_length=0), "bond_length is the length of a bond in page units, a positive number of at most 300"),
    (dict(bond_length=-5.0), "bond_length is the length of a bond .* not -5.0"),
    (dict(bond_length=math.nan), "bond_length is the length of a bond .* not nan"),
    (dict(bond_length=math.inf), "bond_length is the length of a bond .* not inf"),
    (dict(bond_length="30"), "bond_length is the length of a bond .* not '30'"),
    (dict(bond_length=True), "bond_length is the length of a bond .* not True"),
    (dict(bond_length=3000.0), "bond_length is the length of a bond .* at most 300 .* not 3000.0"),
    (dict(bond_length=1e6), "bond_length is the length of a bond .* not 1000000.0"),
    (dict(figure_width_mm=0), "figure_width_mm is the width of the page in millimetres, a positive number, not 0"),
    (dict(figure_width_mm=-10.0), "figure_width_mm is the width of the page .* not -10.0"),
    (dict(figure_width_mm=math.nan), "figure_width_mm is the width of the page .* not nan"),
    (dict(figure_width_mm=math.inf), "figure_width_mm is the width of the page .* not inf"),
    (dict(figure_width_mm="180"), "figure_width_mm is the width of the page .* not '180'"),
    (dict(figure_height_mm=0.0), "figure_height_mm is the height of the page in millimetres, a positive number, not 0.0"),
    (dict(figure_height_mm=math.nan), "figure_height_mm is the height of the page .* not nan"),
    (dict(min_font_pt=math.nan), "min_font_pt is the smallest text that counts as legible, in points: a number that is not negative, not nan"),
    (dict(min_font_pt=-1.0), "min_font_pt is the smallest text .* not -1.0"),
]


@pytest.mark.parametrize("options, said", NOT_A_SIZE, ids=[str(row[0]) for row in NOT_A_SIZE])
@pytest.mark.parametrize("on_unreadable", ["summary", "render"])
def test_a_page_or_a_scale_that_is_no_size_is_refused_before_anything_is_drawn(options, said, on_unreadable, monkeypatch):
    dia = flagged()

    def drawn(*args, **kwargs):
        raise AssertionError("a figure was drawn for it")
    monkeypatch.setattr(Diagram, "_compose", drawn)
    monkeypatch.setattr(Diagram, "_hopeless", drawn)
    with pytest.raises(RenderError, match=said):                  # (one of the package's own: a caller of it is told in words)
        dia.render(on_unreadable=on_unreadable, **options)


def test_sizes_that_are_sizes_are_taken():
    import numpy as np

    dia = fresh_diagram("ca2_celecoxib")
    plain = dia.render()
    assert dia.render(bond_length=None).svg == plain.svg
    assert dia.render(figure_width_mm=np.float64(180.0), figure_height_mm=240, min_font_pt=0).status == "ok"
    for scale in (11.5, np.float64(30.0)):
        assert dia.render(bond_length=scale, on_unreadable="render").quality.bond_length == scale
    # The longest bond that is taken (not drawn here: a layout at that scale takes several hundred megabytes).
    from bindview.diagram import _check_sizes
    from bindview.styles.base import RenderOptions

    assert _check_sizes(RenderOptions(bond_length=300)) is None
    with pytest.raises(RenderError, match="at most 300"):
        _check_sizes(RenderOptions(bond_length=300.5))


def test_the_command_line_says_of_a_scale_beyond_any_page_what_is_wrong(capsys, tmp_path, monkeypatch):
    from bindview.cli import main

    def drawn(*args, **kwargs):
        raise AssertionError("a figure was drawn for it")
    monkeypatch.setattr(Diagram, "_compose", drawn)               # (a layout at that scale would take all the memory there is)
    case = TESTDATA / "ca2_celecoxib"
    out = tmp_path / "figure.svg"
    code = main(["render", str(case / "model.cif.gz"), "--ligand", "CEL", "--interactions", str(case / "interactions.tsv"),
                 "--bond-length", "1e9", "-o", str(out)])
    said = capsys.readouterr()
    assert code == 2 and "bond_length is the length of a bond in page units" in said.err and not out.exists()
    assert issubclass(RenderError, BindviewError)


def test_a_graph_with_a_scale_beyond_any_page_is_refused_before_anything_is_drawn(monkeypatch):
    from bindview.errors import GraphError

    dia = fresh_diagram("ca2_celecoxib")
    graph = dia.render().graph

    def drawn(*args, **kwargs):
        raise AssertionError("a figure was drawn for it")
    monkeypatch.setattr(Diagram, "_compose", drawn)
    for scale, said in ((3000.0, "3000"), (1e6, "1e\\+06")):
        graph.bond_length = scale
        for options in ({}, {"on_unreadable": "render"}):
            with pytest.raises(GraphError, match=f"bond_length of the graph is {said}: .* at most 300"):
                dia.render(graph=graph, **options)


# ---------------------------------------------------------------------------
# What a caller hands in comes out as a colour or as text, never as markup
# ---------------------------------------------------------------------------

NOT_A_COLOUR = ['red" onload="alert(1)', "red;}</style><script>alert(1)</script>", "#fff'", "red}svg{display:none",
                "red/*", "re\nd", "a\\b", "<b>", "&amp;", ""]


@pytest.mark.parametrize("colour", NOT_A_COLOUR)
def test_a_colour_of_a_palette_that_is_no_colour_is_refused(colour):
    from bindview import DARK, LIGHT
    from bindview.svg import svg_and_css

    dia = fresh_diagram("ca2_celecoxib")
    with pytest.raises(ValueError, match="the colour 'hbond' of the palette 'mine' is .*: a colour is"):
        dia.render(theme=LIGHT.with_(name="mine", hbond=colour))
    # ... in a file that follows the reader's theme too, where every colour of both palettes is declared in a style
    # sheet, drawn with or not, and under the name the palette gives it.
    scene = dia.render().scene
    for light, dark in ((LIGHT.with_(name="mine", sch_metal=colour), DARK), (LIGHT, DARK.with_(name="mine", sch_metal=colour))):
        with pytest.raises(ValueError, match="the colour 'sch_metal' of the palette 'mine' is .*: a colour is"):
            svg_and_css(scene, light, css_vars=True, dark_palette=dark)
    with pytest.raises(ValueError, match="the colour 'element_X;}' of the palette 'mine' has a name that"):
        svg_and_css(scene, LIGHT.with_(name="mine", elements={**LIGHT.elements, "X;}": "#000000"}), css_vars=True,
                    dark_palette=DARK)


def test_a_colour_of_a_palette_is_written_as_it_is_given():
    from bindview import LIGHT

    dia = fresh_diagram("ca2_celecoxib")
    for colour in ("#123456", "rgb(18, 52, 86)", "hsl(210 65% 20% / 0.9)", "var(--brand-green, #1a9641)", "rebeccapurple",
                   "color-mix(in srgb, #123456 60%, white)"):
        svg = dia.render(theme=LIGHT.with_(hbond=colour)).svg
        assert f'stroke="{colour}"' in svg, colour
        ET.fromstring(svg)


@pytest.mark.parametrize("odd", ["\x00", "\x01", "\x0b", "\x1f", "\ufffe", "\uffff", "\ud800", "\udfff"],
                         ids=lambda c: f"U+{ord(c):04X}")
def test_a_character_that_no_xml_file_can_hold_is_left_out_of_the_figure(odd, tmp_path):
    ns = "{http://www.w3.org/2000/svg}"
    dia = fresh_diagram("ca2_celecoxib", trusted=False, flag=f"grade {odd}C")
    path = tmp_path / "figure.svg"
    res = dia.render_svg(path, title=f"A{odd}B", subtitle=f"with a tab\tand {odd}more", interactive=True)
    assert res.status == "ok" and odd not in res.svg
    root = ET.fromstring(res.svg)                                 # it is XML
    assert path.read_text(encoding="utf-8") == res.svg            # ... and the file is the figure, whole
    assert root.find(f"{ns}title").text == "AB" and "(grade C)" in root.find(f"{ns}desc").text
    words = ["".join(t.itertext()) for t in root.iter(f"{ns}text")]
    assert "AB" in words and "with a tab\tand more" in words
    assert any("(grade C)" in w for w in words)
