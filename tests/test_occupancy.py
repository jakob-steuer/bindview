"""Occupancy and confidence on the contact record.

``occupancy`` is the share of an ensemble in which a contact is made; the
number does not say which ensemble, the legend does. One field and one visual
channel serve a trajectory's frames and a series' ligands alike, so both are
tested here: records as a caller would supply them from MD, and the consensus
of a set of ligands of one pocket, which produces the same records.
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from dataclasses import replace

import pytest

from conftest import (diagram_of, invented_plddt, overlay_diagram, overlay_members, predicted_diagram, render_of,
                      structure_of)
from bindview import Comparison, Diagram, FingerprintTable, Interaction, parse_interactions
from bindview.confidence import MARK, occupancy_factors
from bindview.errors import InteractionError
from bindview.scene import Line, Text
from scenetools import groups, inside, interactions, walk, words

FRAMES = "1,000 frames of a 100 ns trajectory"
CASE = "ar_bicalutamide"     # hydrogen bonds to Leu704 and Gln711, a halogen bond to Met742, hydrophobic contacts


def with_records(dia: Diagram, change, **kwargs) -> Diagram:
    """The same complex with its records changed by ``change(record) -> dict of fields``."""
    resolved = [replace(r, record=r.record.with_(**change(r.record))) for r in dia.interactions]
    return Diagram(dia.ligand, resolved, structure=dia.structure, ligand_key=dia.ligand_key, name=dia.name,
                   confidence=kwargs.pop("confidence", dia.confidence), **kwargs)


def strokes(res, group):
    return [f.item.paint for f in inside(res, group, Line)]


def link(result, kind, residue):
    (g,) = interactions(result, kind, residue)
    return g


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------

def test_the_record_takes_occupancy_and_confidence():
    rec = Interaction("hbond", 570, "GLN", "A", "NE2", "O1", distance=2.9, occupancy="0.63", confidence=41)
    assert rec.occupancy == 0.63 and rec.confidence == 41.0
    plain = Interaction("hbond", 570, "GLN")
    assert plain.occupancy is None and plain.confidence is None        # one structure: no ensemble, no number
    assert Interaction.from_mapping({"type": "hbond", "residue_number": 1, "residue_name": "SER",
                                     "frequency": 0.5, "contact_confidence": "88.5"}).occupancy == 0.5
    table = ("interaction_type\tresidue_number\tresidue_name\tchain\tprotein_atom\tligand_atom\tdistance\t"
             "occupancy\tconfidence\n"
             "hbond\t712\tGLN\tA\tNE2\tN7\t3.1\t0.82\t\n"
             "hydrophobic\t702\tLEU\tA\tCD1\tC11\t\t0.07\t35\n")
    rows = parse_interactions(table)
    assert [(r.occupancy, r.confidence, r.distance) for r in rows] == [(0.82, None, 3.1), (0.07, 35.0, None)]
    with pytest.raises(InteractionError, match="fraction between 0 and 1.*percentage"):
        Interaction("hbond", 1, "SER", occupancy=63)
    with pytest.raises(InteractionError, match="pLDDT scale"):
        Interaction("hbond", 1, "SER", confidence=120)
    with pytest.raises(InteractionError, match="occupancy is not a number"):
        Interaction("hbond", 1, "SER", occupancy="often")


# ---------------------------------------------------------------------------
# Occupancy from outside: what a trajectory would supply
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("style", ("detailed", "schematic"))
def test_occupancy_scales_the_stroke_and_the_legend_names_the_ensemble(style):
    base = diagram_of(CASE)
    share = {(742, "halogen"): 1.0, (711, "hbond"): 0.25}
    dia = with_records(base, lambda r: {"occupancy": share.get((r.residue_number, r.type.value))}, ensemble=FRAMES)
    res, plain = dia.render(style=style), render_of(CASE, style)
    assert res.status == "ok" and res.quality.valid
    width, opacity = occupancy_factors(0.25, steps=True)              # in steps, the default
    weak, was = strokes(res, link(res, "hbond", "GLN711")), strokes(plain, link(plain, "hbond", "GLN711"))
    assert weak and all(p.width == pytest.approx(q.width * width) for p, q in zip(weak, was))
    assert all(p.opacity == pytest.approx(q.opacity * opacity) for p, q in zip(weak, was))
    assert width < 1.0 and opacity < 1.0                               # thinner and paler, both
    # A contact that is always made is the limiting case: drawn exactly like one with no occupancy, but for its share,
    # which is written on it where the others' are (its line is broken around it).
    assert set(strokes(res, link(res, "halogen", "MET742"))) == set(strokes(plain, link(plain, "halogen", "MET742")))
    partial = dia.render(style=style, occupancy_labels="partial")
    assert strokes(partial, link(partial, "halogen", "MET742")) == strokes(plain, link(plain, "halogen", "MET742"))
    assert strokes(res, link(res, "hbond", "LEU704")) == strokes(plain, link(plain, "hbond", "LEU704"))
    weakest = link(res, "hbond", "GLN711")
    assert weakest.data["occupancy"] == "0.250" and "occupancy 25 %" in weakest.title
    assert "occupancy" not in link(res, "hbond", "LEU704").data          # a record without one
    # The number does not say what it is a share of; the figure does.
    text = words(res)
    assert f"Occupancy (share of {FRAMES}, written on the line):" in text and "90–100 %" in text and "< 50 %" in text
    assert "25 %" in text                                              # the share, where a distance would stand
    assert f"the share of {FRAMES} in which a contact is made" in text
    assert "Occupancy" not in words(plain)
    # And it reaches the graph on the contact, not on the residue.
    hbond = next(c for c in res.graph.contacts if c.type == "hbond" and c.residue.endswith(":GLN711"))
    assert hbond.occupancy == 0.25 and res.graph.ensemble == FRAMES
    assert res.graph.node(hbond.residue).occupancy == 0.25             # drawn at its most persistent contact


def test_a_residue_is_drawn_at_its_most_persistent_contact():
    """Leu701 has hydrophobic contacts with R1881 and nothing else: the residue itself is drawn at their share."""
    base = diagram_of("ar_r1881")
    dia = with_records(base, lambda r: {"occupancy": 0.1 if r.residue_number == 701 else 1.0}, ensemble=FRAMES)
    res, plain = dia.render(), render_of("ar_r1881")

    def lash(result):
        (g,) = [g for g in groups(result, "residue") if g.data["residue"] == "A:LEU701"]
        return [f.item.paint for f in inside(result, g) if hasattr(f.item, "paint") and f.item.paint.stroke]
    faint, full = lash(res), lash(plain)
    assert faint and max(p.width for p in faint) < min(p.width for p in full)
    assert max(p.opacity for p in faint) < min(p.opacity for p in full)


# ---------------------------------------------------------------------------
# Confidence the contact carries itself
# ---------------------------------------------------------------------------

def test_a_contact_can_carry_its_own_confidence():
    """On a structure with a pLDDT (made up for the test from its temperature factors) one contact is given a
    confidence of its own; the others keep the one derived from their two ends. On a structure without a pLDDT a
    supplied confidence is drawn all the same, and nothing is read from the temperature factors."""
    slug = "ar_r1881"
    base = predicted_diagram(slug)
    plain = base.render()
    assert not [w for w in words(plain).split() if w.endswith(MARK)]     # every contact above 50: nothing is qualified
    dia = with_records(base, lambda r: {"confidence": 31.0 if (r.residue_number, r.type.value) == (752, "hbond")
                                        else None})
    res = dia.render()
    own = link(res, "hbond", "ARG752")
    assert own.data["plddt"] == "31.0" and own.data["confidence"] == "very_low"
    # The result counts it in the band it is drawn in.
    assert res.confidence["contacts_by_band"]["very_low"] == 1
    assert sum(res.confidence["contacts_by_band"].values()) == sum(plain.confidence["contacts_by_band"].values())
    assert any(f.item.content.endswith(MARK) for f in inside(res, own, Text))
    # Another is derived: the lower of its two ends, the residue's mean pLDDT and the ligand atom's.
    other = link(res, "hbond", "GLN711")
    (row,) = [r.record for r in base.interactions if (r.record.residue_number, r.type.value) == (711, "hbond")]
    st = invented_plddt(structure_of(slug))
    residue = [a.b_iso for a in st.atoms if (a.chain, a.resnum, a.hetero) == ("A", 711, False)]
    (ligand,) = [a.b_iso for a in st.atoms if a.residue == base.ligand_key and a.name == row.ligand_atom]
    assert float(other.data["plddt"]) == pytest.approx(min(sum(residue) / len(residue), ligand), abs=0.05)
    assert next(c for c in res.graph.contacts if c.residue == "R:A:ARG752" and c.type == "hbond").confidence == 31.0
    # It needs no predicted model: on a structure that declares no pLDDT the supplied number is still drawn ...
    crystal = with_records(diagram_of(slug), lambda r: {"confidence": 25.0 if r.type.value == "hbond" else None})
    assert crystal.confidence is None
    drawn = crystal.render()
    marked = [g for g in groups(drawn, "interaction") if g.data.get("confidence") == "very_low"]
    assert marked and all(g.data["type"] == "hbond" and g.data["plddt"] == "25.0" for g in marked)
    assert not groups(crystal.render(burial=False), "confidence")       # ... and nothing is read from B-factors
    # ... and switching confidence off switches this off too.
    assert not [g for g in groups(crystal.render(confidence=False), "interaction") if "confidence" in g.data]


# ---------------------------------------------------------------------------
# Occupancy from a series: the consensus produces the same records
# ---------------------------------------------------------------------------

SET = "gr_steroids"         # four corticosteroids in the glucocorticoid receptor, each from its own crystal structure


def steroids() -> Comparison:
    return Comparison([overlay_diagram(SET, s) for s, _ in overlay_members(SET)])


def counted() -> Counter:
    """How many ligands of the set have a contact of each kind with each residue, as their tables list them
    (``interactions.tsv``, read here as text)."""
    out: Counter = Counter()
    for _, folder in overlay_members(SET):
        with open(folder / "interactions.tsv", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        out.update({(f"{r['residue_name'].capitalize()}{r['residue_number']}", r["interaction_type"]) for r in rows})
    return out


def test_consensus_counts_are_occupancy_on_contact_records():
    cmp_ = steroids()
    records = cmp_.consensus_contacts()
    assert all(isinstance(r, Interaction) and 0.0 < r.occupancy <= 1.0 for r in records)
    by = {(r.residue_label, r.type.value): r for r in records}
    tables = counted()
    assert {key: r.occupancy for key, r in by.items()} == {key: n / cmp_.n for key, n in tables.items()}
    assert by[("Gln570", "hbond")].occupancy == by[("Arg611", "hbond")].occupancy == 1.0     # every steroid's 3-keto O
    assert by[("Gln570", "hbond")].ligand_atom is None
    assert all(r.confidence is None for r in records)                           # crystal structures: no pLDDT
    keys = [(r.chain, r.residue_number, r.insertion_code) for r in records]
    assert keys == sorted(keys) and len(records) == len(cmp_.table.columns)      # the fixed column order
    # The per-residue rows read their counts off these records: there is no second set of numbers.
    for row in cmp_.consensus():
        assert row.types == tuple((c.type, round(c.occupancy * cmp_.n)) for c in row.contacts)
        assert row.occupancy == row.fraction == row.n_ligands / cmp_.n
        assert all(by[(row.label, c.type.value)].occupancy == c.occupancy for c in row.contacts)
        assert row.levels is None and row.n_confident is None
    # With a pLDDT (made up for the test from the temperature factors) each row counts its ligands by band.
    rated = Comparison([predicted_diagram(s) for s, _ in overlay_members(SET)])
    for row in rated.consensus():
        assert row.levels is not None and sum(row.levels) == row.n_ligands and row.n_confident == row.levels[0]


def test_series_occupancy_on_a_single_diagram():
    """The channel end to end on one member of the set: dexamethasone, each contact weighted by the share of the
    four ligands that make it. The structures have a pLDDT, made up for the test from their temperature factors:
    each stroke is scaled by its occupancy on top of what that confidence did to it, and neither is folded into
    the other."""
    cmp_ = Comparison([predicted_diagram(s) for s, _ in overlay_members(SET)])
    k = cmp_.names.index("dexamethasone")
    dia = cmp_.diagram(k)
    assert dia.ensemble == "the 4 ligands of the set" and dia.n_interactions == cmp_.diagrams[k].n_interactions
    share = {(r.record.residue_label, r.type.value): r.record.occupancy for r in dia.interactions}
    tables = counted()
    assert share == {key: tables[key] / cmp_.n for key in share}
    assert (share[("Gln570", "hbond")], share[("Asn564", "hbond")], share[("Leu563", "hbond")]) == (1.0, 0.75, 0.5)
    assert cmp_.diagrams[k].interactions[0].record.occupancy is None            # the member itself is untouched
    res, plain = dia.render(), cmp_.diagrams[k].render()
    assert res.status == plain.status == "ok" and res.quality.valid

    def widest(result, residue):
        g = link(result, "hbond", residue)
        return max(p.width for p in strokes(result, g)), max(p.opacity for p in strokes(result, g))
    assert widest(res, "GLN570") == widest(plain, "GLN570")                     # every ligand makes it
    for residue, occupancy in (("LEU563", 0.5), ("ASN564", 0.75), ("GLN570", 1.0)):
        factor = occupancy_factors(occupancy, steps=True)                 # in steps, the default
        assert widest(res, residue)[0] == pytest.approx(widest(plain, residue)[0] * factor[0])
        assert widest(res, residue)[1] == pytest.approx(widest(plain, residue)[1] * factor[1])
        drawn, was = link(res, "hbond", residue), link(plain, "hbond", residue)
        assert (drawn.data["confidence"], drawn.data["plddt"]) == (was.data["confidence"], was.data["plddt"])
        assert drawn.data["occupancy"] == f"{occupancy:.3f}"
    assert occupancy_factors(0.2)[0] < occupancy_factors(0.5)[0] < occupancy_factors(1.0)[0] == 1.0
    text = words(res)
    assert "Occupancy (share of the 4 ligands of the set, written on the line):" in text
    assert "Occupancy" not in words(plain) and "50 %" in text and "75 %" in text
    # Continuous: each stroke by its own share, and no share written on it.
    smooth = dia.render(occupancy_style="continuous")
    for residue, occupancy in (("LEU563", 0.5), ("ASN564", 0.75), ("GLN570", 1.0)):
        factor = occupancy_factors(occupancy)
        assert widest(smooth, residue)[0] == pytest.approx(widest(plain, residue)[0] * factor[0])
    assert "Occupancy (share of the 4 ligands of the set):" in words(smooth) and "75 %" not in words(smooth)

    # Same contacts, same atoms: occupancy qualifies the evidence and leaves the chemistry alone.
    def claims(result):
        return sorted((g.data["residue"], g.data["type"], g.data["ligand-atoms"])
                      for g in groups(result, "interaction") + groups(result, "contact"))
    assert claims(res) == claims(plain)
    # The fingerprint carries it too, and it survives the CSV.
    fp = dia.fingerprint()
    assert {(b.residue_label, b.type.value): v.occupancy for b, v in fp.bits.items()}[("Leu563", "hbond")] == 0.5
    back = FingerprintTable.from_csv(FingerprintTable([fp]).to_csv())
    assert back.fingerprints[0].bits == fp.bits and "occupancy" in FingerprintTable([fp]).to_csv().splitlines()[0]


# ---------------------------------------------------------------------------
# The share of an ensemble on every contact (``occupancy_labels``)
# ---------------------------------------------------------------------------

def _ensemble(shares: "dict[tuple[int, str], float]"):
    """The committed case with an occupancy on each of its records."""
    import copy

    base = diagram_of(CASE)
    dia = copy.copy(base)
    dia.interactions = [replace(r, record=r.record.with_(occupancy=shares.get(
        (r.record.residue_number, r.record.type.value), 1.0))) for r in base.interactions]
    dia.ensemble, dia._composed = "ten models", {}
    return dia


def _words(res) -> list[str]:
    return [f.item.content for f in walk(res.scene) if isinstance(f.item, Text)]


def test_the_share_is_written_on_every_line_and_under_the_name_of_a_residue_without_one():
    dia = _ensemble({(711, "hbond"): 0.4, (707, "hydrophobic"): 0.7})
    res = dia.render()
    said = _words(res)
    assert "40 %" in said and "70 %" in said                                     # a line, and an eyelash
    # Made in all ten: the other lines, whatever their kind, and the other eyelashes (a residue with a line has none).
    # A water bridge has its share beside its water.
    lines = sum(1 for r in dia.interactions if r.type.value != "hydrophobic")
    lined = {r.record.residue_number for r in dia.interactions if r.type.value != "hydrophobic"}
    lashes = {r.record.residue_number for r in dia.interactions if r.type.value == "hydrophobic"} - lined
    assert said.count("H₂O 100 %") == sum(r.type.value == "water_bridge" for r in dia.interactions) == 1
    assert [w.removeprefix("H₂O ") for w in said].count("100 %") == (lines - 1) + (len(lashes) - 1)
    assert "Distances in Å." not in " ".join(said)
    assert "the share is written on the line, or under the name of a residue that has none" in " ".join(said)
    # A distance that is asked for stays on a contact that is made in all.
    kept = _words(dia.render(distance_labels="all"))
    assert "40 %" in kept and any(re.fullmatch(r"\d\.\d\d", w) for w in kept) and "Distances in Å." in " ".join(kept)


def test_with_partial_labels_only_a_contact_made_in_fewer_than_all_has_its_share():
    dia = _ensemble({(711, "hbond"): 0.4, (707, "hydrophobic"): 0.7})
    said = _words(dia.render(occupancy_labels="partial"))
    assert "40 %" in said and "70 %" not in said and "100 %" not in said
    assert any(re.fullmatch(r"\d\.\d\d", w) for w in said)
    plain = _words(dia.render(occupancy_style="continuous"))
    assert "40 %" not in plain and "70 %" not in plain
    with pytest.raises(ValueError, match="occupancy_labels"):
        dia.render(occupancy_labels="some")


def test_a_figure_without_occupancy_is_the_same_file_either_way():
    dia = diagram_of(CASE)
    assert dia.render(occupancy_labels="partial").svg == dia.render().svg
