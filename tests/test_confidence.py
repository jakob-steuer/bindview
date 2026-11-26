"""Confidence rendering: what is read, when it may be read, and what it is allowed to change.

A crystal structure has no confidence. The pLDDT drawn here is made up for the tests (``conftest.invented_plddt``:
100 minus an atom's temperature factor) and declared as a model declares it; the AlphaFold database's model of
TECTB declares a pLDDT of its own.
"""

from __future__ import annotations

import gzip
import re
from collections import Counter
from dataclasses import replace

import pytest

from conftest import TESTDATA, invented_plddt, meta_of, predicted_diagram, structure_of
from bindview import Confidence, Diagram, ModelConfidence, load_interactions
from bindview.confidence import LOOKS, MARK, PLAIN, band_of, look_of
from bindview.diagram import _confidence_for
from bindview.scene import Group, Line, Path as ScenePath, Polyline, Rect, Text
from bindview.structure import Structure, parse_mmcif, read_structure
from scenetools import groups, inside, walk

KIT = "kit_imatinib"
#: Two cases with a pLDDT made up for them.
SLUGS = ("kit_imatinib", "dhfr_methotrexate")
AFDB_MODEL = TESTDATA / "interface_pae" / "afdb_tectb" / "model.cif.gz"


def afdb_text() -> str:
    with gzip.open(AFDB_MODEL, "rt", encoding="utf-8") as handle:
        return handle.read()


def invented(slug: str, structure=None, low: float = 25.0, **kwargs) -> Diagram:
    """A case's diagram on ``structure`` (its own model unless given) with a pLDDT made up from it, held at
    ``low`` or above."""
    ligand = meta_of(slug)["ligand"]
    kwargs.setdefault("name", ligand["name"])
    return Diagram.from_structure(invented_plddt(structure if structure is not None else structure_of(slug), low=low),
                                  ligand["residue_name_in_cif"], ligand["smiles"],
                                  load_interactions(TESTDATA / slug / "interactions.tsv"), **kwargs)


#: Imatinib's contact that is made weak: Thr670, its temperature factors set to 80, so that the pLDDT made up from
#: them is the lowest the invention gives, 25.
WEAK = 670


def with_a_weak_residue(**kwargs) -> Diagram:
    st = structure_of(KIT)
    weak = Structure(atoms=[replace(a, b_iso=80.0) if (a.chain, a.resnum) == ("A", WEAK) else a for a in st.atoms],
                     source=st.source, chem_comp_smiles=dict(st.chem_comp_smiles), links=st.links)
    return invented(KIT, weak, **kwargs)


# ---------------------------------------------------------------------------
# The guard: the B-factor column is confidence only when the file says so
# ---------------------------------------------------------------------------

def test_plddt_is_read_only_when_the_file_declares_it():
    declared = read_structure(AFDB_MODEL)
    assert declared.declares_plddt and declared.plddt_scale == 1.0
    assert ("local", "pLDDT", "pLDDT") in declared.qa_metrics
    # The same file without its declaration is no model with a pLDDT, whatever its numbers look like.
    out, n = re.subn(r"loop_\n(_ma_qa_metric\.\w+\s*\n)+(?:(?!#|loop_|_).*\n)+", "", afdb_text(), count=1)
    assert n == 1 and "_ma_qa_metric." not in out.replace("_ma_qa_metric_", "")
    assert not parse_mmcif(out, source="stripped.cif").declares_plddt
    # A crystal structure: the numbers are there, they are temperature factors, and they are not used.
    crystal = structure_of(KIT)
    assert not crystal.declares_plddt and all(a.b_iso is not None for a in crystal.atoms)
    key = crystal.select_ligand(meta_of(KIT)["ligand"]["residue_name_in_cif"])
    assert _confidence_for(crystal, key, "auto", None, None, None) is None


def test_a_global_only_declaration_is_not_enough_and_unit_scale_is_honoured():
    text = afdb_text()
    only_global = text.replace("2 local  pLDDT 1 pLDDT", "2 global pLDDT 1 pLDDT")
    assert only_global != text and not parse_mmcif(only_global).declares_plddt
    unit = text.replace("2 local  pLDDT 1 pLDDT", "2 local  pLDDT 1 'pLDDT in [0,1]'")
    assert unit != text and parse_mmcif(unit).plddt_scale == 100.0


def test_without_a_declaration_the_figure_is_the_unqualified_one():
    """A crystal drawn as it is, and the same atoms with a declared pLDDT drawn with confidence switched off: the
    same figure, so nothing of a confidence leaks through."""
    crystal = Diagram.from_structure(structure_of(KIT), meta_of(KIT)["ligand"]["residue_name_in_cif"],
                                     meta_of(KIT)["ligand"]["smiles"], load_interactions(TESTDATA / KIT / "interactions.tsv"),
                                     name=meta_of(KIT)["ligand"]["name"])
    assert crystal.confidence is None
    res = crystal.render(burial=False)                                  # the buried fraction is not confidence
    assert res.confidence is None and not groups(res, "confidence") and MARK not in res.svg.split("</title>", 1)[1]
    assert res.svg == predicted_diagram(KIT).render(confidence=False, burial=False).svg


def test_the_caller_may_assert_or_supply_confidence_explicitly():
    crystal = structure_of(KIT)
    key = crystal.select_ligand(meta_of(KIT)["ligand"]["residue_name_in_cif"])
    asserted = _confidence_for(crystal, key, "bfactor", None, None, None)
    assert asserted is not None and asserted.origin == "asserted"
    factors = [a.b_iso for a in crystal.residue_atoms(crystal.find_residue("A", WEAK))]
    assert asserted.residue(("A", WEAK, "")) == pytest.approx(sum(factors) / len(factors))    # the column itself

    given = Confidence.from_values({"A:670": 31.0, ("A", 673): 88.0}, {"C1": 40.0})
    assert _confidence_for(crystal, key, given, None, None, None) is given and given.origin == "supplied"
    with pytest.raises(ValueError, match="plddt must be"):
        _confidence_for(crystal, key, "guess", None, None, None)


# ---------------------------------------------------------------------------
# The numbers: read out of the atom records, residue by residue
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("slug", SLUGS)
def test_residue_plddt_is_read_from_the_atom_records(slug):
    """The pLDDT of a residue is the mean of its atoms' in the file, and of the ligand's atoms each its own; every
    residue the contacts name has one, keyed on the numbering the table uses."""
    dia = predicted_diagram(slug)
    conf = dia.confidence
    assert conf is not None and conf.origin == "declared"
    model = invented_plddt(structure_of(slug))
    assert dia.interactions
    for r in dia.interactions:
        values = [a.b_iso for a in model.residue_atoms(r.residue)]
        assert conf.residue(r.residue) == pytest.approx(sum(values) / len(values))
    assert conf.ligand_atoms == {a.name: a.b_iso for a in model.residue_atoms(dia.ligand_key)}


def test_a_contact_is_as_believable_as_its_weaker_end():
    conf = Confidence.from_values({("A", 10): 90.0, ("A", 11): 30.0}, {"C1": 60.0, "C2": 95.0, "C3": 20.0})
    assert conf.contact(("A", 10, ""), ["C1"]) == 60.0          # the ligand end is the weaker one
    assert conf.contact(("A", 11, ""), ["C2"]) == 30.0          # the residue is
    assert conf.contact(("A", 10, ""), ["C1"]) != (90.0 + 60.0) / 2.0
    assert conf.contact(("A", 10, ""), ["C1", "C2"]) == pytest.approx(77.5)    # a ring: mean over its atoms
    assert conf.contact(("A", 99, ""), ["C3"]) == 20.0          # one end unknown: what is known
    assert conf.contact(("A", 99, ""), ["nope"]) is None


def test_bands_are_alphafolds_and_nothing_else():
    assert [band_of(v) for v in (97, 90, 89.9, 70, 69.9, 50, 49.9, 0)] == [
        "very_high", "very_high", "confident", "confident", "low", "low", "very_low", "very_low"]
    assert look_of(None) is PLAIN and look_of(85.0).level == 0 and look_of(60.0).level == 1
    low, very_low = LOOKS["low"], LOOKS["very_low"]
    # Every step down changes weight AND opacity; only the lowest band carries the mark.
    assert 1.0 > low.width > very_low.width and 1.0 > low.opacity > very_low.opacity
    assert very_low.mark and not low.mark and not LOOKS["confident"].mark


# ---------------------------------------------------------------------------
# The drawing
# ---------------------------------------------------------------------------

def _residue(res, number):
    (g,) = [g for g in groups(res, "residue") if re.search(rf"\D{number}$", g.data["residue"])]
    return g


def _strokes(res, group):
    return [f.item.paint for f in inside(res, group, (Line, Polyline, ScenePath)) if f.item.paint.stroke]


@pytest.mark.parametrize("style", ("detailed", "schematic"))
def test_low_plddt_residues_are_drawn_weaker_by_a_greyscale_safe_channel(style):
    """A contact on a residue in the lowest band is visibly weaker than one on a confident residue, by a channel
    that survives printing in grey."""
    res = with_a_weak_residue().render(style=style)
    assert res.status == "ok" and res.quality.valid
    strong, weak = _residue(res, 799), _residue(res, WEAK)
    assert strong.data["confidence"] == "confident" and weak.data["confidence"] == "very_low"
    assert float(weak.data["plddt"]) == pytest.approx(25.0, abs=0.05)

    def name(g):
        return [f.item for f in inside(res, g, Text) if f.in_class("fragment") is None]

    # Not colour: the name of the weak residue is set in regular weight and carries the mark.
    assert all(t.bold for t in name(strong)[:1]) and not any(t.bold for t in name(weak))
    assert any(t.content.endswith(MARK) for t in name(weak)) and not any(MARK in t.content for t in name(strong))
    if style == "detailed":
        # Thinner strokes, not only paler ones.
        assert max(p.width for p in _strokes(res, weak)) < min(p.width for p in _strokes(res, strong))
    else:
        # The outline changes kind: solid for the confident residue, broken for the weak one.
        assert all(not p.dash for p in _strokes(res, strong)) and all(p.dash for p in _strokes(res, weak))


def test_every_contact_takes_the_lower_of_its_two_ends():
    dia = with_a_weak_residue()
    conf = dia.confidence
    res = dia.render()
    drawn = groups(res, "interaction") + groups(res, "contact")
    assert drawn
    for g in drawn:
        chain, rest = g.data["residue"].split(":")
        number = int(re.search(r"(\d+)$", rest).group(1))
        names = g.data["ligand-atoms"].split(",")
        want = min(conf.residue((chain, number, "")), conf.ligand(names))
        assert float(g.data["plddt"]) == pytest.approx(want, abs=0.05)
        assert g.data["confidence"] == band_of(want)
    # The contacts of the weak residue are in the lowest band and carry the mark; no other link does.
    links = groups(res, "interaction")
    marked = {g.data["residue"] for g in links if any(MARK in f.item.content for f in inside(res, g, Text))}
    assert marked == {g.data["residue"] for g in links if g.data["confidence"] == "very_low"} == {f"A:THR{WEAK}"}
    assert res.confidence["contacts_by_band"] == dict(Counter(
        band_of(min(conf.residue(r.residue), conf.ligand([dia.ligand.names[i] for i in r.ligand.atoms])))
        for r in dia.interactions))
    # A model in which every atom is confident is left exactly as it was: no mark, nothing below full strength.
    good = invented(KIT, low=90.0).render()
    assert good.confidence["contacts_by_band"].keys() == {"very_high"}
    assert not [f for f in walk(good.scene) if isinstance(f.item, Text) and f.item.content.endswith(MARK)]
    assert all(f.item.paint.opacity == 1.0 for g in groups(good, "interaction") for f in inside(good, g, Line))


@pytest.mark.parametrize("slug", SLUGS)
def test_model_level_evidence_is_printed_at_the_top(slug):
    """The figure carries the evidence of the whole model where a reader cannot miss it: the ipTM that was
    handed in, and the mean pLDDT of the pocket."""
    iptm = 0.83
    dia = predicted_diagram(slug, model_confidence={"iptm": iptm})
    pocket = dia.confidence.pocket_plddt
    res = dia.render(title="A title")
    (strip,) = groups(res, "confidence")
    text = " ".join(f.item.content for f in inside(res, strip, Text))
    assert f"ipTM {iptm:.2f}" in text and f"pocket pLDDT {pocket:.0f}" in text
    assert strip.data["iptm"] == f"{iptm:.2f}" and strip.data["pocket-plddt"] == f"{pocket:.1f}"
    # Above the drawing, directly under the title -- not in a footnote.
    (diagram,) = [g for g in res.scene.items if isinstance(g, Group) and g.cls == "diagram"]
    strip_y = max(f.box()[3] for f in inside(res, strip, Text))
    assert strip_y < min(f.box()[1] for f in inside(res, diagram))
    assert res.confidence["iptm"] == iptm and res.confidence["pocket_plddt"] == round(pocket, 2)
    assert f"ipTM {iptm:.2f}" in res.scene.description


def test_model_confidence_comes_from_the_sidecar_by_name_only(tmp_path):
    for name, sidecar in (("job_model.cif", "job_summary_confidences.json"), ("model.cif", "summary_confidences.json")):
        (tmp_path / name).write_text("data_x\n")
        assert ModelConfidence.beside(tmp_path / name) is None            # nothing there yet
        (tmp_path / sidecar).write_text('{"iptm": 0.5, "ptm": 0.6, "ranking_score": 0.7}')
        assert ModelConfidence.beside(tmp_path / name) == tmp_path / sidecar
    (tmp_path / "other.cif").write_text("data_x\n")
    assert ModelConfidence.beside(tmp_path / "other.cif") is None         # never "whatever JSON is around"
    m = ModelConfidence.from_json(tmp_path / "summary_confidences.json")
    assert (m.iptm, m.ptm, m.ranking_score, m.trusted, m.flag) == (0.5, 0.6, 0.7, True, None)


def test_a_verdict_and_a_number_written_as_text_are_read_as_what_they_say(tmp_path):
    """A table or a hand-written JSON gives ``"false"`` and ``"0.83"`` as text. Any text that is not
    empty is true to Python, so the word is read and not its truth: a model that is not to be trusted
    is not drawn as trusted. A number as text is that number, and is not left out."""
    from bindview.errors import BindviewError

    assert ModelConfidence.from_mapping({"iptm": 0.3, "trusted": "false"}).trusted is False
    assert [ModelConfidence.from_mapping({"trusted": said}).trusted for said in (
        "False", " no ", False, 0, "true", "Yes", True, 1, None)] == [False] * 4 + [True] * 5
    assert ModelConfidence.from_mapping({}).trusted is True
    with pytest.raises(BindviewError, match="'trusted' is 'maybe'.*true or false"):
        ModelConfidence.from_mapping({"trusted": "maybe"})
    assert ModelConfidence.from_mapping({"iptm": "0.83", "ptm": " 0.9 ", "ranking_score": None}) == ModelConfidence(
        iptm=0.83, ptm=0.9)
    assert ModelConfidence.from_mapping({"iptm": 1, "ptm": 0.5}) == ModelConfidence(iptm=1.0, ptm=0.5)
    for key, value in (("iptm", "high"), ("ptm", [0.9]), ("ranking_score", True), ("iptm", "nan")):
        with pytest.raises(BindviewError, match=f"{key!r} is .*, which is no number"):
            ModelConfidence.from_mapping({key: value})
    (tmp_path / "summary_confidences.json").write_text('{"iptm": "0.41", "trusted": "false", "flag": "grade C"}')
    read = ModelConfidence.from_json(tmp_path / "summary_confidences.json")
    assert (read.iptm, read.trusted, read.flag) == (0.41, False, "grade C")


@pytest.mark.parametrize("style", ("detailed", "schematic"))
def test_an_untrusted_model_is_marked_not_merely_faint(style):
    """A model that its caller says is not to be trusted is marked as such, by shape and not by shade, and is
    never presented as equivalent to a confident prediction."""
    flagged = predicted_diagram(KIT, trusted=False, flag="grade C").render(style=style)
    (banner,) = groups(flagged, "model-flag")
    words = " ".join(f.item.content for f in inside(flagged, banner, Text))
    assert "not trustworthy" in words and "grade C" in words
    # A filled bar across the top of the page and a frame around all of it: shape, not shade.
    (bar,) = [f.item for f in inside(flagged, banner, Rect)]
    assert bar.w == pytest.approx(flagged.scene.width) and bar.y == 0.0 and bar.paint.fill == "warning"
    (frame,) = [f.item for g in groups(flagged, "model-flag-frame") for f in inside(flagged, g, Rect)]
    assert frame.paint.stroke == "warning" and frame.paint.width >= 3.0
    assert frame.w > flagged.scene.width - 4.0 and frame.h > flagged.scene.height - 4.0
    assert flagged.confidence["trusted"] is False and flagged.confidence["flag"] == "grade C"
    assert any("not trustworthy" in w and "grade C" in w for w in flagged.warnings)
    assert "not trustworthy" in flagged.scene.description
    # bindview does not decide this by itself: the same model, unflagged, carries no banner.
    plain = predicted_diagram(KIT).render(style=style)
    assert not groups(plain, "model-flag") and plain.confidence["trusted"] is True


@pytest.mark.parametrize("slug", SLUGS)
def test_confidence_changes_the_evidence_not_the_chemistry(slug):
    dia = predicted_diagram(slug)

    def claims(res):
        return sorted((g.cls, g.data["residue"], g.data["type"], g.data["ligand-atoms"], g.data.get("distance", ""))
                      for g in groups(res, "interaction") + groups(res, "contact"))
    for style in ("detailed", "schematic"):
        with_conf, without = dia.render(style=style), dia.render(style=style, confidence=False)
        assert with_conf.status == without.status == "ok"
        assert claims(with_conf) == claims(without) and len(claims(with_conf)) > 0
        assert with_conf.quality.n_interactions == without.quality.n_interactions == dia.n_interactions


def test_confidence_survives_the_summary_fallback():
    """Drawn 40 mm wide, the figure is the summary of the residues, and the confidence stays on it."""
    res = predicted_diagram(KIT).render(figure_width_mm=40.0)
    assert res.figure == "summary" and groups(res, "confidence")
    words = [f.item.content for f in walk(res.scene) if isinstance(f.item, Text)]
    assert "pLDDT" in words and any("pocket pLDDT" in w for w in words)


def test_confidence_rendering_is_deterministic():
    def fresh():
        return predicted_diagram(KIT, trusted=False, flag="grade C").render(style="schematic", theme="dark").svg
    assert fresh() == fresh()
