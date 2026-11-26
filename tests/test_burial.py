"""Buried fraction: the surface maths, where the number is reported, and the warning below
``BURIAL_WARNING_BELOW``."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from conftest import (chain_diagram, diagram_of, meta_of, model_of, overlay_diagram, overlay_members,
                      predicted_diagram, render_of, structure_of)
from bindview import Comparison, Diagram
from bindview.cli import main as cli
from bindview.exposure import (BURIAL_WARNING_BELOW, PROBE, SASA_POINTS, Burial, burial_warning, ligand_burial,
                             structure_burial)
from bindview.scene import Text
from bindview.structure import Atom, Structure
from scenetools import groups, inside, walk

KIT = "kit_imatinib"


def words(res) -> str:
    return " ".join(f.item.content for f in walk(res.scene) if isinstance(f.item, Text))


def part_of(st: Structure, keep) -> Structure:
    """The atoms of ``st`` that ``keep`` accepts, as a structure of their own."""
    from dataclasses import replace

    return Structure(atoms=[replace(a, index=i) for i, a in enumerate(a for a in st.atoms if keep(a))], source="part")


def percent(fraction: float) -> str:
    return f"{100.0 * fraction:.0f} %"


def sulfate() -> Diagram:
    """The sulfate of 1Z95 as the ligand: an ion of the crystal's buffer on the protein's surface, with the plain
    distance table."""
    return Diagram.from_structure(structure_of("ar_bicalutamide"), "SO4", "[O-]S(=O)(=O)[O-]", "distances")


# ---------------------------------------------------------------------------
# The surface
# ---------------------------------------------------------------------------

def test_surface_of_simple_bodies_is_what_geometry_says():
    none = np.zeros((0, 3))
    one = ligand_burial(np.zeros((1, 3)), ["C"], none, [])
    assert one.sasa_free == pytest.approx(4.0 * math.pi * (1.7 + PROBE) ** 2) and one.buried_fraction == 0.0
    far = ligand_burial(np.array([[0.0, 0, 0], [50.0, 0, 0]]), ["C", "O"], none, [])
    assert far.sasa_free == pytest.approx(4.0 * math.pi * ((1.7 + PROBE) ** 2 + (1.52 + PROBE) ** 2))
    # Two overlapping spheres: each loses the cap the other covers (to the accuracy of 240 points).
    d, r = 1.5, 1.7 + PROBE
    pair = ligand_burial(np.array([[0.0, 0, 0], [d, 0, 0]]), ["C", "C"], none, [])
    assert pair.sasa_free == pytest.approx(2.0 * 2.0 * math.pi * r * (r + d / 2.0), rel=0.02)
    assert pair.atom_free[0] == pytest.approx(pair.atom_free[1], rel=0.02)
    # An atom inside a shell of atoms has no accessible surface left.
    shell = 4.0 * np.array([[x, y, z] for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1) if (x, y, z) != (0, 0, 0)])
    caged = ligand_burial(np.zeros((1, 3)), ["C"], shell / np.linalg.norm(shell, axis=1)[:, None] * 3.6, ["C"] * 26)
    assert caged.sasa_complex == 0.0 and caged.buried_fraction == 1.0 and caged.exposed_fraction == 0.0
    assert ligand_burial(none, [], none, []).buried_fraction == 0.0


def test_burial_is_monotone_converged_and_deterministic():
    dia = diagram_of(KIT)
    st, key = dia.structure, dia.ligand_key
    b = structure_burial(st, key)
    assert b == structure_burial(st, key) and b.n_points == SASA_POINTS and b.probe == PROBE
    assert len(b.atoms) == len(b.atom_free) == len(b.atom_complex) == dia.ligand.n_atoms and list(b.atoms) == dia.ligand.names
    assert all(c <= f + 1e-9 for c, f in zip(b.atom_complex, b.atom_free))            # cover only ever removes surface
    assert b.sasa_complex == pytest.approx(sum(b.atom_complex)) and 0.0 < b.buried_fraction < 1.0
    fine = structure_burial(st, key, n_points=1920)
    assert abs(fine.buried_fraction - b.buried_fraction) < 0.01                      # 240 points are enough
    # Taking protein away exposes the ligand; taking all of it away leaves the free ligand.
    half = part_of(st, lambda a: a.residue == key or a.resnum % 2 == 0)
    alone = part_of(st, lambda a: a.residue == key)
    assert structure_burial(half, key).buried_fraction < b.buried_fraction
    assert structure_burial(alone, key).buried_fraction == 0.0 and structure_burial(alone, key).sasa_free == b.sasa_free


def test_water_is_solvent_and_does_not_bury_anything():
    dia = diagram_of(KIT)
    st, key = dia.structure, dia.ligand_key
    lig = st.residue_atoms(key)
    waters = [Atom(index=len(st.atoms) + i, name="O", element="O", resname="HOH", chain="W", resnum=i + 1, icode="",
                   x=a.x + 2.8, y=a.y, z=a.z, hetero=True) for i, a in enumerate(lig)]
    wet = Structure(atoms=list(st.atoms) + waters, source="wet")
    assert structure_burial(wet, key) == structure_burial(st, key)
    ions = [Atom(index=w.index, name="NA", element="Na", resname="NA", chain="W", resnum=w.resnum, icode="",
                 x=w.x, y=w.y, z=w.z, hetero=True) for w in waters]
    salted = Structure(atoms=list(st.atoms) + ions, source="salted")                   # anything else does count
    assert structure_burial(salted, key).buried_fraction > structure_burial(st, key).buried_fraction


# ---------------------------------------------------------------------------
# Where it is reported
# ---------------------------------------------------------------------------

def test_burial_is_reported_on_every_result_and_in_json(tmp_path, capsys):
    dia = diagram_of(KIT)
    measured = structure_burial(dia.structure, dia.ligand_key)
    res = render_of(KIT)
    assert res.burial["buried_fraction"] == pytest.approx(measured.buried_fraction, abs=0.0005)
    assert measured.buried_fraction >= BURIAL_WARNING_BELOW and res.burial["warning"] is False
    assert 0 < res.burial["sasa_complex"] < res.burial["sasa_free"] and res.burial["warn_below"] == BURIAL_WARNING_BELOW
    assert res.graph.meta["burial"] == res.burial
    empty = Diagram.from_structure(dia.structure, dia.ligand_key, dia.ligand.smiles, [])
    small = dia.render(figure_width_mm=40.0)
    assert empty.render().status == "empty" and small.figure == "summary"
    for figure in (empty.render(), small):                                  # empty figure, summary table: still reported
        assert figure.burial["sasa_free"] > 0
    out = tmp_path / "x.svg"
    assert cli(["render", str(model_of(KIT)), "--ligand", meta_of(KIT)["ligand"]["residue_name_in_cif"],
                "--smiles-from", "meta.json", "-o", str(out), "--json", "-q"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["burial"]["buried_fraction"] == pytest.approx(measured.buried_fraction, abs=0.0005)
    assert set(printed["burial"]) == {"sasa_free", "sasa_complex", "buried_fraction", "exposed_fraction", "probe",
                                      "n_points", "warn_below", "warning"}


def test_the_figure_shows_burial_unless_asked_not_to():
    dia = diagram_of(KIT)
    fraction = dia.burial().buried_fraction
    plain = dia.render(burial=False)
    assert "buried" not in words(plain) and not groups(plain, "confidence")
    shown = render_of(KIT)
    assert dia.render(burial=True).svg == shown.svg                          # it is the default
    # A crystal has no confidence to stand beside: the number stands alone.
    (strip,) = groups(shown, "confidence")
    text = " ".join(f.item.content for f in inside(shown, strip, Text))
    assert text == f"Ligand buried {percent(fraction)}"
    assert strip.data["buried-fraction"] == f"{fraction:.3f}" and not groups(shown, "burial-warning")
    assert shown.status == plain.status == "ok" and shown.quality.valid
    # Nothing but the provenance line moved: same contacts on the same atoms.
    def claims(res):
        return sorted((g.data["residue"], g.data["type"], g.data["ligand-atoms"]) for g in groups(res, "interaction"))
    assert claims(shown) == claims(plain)
    # With a pLDDT (made up here) the number stands beside its readout.
    predicted = predicted_diagram(KIT).render()
    (strip,) = groups(predicted, "confidence")
    text = " ".join(f.item.content for f in inside(predicted, strip, Text))
    assert "ligand pLDDT" in text and text.endswith(f"ligand buried {percent(fraction)}")


def test_a_poorly_buried_ligand_is_warned_about_in_words():
    dia = sulfate()
    fraction = dia.burial().buried_fraction
    low = dia.render()
    assert fraction < BURIAL_WARNING_BELOW and low.burial["warning"] is True
    (note,) = [w for w in low.warnings if "buried" in w]
    left = 100 - int(percent(fraction).split()[0])
    assert f"only {percent(fraction)} buried" in note and f"{left} % of its surface stays solvent-exposed" in note
    assert "may be incomplete" in note and "oligomer" in note and "cofactor" in note
    assert "may not sit in a pocket" in note                                  # both readings, neither chosen
    assert not [w for w in render_of(KIT).warnings if "buried" in w]
    off = dia.render(burial=False)
    assert "buried" not in words(off)                                         # with burial=False, a note on the result only
    (line,) = groups(low, "burial-warning")                                   # ... and in the figure, the default
    said = " ".join(f.item.content for f in inside(low, line, Text))
    assert "open to solvent" in said and "may be incomplete" in said and line.data["buried-fraction"] == f"{fraction:.3f}"
    assert all(f.item.fill == "warning" for f in inside(low, line, Text))
    # The function behind it, at the edges.
    assert burial_warning(None) is None and burial_warning(Burial(100.0, 30.0, (), ())) is None      # exactly 0.70
    assert burial_warning(Burial(100.0, 30.1, (), ())) is not None and burial_warning(Burial(0.0, 0.0, (), ())) is None


def test_a_chain_of_residues_is_measured_and_not_warned_about():
    """A peptide or a strand lies along a surface: its buried fraction is measured and printed, and no warning is
    drawn from it. Hirudin's tail on thrombin and a hairpin of the riboswitch are both below the cut."""
    from conftest import nucleic_chain

    for res in (chain_diagram("thrombin_hirudin").render(style="detailed"), nucleic_chain().render(style="detailed")):
        assert res.burial["buried_fraction"] < BURIAL_WARNING_BELOW                  # below the cut all the same
        assert res.burial["warning"] is False and res.burial["warn_below"] is None
        assert not [w for w in res.warnings if "buried" in w] and not groups(res, "burial-warning")
        assert "buried" in words(res)                                                 # the number stays
    # By what it is, not by how it is drawn: the same peptide asked to be drawn as atoms is still a chain.
    from bindview.backends import choose_backend

    lig = chain_diagram("thrombin_hirudin").ligand
    assert choose_backend(lig, "atoms").decision.is_chain and choose_backend(lig, "auto").decision.is_chain
    assert not sulfate().backend.decision.is_chain                                    # a small molecule is warned of


def test_comparisons_report_burial_per_ligand():
    """The two ligands of the androgen receptor: each with its own buried fraction, under its name."""
    diagrams = [overlay_diagram("ar_ligands", slug) for slug, _ in overlay_members("ar_ligands")]
    cmp_ = Comparison(diagrams)
    res = cmp_.render(burial=False)
    assert set(res.burial) == {d.name for d in diagrams}
    for d in diagrams:
        assert res.burial[d.name]["buried_fraction"] == pytest.approx(d.burial().buried_fraction, abs=0.0005)
    assert "buried" not in words(res)
    shown = words(cmp_.render())                                              # in the figure: the default
    assert all(f"Ligand buried {percent(d.burial().buried_fraction)}" in shown for d in diagrams)


def test_the_warning_is_no_detector_of_an_incomplete_site():
    """Carbon monoxide in the pocket of haemoglobin's alpha chain, its haem taken away: the site is incomplete,
    and the ligand is enclosed all the same, so nothing is said."""
    st = structure_of("hb_carbon_monoxide")
    without = part_of(st, lambda a: a.resname != "HEM")
    whole, open_ = structure_burial(st, st.select_ligand("A:143")), structure_burial(without, without.select_ligand("A:143"))
    assert open_.sasa_complex > whole.sasa_complex                            # the haem did cover it
    assert open_.buried_fraction >= BURIAL_WARNING_BELOW and burial_warning(open_) is None
