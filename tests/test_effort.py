"""What a figure costs: a layout is made once for all figures that it is the layout of, and a chain that is too
crowded can be given up early (``effort="quick"``)."""

from __future__ import annotations

import copy
from dataclasses import replace

import pytest

from conftest import chain_diagram, diagram_of, fresh_diagram, one_residue_peptide
from bindview import Diagram
from bindview import diagram as module
from bindview.layout import LayoutParams, LigandField, layout_bodies, layout_key
from bindview.layout import engine
from bindview.styles.base import RenderOptions


#: The nonamer of HLA-A2 (3MRE, chain P) as a chain ligand: its first layout leaves residues without a place.
PEPTIDE = "mhc_peptide"
_kept: dict = {}


def fresh(case: str, **how) -> Diagram:
    """A diagram of its own: a case of ``testdata/cases.json``, or the peptide as a chain ligand."""
    return chain_diagram(case, **how) if case == PEPTIDE else fresh_diagram(case, **how)


def kept(case: str) -> Diagram:
    """The same, built once for the module: do not change it."""
    if case != PEPTIDE:
        return diagram_of(case)
    if case not in _kept:
        _kept[case] = fresh(case)
    return _kept[case]


def twin(dia: Diagram) -> Diagram:
    """A diagram as other tests have drawn it, with what is kept of it, to do with as one likes."""
    dia = copy.copy(dia)
    dia._composed = dict(dia._composed)
    return dia


def one_residue() -> Diagram:
    """The same nonamer written as one hetero residue with its SMILES, with the plain distance table: too crowded to
    be drawn with its atoms, after six layouts. A diagram of its own."""
    from bindview.distance_table import ligand_contacts

    model, smiles = one_residue_peptide(PEPTIDE)
    return Diagram.from_structure(model, "LIG", smiles=smiles, interactions=ligand_contacts(model, "LIG", smiles))


def one_residue_kept() -> Diagram:
    """The same, built once for the module: do not change it."""
    if "one residue" not in _kept:
        _kept["one residue"] = one_residue()
    return _kept["one residue"]


@pytest.fixture
def runs(monkeypatch):
    """The layouts that are made, as they are made."""
    made: list[engine.LayoutResult] = []
    run = engine._Engine.run

    def counted(self):
        result = run(self)
        made.append(result)
        return result

    monkeypatch.setattr(engine._Engine, "run", counted)
    return made


# ---------------------------------------------------------------------------
# One layout for all the figures that it is the layout of
# ---------------------------------------------------------------------------

#: What is drawn, written or said differently, with everything where it was.
MOVES_NOTHING = [dict(plddt_colors=True), dict(plddt_colors="bonds"), dict(title="A title", subtitle="and more"), dict(bare=True),
                 dict(colors="contrast"), dict(theme="auto", interactive=True, stylesheet="external"),
                 dict(charge_halos=True), dict(absent=[673], absent_label="a construct"), dict(atom_labels="names"),
                 dict(on_unreadable="none"), dict(on_unreadable="render"), dict(effort="quick"), dict(legend=False)]


@pytest.mark.parametrize("style", ["detailed", "schematic"])
def test_a_figure_that_moves_nothing_is_not_laid_out_again(style, runs):
    dia = fresh("kit_imatinib")
    first = dia.render(style=style)
    assert len(runs) == 1
    where = {n.id: (n.x, n.y, n.angle) for n in first.graph.nodes if n.kind == "residue"}
    drawn = [(options, dia.render(style=style, **options)) for options in MOVES_NOTHING]
    assert len(runs) == 1
    for options, res in drawn:
        assert {n.id: (n.x, n.y, n.angle) for n in res.graph.nodes if n.kind == "residue"} == where, options
    for options, res in drawn[:1] + drawn[5:9:3]:                 # ... and it is the file that a diagram of its own gives
        own = fresh("kit_imatinib")
        assert res.svg == own.render(style=style, **options).svg, options
    del runs[1:]
    dia.render(style=style, distance_labels="all")                # the lengths need room: another layout
    assert len(runs) == 2


def test_what_a_layout_depends_on_is_in_its_key():
    dia = diagram_of("kit_imatinib")
    result = dia.render()
    comp = next(c for c in dia._composed.values() if getattr(c, "layout", None) is not None and c.quality == result.quality)
    _, _, bodies, _, fld, params, _, _ = comp.parts
    key = layout_key(fld, bodies, params)
    assert key == layout_key(fld, bodies, LayoutParams(unit=params.unit))
    body = bodies[0]
    payload, body.payload = body.payload, None
    assert layout_key(fld, bodies, params) == key                 # what is drawn of a residue is not where it is
    body.payload = payload
    for change in (dict(ligand_clearance=body.ligand_clearance + 1.0), dict(priority=body.priority + 1.0),
                   dict(shapes_r=body.shapes_r + 0.5), dict(axis=(0.0, 1.0)), dict(key="another")):
        assert layout_key(fld, [replace(body, **change), *bodies[1:]], params) != key, change
    for k in range(len(body.ports)):
        for change in (dict(ideal=1.0), dict(target=(0.0, 0.0)), dict(fan=0.123), dict(label_size=(1.0, 2.0)), dict(drawn=not body.ports[k].drawn)):
            ports = list(body.ports)
            ports[k] = replace(ports[k], **change)
            assert layout_key(fld, [replace(body, ports=ports), *bodies[1:]], params) != key, change
    assert layout_key(fld, bodies[::-1], params) != key
    assert layout_key(fld, bodies, replace(params, w_ink=params.w_ink + 1.0)) != key
    assert layout_key(LigandField(fld.dep, reach=100.0, step=fld.step), bodies, params) != key


# ---------------------------------------------------------------------------
# What is kept of the figures of a diagram has a bound
# ---------------------------------------------------------------------------

def held(dia: Diagram) -> dict[str, int]:
    """How many a diagram keeps: figures as they were drawn, fields round the ligand, placements of the residues."""
    count = {"figure": 0, "field": 0, "layout": 0}
    for key in dia._composed:
        count[key[0] if key[0] in ("field", "layout") else "figure"] += 1
    return count


def test_a_diagram_drawn_under_many_titles_keeps_a_few_of_them(runs):
    dia = fresh_diagram("ca2_celecoxib")
    first = dia.render(title="Figure 0")
    for k in range(1, 100):
        dia.render(title=f"Figure {k}")
    assert held(dia)["figure"] <= 40 and len(runs) == 1
    assert held(dia) == {"figure": module._KEPT_AT_MOST["figure"], "field": 1, "layout": 1}
    # One that went is drawn again, and on the layout that stayed.
    assert dia.render(title="Figure 0").svg == first.svg and len(runs) == 1


def test_the_layout_that_was_used_longest_ago_goes_first(runs, monkeypatch):
    monkeypatch.setitem(module._KEPT_AT_MOST, "layout", 2)
    monkeypatch.setitem(module._KEPT_AT_MOST, "field", 2)
    monkeypatch.setitem(module._KEPT_AT_MOST, "figure", 1)        # (so that every figure asks for its layout)
    dia = fresh_diagram("ca2_celecoxib")
    for scale in (30.0, 26.0, 30.0, 22.0):                        # 26 is now the one that was used longest ago
        dia.render(bond_length=scale)
    assert held(dia) == {"figure": 1, "field": 2, "layout": 2} and len(runs) == 3
    dia.render(bond_length=30.0)
    assert len(runs) == 3                                         # kept
    dia.render(bond_length=26.0)
    assert len(runs) == 4                                         # gone, and laid out again


# ---------------------------------------------------------------------------
# A chain that is too crowded
# ---------------------------------------------------------------------------

def test_a_crowded_chain_is_given_up_after_two_looks(runs, monkeypatch):
    """The nonamer as one residue: six layouts and a table. Its first look leaves three of its 21 residues without a
    place, under the bound at which a chain counts as crowded (four): with the bounds set at three and one, it is
    given up after the first look and the roomiest, each left half way, and a table is drawn."""
    monkeypatch.setattr(module, "_crowded", lambda n: 3)          # (so that this one is given up)
    monkeypatch.setattr(module, "_hopeless", lambda n: 1)
    dia = one_residue()
    none = dia.render(effort="quick", on_unreadable="none")
    assert none.status == "unreadable" and none.svg is None
    assert len(runs) == 2 and all(r.abandoned for r in runs) and runs[0].left >= 3 and runs[1].left >= 1
    message = none.warnings[-1]
    assert f"{runs[-1].left} residue(s) could not be placed" in message
    assert "(effort='quick': with the most room that the page allows, and not tried any further)" in message
    table = dia.render(effort="quick")
    assert table.figure == "summary" and len(runs) == 2           # the verdict is kept
    assert "not tried any further" in table.warnings[-1]
    drawn = dia.render(style="schematic", effort="quick")         # a figure that can be drawn is
    assert drawn.status == "ok" and not runs[-1].abandoned and len(runs) == 3
    assert drawn.svg == dia.render(style="schematic").svg and len(runs) == 3


def test_a_chain_that_is_drawn_is_the_same_file_either_way(runs):
    """The nonamer as a chain: its first layout leaves residues without a place, fewer than a crowded chain does, so
    no look at the roomiest is needed and it is laid out as it is with full effort."""
    quick = fresh(PEPTIDE).render(effort="quick")
    n = len({i.record.residue_id for i in kept(PEPTIDE).interactions})
    assert quick.status == "ok" and len(runs) > 1 and not any(r.abandoned for r in runs)
    assert 0 < runs[0].left < module._crowded(n) and runs[-1].left == 0
    assert quick.svg == kept(PEPTIDE).render().svg


def test_a_look_at_the_roomiest_changes_nothing_of_what_is_drawn(runs, monkeypatch):
    fresh(PEPTIDE).render()
    patient = len(runs)
    del runs[:]
    monkeypatch.setattr(module, "_crowded", lambda n: 2)          # (so that this one is looked at)
    monkeypatch.setattr(module, "_hopeless", lambda n: 9)
    quick = fresh(PEPTIDE).render(effort="quick")
    assert [r.abandoned for r in runs] == [True, False] + [False] * patient   # the first, the roomiest, and then as ever
    assert quick.svg == kept(PEPTIDE).render().svg and quick.quality == kept(PEPTIDE).render().quality


def test_what_was_given_up_is_not_what_a_patient_figure_is_made_of(runs, monkeypatch):
    """... in whichever order the two are asked for. The nonamer as one residue, with the bounds of a crowded chain
    set so that it is given up."""
    monkeypatch.setattr(module, "_crowded", lambda n: 1)          # (so that this one is given up)
    monkeypatch.setattr(module, "_hopeless", lambda n: 1)
    whole = one_residue_kept().render()                           # six layouts, as with full effort
    assert whole.status == "unreadable" and "not tried any further" not in whole.warnings[-1]
    del runs[:]
    dia = one_residue()
    first = dia.render(effort="quick", on_unreadable="none")
    assert first.status == "unreadable" and runs[0].abandoned and len(runs) == 2
    assert "not tried any further" in first.warnings[-1]
    patient = dia.render()
    assert patient.svg == whole.svg and patient.warnings == whole.warnings
    del runs[:]
    again = dia.render(effort="quick", on_unreadable="none")      # (kept, and not the patient one's)
    assert again.status == "unreadable" and again.warnings == first.warnings and again.quality == first.quality
    assert not runs
    other = twin(one_residue_kept())
    late = other.render(effort="quick", on_unreadable="none")
    assert late.status == "unreadable" and late.warnings == first.warnings and late.quality == first.quality
    # A figure that is to be drawn whatever comes of it is not given up.
    forced = other.render(effort="quick", on_unreadable="render")
    assert forced.figure == "diagram" and "not tried any further" not in forced.warnings[-1]


def test_a_ligand_drawn_as_atoms_has_one_effort(runs):
    quick = fresh("kit_imatinib").render(effort="quick")
    assert quick.status == "ok" and runs[0].left == 0 and not runs[0].abandoned
    assert quick.svg == diagram_of("kit_imatinib").render().svg
    with pytest.raises(ValueError, match="effort is 'full' or 'quick'"):
        RenderOptions(effort="none")
    assert (module._crowded(10), module._hopeless(10), module._crowded(61), module._hopeless(61)) == (4, 3, 7, 5)


def test_the_engine_leaves_a_layout_only_when_it_is_told_to():
    dia = diagram_of("kit_imatinib")
    result = dia.render()
    comp = next(c for c in dia._composed.values() if getattr(c, "layout", None) is not None and c.quality == result.quality)
    _, _, bodies, _, fld, params, _, _ = comp.parts
    told = layout_bodies(fld, bodies, replace(params, give_up=1))
    assert told.left == 0 and not told.abandoned
    assert told.poses == layout_bodies(fld, bodies, replace(params, give_up=0)).poses
