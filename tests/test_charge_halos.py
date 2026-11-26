"""``charge_halos``: the shading of a charged group, off unless ``True`` is passed.

A salt bridge and a pi-cation contact are made with a group, not with one of its atoms, and the group can be
shaded in the colour of the contact. These tests hold that the shading is all that the option changes, with
its name in the key: every line ends where it does without.
"""

from __future__ import annotations

import re

import pytest

from conftest import TESTDATA, diagram_of, meta_of, model_of, nucleic_diagram
from bindview import STYLES, Diagram
from bindview.cli import main
from bindview.interactions import load_interactions


def shading(svg: str) -> int:
    """How many strokes of shading a figure has."""
    group = re.search(r'<g class="halos">(.*?)</g>', svg, flags=re.S)
    return group.group(1).count("<line") if group else 0


def but_for_the_shading(svg: str) -> str:
    return re.sub(r'\s*<g class="halos">.*?</g>', "", svg, count=1, flags=re.S)


@pytest.mark.parametrize("style", STYLES)
def test_the_shading_comes_and_nothing_else_changes(style):
    dia = nucleic_diagram()                      # three salt bridges and three pi-cation contacts
    keyed, plain = dia.render(style=style, charge_halos=True), dia.render(style=style)
    drawn = dia.render(style=style, charge_halos=True, charge_key=False)
    assert dia.render(style=style, charge_halos=False).svg == plain.svg           # the default, said aloud
    assert shading(drawn.svg) > 0 and shading(plain.svg) == 0
    assert but_for_the_shading(drawn.svg) == but_for_the_shading(plain.svg)
    assert (plain.status, plain.warnings) == (drawn.status, drawn.warnings)
    # The key names the shading where there is one (both kinds shade a group here), and not otherwise.
    assert keyed.svg.count("Charged group (") == 2 and "Charged group" not in drawn.svg + plain.svg


@pytest.mark.parametrize("case", ["dhfr_methotrexate", "kit_imatinib"])
def test_a_charged_group_of_a_protein_is_shaded_or_not_the_same_way(case):
    """Methotrexate's two salt bridges; imatinib's salt bridges and its two rings over a charge."""
    dia = diagram_of(case)
    drawn, plain = dia.render(charge_halos=True, charge_key=False), dia.render()
    assert shading(drawn.svg) > 0 and shading(plain.svg) == 0
    assert but_for_the_shading(drawn.svg) == but_for_the_shading(plain.svg)
    assert "Charged group" in dia.render(charge_halos=True).svg and "Charged group" not in plain.svg


def test_a_figure_without_a_charged_group_is_the_same_file():
    dia = diagram_of("ca2_celecoxib")            # hydrogen bond, ring stacking and hydrophobic contacts only
    assert shading(dia.render(charge_halos=True).svg) == 0
    assert dia.render(charge_halos=True).svg == dia.render().svg


def test_the_command_line_draws_it_on_request(tmp_path):
    drawn, plain, said = tmp_path / "figure.svg", tmp_path / "plain.svg", tmp_path / "said.svg"
    case = "dhfr_methotrexate"
    meta = meta_of(case)
    ligand = meta["ligand"]["residue_name_in_cif"]
    asked = ["render", str(model_of(case)), "--ligand", ligand, "--smiles-from", "meta.json", "-q"]
    assert main(asked + ["-o", str(drawn), "--charge-halos"]) == 0
    assert main(asked + ["-o", str(plain)]) == 0
    assert main(asked + ["-o", str(said), "--no-charge-halos"]) == 0
    same = Diagram.from_structure(model_of(case), ligand, smiles=meta["ligand"]["smiles"],
                                  interactions=load_interactions(TESTDATA / case / "interactions.tsv"))
    assert drawn.read_text() == same.render(charge_halos=True).svg
    assert plain.read_text() == said.read_text() == same.render().svg
    assert shading(drawn.read_text()) > 0 and shading(plain.read_text()) == 0
