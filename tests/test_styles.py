"""The settings of a style, one by one, and the lining of the pocket."""

from __future__ import annotations

import re

import pytest

from bindview import Comparison, Diagram
from bindview.errors import GraphError
from bindview.interactions import Interaction, InteractionType
from bindview.resolve import ProteinSite, ResolvedInteraction
from bindview.styles.base import STYLE_SETTINGS, RenderOptions, Settings, contact_face
from bindview.theme import get_palette
from bindview import load_interactions
from conftest import (CASES, TESTDATA, chain_diagram, diagram_of, meta_of, overlay_diagram, overlay_members, render_of,
                      structure_of, trimer, without_waters)

_chain: dict = {}


def subject(name: str):
    """A case's diagram (built once), or ``"hirudin"``: the ten residues of hirudin on thrombin (2UUF, chain H) as a
    chain ligand, with the plain distance table."""
    if name != "hirudin":
        return diagram_of(name)
    if not _chain:
        _chain["hirudin"] = chain_diagram("thrombin_hirudin")
    return _chain["hirudin"]


def lined_of(case: str):
    """The schematic figure of a case, its lining from the contacts (the style's own)."""
    return render_of(case, "schematic") if case != "hirudin" else subject(case).render(style="schematic")


def residues(svg: str) -> set[str]:
    return set(re.findall(r'<g class="residue"[^>]*data-residue="([^"]+)"', svg))


def stretches(svg: str) -> list[tuple[str, set[str]]]:
    """``(surface, residues)`` of every stretch of the lining."""
    return [(dict(re.findall(r'data-([a-z]+)="([^"]*)"', tag))["surface"],
             set(dict(re.findall(r'data-([a-z]+)="([^"]*)"', tag))["residues"].split()))
            for tag in re.findall(r'<g class="lining"[^>]*>', svg)]


def surfaces(svg: str) -> dict[str, set[str]]:
    """The surfaces each residue lines the pocket with."""
    by: dict[str, set[str]] = {}
    for surface, who in stretches(svg):
        for r in who:
            by.setdefault(r, set()).add(surface)
    return by


def forms(svg: str) -> dict[str, str]:
    return {m.group("who"): m.group("form") for m in re.finditer(
        r'<g class="residue"[^>]*data-form="(?P<form>[^"]+)"[^>]*data-residue="(?P<who>[^"]+)"', svg)}


# -- the lining --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("case", CASES)
def test_every_residue_that_makes_a_contact_has_a_stretch(case):
    svg = lined_of(case).svg
    lined = set().union(*(who for _, who in stretches(svg)))
    assert residues(svg) and {r.split("~")[0] for r in residues(svg)} <= lined


def test_a_contact_of_the_backbone_is_not_the_colour_of_the_residues_class():
    """Bicalutamide: Met742's halogen bond is made by the oxygen of its backbone, and so is one of Leu704's two
    contacts (the other is hydrophobic); Arg752 gives its hydrogen bond with its charged side chain."""
    assert surfaces(lined_of("ar_bicalutamide").svg) == {
        "A:MET742": {"polar"}, "A:LEU704": {"polar", "hydrophobic"}, "A:GLN711": {"polar"}, "A:ARG752": {"positive"},
        "A:LEU707": {"hydrophobic"}, "A:LEU741": {"hydrophobic"}, "A:PHE764": {"hydrophobic"},
        "A:ILE898": {"hydrophobic"}, "A:ILE899": {"hydrophobic"}}


def _made(kind: InteractionType, name: str, site: ProteinSite) -> ResolvedInteraction:
    record = Interaction(type=kind, residue_number=1, residue_name=name, chain="A")
    return ResolvedInteraction(record=record, index=0, residue=None, ligand=None, protein=site, method="named")


@pytest.mark.parametrize("kind, name, site, face", [
    (InteractionType.HYDROPHOBIC, "LYS", ProteinSite("atom", ("CB",)), "hydrophobic"),
    (InteractionType.HBOND, "LEU", ProteinSite("atom", ("O",)), "polar"),
    (InteractionType.HBOND, "ASP", ProteinSite("atom", ("OD1",)), "negative"),
    (InteractionType.HBOND, "ASP", ProteinSite("atom", ("N",)), "polar"),
    (InteractionType.HBOND, "TRP", ProteinSite("atom", ("NE1",)), "polar"),
    (InteractionType.HALOGEN, "MET", ProteinSite("atom", ("SD",)), "polar"),
    (InteractionType.SALTBRIDGE, "ARG", ProteinSite("group", ("NE", "NH1", "NH2"), 1), "positive"),
    (InteractionType.PI_CATION, "PHE", ProteinSite("ring", ("CG", "CD1")), "hydrophobic"),
    (InteractionType.PI_STACKING, "HIS", ProteinSite("ring", ("CG", "ND1")), "hydrophobic"),
    (InteractionType.HBOND, "G", ProteinSite("atom", ("N7",)), "polar"),
    (InteractionType.HBOND, "A", ProteinSite("atom", ("OP1",)), "negative"),
    (InteractionType.METAL, "ZN", ProteinSite("atom", ("ZN",)), "metal"),
    (InteractionType.HBOND, "SER", ProteinSite("residue"), "polar"),
])
def test_the_colour_is_of_the_part_that_makes_the_contact(kind, name, site, face):
    assert contact_face(_made(kind, name, site)) == face


def test_the_key_names_the_colours_that_are_drawn():
    """Celecoxib's pocket is lined by hydrophobic residues and by the backbone of Thr199: no charge."""
    svg = lined_of("ca2_celecoxib").svg
    assert {s for s, _ in stretches(svg)} == {"hydrophobic", "polar"}
    key = re.search(r'<g class="legend".*?</g>', svg, re.S).group(0)
    words = re.findall(r">([^<]+)</text>", key)
    at = words.index("Pocket lining:")
    assert words[at + 1:at + 3] == ["hydrophobic", "polar"] and "charged (+)" not in words


def test_boxes_on_a_plane_have_no_lining():
    made = trimer("A", "B", layout="plane").render(style="schematic")
    assert made.svg is not None and stretches(made.svg) == []


def test_the_lining_is_all_that_the_option_changes():
    def rest(svg: str) -> str:
        svg = re.sub(r'<g class="lining".*?</g>', "", svg, flags=re.S)
        svg = re.sub(r'<polyline[^>]*stroke-width="4.4"[^>]*/>', "", svg)
        return re.sub(r"\s+", " ", re.search(r'<g class="diagram".*?(?=<g class="legend")', svg, re.S).group(0))
    for case in ("kit_imatinib", "dhfr_methotrexate", "hirudin"):
        dia = subject(case)
        assert rest(lined_of(case).svg) == rest(dia.render(style="schematic", lining="nearest").svg)
        assert dia.render(style="schematic", lining="contacts").svg == lined_of(case).svg
    assert stretches(diagram_of("kit_imatinib").render(style="schematic", lining="nearest").svg) == []


# -- the settings --------------------------------------------------------------------------------------------

def by_hand(settings: Settings) -> dict:
    return dict(residues=settings.residues, lines=settings.lines.name, exposure=settings.exposure,
                lining=settings.lining)


@pytest.mark.parametrize("case", ["kit_imatinib", "dhfr_methotrexate", "hirudin"])
def test_one_style_set_by_hand_is_the_other(case):
    dia = subject(case)
    made = dia.render(style="detailed", distance_labels="none", **by_hand(STYLE_SETTINGS["schematic"]))
    assert made.svg == lined_of(case).svg
    plain = dia.render(style="detailed", distance_labels="none")
    other = dia.render(style="schematic", **by_hand(STYLE_SETTINGS["detailed"]))
    assert other.svg == plain.svg


def test_each_setting_alone():
    dia = diagram_of("kit_imatinib")
    plain = render_of("kit_imatinib", "detailed").svg
    assert stretches(plain) == []
    # The halo of an atom open to the solvent: celecoxib has such atoms (imatinib has none).
    assert 'fill="#c9ced6"' not in render_of("ca2_celecoxib").svg
    assert 'fill="#c9ced6"' in diagram_of("ca2_celecoxib").render(exposure=True).svg
    # The lining of the detailed figure is that of its contacts, as in the schematic one: residue by residue.
    assert surfaces(dia.render(lining="contacts").svg) == surfaces(lined_of("kit_imatinib").svg)
    assert dia.render(lining=True).svg == dia.render(lining="contacts").svg
    assert set(forms(dia.render(residues="bubbles").svg).values()) == {"bubble"}
    assert "H-bond (side chain)" in dia.render(lines="schematic").svg and "H-bond" not in plain
    schematic = dia.render(style="schematic", exposure=False, lining=False).svg
    assert 'fill="#c9ced6"' not in schematic and stretches(schematic) == []
    # By name: a chip, but for a residue whose contacts are all hydrophobic, which stays an eyelash.
    kinds: dict[str, set] = {}
    for row in load_interactions(TESTDATA / "kit_imatinib" / "interactions.tsv"):
        kinds.setdefault(f"{row.chain}:{row.residue_name}{row.residue_number}", set()).add(row.type.value)
    assert forms(dia.render(residues="names").svg) == {r: "eyelash" if k == {"hydrophobic"} else "chip"
                                                       for r, k in kinds.items()}


def test_residues_one_by_one():
    dia = diagram_of("kit_imatinib")
    r = dia.render(style="schematic",
                   residues={"Cys673": "atoms", "A:THR670": "names", "Nope1": "atoms"})
    assert r.status == "ok" and r.quality.overlaps == 0
    drawn = forms(r.svg)
    assert drawn["A:CYS673"] == "fragment" and drawn["A:THR670"] == "chip"
    assert {v for k, v in drawn.items() if k not in ("A:CYS673", "A:THR670")} == {"bubble"}
    assert any("Nope1" in w and "no residue" in w for w in r.warnings)
    # One set of lines in the figure, whoever draws the residue, and one key.
    assert r.svg.count(">Pocket lining:<") == 1 and ">Hydrogen bond<" not in r.svg and ">H-bond (side chain)<" in r.svg
    rest = forms(dia.render(residues={"Cys673": "atoms", "*": "bubbles"}).svg)
    assert rest.pop("A:CYS673") == "fragment" and set(rest.values()) == {"bubble"}


def test_a_setting_that_does_not_exist_is_refused():
    for options, words in ((dict(residues="balls"), "residues is one of"), (dict(lines="fine"), "lines is one of"),
                           (dict(lining="left"), "lining is one of"), (dict(residues={"Met743": "x"}), "Met743"),
                           (dict(solvent_mark="dot"), "solvent_mark")):
        with pytest.raises(ValueError, match=words):
            RenderOptions(**options)
    with pytest.raises(ValueError, match="unknown style"):
        diagram_of("kit_imatinib").render(style="sepia")


@pytest.mark.parametrize("name, valid", [("distance_labels", ("auto", "all", "none")),
                                         ("show_chain", ("auto", "always", "never")),
                                         ("on_unreadable", ("summary", "render", "none")),
                                         ("coords", ("auto", "coordgen", "rdkit"))])
def test_a_word_that_an_option_does_not_know_is_refused_with_the_words_it_knows(name, valid):
    for wrong in ("xyzzy", None, valid[0].upper(), 1):
        with pytest.raises(ValueError) as refused:
            RenderOptions(**{name: wrong})
        said = str(refused.value)
        assert said.startswith(f"{name} is one of ") and said.endswith(f"not {wrong!r}")
        assert all(repr(word) in said for word in valid)
    for word in valid:
        assert getattr(RenderOptions(**{name: word}), name) == word


def test_a_theme_that_does_not_exist_is_refused_with_every_theme_there_is():
    with pytest.raises(ValueError) as refused:
        diagram_of("kit_imatinib").render(theme="drak")
    said = str(refused.value)
    assert said.startswith("unknown theme 'drak'") and all(word in said for word in ("'light'", "'dark'", "'auto'", "Palette"))
    assert diagram_of("kit_imatinib").render(theme="auto").theme == "auto"
    # "auto" is a figure with both palettes and no palette itself: asked for as one, that is what is said.
    with pytest.raises(ValueError, match="'auto' is a figure with both palettes"):
        get_palette("auto")


def test_a_graph_made_with_other_residues_is_refused():
    dia = diagram_of("kit_imatinib")
    graph = dia.render(residues="bubbles").graph
    again = dia.render(residues="bubbles", graph=graph)
    assert again.svg == dia.render(residues="bubbles").svg
    with pytest.raises(GraphError, match="other options"):
        dia.render(graph=graph)


# -- a predicted water -----------------------------------------------------------------------------------------

def dry(case: str):
    """A case as a model without water would give it: its waters taken out, and with them the rows of its table
    that go through one."""
    m = meta_of(case)["ligand"]
    rows = [r for r in load_interactions(TESTDATA / case / "interactions.tsv") if r.type.value != "water_bridge"]
    return Diagram.from_structure(without_waters(structure_of(case)), ligand=m["residue_name_in_cif"],
                                  smiles=m["smiles"], interactions=rows)


@pytest.mark.parametrize("style", ["detailed", "schematic"])
def test_a_predicted_water_stands_on_a_glow_or_in_a_box(style):
    """R1881 without its waters: one site where a water would bridge it to the protein is predicted."""
    dia = dry("ar_r1881")
    glow, filled, outline = (dia.render(style=style, solvent_sites=True, solvent_mark=m).svg
                             for m in ("glow", "filled", "outline"))
    assert dia.render(style=style, solvent_sites=True).svg == glow
    assert glow.count('opacity="0.07"') == 2 and 'stroke-width="1.9"' not in glow       # in the figure and in its key
    assert filled.count('opacity="0.24"') == 2 and 'stroke-width="1.9"' in filled       # in the figure and in its key
    assert 'opacity="0.24"' not in outline and 'stroke-width="0.85"' in outline
    # Nothing moves for it: the same box in the same place.
    box = r'<rect x="(-?[\d.]+)" y="(-?[\d.]+)" width="(5[\d.]+)"'
    assert re.search(box, filled).groups() == re.search(box, outline).groups()


def test_a_comparison_says_that_it_has_not_these_settings():
    made = Comparison([overlay_diagram("ar_ligands", slug) for slug, _ in overlay_members("ar_ligands")]).render(
        "difference", lining="contacts", residues="bubbles")
    assert any("residues, lining" in w and "comparison" in w for w in made.warnings)
