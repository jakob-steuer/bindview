"""Two things a figure can be asked for in colour: a scheme (``colors``), and pLDDT in AlphaFold's four
colours (``plddt_colors``).

A figure that asks for neither is the figure it was. A scheme changes colours and nothing else, and its
classes of residues are held to a measure: the worst of all pairs further apart than in the standard
palette, for full colour vision and for three kinds of colour blindness. Colouring by pLDDT fills
residues and sets atoms on the colour of their band, draws nothing weaker, and says so where there is no
pLDDT to colour by.
"""

from __future__ import annotations

import itertools
import math
import re
import xml.etree.ElementTree as ET

import pytest

from conftest import (TESTDATA, chain_diagram, diagram_of, invented_plddt, meta_of, model_of, predicted_diagram,
                      structure_of)
from bindview import LIGHT, Comparison, Diagram, load_interactions
from bindview.cli import main
from bindview.confidence import BANDS, band_of
from bindview.styles.base import LINING_WIDTH
from bindview.styles.ligand_draw import coloured_bond_items
from bindview.theme import (PLDDT_COLORS, PLDDT_TEXT, Metrics, SCHEMES, CONTRAST, CONTRAST_DARK, fixed_tokens,
                          get_palette, scheme)

SVG = "{http://www.w3.org/2000/svg}"
CLASSES = ("hydrophobic", "polar", "positive", "negative", "glycine")


def doubtful(**options) -> Diagram:
    """Imatinib in c-Kit with a pLDDT that is made up and below 50 on every atom (:func:`conftest.invented_plddt`,
    held at 45 at most): every contact of it in the lowest band."""
    m = meta_of("kit_imatinib")["ligand"]
    return Diagram.from_structure(invented_plddt(structure_of("kit_imatinib"), high=45.0),
                                  ligand=m["residue_name_in_cif"], smiles=m["smiles"], name=m["name"],
                                  interactions=load_interactions(TESTDATA / "kit_imatinib" / "interactions.tsv"),
                                  **options)


@pytest.fixture(scope="module")
def steroid() -> Diagram:
    """Bicalutamide in the androgen receptor with a made-up pLDDT (100 minus the temperature factor): its residues
    and its atoms lie in two bands."""
    return predicted_diagram("ar_bicalutamide")


def residues(svg: str):
    """Every residue of a figure: its group, and the first shape in it that is filled (a bubble, or
    the plate under a name), where it has one."""
    for group in ET.fromstring(svg).iter(SVG + "g"):
        if group.get("class") == "residue":
            yield group, next((e for e in group if e.tag in (SVG + "path", SVG + "rect")
                               and e.get("fill", "none") != "none"), None)


# -- how far two colours are apart ----------------------------------------------
# Distance in OKLab (Ottosson 2020), times 100; a kind of colour blindness is simulated first, on linear
# RGB, with the matrices of Machado, Oliveira and Fernandes (2009) at full severity.

MACHADO = {
    "protan": ((0.152286, 1.052583, -0.204868), (0.114503, 0.786281, 0.099216), (-0.003882, -0.048116, 1.051998)),
    "deutan": ((0.367322, 0.860646, -0.227968), (0.280085, 0.672501, 0.047413), (-0.011820, 0.042940, 0.968881)),
    "tritan": ((1.255528, -0.076749, -0.178779), (-0.078411, 0.930809, 0.147602), (0.004733, 0.691367, 0.303900)),
}


def linear(colour: str) -> tuple[float, ...]:
    values = [int(colour[i:i + 2], 16) / 255.0 for i in (1, 3, 5)]
    return tuple(v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in values)


def oklab(rgb) -> tuple[float, float, float]:
    r, g, b = rgb
    lms = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b,
           0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b,
           0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b)
    l, m, s = (v ** (1.0 / 3.0) for v in lms)
    return (0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
            1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
            0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s)


def apart(a: str, b: str, kind: "str | None" = None) -> float:
    def seen(colour: str):
        rgb = linear(colour)
        if kind is not None:
            rgb = tuple(min(1.0, max(0.0, sum(k * v for k, v in zip(row, rgb)))) for row in MACHADO[kind])
        return oklab(rgb)

    return 100.0 * math.dist(seen(a), seen(b))


def worst(colours, kind: "str | None" = None) -> float:
    return min(apart(a, b, kind) for a, b in itertools.combinations(colours, 2))


def contrast(a: str, b: str) -> float:
    def luminance(colour: str) -> float:
        r, g, b_ = linear(colour)
        return 0.2126 * r + 0.7152 * g + 0.0722 * b_

    hi, lo = sorted((luminance(a), luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


# ---------------------------------------------------------------------------
# A figure that asks for neither
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("style", ["detailed", "schematic"])
def test_a_figure_that_asks_for_neither_is_the_figure_it_was(style):
    dia = diagram_of("kit_imatinib")
    for theme in ("light", "dark", "auto"):
        plain = dia.render(style=style, theme=theme).svg
        assert dia.render(style=style, theme=theme, colors="standard", plddt_colors=False).svg == plain
        assert not any(colour in plain for colour in PLDDT_COLORS.values())
        assert not any("plddt" in token for token in re.findall(r"--lf-[a-z-]+", plain))    # no token was added
    assert not set(fixed_tokens()) & set(LIGHT.tokens())


# ---------------------------------------------------------------------------
# The colour scheme
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("style", ["detailed", "schematic"])
def test_a_scheme_changes_colours_and_nothing_else(style):
    dia = diagram_of("kit_imatinib")
    standard, scheme_ = dia.render(style=style), dia.render(style=style, colors="contrast")
    assert scheme_.status == "ok" and scheme_.svg != standard.svg
    assert scheme_.graph.to_json() == standard.graph.to_json()              # laid out as it was

    def uncoloured(svg: str) -> str:
        return re.sub(r"#[0-9a-fA-F]{6}\b", "#", svg)

    assert uncoloured(scheme_.svg) == uncoloured(standard.svg)


def test_a_scheme_has_a_palette_for_each_theme():
    """Celecoxib's residues are of two classes, hydrophobic and polar."""
    dia = diagram_of("ca2_celecoxib")
    light = dia.render(style="schematic", colors="contrast")
    dark = dia.render(style="schematic", colors="contrast", theme="dark")
    auto = dia.render(style="schematic", colors="contrast", theme="auto")
    assert (light.theme, dark.theme, auto.theme) == ("light", "dark", "auto")       # the theme, whatever the scheme
    for figure, palette in ((light, CONTRAST), (dark, CONTRAST_DARK)):
        fills = {filled.get("fill") for _, filled in residues(figure.svg)}
        assert fills == {palette.residue_classes["hydrophobic"], palette.residue_classes["polar"]}
        assert f'fill="{palette.background}"' in figure.svg
    # One file for both themes: the light palette as it stands, the dark one behind the reader's setting.
    assert f"--lf-class-hydrophobic:{CONTRAST.residue_classes['hydrophobic']}" in auto.svg
    assert f"--lf-class-hydrophobic:{CONTRAST_DARK.residue_classes['hydrophobic']}" in auto.svg
    assert scheme("contrast") == (CONTRAST, CONTRAST_DARK) and set(SCHEMES) == {"standard", "contrast"}


def test_colours_that_are_no_scheme_are_refused():
    dia = diagram_of("ca2_celecoxib")
    with pytest.raises(ValueError, match="unknown colours 'rainbow'; the schemes are 'standard', 'contrast'"):
        dia.render(colors="rainbow")
    mine = LIGHT.with_(name="mine", hbond="#000000")
    assert dia.render(theme=mine).theme == "mine"
    with pytest.raises(ValueError, match="a Palette of your own is its colours"):
        dia.render(theme=mine, colors="contrast")
    with pytest.raises(ValueError, match="unknown theme"):
        get_palette("sepia", "contrast")


#: The worst pair of the five classes, standard and then contrast, for full vision, protanopia, deuteranopia and
#: tritanopia, to a tenth: the values the documentation of the colours gives.
WORST = {"light": ["11.1 | 14.6", "7.7 | 7.0", "5.2 | 9.3", "6.9 | 10.8"],
         "dark": ["8.6 | 12.8", "6.3 | 11.3", "3.0 | 10.3", "4.7 | 5.0"]}
#: How near a residue's colour comes to that of the nearest line, standard and then contrast.
NEAREST = {"light": "8.0 and 9.1", "dark": "10.4 and 10.8"}


def test_the_classes_of_the_scheme_are_further_apart_than_the_standard_ones():
    """The worst of all pairs of the scheme's five fills is further apart than in the standard palette, for full
    colour vision and for red-green colour blindness taken as the worse of its two kinds; and a name stands on
    every fill at 4.5:1 or better."""
    lines = ("hbond", "hydrophobic", "saltbridge", "pi_stacking", "pi_cation", "halogen", "metal", "water_bridge")

    def nearest(palette) -> float:
        return min(apart(palette.protein_bond, getattr(palette, token)) for token in lines)

    for theme in ("light", "dark"):
        ours, theirs = get_palette(theme, "contrast"), get_palette(theme)
        mine = [ours.residue_classes[c] for c in CLASSES]
        std = [theirs.residue_classes[c] for c in CLASSES]
        assert worst(mine) > worst(std)
        assert (min(worst(mine, "protan"), worst(mine, "deutan"))
                > min(worst(std, "protan"), worst(std, "deutan")))
        assert worst(mine, "tritan") > worst(std, "tritan")                 # not asked for by the rule; so it is
        cells = [f"{worst(std, kind):.1f} | {worst(mine, kind):.1f}" for kind in (None, "protan", "deutan", "tritan")]
        assert cells == WORST[theme]
        assert min(contrast(ours.bubble_text, c) for c in ours.residue_classes.values()) >= 4.5
        # What is drawn of a residue is no nearer to the colour of any line than the standard's grey is ...
        assert nearest(ours) > nearest(theirs)
        assert f"{nearest(theirs):.1f} and {nearest(ours):.1f}" == NEAREST[theme]
        # ... and a line means what it meant: the scheme does not touch the colours of contacts.
        assert all(getattr(ours, token) == getattr(theirs, token) for token in lines)
    # Said, and not left to be found: for protanopia alone the light scheme is a little closer.
    light = [CONTRAST.residue_classes[c] for c in CLASSES]
    assert worst(light, "protan") < worst([LIGHT.residue_classes[c] for c in CLASSES], "protan")


# ---------------------------------------------------------------------------
# pLDDT in AlphaFold's colours
# ---------------------------------------------------------------------------

def test_the_colours_of_the_bands_are_alphafolds():
    assert PLDDT_COLORS == {"very_high": "#0053d6", "confident": "#65cbf3", "low": "#ffdb13", "very_low": "#ff7d45"}
    assert list(PLDDT_COLORS) == [name for name, _ in BANDS] == list(PLDDT_TEXT)
    assert all(contrast(PLDDT_TEXT[band], colour) >= 4.5 for band, colour in PLDDT_COLORS.items())


@pytest.mark.parametrize("style", ["schematic", "detailed"])
def test_a_residue_is_filled_with_the_colour_of_its_band(steroid, style):
    plain, coloured = steroid.render(style=style), steroid.render(style=style, plddt_colors=True)
    assert coloured.status == plain.status == "ok"
    seen = set()
    for group, filled in residues(coloured.svg):
        band = group.get("data-confidence")
        seen.add(band)
        assert filled.get("fill") == PLDDT_COLORS[band], group.get("data-residue")
        name = next(e for e in group if e.tag == SVG + "text")
        assert name.get("fill") == PLDDT_TEXT[band]
    assert len(seen) > 1                                    # a pocket with a sure part and a doubtful one
    # The bands themselves are what they were: the option changes how they are shown, not what they are.
    assert ([g.get("data-confidence") for g, _ in residues(coloured.svg)]
            == [g.get("data-confidence") for g, _ in residues(plain.svg)])
    assert coloured.confidence == plain.confidence


@pytest.mark.parametrize("style", ["schematic", "detailed"])
def test_nothing_is_drawn_weaker_for_its_plddt(style):
    dia = doubtful()                                        # every contact below pLDDT 50
    plain, coloured = dia.render(style=style), dia.render(style=style, plddt_colors=True)

    def texts(svg: str) -> list[str]:
        return ["".join(t.itertext()) for t in ET.fromstring(svg).iter(SVG + "text")]

    assert any(t.endswith("?") for t in texts(plain.svg)) and "Confidence:" in texts(plain.svg)
    said = texts(coloured.svg)
    assert not any(t.endswith("?") for t in said)
    assert "Confidence:" not in said and "pLDDT:" in said
    assert [t for t in said if t in ("≥ 90", "70–90", "50–70", "< 50")] == ["≥ 90", "70–90",
                                                                                         "50–70", "< 50"]
    words = " ".join(said)
    assert "Colour follows the model's pLDDT, in AlphaFold's four bands" in words
    assert "Stroke weight and opacity follow the model's pLDDT" not in words

    def paler(svg: str) -> list[str]:
        """Everything that is drawn paler than full strength. Left out: what lies behind the figure and
        is pale by design (the lining of the pocket, the halos of exposed atoms, and the colour the
        atoms stand on), which has as many pieces as it has colours, and the lining's samples in the key."""
        out = []

        def walk(element) -> None:
            if element.get("class") in ("plddt", "backdrop"):
                return
            if element.tag == SVG + "line" and element.get("stroke-width") == f"{LINING_WIDTH:g}":
                return
            out.extend(f"{key}={element.get(key)}" for key in ("opacity", "stroke-opacity", "fill-opacity")
                       if element.get(key))
            for child in element:
                walk(child)

        walk(ET.fromstring(svg))
        return sorted(out)

    # What is paler in the coloured figure is paler for another reason than pLDDT: it is as pale in the
    # figure of the same contacts drawn without any confidence. The plain figure has more of it.
    without = doubtful(plddt=None).render(style=style)
    assert paler(coloured.svg) == paler(without.svg)
    assert len(paler(plain.svg)) > len(paler(coloured.svg))


def test_the_atoms_of_the_ligand_stand_on_the_colours_of_their_bands(steroid):
    for style in ("schematic", "detailed"):
        figure = steroid.render(style=style, plddt_colors=True)
        root = ET.fromstring(figure.svg)
        under = [g for g in root.iter(SVG + "g") if g.get("class") == "plddt"]
        assert under and all(g.get("opacity") == "0.5" for g in under)
        whole = next(g for g in root.iter(SVG + "g") if g.get("class") == "diagram")
        mine = [g for g in whole if g.get("class") == "plddt"]              # the ligand's own: one group
        assert len(mine) == 1
        discs = list(mine[0].iter(SVG + "circle"))
        assert len(discs) >= steroid.ligand.n_atoms
        assert {c.get("fill") for c in discs} <= set(PLDDT_COLORS.values())
        assert len({c.get("fill") for c in discs}) > 1                      # the tail is not the core
        # All of it lies under the ligand and under every line: nothing is drawn over by colour.
        order = [g.get("class") for g in root.iter(SVG + "g") if g.get("class") in ("plddt", "ligand", "interactions")]
        last = max(i for i, c in enumerate(order) if c == "plddt")
        assert last < order.index("ligand") and last < order.index("interactions")
        # What is drawn of a residue stands on colour too, where it is drawn with atoms.
        assert (len(under) > 1) is (style == "detailed")


def test_without_a_plddt_the_option_changes_nothing_and_says_so():
    """A crystal structure declares no pLDDT, and a model that declares one may be read without it; a model with its
    confidence switched off has none drawn."""
    for dia in (diagram_of("kit_imatinib"), predicted_diagram("kit_imatinib", plddt=None)):
        plain, asked = dia.render(style="schematic"), dia.render(style="schematic", plddt_colors=True)
        assert asked.svg == plain.svg
        assert asked.warnings[len(plain.warnings):] == [
            "plddt_colors was asked for, and this figure has no pLDDT to colour by: none is read from this model "
            "(it declares none, or plddt is off)"]
    dia = predicted_diagram("kit_imatinib")
    off = dia.render(plddt_colors=True, confidence=False)
    assert off.svg == dia.render(confidence=False).svg
    assert off.warnings[-1].endswith("confidence is switched off (confidence=False)")


def test_a_ligand_that_is_a_chain_has_its_boxes_filled():
    """The ten residues of hirudin on thrombin as a chain ligand, with a made-up pLDDT."""
    dia = chain_diagram("thrombin_hirudin", structure=invented_plddt(structure_of("thrombin_hirudin")))
    plain, coloured = dia.render(style="schematic"), dia.render(style="schematic", plddt_colors=True)
    assert coloured.status == plain.status

    def units(svg: str):
        return [(g.get("data-confidence"), g.find(SVG + "rect").get("fill"), g.find(SVG + "text").get("fill"))
                for g in ET.fromstring(svg).iter(SVG + "g") if g.get("class") == "unit"]

    assert len(units(coloured.svg)) == len(units(plain.svg)) > 0
    assert all(fill == PLDDT_COLORS[band] and ink == PLDDT_TEXT[band] for band, fill, ink in units(coloured.svg))
    assert all(fill == LIGHT.background for _, fill, _ in units(plain.svg))


def a_pair() -> Comparison:
    """The two ligands of the androgen receptor, each with a made-up pLDDT: some of their contacts are in the low
    band, drawn thinner and paler."""
    return Comparison([predicted_diagram("ar_r1881"), predicted_diagram("ar_bicalutamide")])


def test_a_comparison_takes_the_scheme():
    pair = a_pair()
    plain = pair.render_svg()
    scheme_ = pair.render_svg(colors="contrast")
    assert scheme_.status == plain.status and scheme_.svg != plain.svg
    assert CONTRAST.residue_classes["hydrophobic"] in scheme_.svg


def test_a_comparison_says_that_it_does_not_colour_by_plddt_and_draws_it_by_weight():
    pair = a_pair()
    plain = pair.render_svg()
    assert 'opacity="0.62"' in plain.svg                                       # the low band, drawn paler
    asked = pair.render_svg(plddt_colors=True)
    assert asked.svg == plain.svg
    assert asked.warnings[-1] == ("plddt_colors is not drawn in a comparison: its figures show pLDDT by stroke "
                                  "weight and opacity")


def test_the_command_line_takes_both(tmp_path):
    """On the bicalutamide entry with a local pLDDT declared in it, as a model declares one: its temperature factors
    are then read as a pLDDT (made up, in several bands)."""
    import gzip
    import shutil

    text = gzip.open(model_of("ar_bicalutamide"), "rt", encoding="utf-8").read()
    model = tmp_path / "model.cif"
    model.write_text(text.rstrip("\n") + "\n#\nloop_\n_ma_qa_metric.id\n_ma_qa_metric.mode\n_ma_qa_metric.name\n"
                     "_ma_qa_metric.type\n1 local pLDDT pLDDT\n#\n", encoding="utf-8")
    for name in ("meta.json", "interactions.tsv"):
        shutil.copy(TESTDATA / "ar_bicalutamide" / name, tmp_path / name)
    out = tmp_path / "figure.svg"
    ligand = meta_of("ar_bicalutamide")["ligand"]
    assert main(["render", str(model), "--ligand", ligand["residue_name_in_cif"], "--smiles-from", "meta.json",
                 "-o", str(out), "-q", "--style", "schematic", "--colors", "contrast", "--plddt-colors"]) == 0
    svg = out.read_text()
    assert sum(colour in svg for colour in PLDDT_COLORS.values()) > 1 and f'fill="{CONTRAST.text}"' in svg
    same = Diagram.from_structure(model, ligand["residue_name_in_cif"], smiles=ligand["smiles"],
                                  interactions=load_interactions(tmp_path / "interactions.tsv"))
    assert svg == same.render(style="schematic", colors="contrast", plddt_colors=True).svg


# ---------------------------------------------------------------------------
# pLDDT as the colour of the bonds (``plddt_colors="bonds"``)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("style", ("detailed", "schematic"))
def test_the_bonds_have_the_colours_of_their_atoms(style):
    from bindview.scene import Line, Polygon
    from scenetools import walk
    from scenetools import groups, inside

    dia = predicted_diagram("ar_bicalutamide")                    # a made-up pLDDT, in two bands
    res, behind, plain = (dia.render(style=style, plddt_colors=c) for c in ("bonds", True, False))
    assert res.status == behind.status == "ok"
    assert dia.render(style=style, plddt_colors="behind").svg == behind.svg
    bands = {n: band_of(v) for n, v in dia.confidence.ligand_atoms.items()}
    used = {f"plddt_{b}" for b in bands.values()}
    (bonds,) = groups(res, "bonds")
    strokes = [f.item.paint.stroke or f.item.paint.fill for f in inside(res, bonds) if isinstance(f.item, (Line, Polygon))]
    assert strokes and set(strokes) <= used and "ligand_bond" not in strokes
    # Nothing stands behind the ligand, and nothing is drawn weaker: every bond at its full width.
    assert not groups(res, "plddt") and groups(behind, "plddt")
    widths = {f.item.paint.width for f in inside(res, bonds) if isinstance(f.item, Line) and not f.item.paint.dash}
    (whole,) = groups(dia.render(style=style, confidence=False), "bonds")
    assert max(widths) == max(f.item.paint.width for f in inside(dia.render(style=style, confidence=False), whole)
                              if isinstance(f.item, Line))
    # The letters of the atoms keep the colours of their elements, and the lines of the contacts theirs.
    letters = {f.item.fill for f in walk(res.scene) if type(f.item).__name__ == "Text" and f.item.fill.startswith("element_")}
    assert letters == {f.item.fill for f in walk(behind.scene) if type(f.item).__name__ == "Text"
                       and f.item.fill.startswith("element_")}
    assert "pLDDT:" in " ".join(f.item.content for f in walk(res.scene) if type(f.item).__name__ == "Text")
    if style == "detailed":
        # A residue that is drawn with atoms says it with its bonds and has no plate behind its name.
        drawn = [g for g in groups(res, "residue") if g.data["form"] == "fragment"]
        lashes = [g for g in groups(res, "residue") if g.data["form"] == "eyelash"]
        assert drawn and lashes
        for g in drawn:
            kinds = {type(f.item).__name__ for f in inside(res, g)}
            assert "Rect" not in kinds
            own = {f.item.paint.stroke for f in inside(res, g) if isinstance(f.item, Line)}
            assert own and all(t.startswith("plddt_") for t in own)
        assert all("Rect" in {type(f.item).__name__ for f in inside(res, g)} for g in lashes)
    assert plain.svg != res.svg


def test_a_bond_between_two_bands_is_half_and_half():
    from dataclasses import replace as changed

    from bindview.scene import Line

    dia = diagram_of("kit_imatinib")
    dep = dia.backend.depict(Metrics(), "auto")
    bond = next(b for b in dep.bonds if b.order == 1 and b.stereo not in ("wedge", "hash")
                and dep.atoms[b.a].label is None and dep.atoms[b.b].label is None)
    bands: "list[str | None]" = ["very_high"] * len(dep.atoms)
    one = coloured_bond_items(dep, bond, bands)
    assert [it.paint.stroke for it in one if isinstance(it, Line)] == ["plddt_very_high"]
    bands[bond.b] = "low"
    first, second = [it for it in coloured_bond_items(dep, bond, bands) if isinstance(it, Line)]
    assert (first.paint.stroke, second.paint.stroke) == ("plddt_very_high", "plddt_low")
    assert (first.x2, first.y2) == (second.x1, second.y1)                        # they meet in the middle
    (pa, pb) = dep.pos(bond.a), dep.pos(bond.b)
    assert (first.x2, first.y2) == pytest.approx(((pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2))
    bands[bond.b] = None                                                         # no value: the bond's own colour
    assert coloured_bond_items(dep, bond, bands)[1].paint.stroke == "ligand_bond"
    for stereo in ("wedge", "hash"):
        items = coloured_bond_items(dep, changed(bond, stereo=stereo), bands)
        assert {"plddt_very_high", "ligand_bond"} == {it.paint.stroke for it in items}
    with pytest.raises(ValueError, match="plddt_colors"):
        dia.render(plddt_colors="letters")


def test_a_palette_gives_the_colour_of_an_element_and_of_a_kind_of_contact():
    """What a caller asks a palette who draws something of their own beside a figure: the colour of a letter
    and of a line. They are the colours the figure is drawn in."""
    svg = diagram_of("kit_imatinib").render().svg
    for palette in (LIGHT, get_palette("dark")):
        assert palette.interaction("hbond") == palette.hbond
        assert palette.interaction("pi_stacking") == palette.pi_stacking
        assert palette.element("N") == palette.elements["N"]
        assert palette.element("C") == palette.ligand_carbon and palette.element("C", on_protein=True) == palette.protein_bond
        assert palette.element("Xx") == palette.elements.get("*", palette.text)        # an element it has no colour for
    assert LIGHT.interaction("hbond") in svg and LIGHT.element("N") in svg
    with pytest.raises(ValueError, match="no kind of contact"):
        LIGHT.interaction("background")


def test_a_comparison_and_base_pairs_report_the_theme_whatever_the_scheme():
    """The theme of a result is the theme that was asked for, with a colour scheme too, for a comparison and for a
    figure of base pairs; a palette of the caller's is named."""
    from conftest import nucleic_basepairs

    pair = Comparison([diagram_of("ar_bicalutamide"), diagram_of("ar_r1881")])
    pairs = nucleic_basepairs()
    for figure in (pair, pairs):
        told = [figure.render(theme=theme, colors="contrast").theme for theme in ("light", "dark", "auto")]
        assert told == ["light", "dark", "auto"]
    assert pairs.render(theme=LIGHT).theme == LIGHT.name and pair.render(theme=LIGHT).theme == LIGHT.name

