"""Residues that another protein does not have: ``render(absent=[...], absent_label=...)``."""

import copy
import re
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from conftest import CASES, TESTDATA, chain_diagram, diagram_of, predicted_diagram, render_of
from bindview import LIGHT, Comparison, Diagram, load_interactions
from bindview import diagram as diagram_module
from bindview.cli import _absent_given, build_parser, main
from bindview.errors import BindviewError
from bindview.styles.absence import FROM, OPACITY, absent_option
from bindview.styles.base import ABSENT_OPACITY, RenderOptions

SVG = "{http://www.w3.org/2000/svg}"
#: Bicalutamide in the androgen receptor.
AR = "ar_bicalutamide"
#: The residues of its table, by number: what touches the ligand there.
CONTACTS = sorted({(r.residue_number, f"{r.chain}:{r.residue_name}{r.residue_number}")
                   for r in load_interactions(TESTDATA / AR / "interactions.tsv")})
#: Four of them: Leu704, Leu707, Gln711 and Met742.
FOUR = [704, 707, 711, 742]
LOST = [ident for number, ident in CONTACTS if number in FOUR]
N = len(CONTACTS)


def groups(svg: str, kind: str) -> "list[ET.Element]":
    return [g for g in ET.fromstring(svg).iter(f"{SVG}g") if g.get("class", "").split(" ")[0] == kind]


def marked(svg: str) -> "set[str]":
    return {g.get("data-residue") for g in groups(svg, "absence")}


def without_mask(svg: str) -> str:
    """The figure with everything taken out that the mask puts into the drawing (its words stay)."""
    svg = re.sub(r'\n *<g class="absence".*?\n *</g>', "", svg, flags=re.S)
    svg = re.sub(rf' data-{FROM}="[^"]*"', "", svg)
    return re.sub(r'\n *<g class="backdrop">\n *</g>', "", svg)


@pytest.mark.parametrize("style", ["detailed", "schematic"])
def test_what_is_lost_is_marked_and_the_drawing_is_what_it_was(style):
    plain = render_of(AR, style)
    res = diagram_of(AR).render(style=style, absent=FOUR, absent_label="AR-V7")
    assert marked(res.svg) == set(LOST) and len(LOST) == 4
    said = {g.get("data-residue"): g.get(f"data-{FROM}") for g in groups(res.svg, "residue")}
    assert {i for i, v in said.items() if v == "AR-V7"} == marked(res.svg) and len(said) == N
    # ... and their contacts say so as well, and no other's does.
    for kind in ("interaction", "contact"):
        for g in groups(res.svg, kind):
            assert (g.get(f"data-{FROM}") == "AR-V7") == (g.get("data-residue") in marked(res.svg))
    # Nothing has moved, nothing is drawn otherwise: the diagram is the one without the mask, on its marks.
    def drawing(svg: str) -> str:
        return svg[svg.index('<g class="diagram"'):svg.index('<g class="legend"')]
    assert drawing(without_mask(res.svg)) == drawing(plain.svg)
    assert [(n.id, n.x, n.y) for n in res.graph.nodes if n.kind != "legend"] == \
        [(n.id, n.x, n.y) for n in plain.graph.nodes if n.kind != "legend"]
    assert diagram_of(AR).render(style=style).svg == plain.svg          # a figure without a mask is what it was


#: The cases whose drawing is wider than its key. The key wraps at the drawing's width or at 300 units, whichever is
#: more: the page of a smaller drawing (carbon monoxide and its one residue) is as wide as its key, and the line that
#: the mask adds to the key can widen it.
WIDE_ENOUGH = [c for c in CASES if c != "hb_carbon_monoxide"]


@pytest.mark.parametrize("case", WIDE_ENOUGH)
@pytest.mark.parametrize("style", ["detailed", "schematic"])
def test_the_figure_lies_where_the_one_without_the_mask_does(case, style):
    """A page that changes from one to the other sees nothing jump, and what is placed by ``atoms`` fits both."""
    plain, res = render_of(case, style), diagram_of(case).render(style=style, absent=range(1, 2000), absent_label="X")
    assert len(marked(res.svg)) == res.quality.n_residues and res.atoms == plain.atoms
    width, height = res.quality.width, res.quality.height
    assert width == plain.quality.width
    # The marks at the edge are in the margin.
    dx, dy = (float(v) for v in re.search(r'<g class="diagram" transform="translate\(([-\d.]+) ([-\d.]+)\)"', res.svg).groups())
    for g in groups(res.svg, "absence"):
        for it in g:
            xs = [float(it.get(k)) for k in ("x1", "x2", "x", "cx") if it.get(k)]
            ys = [float(it.get(k)) for k in ("y1", "y2", "y", "cy") if it.get(k)]
            reach = float(it.get("r") or it.get("stroke-width") or 0.0) / (1.0 if it.get("r") else 2.0)
            xs += [float(it.get("x")) + float(it.get("width"))] if it.tag.endswith("rect") else []
            for p in (it.get("points") or "").split():
                xs.append(float(p.split(",")[0]))
                ys.append(float(p.split(",")[1]))
            assert min(xs) + dx - reach >= 0 and max(xs) + dx + reach <= width and min(ys) + dy - reach >= 0, (case, g.get("data-residue"))


def test_the_figure_says_how_many_they_are():
    def said(**options):
        res = diagram_of(AR).render(absent_label="AR-V7", **options)
        return next((n for n in res.notes if "contact residue" in n), None), res.data.get("absent")
    sentence, data = said(absent=FOUR)
    assert sentence == f"4 of {N} contact residues are absent from AR-V7."
    assert data == {"from": "AR-V7", "contact_residues": N, "absent": LOST,
                    "present": [ident for number, ident in CONTACTS if number not in FOUR]}
    assert said(absent=[704])[0] == f"1 of {N} contact residues is absent from AR-V7."
    assert said(absent=range(628, 921))[0] == f"All {N} contact residues are absent from AR-V7."
    # A comparison in which none is lost is a finding; no comparison is none.
    assert said(absent=[])[0] == said(absent=[1, 2, 3])[0] == f"None of the {N} contact residues is absent from AR-V7."
    assert said() == (None, None)
    res = diagram_of(AR).render(absent=[704])
    assert any(n == f"1 of {N} contact residues is absent from the other protein." for n in res.notes)
    assert groups(res.svg, "absence")[0].get(f"data-{FROM}") == "true"
    # In the figure, under the key, before the other notes; and for a bare figure in the words of its caption.
    svg = diagram_of(AR).render(absent=FOUR, absent_label="AR-V7").svg
    texts = [t.text for t in groups(svg, "footnotes")[0].iter(f"{SVG}text")]
    assert texts[0] == f"4 of {N} contact residues are absent from AR-V7."
    assert any(t.text == "Absent from AR-V7" for t in groups(svg, "legend")[0].iter(f"{SVG}text"))
    bare = diagram_of(AR).render(absent=FOUR, absent_label="AR-V7", bare=True)
    assert "contact residues" not in bare.svg and any(f"4 of {N}" in n for n in bare.notes) and len(marked(bare.svg)) == 4
    assert not any("Absent from" in (t.text or "") for t in ET.fromstring(
        diagram_of(AR).render(absent=[1], absent_label="AR-V7").svg).iter(f"{SVG}text"))      # no key to nothing


def test_the_one_contact_residue_there_is():
    one = [i for i in diagram_of(AR).interactions if i.record.residue_number == 711]
    dia = Diagram.__new__(Diagram)
    dia.__dict__.update(diagram_of(AR).__dict__)
    dia.interactions, dia._composed = one, {}
    assert "The only contact residue is absent from X." in dia.render(absent=[711], absent_label="X").notes
    assert "The only contact residue is not absent from X." in dia.render(absent=[704], absent_label="X").notes


def test_a_residue_is_named_as_it_is_everywhere_else():
    def who(names):
        return marked(diagram_of(AR).render(absent=names).svg)
    assert who([711]) == who(["711"]) == who([711.0]) == who(["Gln711"]) == who(["GLN711"]) == who(["A:711"]) \
        == who(["A:GLN711"]) == who("711") == who(711) == who(iter([711])) == {"A:GLN711"}
    assert who(["B:711"]) == who(["Leu711"]) == set()                     # another chain; a residue that is not there
    assert absent_option([743, "702", 702.0]) == absent_option({702, 743}) == ("702", "743")
    with pytest.raises(ValueError, match="by their number or name, not 702.5"):
        diagram_of(AR).render(absent=[702.5])
    with pytest.raises(ValueError, match="absent_style is one of 'marked', 'faint', not 'gone'"):
        diagram_of(AR).render(absent=[702], absent_style="gone")


def test_faint_sets_back_what_is_lost():
    plain = render_of(AR)
    res = diagram_of(AR).render(absent=FOUR, absent_label="AR-V7", absent_style="faint")
    assert not marked(res.svg) and "Faint: absent from AR-V7" in res.svg
    for g in groups(res.svg, "interaction"):
        lost = g.get(f"data-{FROM}") == "AR-V7"
        strengths = {float(line.get("opacity", "1")) for line in g.iter(f"{SVG}line")}
        assert (max(strengths) <= ABSENT_OPACITY) == lost
    names = {t.text: t for g in groups(res.svg, "residue") for t in g.iter(f"{SVG}text")}
    assert names["Gln711"].get("font-style") == "italic" and names["Gln711"].get("font-weight") is None
    assert names["Phe764"].get("font-style") is None and names["Phe764"].get("font-weight") == "bold"
    assert [(n.id, n.x, n.y) for n in res.graph.nodes if n.kind != "legend"] == \
        [(n.id, n.x, n.y) for n in plain.graph.nodes if n.kind != "legend"]
    bubbles = diagram_of(AR).render(style="schematic", absent=FOUR, absent_style="faint").svg
    lines = [g for g in groups(render_of(AR, "schematic").svg, "interaction") if g.get("data-residue") in LOST]
    assert marked(bubbles) == set() and bubbles.count(f'data-{FROM}="true"') == 4 + len(lines) > 4   # bubbles, lines
    # The pocket is not lined by what is set back.
    lined = {r for g in groups(bubbles, "lining") for r in g.get("data-residues").split()}
    assert lined and not lined & set(LOST)


def test_a_mask_costs_no_layout(monkeypatch):
    laid = []
    real = diagram_module.layout_bodies
    monkeypatch.setattr(diagram_module, "layout_bodies", lambda *a: laid.append(1) or real(*a))
    dia = diagram_of(AR)
    monkeypatch.setattr(dia, "_composed", {})
    dia.render()
    for mask in (FOUR, [704], range(600, 900)):
        dia.render(absent=mask, absent_label="AR-V7")
        dia.render(absent=mask, absent_style="faint", interactive=True, theme="auto", stylesheet="scoped")
    assert len(laid) == 1
    # What is kept is kept in one place: a copy that is given other contacts empties it, and lays out its own.
    other = copy.copy(dia)
    other.interactions, other._composed = dia.interactions[:4], {}
    kept = set(dia._composed)
    other.render()
    assert len(laid) == 2 and set(dia._composed) == kept


def test_where_it_is_drawn_and_where_it_is_said_that_it_is_not():
    # Every theme, a palette of one's own, a figure that answers the pointer, and the confidence in colours.
    own = LIGHT.with_(extra={})
    svg = diagram_of(AR).render(theme=own, absent=FOUR).svg
    assert "#facc15" in svg and f'opacity="{OPACITY:g}"' in svg
    auto = diagram_of(AR).render(theme="auto", absent=FOUR).svg
    assert "--lf-absent:#facc15" in auto and "var(--lf-absent,#facc15)" in auto
    assert "--lf-absent" not in diagram_of(AR).render(theme="auto").svg
    live = diagram_of(AR).render(interactive=True, absent=FOUR).svg
    for g in groups(live, "absence"):                                  # it steps back, and stays, with its residue
        number = re.search(r"\d+$", g.get("data-residue")).group()
        assert {"lf-part", f"lf-t-r{number}", f"lf-o-r{number}"} <= set(g.get("class").split())
        assert g.get("tabindex") is None
    # Marks on a figure in the colours of a pLDDT (made up here) are better faint, and it says so where it has some.
    model = predicted_diagram(AR)
    warned = model.render(absent=FOUR, plddt_colors=True)
    assert any("absent_style='faint'" in w for w in warned.warnings)
    for options in (dict(absent=FOUR), dict(absent=FOUR, plddt_colors=True, absent_style="faint"), dict(absent=[1], plddt_colors=True)):
        assert not any("absent_style" in w for w in model.render(**options).warnings)
    pair = Comparison([diagram_of(AR), diagram_of("ar_r1881")])
    assert any(w.startswith("absent:") for w in pair.render("difference", absent=FOUR).warnings)


def test_a_table_marks_them_too():
    res = diagram_of(AR).render(absent=FOUR, absent_label="AR-V7", figure_width_mm=20.0)
    assert res.figure == "summary" and res.data["absent"]["absent"] == LOST
    assert marked(res.svg) == set(LOST)
    assert any(n == f"Marked: absent from AR-V7. 4 of {N} contact residues are absent from AR-V7." for n in res.notes)
    faint = diagram_of(AR).render(absent=FOUR, absent_style="faint", figure_width_mm=20.0)
    assert not marked(faint.svg) and any(n.startswith("In italics: absent from the other protein.") for n in faint.notes)


def test_beside_a_chain_and_by_the_command_line():
    dia = chain_diagram("thrombin_hirudin")                       # hirudin on thrombin, a chain ligand
    three = sorted({r.record.residue_number for r in dia.interactions})[:3]
    res = dia.render(style="schematic", absent=three, absent_label="a construct")
    assert len(res.data["absent"]["absent"]) == 3 and marked(res.svg) == set(res.data["absent"]["absent"])
    assert _absent_given("702, 705-707,A:744, B:1-2") == ["702", "705", "706", "707", "A:744", "B:1", "B:2"]
    assert _absent_given(None) is None and _absent_given("") == []
    args = build_parser().parse_args(["render", "x.cif", "--ligand", "L", "--absent", "702", "--absent-label", "AR-V7",
                                      "--absent-style", "faint"])
    assert (args.absent, args.absent_label, args.absent_style) == ("702", "AR-V7", "faint")
    assert RenderOptions(absent=[3, 1]) == RenderOptions(absent=(1, 3))          # options are the key of what is kept


@pytest.mark.parametrize("text, word", [("712-705", "705-712"), ("A:9-3", "A:3-9"), (",", "empty"), ("702,,705", "empty"),
                                        ("702,", "empty"), (" , ", "empty")])
def test_a_list_on_the_command_line_that_names_nothing_is_an_error(text, word, tmp_path, capsys):
    """A range written backwards holds no number and a comma alone no name: neither is "none is absent"."""
    with pytest.raises(BindviewError, match=f"--absent {re.escape(text)!s}.*{word}"):
        _absent_given(text)
    # ... and it is said before anything is read or written.
    out = tmp_path / "figure.svg"
    assert main(["render", str(tmp_path / "no_such_model.cif"), "--ligand", "LIG", "--absent", text, "-o", str(out)]) == 2
    assert "--absent" in capsys.readouterr().err and not out.exists()
    assert _absent_given("705-705") == ["705"] and _absent_given("") == []      # one number; none, said on purpose


def test_a_number_is_a_number_whatever_holds_it():
    """One number as well as a list of them, of Python or of NumPy; a gap in a column of numbers is skipped."""
    assert absent_option(702.0) == absent_option(np.int64(702)) == absent_option(np.float64(702.0)) == ("702",)
    gap = float("nan")
    assert absent_option([702.0, gap, 705.0]) == absent_option(np.array([702.0, gap, 705.0])) == ("702", "705")
    assert absent_option(np.array([702, 705], dtype=np.float32)) == absent_option(np.array([705, 702])) == ("702", "705")
    assert RenderOptions(absent=702.0).absent == ("702",)
    # What is neither a number nor a name is refused, and so is a gap that is all there is.
    for wrong in (True, [702, False], np.bool_(True), b"702", [b"702"], gap, float("inf"), [float("-inf")], 702.5):
        with pytest.raises(ValueError, match="by their number or name, not"):
            absent_option(wrong)
