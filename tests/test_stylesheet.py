"""Several figures in one page: where the style sheet of each is (``render(stylesheet=...)``)."""

import re
import xml.etree.ElementTree as ET

import pytest

import bindview
from conftest import diagram_of, render_of
from bindview import Comparison
from bindview.interactive import ROOT_CLASS

SVG = "{http://www.w3.org/2000/svg}"


def sheets(svg: str) -> "list[str]":
    return re.findall(r"<style>(.*?)</style>", svg, flags=re.S)


def scope_of(svg: str) -> str:
    (found,) = [c for c in ET.fromstring(svg).get("class").split() if c.startswith("lf-c")]
    return found


def unscoped(svg: str, scope: str) -> str:
    return svg.replace(f".{scope}", "").replace(f" {scope}", "")


def test_the_sheet_is_in_the_figure_unless_it_is_asked_elsewhere():
    plain = diagram_of("ar_r1881").render(theme="auto")
    assert diagram_of("ar_r1881").render(theme="auto", stylesheet="inline").svg == plain.svg and plain.css is None
    assert len(sheets(plain.svg)) == 1 and sheets(plain.svg)[0].startswith("svg.bindview{--lf-")
    # A figure that has no sheet has none to put elsewhere.
    for where in ("scoped", "external"):
        res = diagram_of("ar_r1881").render(stylesheet=where)
        assert res.svg == render_of("ar_r1881").svg and res.css is None
    with pytest.raises(ValueError, match="stylesheet is 'inline', 'scoped' or 'external', not 'apart'"):
        diagram_of("ar_r1881").render(stylesheet="apart")


def test_the_colours_of_one_figure_are_not_those_of_the_next():
    def drawn(**options):
        return diagram_of("ar_r1881").render(theme="auto", stylesheet="scoped", **options).svg
    standard, ours = drawn(), drawn(colors="contrast")
    assert scope_of(standard) != scope_of(ours)
    for svg in (standard, ours):
        # Every rule of the sheet is of that class, and the class is named by the rules: no counter, no clock.
        (sheet,) = sheets(svg)
        selectors = re.findall(r"(svg[^{]*)\{", sheet)
        assert len(selectors) == 3 and all(f"svg.bindview.{scope_of(svg)}" in s for s in selectors)
        assert re.fullmatch(r"lf-c[0-9a-f]{8}", scope_of(svg)) and " id=" not in svg
    assert drawn() == standard                                              # the same figure is the same file
    other = diagram_of("ar_bicalutamide").render(theme="auto", stylesheet="scoped").svg
    assert scope_of(other) == scope_of(standard)                            # and two in the same colours share them
    # It is the inline figure but for the class.
    inline = diagram_of("ar_r1881").render(theme="auto", colors="contrast").svg
    assert unscoped(ours, scope_of(ours)) == inline
    # What overrides the palette for one figure still does (data-theme on the root), more specifically than a
    # figure beside it that declares its colours for the whole page.
    assert f"svg.bindview.{scope_of(ours)}[data-theme=dark]" in ours
    assert f"svg.bindview.{scope_of(ours)}:not([data-theme=light])" in ours


def test_a_sheet_for_the_page_is_written_once():
    figures = [diagram_of(case).render(theme="auto", colors="contrast", interactive=True, stylesheet="external", style=style)
               for case in ("ar_r1881", "ar_bicalutamide") for style in ("detailed", "schematic")]
    for res in figures:
        assert not sheets(res.svg) and res.css.endswith("\n")
        assert ROOT_CLASS in ET.fromstring(res.svg).get("class") and scope_of(res.svg) in res.css
        # One rule on a line.
        for line in res.css.splitlines():
            assert line.count("{") == (2 if line.startswith("@media") else 1) and line.endswith("}")
    page = bindview.stylesheet(figures)
    lines = page.splitlines()
    assert len(lines) == len(set(lines)) and set(lines) == {line for r in figures for line in r.css.splitlines()}
    assert bindview.stylesheet([r.css for r in figures]) == page                 # results, or their sheets
    assert len({scope_of(r.svg) for r in figures}) == 1 and page.count("--lf-background:#ffffff") == 1
    # The four share most of it: the two ligands lie in one pocket of the androgen receptor, and its residues have one
    # rule each.
    assert len(page) < 0.75 * sum(len(r.css) for r in figures)
    # No rule of it is of one figure alone: each says of a residue, by its number, what is so wherever it is drawn.
    assert not re.search(r"\[data-active=[^]]*\][^{]*\[data-active=", page)
    # The drawing is that of the inline figure.
    inline = diagram_of("ar_r1881").render(theme="auto", colors="contrast", interactive=True)
    assert re.sub(r"\n <style>.*?</style>", "", inline.svg, flags=re.S) == unscoped(figures[0].svg, scope_of(figures[0].svg))
    assert bindview.stylesheet([]) == "" and bindview.stylesheet([render_of("ar_r1881")]) == ""


def test_a_comparison_has_it_too():
    pair = Comparison([diagram_of("ar_r1881"), diagram_of("ar_bicalutamide")])
    plain, scoped = pair.render("difference", theme="auto"), pair.render("difference", theme="auto", stylesheet="scoped")
    assert unscoped(scoped.svg, scope_of(scoped.svg)) == plain.svg
    apart = pair.render("difference", theme="auto", stylesheet="external")
    assert not sheets(apart.svg) and apart.css.startswith(f"svg.bindview.{scope_of(apart.svg)}{{")
