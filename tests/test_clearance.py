"""A contact's line, kept off words that are not its own: ``break_lines``, on unless ``False`` is passed.

A line is broken around a word it runs over and nothing else is drawn: what is tested is what counts as
running over a word, that a figure without such a line is the same file either way, that a figure with
one loses exactly that, and that ``break_lines=False`` draws the figure as it is before any line is broken.
Whether a broken line reads better was looked at in the figures, not here.
"""

import copy

import pytest

from conftest import TESTDATA, diagram_of, meta_of, model_of, overlay_diagram, overlay_members, render_of, trimer
from bindview import Comparison, Diagram
from bindview.cli import main
from bindview.clearance import INSIDE, ROOM, keep_lines_off_words
from bindview.interactions import load_interactions
from bindview.scene import Group, Line, Paint, Text, text_box

INK = Paint(stroke="hbond")


def contact(*items) -> Group:
    return Group("interaction hbond", list(items))


def pieces(group: Group) -> "list[tuple[float, float]]":
    return [(round(it.x1, 2), round(it.x2, 2)) for it in group.items if isinstance(it, Line)]


def word(text: str, x: float, y: float, size: float = 12.0) -> Text:
    return Text.plain(x, y, text, size, "text")


def lines_over_words(result) -> "list[str]":
    """The words of a figure that a contact's line runs over: counted here in the test's own way, on the
    scene, so that the library's function is not checked against itself."""
    words, lines = [], []

    def walk(items, dx, dy, mine):
        for it in items:
            if isinstance(it, Group):
                tx, ty = it.translate or (0.0, 0.0)
                walk(it.items, dx + tx, dy + ty, id(it) if it.cls.split(" ")[0] == "interaction" else mine)
            elif isinstance(it, Text) and it.content.strip():
                b = text_box(it)
                words.append((it.content, (b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy), mine))
            elif isinstance(it, Line) and mine and it.paint.stroke is not None:
                lines.append(((it.x1 + dx, it.y1 + dy), (it.x2 + dx, it.y2 + dy), mine))

    walk(result.scene.items, 0.0, 0.0, 0)
    out = []
    for name, box, whose in words:
        x0, y0, x1, y1 = box[0] + INSIDE, box[1] + INSIDE, box[2] - INSIDE, box[3] - INSIDE
        for a, b, mine in lines:
            if mine == whose:
                continue
            # the line is inside the box somewhere strictly between its two ends
            steps = [k / 400.0 for k in range(1, 400)]
            inside = [t for t in steps if x0 < a[0] + (b[0] - a[0]) * t < x1 and y0 < a[1] + (b[1] - a[1]) * t < y1]
            if inside and inside[0] > 0.02 and inside[-1] < 0.98:
                out.append(name)
    return out


# -- what counts ---------------------------------------------------------------

def test_a_line_is_broken_around_a_word_it_runs_over():
    n = word("N", 50.0, 4.0)
    box = text_box(n)
    g = contact(Line(0.0, 0.0, 100.0, 0.0, INK))
    content = Group("diagram", [Group("ligand", [n]), Group("interactions", [g])])
    assert keep_lines_off_words(content) == 1
    assert pieces(g) == [(0.0, round(box[0] - ROOM[0], 2)), (round(box[2] + ROOM[0], 2), 100.0)]
    assert all(it.paint is INK for it in g.items)                 # the pieces are the line: dashes, weight, colour


def test_the_word_may_sit_in_a_group_that_was_moved():
    n = word("N", 0.0, 4.0)
    g = contact(Line(0.0, 0.0, 100.0, 0.0, INK))
    content = Group("diagram", [Group("ligand", [Group("atom", [n], translate=(50.0, 0.0))]),
                                Group("interactions", [g])])
    assert keep_lines_off_words(content) == 1
    assert len(pieces(g)) == 2


def test_a_line_that_ends_at_a_word_is_pointing_at_it_and_stays_whole():
    for x in (0.0, 100.0, 97.0):                                  # at either end, and just short of one
        g = contact(Line(0.0, 0.0, 100.0, 0.0, INK))
        content = Group("diagram", [Group("ligand", [word("OH", x, 4.0)]), Group("interactions", [g])])
        assert keep_lines_off_words(content) == 0
        assert pieces(g) == [(0.0, 100.0)]


def test_a_line_is_not_broken_around_its_own_distance_a_second_time():
    g = contact(Line(0.0, 0.0, 40.0, 0.0, INK), word("2.98", 50.0, 4.0, 9.0), Line(60.0, 0.0, 100.0, 0.0, INK))
    content = Group("diagram", [Group("interactions", [g])])
    assert keep_lines_off_words(content) == 0


def test_the_distance_on_another_line_is_a_word_like_any_other():
    crossed = contact(Line(50.0, -40.0, 50.0, 40.0, INK))
    other = contact(Line(0.0, 0.0, 38.0, 0.0, INK), word("2.98", 50.0, 3.0, 9.0), Line(62.0, 0.0, 100.0, 0.0, INK))
    content = Group("diagram", [Group("interactions", [other, crossed])])
    assert keep_lines_off_words(content) == 1
    assert len([it for it in crossed.items if isinstance(it, Line)]) == 2
    assert pieces(other) == [(0.0, 38.0), (62.0, 100.0)]


def test_a_line_that_touches_the_edge_of_a_word_is_not_running_over_it():
    n = word("N", 50.0, 4.0)
    box = text_box(n)
    g = contact(Line(0.0, box[1] + 0.2, 100.0, box[1] + 0.2, INK))          # along the top edge, inside by 0.2
    content = Group("diagram", [Group("ligand", [n]), Group("interactions", [g])])
    assert keep_lines_off_words(content) == 0


def test_a_short_line_that_would_all_but_vanish_stays_whole():
    wide = word("Glu565", 10.0, 4.0)
    g = contact(Line(-15.0, 0.0, 35.0, 0.0, INK))                 # 50 long, and the word with its room takes more than half
    content = Group("diagram", [Group("residues", [wide]), Group("interactions", [g])])
    assert keep_lines_off_words(content) == 0
    assert pieces(g) == [(-15.0, 35.0)]


def test_only_the_lines_of_contacts_are_broken():
    n = word("N", 50.0, 4.0)
    bond = Group("bonds", [Line(0.0, 0.0, 100.0, 0.0, Paint(stroke="bond"))])
    content = Group("diagram", [Group("ligand", [bond, n])])
    assert keep_lines_off_words(content) == 0
    assert pieces(bond) == [(0.0, 100.0)]


# -- on figures ----------------------------------------------------------------

def test_a_line_over_a_word_is_broken_unless_the_figure_is_asked_whole():
    """R1881's schematic figure: the line of one contact runs over the water that another one goes through."""
    dia = diagram_of("ar_r1881")
    drawn = render_of("ar_r1881", "schematic")
    whole = dia.render(style="schematic", break_lines=False)
    assert lines_over_words(whole) == ["H\u2082O"]
    assert lines_over_words(drawn) == []
    assert dia.render(style="schematic", break_lines=True).svg == drawn.svg      # the default, said aloud
    # Nothing but the one line differs: every other line of the two files is the same line.
    a, b = whole.svg.splitlines(), drawn.svg.splitlines()
    assert len(b) == len(a) + 1
    assert len(set(a) - set(b)) == 1 and len(set(b) - set(a)) == 2


@pytest.mark.parametrize("case,style", [("kit_imatinib", "detailed"), ("kit_imatinib", "schematic"),
                                        ("ca2_celecoxib", "detailed"), ("hb_carbon_monoxide", "schematic")])
def test_a_figure_in_which_no_line_runs_over_a_word_is_the_same_file(case, style):
    whole = diagram_of(case).render(style=style, break_lines=False)
    assert lines_over_words(whole) == []
    assert render_of(case, style).svg == whole.svg


def test_the_layout_and_what_is_measured_do_not_change():
    """Bicalutamide's schematic figure, in which a line is broken."""
    dia = diagram_of("ar_bicalutamide")
    whole, drawn = dia.render(style="schematic", break_lines=False), dia.render(style="schematic")
    assert drawn.svg != whole.svg
    assert drawn.quality == whole.quality and drawn.atoms == whole.atoms
    assert drawn.graph.to_json() == whole.graph.to_json()


#: Figures in which a line runs over a word: two ligands of the androgen receptor, and the interface of chains A and B
#: of the TNF trimer (over a distance in the one style, over the mark of a low confidence in the other).
OVER_A_WORD = {"ar_r1881": lambda: diagram_of("ar_r1881"), "ar_bicalutamide": lambda: diagram_of("ar_bicalutamide"),
               "tnfa_ab": lambda: trimer("A", "B")}


@pytest.mark.parametrize("name, style", [("ar_r1881", "schematic"), ("ar_bicalutamide", "schematic"),
                                         ("tnfa_ab", "detailed"), ("tnfa_ab", "schematic")])
def test_asked_whole_a_figure_is_the_figure_before_its_lines_are_broken(name, style):
    """``break_lines=False`` draws every line whole: the lines of its contacts broken around the words they run
    over, it is the figure that is drawn by default, and nothing else of it differs."""
    dia = OVER_A_WORD[name]()
    whole, drawn = dia.render(style=style, break_lines=False), dia.render(style=style)
    over = lines_over_words(whole)
    assert over and not lines_over_words(drawn)
    scene = copy.deepcopy(whole.scene)
    (content,) = [g for g in scene.items if isinstance(g, Group) and g.cls == "diagram"]
    assert keep_lines_off_words(content) == len(over)
    assert scene == drawn.scene
    assert (whole.status, whole.quality, whole.warnings) == (drawn.status, drawn.quality, drawn.warnings)


def test_a_comparison_draws_its_lines_as_laid_out_and_says_so_where_it_was_asked():
    members = overlay_members("ar_ligands")
    pair = Comparison([overlay_diagram("ar_ligands", slug) for slug, _ in members], names=[slug for slug, _ in members])
    plain = pair.render("difference")
    asked, whole = pair.render("difference", break_lines=True), pair.render("difference", break_lines=False)
    assert asked.svg == plain.svg == whole.svg
    assert any("break_lines is not done in a comparison" in w for w in asked.warnings)
    assert not any("break_lines" in w for w in [*plain.warnings, *whole.warnings])


def test_the_command_line_draws_the_lines_whole_on_request(tmp_path):
    """R1881's schematic figure, in which a line is broken."""
    drawn, whole = tmp_path / "figure.svg", tmp_path / "whole.svg"
    ligand = meta_of("ar_r1881")["ligand"]
    asked = ["render", str(model_of("ar_r1881")), "--ligand", ligand["residue_name_in_cif"], "--smiles-from",
             "meta.json", "-q", "--style", "schematic"]
    assert main(asked + ["-o", str(drawn)]) == 0
    assert main(asked + ["-o", str(whole), "--no-break-lines"]) == 0
    same = Diagram.from_structure(model_of("ar_r1881"), ligand["residue_name_in_cif"], smiles=ligand["smiles"],
                                  interactions=load_interactions(TESTDATA / "ar_r1881" / "interactions.tsv"))
    assert drawn.read_text() == same.render(style="schematic").svg
    assert whole.read_text() == same.render(style="schematic", break_lines=False).svg
    assert drawn.read_text() != whole.read_text()
