"""A figure that answers the pointer: ``render(interactive=True)``.

The option adds hooks to a figure, and for a ligand that is drawn as atoms what is hidden until it is pointed
at (the atoms of a residue that is drawn without them, a place on every atom of the ligand). What is tested is
that taking all of it out again gives the figure that was not asked for it, byte for byte, and that the hooks
tie the right things together. That a browser then follows the rules was looked at in a browser, not here.
"""

import re
import xml.etree.ElementTree as ET

import pytest

from dataclasses import replace
from itertools import cycle

from conftest import TESTDATA, chain_diagram, diagram_of, fresh_diagram, meta_of, model_of, render_of, structure_of
from bindview import Comparison, Diagram
from bindview.cli import main
from bindview.interactions import load_interactions
from bindview.interactive import ACTIVE, DIMMED, HEAVY, NOTHING_TO_POINT_AT, NUMBER, ROOT_CLASS, hooks
from bindview.scene import Group, Line, Paint, Scene
from bindview.svg import to_svg
from bindview.theme import LIGHT

SVG = "{http://www.w3.org/2000/svg}"
#: c-Kit with imatinib (1T46): thirteen residues in its table, six of them with hydrophobic contacts only.
REFERENCE = "kit_imatinib"
ROWS = load_interactions(TESTDATA / REFERENCE / "interactions.tsv")
#: Each residue of the reference case's table, and the kinds of its contacts.
KINDS: "dict[str, set[str]]" = {}
for _row in ROWS:
    KINDS.setdefault(f"{_row.chain}:{_row.residue_name}{_row.residue_number}", set()).add(_row.type.value)
_kept: dict = {}


def subject(name: str) -> Diagram:
    """A case's diagram; or ``"zinc"``, the zinc of carbonic anhydrase II as a ligand of one atom (no SMILES, the plain
    distance table); or ``"hirudin"``, the ten residues of hirudin on thrombin as a chain ligand. Built once."""
    if name not in ("zinc", "hirudin"):
        return diagram_of(name)
    if name not in _kept:
        _kept[name] = (Diagram.from_structure(model_of("ca2_celecoxib"), ligand="ZN", interactions="distances")
                       if name == "zinc" else chain_diagram("thrombin_hirudin"))
    return _kept[name]


def plain_of(name: str, style: str):
    return render_of(name, style) if name not in ("zinc", "hirudin") else subject(name).render(style=style)


def without_hooks(svg: str) -> str:
    """The figure with everything taken out that ``interactive`` put in."""
    svg = svg.replace(f' class="bindview {ROOT_CLASS}"', ' class="bindview"')
    svg = re.sub(rf"\n <style>svg\.bindview\.{ROOT_CLASS}[^<]*</style>", "", svg)
    svg = re.sub(r'\n <g class="lf-over">.*?\n </g>(?=\n</svg>)', "", svg, flags=re.S)
    svg = re.sub(r' lf-part[^"]*"', '"', svg)
    svg = re.sub(r' data-residue-number="[^"]*"', "", svg)
    svg = svg.replace('<line class="lf-under" ', "<line ")
    return re.sub(r' tabindex="-?\d"', "", svg)


def groups(svg: str, kind: str) -> "list[ET.Element]":
    return [g for g in ET.fromstring(svg).iter(f"{SVG}g") if g.get("class", "").split(" ")[0] == kind]


def ident_of(row) -> str:
    return f"{row.chain}:{row.residue_name}{row.residue_number}"


def tokens(group: ET.Element, prefix: str) -> "list[str]":
    return [c[len(prefix):] for c in group.get("class", "").split() if c.startswith(prefix)]


def rule(who: str) -> str:
    return f".lf-part:not(.lf-o-{who},.lf-p-{who}){{opacity:{DIMMED:g}}}"


@pytest.mark.parametrize("case, style", [(REFERENCE, "schematic"), (REFERENCE, "detailed"), ("zinc", "schematic"),
                                         ("hirudin", "schematic")])
def test_an_interactive_figure_is_the_figure_with_hooks_and_nothing_else(case, style):
    plain = plain_of(case, style)
    asked = subject(case).render(style=style, interactive=True)
    assert subject(case).render(style=style, interactive=False).svg == plain.svg     # off unless asked for
    assert asked.svg != plain.svg and without_hooks(asked.svg) == plain.svg
    assert (asked.status, asked.figure, asked.warnings) == (plain.status, plain.figure, plain.warnings)
    assert ET.fromstring(asked.svg).get("class") == f"bindview {ROOT_CLASS}"
    assert "<script" not in asked.svg


def test_a_residue_and_its_contacts_answer_together():
    svg = diagram_of(REFERENCE).render(style="schematic", interactive=True).svg
    residues = groups(svg, "residue")
    assert len(residues) == len(KINDS)
    who = {}
    for g in residues:
        (mine,) = tokens(g, "lf-t-")
        assert tokens(g, "lf-o-") == [mine] and g.get("tabindex") == "0"
        assert tokens(g, "lf-p-") and all(t[0] == "a" for t in tokens(g, "lf-p-"))      # atoms of the ligand
        who[g.get("data-residue")] = mine
    assert who == {i: "r" + re.search(r"\d+$", i).group() for i in who}     # each is called by its number
    # A line for each row that is not hydrophobic (no two of them join one residue to one atom of the ligand).
    contacts = groups(svg, "interaction")
    assert len(contacts) == sum(r.type.value != "hydrophobic" for r in ROWS)
    for g in contacts:                                                     # pointing at a contact is pointing at its residue
        mine = who[g.get("data-residue")]
        assert tokens(g, "lf-t-") == [mine] and tokens(g, "lf-o-")[0] == mine and g.get("tabindex") == "-1"
        assert all(t[0] == "a" for t in tokens(g, "lf-o-")[1:])
    style = ET.fromstring(svg).findall(f"{SVG}style")[-1].text
    # While anything is pointed at the lines of contacts are heavy: those that stepped back are faint all the same.
    assert (f":has(.lf-part:is(:hover,:focus)) .lf-part:is(.interaction,.contact) :is(line,polyline,path)"
            f":not(.lf-under){{stroke-width:{HEAVY:g}px}}") in style and DIMMED < 0.2
    for mine in who.values():
        assert f":has(.lf-t-{mine}:focus)" in style and f":has(.lf-t-{mine}:hover)" in style and rule(mine) in style
    # What steps back is a residue or a contact (or the place on an atom, which is no ink): the ligand, the key
    # and the notes are no part of it.
    assert not [g for kind in ("atom", "ligand", "legend", "footnotes") for g in groups(svg, kind) if tokens(g, "lf-")]


def closeups(svg: str) -> "dict[str, ET.Element]":
    return {g.get("data-residue"): g for g in groups(svg, "more")}


def test_a_bubble_has_its_atoms_which_take_its_place():
    svg = diagram_of(REFERENCE).render(style="schematic", interactive=True).svg
    style = ET.fromstring(svg).findall(f"{SVG}style")[-1].text
    who = {g.get("data-residue"): tokens(g, "lf-t-")[0] for g in groups(svg, "residue")}
    more = closeups(svg)
    assert set(more) == set(who)
    for ident, g in more.items():
        mine = who[ident]
        assert g.get("display") == "none" and tokens(g, "lf-m-") == [mine]          # hidden, for every reader
        at = f":is(:has(.lf-t-{mine}:focus),:not(:has(.lf-part:focus)):has(.lf-t-{mine}:hover))"
        assert f"{at} .lf-m-{mine}{{display:inline}}" in style
        # ... and its bubble and its lines step back: in a figure that has these atoms, and in no other on a page.
        assert f"svg.bindview.{ROOT_CLASS}:has(.lf-m-{mine}){at} .lf-part.lf-o-{mine}{{opacity:{DIMMED:g}}}" in style
        # Whoever reads the residues and the contacts of a figure by their class finds each of them once.
        kinds = {c.get("class", "").split(" ")[0] for c in g.iter(f"{SVG}g")}
        assert "more-residue" in kinds and not kinds & {"residue", "interaction", "contact"}
    texts = {i: [t.text for t in g.iter(f"{SVG}text")] for i, g in more.items()}
    # The part that makes the contact and the whole side chain, in the lines of the figure, with the distance: the
    # backbone nitrogen of Cys673 and its sulphur; the hydroxyl of Thr670.
    distance = {ident_of(r): f"{r.distance:.2f}" for r in ROWS if r.type.value == "hbond" and r.residue_number in (670, 673)}
    assert {"N", "S", "Cys673", distance["A:CYS673"]} <= set(texts["A:CYS673"])
    assert {"O", distance["A:THR670"]} <= set(texts["A:THR670"])
    assert "Leu595" in texts["A:LEU595"] and "more-contact hydrophobic" in svg
    hbond = next(g for g in more["A:CYS673"].iter(f"{SVG}g") if "hbond" in g.get("class", ""))
    drawn = next(g for g in groups(svg, "interaction") if "hbond" in g.get("class") and g.get("data-residue") == "A:CYS673")
    assert hbond.find(f"{SVG}line").get("stroke") == drawn.find(f"{SVG}line").get("stroke")


def test_what_is_drawn_with_atoms_needs_none_and_an_eyelash_does():
    svg = diagram_of(REFERENCE).render(style="detailed", interactive=True).svg
    assert set(closeups(svg)) == {r for r, kinds in KINDS.items() if kinds == {"hydrophobic"}}
    mixed = diagram_of(REFERENCE).render(style="schematic", interactive=True, residues={"Cys673": "atoms"}).svg
    assert "A:CYS673" not in closeups(mixed) and "A:THR670" in closeups(mixed)
    # Beside a chain a residue has its place and no other: none.
    assert not closeups(subject("hirudin").render(style="schematic", interactive=True).svg)


@pytest.mark.parametrize("case", [REFERENCE, "dhfr_methotrexate"])
def test_the_atoms_stay_within_the_page_and_off_the_ligand(case):
    res = diagram_of(case).render(style="schematic", interactive=True)
    root = ET.fromstring(res.svg)
    width, height = (float(v) for v in root.get("viewBox").split()[2:])
    for ident, g in closeups(res.svg).items():
        dx, dy = (float(v) for v in re.match(r"translate\(([-\d.]+) ([-\d.]+)\)", g.get("transform")).groups())
        for t in g.iter(f"{SVG}text"):
            x, y = float(t.get("x")) + dx, float(t.get("y")) + dy
            assert 0 < x < width and 0 < y < height, (ident, t.text)
        names = {t.text: (float(t.get("x")) + dx, float(t.get("y")) + dy) for t in g.iter(f"{SVG}text")}
        for x, y in names.values():
            assert min((x - a) ** 2 + (y - b) ** 2 for a, b in res.atoms.values()) > 8 ** 2, ident


def test_every_atom_of_the_ligand_can_be_pointed_at():
    dia = diagram_of(REFERENCE)
    res = dia.render(style="schematic", interactive=True)
    hits = {g.get("data-atom"): g for g in groups(res.svg, "hit")}
    assert set(hits) == set(dia.ligand.names)
    reached = {g.get("data-residue"): set(g.get("data-ligand-atoms").split(",")) for g in groups(res.svg, "interaction")}
    style = ET.fromstring(res.svg).findall(f"{SVG}style")[-1].text
    for name, g in hits.items():
        (ring,) = g.findall(f"{SVG}circle")
        assert ring.get("fill") == "none" and ring.get("stroke") is None             # no ink, for every reader
        title = g.find(f"{SVG}title").text
        assert title.startswith(f"{name}: ")
        mine = tokens(g, "lf-t-")
        assert bool(mine) == ("no contact" not in title)
        if mine:
            assert rule(mine[0]) in style
            kept = {p.get("data-residue") for p in groups(res.svg, "residue") if mine[0] in tokens(p, "lf-p-")}
            assert {r for r, atoms in reached.items() if name in atoms} <= kept
            assert kept == set(g.get("data-residues").split(","))             # one without a line too
    assert "hydrogen bond with Cys673" in hits["N3"].find(f"{SVG}title").text


def test_a_residue_of_a_chain_is_tied_to_the_residues_it_touches():
    """Hirudin on thrombin: a box for each residue of its chain in the model."""
    svg = subject("hirudin").render(style="schematic", interactive=True).svg
    residues = {tokens(g, "lf-t-")[0]: g for g in groups(svg, "residue")}
    units = {tokens(g, "lf-t-")[0]: g for g in groups(svg, "unit")}
    from bindview.exposure import WATER_NAMES

    chain = meta_of("thrombin_hirudin")["peptide_chain"]
    n = len({(a.resnum, a.icode) for a in structure_of("thrombin_hirudin").atoms
             if a.chain == chain and a.resname.upper() not in WATER_NAMES})
    assert len(units) == n and sorted(units, key=lambda t: int(t[1:])) == [f"u{k}" for k in range(n)]
    atoms = {u: set(g.get("data-atoms").split(",")) for u, g in units.items()}
    by_ident = {}
    for r, g in residues.items():
        by_ident.setdefault(g.get("data-residue"), r)
    contacts = groups(svg, "interaction")
    assert contacts
    for g in contacts:
        r = by_ident[g.get("data-residue")]
        reached = {u for u, own in atoms.items() if own & set(g.get("data-ligand-atoms").split(","))}
        assert len(reached) == 1                                           # a contact is made by one residue of the chain
        (u,) = reached
        assert tokens(g, "lf-t-") == [r] and set(tokens(g, "lf-o-")) == {r, u}
        # Each of the two is kept when the other is pointed at.
        assert r in tokens(units[u], "lf-p-")
        assert any(u in tokens(part, "lf-p-") for part in groups(svg, "residue") if tokens(part, "lf-t-") == [r])
    # A residue of the chain that touches nothing is tied to nothing, and still answers for itself.
    alone = [u for u, g in units.items() if not tokens(g, "lf-p-")]
    assert alone and all(tokens(units[u], "lf-o-") == [u] for u in alone)


def test_a_figure_with_nothing_to_point_at_is_drawn_as_it_is_and_says_so(tmp_path):
    """Carbon monoxide in haemoglobin with a table that has only its header: nothing was found."""
    empty = tmp_path / "interactions.tsv"
    empty.write_text((TESTDATA / "hb_carbon_monoxide" / "interactions.tsv").read_text().splitlines()[0] + "\n")
    m = meta_of("hb_carbon_monoxide")["ligand"]
    dia = Diagram.from_structure(model_of("hb_carbon_monoxide"), m["residue_name_in_cif"], smiles=m["smiles"],
                                 interactions=load_interactions(empty))
    plain = dia.render(style="schematic")
    asked = dia.render(style="schematic", interactive=True)
    assert plain.status == "empty" and asked.svg == plain.svg
    assert asked.warnings == plain.warnings + [NOTHING_TO_POINT_AT]


def test_a_comparison_answers_too_and_a_residue_is_one_in_both_of_its_panels():
    pair = Comparison([diagram_of("ar_r1881"), diagram_of("ar_bicalutamide")])
    plain, asked = pair.render("difference"), pair.render("difference", interactive=True)
    assert without_hooks(asked.svg) == plain.svg and asked.warnings == plain.warnings
    names = {}
    for g in groups(asked.svg, "residue"):
        names.setdefault(g.get("data-residue"), set()).add(tokens(g, "lf-t-")[0])
    assert names and all(len(n) == 1 for n in names.values())
    assert len(groups(asked.svg, "residue")) > len(names)                  # the same residue, drawn for either ligand


def test_the_command_line_takes_it(tmp_path):
    out = tmp_path / "figure.svg"
    ligand = meta_of(REFERENCE)["ligand"]
    assert main(["render", str(model_of(REFERENCE)), "--ligand", ligand["residue_name_in_cif"], "--smiles-from",
                 "meta.json", "-o", str(out), "-q", "--style", "schematic", "--interactive"]) == 0
    same = Diagram.from_structure(model_of(REFERENCE), ligand["residue_name_in_cif"], smiles=ligand["smiles"],
                                  interactions=load_interactions(TESTDATA / REFERENCE / "interactions.tsv"))
    assert out.read_text() == same.render(style="schematic", interactive=True).svg


# ---------------------------------------------------------------------------
# What the page around a figure can ask for: data-active on its root
# ---------------------------------------------------------------------------

def sheet(svg: str) -> str:
    return ET.fromstring(svg).findall(f"{SVG}style")[-1].text


@pytest.mark.parametrize("style", ["schematic", "detailed"])
def test_a_page_can_ask_for_what_pointing_at_a_residue_is(style):
    svg = diagram_of(REFERENCE).render(style=style, interactive=True).svg
    numbers = {g.get("data-residue"): g.get("data-residue-number") for g in groups(svg, "residue")}
    assert numbers == {i: re.search(r"\d+$", i).group() for i in numbers} and len(numbers) == len(KINDS)
    for kind in ("interaction", "contact"):                                   # ... and a contact says whose it is
        for g in groups(svg, kind):
            assert g.get("data-residue-number") == numbers[g.get("data-residue")]
    rules = sheet(svg)
    for ident, n in numbers.items():
        # The same thing steps back as where it is pointed at, the same atoms are shown, the lines are as heavy.
        asked = f'svg.{ROOT_CLASS}[{ACTIVE}="{n}"]'
        assert f"{asked}:has(.lf-t-r{n}) .lf-part:not(.lf-o-r{n},.lf-p-r{n}){{opacity:var(--lf-faint,1)}}" in rules
        shown = ident in closeups(svg)
        assert (f"{asked} .lf-m-r{n}{{display:var(--lf-shown,none)}}" in rules) == shown
        assert (f"{asked}:has(.lf-m-r{n}) .lf-part.lf-o-r{n}{{opacity:var(--lf-faint,1)}}" in rules) == shown
        assert f'[{ACTIVE}="{n}"]:has(.lf-t-r{n})' in rules
    # Whoever points into the figure, or has clicked in it, is followed first: then nothing is asked to be faint ...
    assert (f"svg.{ROOT_CLASS}:not(:has(.lf-part:is(:hover,:focus))){{--lf-faint:{DIMMED:g};--lf-shown:inline}}") in rules
    assert rules.count("--lf-faint:") == 1 and "opacity:var(--lf-faint,1)" in rules
    # ... and no script is needed, nor an attribute that the figure does not come with.
    assert "<script" not in svg and f" {ACTIVE}=" not in svg.split("<style>")[0] and " id=" not in svg


def test_only_a_figure_that_answers_says_the_numbers():
    for style in ("schematic", "detailed"):
        assert "data-residue-number" not in render_of(REFERENCE, style).svg
    pair = Comparison([diagram_of("ar_r1881"), diagram_of("ar_bicalutamide")])
    live = pair.render("difference", interactive=True).svg
    both = [{str(r.residue_number) for r in load_interactions(TESTDATA / case / "interactions.tsv")}
            for case in ("ar_r1881", "ar_bicalutamide")]
    assert {g.get("data-residue-number") for g in groups(live, "residue")} == both[0] | both[1]
    assert "711" in both[0] & both[1] and f'[{ACTIVE}="711"]:has(.lf-t-r711)' in sheet(live)


def residue(ident: str, number: str) -> Group:
    return Group("residue", [Line(0, 0, 1, 1, Paint(stroke="text"))], data={"residue": ident, NUMBER: number})


def made(*pairs) -> "tuple[str, dict[str, str]]":
    """The rules of a figure of these residues, and what each is called."""
    scene = Scene(10, 10)
    scene.items = [residue(i, n) for i, n in pairs]
    found = hooks(scene)
    return found.css, {g.data["residue"]: re.search(r"lf-t-(\S+)", found.classes[id(g)]).group(1) for g in scene.items}


def test_a_number_that_two_chains_have_lights_both_and_one_of_them_by_its_chain():
    css, who = made(("A:ASP25", "25"), ("B:ASP25", "25"), ("A:GLY27", "27"))
    # Each of the two is called with its chain, and by the number as well: that is what the number alone keeps.
    assert who == {"A:ASP25": "r25_A", "B:ASP25": "r25_B", "A:GLY27": "r27"}
    _, classes = figure(("A:ASP25", "25"), ("B:ASP25", "25"), ("A:GLY27", "27"))
    assert classes["B:ASP25"] == {"lf-part", "lf-t-r25_B", "lf-t-r25", "lf-o-r25_B", "lf-o-r25"}
    assert classes["A:GLY27"] == {"lf-part", "lf-t-r27", "lf-o-r27"}
    root = f"svg.{ROOT_CLASS}"
    assert f'{root}[{ACTIVE}="25"]:has(.lf-t-r25) .lf-part:not(.lf-o-r25,.lf-p-r25)' in css
    assert f'{root}[{ACTIVE}="B:25"]:has(.lf-t-r25_B) .lf-part:not(.lf-o-r25_B,.lf-p-r25_B)' in css
    assert f'[{ACTIVE}="27"]:has(.lf-t-r27)' in css and '"A:27"' not in css     # (one is meant by its number alone)


def test_a_number_is_said_with_its_insertion_code_and_is_never_read_off_a_name():
    # (A residue's own name may end in a digit: PO4 301 is written A:PO4301.)
    css, who = made(("A:TYR60A", "60A"), ("A:PRO60B", "60B"), ("A:PO4301", "301"), ("A:SER-3", "-3"))
    assert who == {"A:TYR60A": "r60A", "A:PRO60B": "r60B", "A:PO4301": "r301", "A:SER-3": "r-3"}
    for number in ("60A", "60B", "301", "-3"):
        assert f'[{ACTIVE}="{number}"]:has(.lf-t-r{number})' in css


def test_the_rules_of_one_figure_do_not_reach_into_another():
    """A style sheet in a figure is read by the page: each rule is of a figure that has the residue."""
    css, _ = made(("A:GLN712", "712"), ("A:MET743", "743"))
    for selector in re.findall(r"([^{}]+)\{", css):
        if ACTIVE in selector and ":is(" not in selector:
            number = re.search(rf'{ACTIVE}="([^"]+)"', selector).group(1)
            assert f":has(.lf-t-r{number})" in selector or f".lf-m-r{number}" in selector
    # What a residue is called follows from its number: the rule for 712 is the same one in every figure.
    other, _ = made(("A:LEU701", "701"), ("A:GLN712", "712"))
    mine = {r for r in re.findall(r"[^{}]+\{[^{}]*\}", css) if "r712" in r and ":is([" not in r}
    assert mine and all(r in other for r in mine)


def figure(*pairs) -> "tuple[str, dict[str, set[str]]]":
    """The rules of a figure of these residues, and the classes of each of them."""
    scene = Scene(10, 10)
    scene.items = [residue(i, n) for i, n in pairs]
    found = hooks(scene)
    return found.css, {g.data["residue"]: set(found.classes[id(g)].split()) for g in scene.items}


def asked_for(rules: str, said: str, classes: "dict[str, set[str]]") -> "set[str] | None":
    """What a page with these rules keeps of a figure that has ``data-active=said`` on its root: the residues that
    do not step back, or None where no rule speaks of the figure, which is then left as it is."""
    there = set().union(*classes.values())
    kept = None
    for has, keep in re.findall(rf'\[{ACTIVE}="{re.escape(said)}"\]((?::has\(\.[^)]+\))+) \.lf-part:not\(([^)]*)\)\{{opacity',
                                rules):
        if all(c in there for c in re.findall(r":has\(\.([^)]+)\)", has)):
            names = {c.lstrip(".") for c in keep.split(",")}
            now = {ident for ident, own in classes.items() if own & names}
            kept = now if kept is None else kept & now
    return kept


def test_a_rule_that_names_a_chain_says_the_same_in_every_figure_of_the_page():
    """Residue 25 in chains A and B of one figure, in C and D of the next, in A, B and C of a third, and in chain A
    alone of a fourth: the page reads the rules of all four for each."""
    (x, in_x), (y, in_y) = figure(("A:ASP25", "25"), ("B:ASP25", "25")), figure(("C:ASP25", "25"), ("D:ASP25", "25"))
    z, in_z = figure(("A:ASP25", "25"), ("B:ASP25", "25"), ("C:ASP25", "25"))
    one, in_one = figure(("A:ASP25", "25"), ("A:GLY27", "27"))
    page = x + y + z + one
    assert asked_for(page, "A:25", in_x) == {"A:ASP25"} and asked_for(page, "B:25", in_x) == {"B:ASP25"}
    assert asked_for(page, "C:25", in_y) == {"C:ASP25"} and asked_for(page, "C:25", in_z) == {"C:ASP25"}
    # A figure that has no A:25 is left as it is, whatever its residues are called in its classes.
    assert asked_for(page, "A:25", in_y) is None and asked_for(page, "B:25", in_y) is None
    assert asked_for(page, "D:25", in_x) is None and asked_for(page, "D:25", in_z) is None
    # The number alone is every residue of that number, however many chains the figure beside it has.
    for classes in (in_x, in_y, in_z):
        assert asked_for(page, "25", classes) == set(classes)
    assert asked_for(page, "25", in_one) == {"A:ASP25"} and asked_for(page, "27", in_one) == {"A:GLY27"}
    assert asked_for(page, "27", in_x) is None


# ---------------------------------------------------------------------------
# What lies under the line of a contact: the glow of its PAE, the highlight of a covalent bond
# ---------------------------------------------------------------------------

DRAWN = tuple(f"{SVG}{tag}" for tag in ("line", "polyline", "path"))


def drawn_heavier(svg: str) -> "list[ET.Element]":
    """What the sheet draws at ``HEAVY`` while a residue is pointed at, as its rule names it."""
    (rule,) = re.findall(r"\.lf-part:is\(\.interaction,\.contact\) (:is\(line,polyline,path\)[^{]*)"
                         rf"\{{stroke-width:{HEAVY:g}px\}}", sheet(svg).split(f"[{ACTIVE}=")[0])
    left_out = set(re.findall(r":not\(\.([\w-]+)\)", rule))
    return [el for kind in ("interaction", "contact") for g in groups(svg, kind) if "lf-part" in g.get("class").split()
            for el in g.iter() if el.tag in DRAWN and not left_out & set(el.get("class", "").split())]


def wide_in(group: ET.Element) -> "list[float]":
    return [float(el.get("stroke-width")) for el in group.iter()
            if el.tag in DRAWN and float(el.get("stroke-width", 0)) > HEAVY]


def wide_lines(svg: str) -> "list[float]":
    return [width for kind in ("interaction", "contact") for g in groups(svg, kind) for width in wide_in(g)]


def with_a_glow(case: str) -> Diagram:
    """A complex from the PDB, each contact of its table given a predicted aligned error."""
    m, folder = meta_of(case), TESTDATA / case
    records = [replace(r, pae=pae) for r, pae in zip(load_interactions(folder / "interactions.tsv"),
                                                     cycle((1.5, 7.0, 12.0, 22.0)))]
    return Diagram.from_structure(model_of(case), ligand=m["ligand"]["residue_name_in_cif"],
                                  smiles=m["ligand"]["smiles"], interactions=records)


@pytest.mark.parametrize("what, style", [("glow", "detailed"), ("glow", "schematic"), ("covalent", "detailed"),
                                         ("covalent", "schematic"), ("a glow in pieces", "schematic")])
def test_what_lies_under_a_contact_is_not_drawn_thinner_while_a_residue_is_pointed_at(what, style):
    """The rule that draws a contact heavier names every line of its group. The glow of a PAE and the highlight
    under a covalent bond are lines of that group and wider than the rule makes a line: they keep their width,
    and so do the pieces of a glow that is broken around a word."""
    dia = (fresh_diagram("egfr_osimertinib", covalent=True) if what == "covalent" else
           with_a_glow("kit_imatinib" if what == "glow" else "ar_bicalutamide"))
    options = {} if what == "covalent" else {"pae_colors": True}
    plain, live = dia.render(style=style, **options), dia.render(style=style, interactive=True, **options)
    under = wide_lines(plain.svg)
    assert under and (what != "covalent" or len(under) == 1)         # the figure has what the rule must leave
    if what == "a glow in pieces":                                   # ... and here one of them is in two
        assert any(len(wide_in(g)) > 1 for g in groups(plain.svg, "interaction"))
    heavier = drawn_heavier(live.svg)
    assert heavier and all(float(el.get("stroke-width")) <= HEAVY for el in heavier)
    # ... and nothing else is left as it is: every other line of a contact is drawn heavier, as it was.
    assert len(heavier) + len(under) == sum(1 for kind in ("interaction", "contact") for g in groups(live.svg, kind)
                                            for el in g.iter() if el.tag in DRAWN)
    # Nothing but the hooks tells the two files apart, and the figure that was not asked for it has none of them.
    assert without_hooks(live.svg) == plain.svg and "lf-under" not in plain.svg


# ---------------------------------------------------------------------------
# Insertion codes and chain names that are no letters
# ---------------------------------------------------------------------------

def said_in(rules: str) -> "set[str]":
    """Every value of ``data-active`` that the rules speak of, as a page would have to write it."""
    strings = re.findall(rf'\[{ACTIVE}="((?:[^"\\]|\\.)*)"\]', rules)
    return {re.sub(r"\\([0-9a-f]{1,6}) ?", lambda m: chr(int(m.group(1), 16)), text) for text in strings}


@pytest.mark.parametrize("odd", ["'", "*", "<", '"', "&", "\\", "]", " ", "_"])
def test_a_code_or_a_chain_that_is_no_letter_leaves_the_file_and_its_rules_whole(odd):
    """An insertion code, and the name of a chain where two residues share a number, go into class names and
    into the rules: whatever they are, the file is XML, a class is one word, and a rule means the residue."""
    scene = Scene(10, 10)
    scene.items = [residue(f"A:TYR60{odd}", f"60{odd}"), residue("A:TYR60", "60"),
                   residue(f"A{odd}:ASP25", "25"), residue("B:ASP25", "25")]
    svg = to_svg(scene, LIGHT, interactive=True)
    root = ET.fromstring(svg)                                              # (not well-formed: an error here)
    drawn = [g for g in root.iter(f"{SVG}g")]
    assert [g.get("data-residue") for g in drawn] == [g.data["residue"] for g in scene.items]
    names = [set(g.get("class").split()) for g in drawn]
    assert all(re.fullmatch(r"[A-Za-z][-_A-Za-z0-9]*", c) for own in names for c in own)
    mine = [{c for c in own if c.startswith("lf-t-")} for own in names]
    for a, b in ((0, 1), (2, 3)):                                          # told apart, with the code and by the chain
        assert mine[a] - mine[b] and mine[b] - mine[a]
    rules = root.find(f"{SVG}style").text
    assert said_in(rules) == {f"60{odd}", "60", "25", f"A{odd}:25", "B:25"}
    # Each rule asks for a class that the residue it speaks of has, and the others have not.
    for said, who in ((f"60{odd}", 0), ("60", 1), (f"A{odd}:25", 2), ("B:25", 3)):
        (asked,) = {c for sel in re.findall(r"[^{}]+(?=\{opacity)", rules) if said_in(sel) == {said}
                    for c in re.findall(r":has\(\.(lf-t-[^)]+)\)", sel)}
        assert [asked in own for own in names] == [k == who for k in range(4)]
