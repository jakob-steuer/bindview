#!/usr/bin/env python
"""The pictures of the README and the manual: ``docs/images``.

    python scripts/build_docs_images.py                    # write them
    python scripts/build_docs_images.py --check            # are those on disk what the code draws?
    python scripts/build_docs_images.py --only rna water   # these alone

The pictures of the README (``docs/images/readme``) are drawn by the code that stands under them there, which this
script runs; those of the manual by the calls below, on the test systems of ``testdata`` (``tests/conftest.py``), in
both themes and ``bare`` (a page says in its own words what a figure shows). The badges (``docs/images/badges``) are
read off ``pyproject.toml``. The sketches of ``prototype`` have one theme and stand on white. Every input is in the
repository.
"""
from __future__ import annotations

import argparse
import ast
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "images"
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]


def _figures() -> dict:
    """The pictures of the manual, by name: each a call that takes the theme and gives a result of the library.

    The pLDDT of ``plddt_lines`` and ``plddt_bonds`` is MADE UP (:func:`conftest.invented_plddt`: 100 minus the
    temperature factor of each atom): a crystal structure has none, and the repository holds no predicted complex
    with a ligand."""
    from conftest import (bioemu_samples, diagram_of, meta_of, model_of, nucleic_basepairs, nucleic_diagram,
                          predicted_diagram, structure_of, without_waters)

    from bindview import Diagram

    def kit():
        return diagram_of("kit_imatinib")             # c-Kit with imatinib (1T46), its table of contacts by PLIP

    def covalent(**o):                                # the figure of the quick start in the README
        ligand = meta_of("egfr_osimertinib")["ligand"]
        return Diagram.from_structure(model_of("egfr_osimertinib"), ligand["residue_name_in_cif"], ligand["smiles"],
                                      "distances", covalent=True).render(bare=True, **o)

    def arranged(**o):                                # the moves of the manual's section on arranging by hand
        graph = kit().render().graph
        graph.move("Leu595", 40, -30)
        graph.rotate("Thr670", 35)
        graph.flip("Glu640")
        graph.move("legend", 0, 20)
        return kit().render(graph=graph, on_unreadable="render", bare=True, **o)

    def water(**o):                                   # the crystal's waters taken out, as a predicted model has none
        ligand = meta_of("ar_r1881")["ligand"]
        dry = without_waters(structure_of("ar_r1881"))
        return Diagram.from_structure(dry, ligand["residue_name_in_cif"], ligand["smiles"], "distances",
                                      name=ligand["name"]).render(solvent_sites=True, bare=True, **o)

    #: Residues 788 to 829 of c-Kit missing, as the manual's section on absent residues has them.
    construct = dict(absent=range(788, 830), absent_label="a shorter construct", bare=True)
    return {
        "covalent": covalent,
        "ensemble": lambda **o: bioemu_samples().consensus().render(bare=True, bond_length=26.0, **o),
        "arranged": arranged,
        "detailed": lambda **o: kit().render(bare=True, **o),
        "schematic": lambda **o: kit().render(style="schematic", bare=True, **o),
        "mixed": lambda **o: kit().render(style="schematic", residues={"Thr670": "atoms", "Cys673": "atoms",
                                                                         "*": "bubbles"}, bare=True, **o),
        "plddt_lines": lambda **o: predicted_diagram("kit_imatinib").render(bare=True, **o),
        "plddt_bonds": lambda **o: predicted_diagram("kit_imatinib").render(plddt_colors="bonds", bare=True, **o),
        "water": water,
        "absent": lambda **o: kit().render(**construct, **o),
        "absent_faint": lambda **o: kit().render(absent_style="faint", **construct, **o),
        "rna": lambda **o: nucleic_diagram().render(bare=True, **o),
        "basepairs": lambda **o: nucleic_basepairs().render(**o),
    }


#: A block of Python in the README that draws: ``<!-- draws: name name -->`` before it, or ``<!-- runs -->``.
BLOCK = re.compile(r"<!-- (?:draws: ([\w ]+)|runs) -->\n```python\n(.*?)```", re.S)


def readme_blocks() -> "list[tuple[list[str], str]]":
    """The blocks of the README that are run, with the names of what each draws."""
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    return [((names or "").split(), code) for names, code in BLOCK.findall(text)]


def _panels(only: "set[str] | None" = None) -> "dict[str, str]":
    """What the code in the README draws, by file: every statement of a block that is a call and is given to nobody is
    a figure, in the order of the names. One of the library is drawn for both themes, without its words, and once more
    as it is opened by itself: following the reader's theme and answering the pointer. A sketch is one file."""
    from bindview import BasePairs
    from bindview.errors import BindviewError

    space: dict = {"__name__": "readme"}
    out: "dict[str, str]" = {}
    here = os.getcwd()
    os.chdir(ROOT)
    try:
        for names, code in readme_blocks():
            left = list(names)
            for node in ast.parse(code).body:
                call = node.value if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) else None
                drawn = call is not None and not ast.unparse(call.func).startswith("sys.")
                name = left.pop(0) if drawn else ""
                if drawn and only is not None and name not in only:
                    continue
                try:
                    if not drawn:
                        exec(compile(ast.Module([node], []), "<README>", "exec"), space)          # noqa: S102 - our own page
                        continue
                    draw = eval(compile(ast.Expression(call.func), "<README>", "eval"), space)      # noqa: S307
                    args = [eval(compile(ast.Expression(x), "<README>", "eval"), space) for x in call.args]      # noqa: S307
                    given = {k.arg: eval(compile(ast.Expression(k.value), "<README>", "eval"), space) for k in call.keywords}  # noqa: S307
                except (OSError, BindviewError) as why:
                    print(f"{name or 'a block of the README'}: not drawn ({why})")
                    continue
                if getattr(draw, "__name__", "") == "render":
                    bare = {} if isinstance(draw.__self__, BasePairs) else {"bare": True}
                    for theme in ("light", "dark"):
                        out[f"readme/{name}.{theme}"] = draw(*args, **{**given, "theme": theme, **bare}).svg
                    out[f"readme/{name}.live"] = draw(*args, **{**given, "theme": "auto", "interactive": True}).svg
                else:
                    out[f"readme/{name}"] = _on_white(draw(*args, **given).svg)
            if left:
                raise AssertionError(f"the README names {left} and draws nothing for them")
    finally:
        os.chdir(here)
    return out


def _badges() -> "dict[str, str]":
    """The badges at the top of the README, from pyproject.toml: pictures of our own, so that the page asks no other
    site for them."""
    import tomllib

    from bindview.textmetrics import text_width

    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    needs = {re.match(r"[\w-]+", d).group(0): d[re.match(r"[\w-]+", d).end():].replace(">=", "\u2265 ") for d in project["dependencies"]}
    said = {"version": ("bindview", project["version"], "#0f766e"),
            "python": ("Python", project["requires-python"].replace(">=", "\u2265 "), "#2f5fd0"),
            "rdkit": ("RDKit", needs["rdkit"], "#475569"), "numpy": ("NumPy", needs["numpy"], "#475569"),
            "licence": ("licence", project["license"], "#475569")}
    out = {}
    for name, (left, right, colour) in said.items():
        a, b = text_width(left, 11.0) + 14.0, text_width(right, 11.0, bold=True) + 14.0
        out[f"badges/{name}"] = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{a + b:.1f}" height="20" viewBox="0 0 {a + b:.1f} 20" role="img" '
            f'font-family="Helvetica, Arial, \'Liberation Sans\', sans-serif" font-size="11"><title>{left} {right}</title>'
            f'<rect width="{a + b:.1f}" height="20" rx="4" fill="{colour}"/>'
            f'<path d="M4 0H{a:.1f}V20H4a4 4 0 0 1-4-4V4a4 4 0 0 1 4-4Z" fill="#1f2933"/>'
            f'<text x="{a / 2:.1f}" y="13.9" text-anchor="middle" fill="#ffffff">{left}</text>'
            f'<text x="{a + b / 2:.1f}" y="13.9" text-anchor="middle" fill="#ffffff" font-weight="bold">{right}</text></svg>\n')
    return out


def _on_white(svg: str, pad: float = 14.0) -> str:
    """A sketch on a ground of its own: it is drawn for a white page."""
    x, y, w, h = (float(v) for v in re.search(r'viewBox="([-\d.]+) ([-\d.]+) ([\d.]+) ([\d.]+)"', svg).groups())
    box = f'{x - pad:g} {y - pad:g} {w + 2 * pad:g} {h + 2 * pad:g}'
    head = re.match(r"<svg[^>]*>", svg).group(0)
    new = re.sub(r' (?:width|height)="[^"]*"', "", re.sub(r'viewBox="[^"]*"', f'viewBox="{box}"', head))
    ground = f'<rect x="{x - pad:g}" y="{y - pad:g}" width="{w + 2 * pad:g}" height="{h + 2 * pad:g}" rx="8" fill="#ffffff"/>'
    return svg.replace(head, new.replace("<svg ", f'<svg width="{w + 2 * pad:g}" height="{h + 2 * pad:g}" ', 1) + ground, 1)


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="compare with the files on disk instead of writing them")
    ap.add_argument("--only", nargs="+", metavar="NAME", help="these figures only")
    args = ap.parse_args(argv)
    only = set(args.only) if args.only else None
    wrong = 0
    made: "dict[str, str]" = {}
    for name, draw in _figures().items():
        if only is None or name in only:
            for theme in ("light", "dark"):
                made[f"{name}.{theme}"] = draw(theme=theme).svg
    named = {n for names, _ in readme_blocks() for n in names}
    if only is None or only & named:
        made.update(_panels(only))
    made.update({k: v for k, v in _badges().items() if only is None or "badges" in only})
    for name, svg in made.items():
        path = OUT / f"{name}.svg"
        if not args.check:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(svg, encoding="utf-8")
        elif not path.exists() or path.read_text(encoding="utf-8") != svg:
            wrong += 1
            print(f"{path.relative_to(ROOT)} is NOT what the code draws")
    if only is None:
        for path in sorted(OUT.rglob("*.svg")):
            if path.relative_to(OUT).with_suffix("").as_posix() not in made:
                wrong += args.check
                print(f"{path.relative_to(ROOT)} is drawn by nothing")
    print("compared" if args.check else f"wrote {OUT.relative_to(ROOT)}")
    return 1 if wrong else 0


if __name__ == "__main__":
    sys.exit(main())
