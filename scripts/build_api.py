#!/usr/bin/env python
"""The API page (``docs/api.md``), read off the library: its names, their signatures and the first words of their help.

    python scripts/build_api.py            # write it
    python scripts/build_api.py --check    # is the page on disk what the library says?
"""
from __future__ import annotations

import argparse
import dataclasses
import inspect
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "api.md"
sys.path.insert(0, str(ROOT / "src"))

import bindview  # noqa: E402
from bindview import arrange, distance_table, report  # noqa: E402
from bindview.options import GROUPS  # noqa: E402
from bindview.styles.base import RenderOptions  # noqa: E402

#: The page, in order: a heading, and the names under it.
PARTS = (
    ("A ligand in its pocket", ("Diagram", "RenderResult", "Quality", "stylesheet")),
    ("Contacts", ("Interaction", "InteractionType", "load_interactions", "parse_interactions", "detect_interactions")),
    ("Structures", ("Structure", "read_structure", "read_models")),
    ("The layout", ("Graph",)),
    ("Several models, several ligands", ("Ensemble", "EnsembleContact", "ResidueShare", "Comparison", "Difference", "Fingerprint",
                                         "FingerprintTable")),
    ("Chains", ("Interface", "Participant", "InterfaceContact", "load_interface_contacts", "BasePairs", "BasePair", "BaseStack",
                "read_mmcif_pairs", "read_fr3d_pairs", "read_fr3d_stacks", "read_rnapolis_pairs", "read_rnapolis_stacks",
                "pairs_from_dot_bracket", "pairs_from_contacts", "pairs_within_from_table", "Peptide", "perceive_peptide")),
    ("Confidence and colours", ("Confidence", "ModelConfidence", "Palette", "Metrics")),
    ("Errors", ("BindviewError", "StructureError", "LigandNotFoundError", "LigandBuildError", "InteractionError",
                "DetectionUnavailableError", "RenderError", "GraphError", "EnsembleError")),
)
#: Public names that have no entry of their own, and why.
LEFT_OUT = {"STYLES": "the names of the styles", "VIEWS": "the names of the views of a comparison", "LIGHT": "a Palette",
            "DARK": "a Palette", "LigandBackend": "how a ligand is drawn: of use inside", "SmallMolecule": "likewise",
            "Polymer": "likewise", "options": "shown at the top", "__version__": "a number"}
#: Modules whose functions are used by name.
MODULES = (("bindview.distance_table", distance_table, ("ligand_contacts", "contact_table", "interface_contacts")),
           ("bindview.arrange", arrange, ("page", "serve")), ("bindview.report", report, ("survey",)))


def plain(text: str) -> str:
    """Help as Markdown: a name in double quotes of the help in single ones, and no roles."""
    text = re.sub(r":(?:class|func|meth|mod|data|attr|exc):`~?([^`<]+?)(?:\s*<[^>]*>)?`", lambda m: f"`{m.group(1)}`", text)
    return text.replace("``", "`")


def first(obj) -> str:
    """The first paragraph of an object's help, on one line."""
    doc = inspect.getdoc(obj) or ""
    return plain(" ".join(doc.split("\n\n")[0].split()))


def call(name: str, function) -> str:
    """A call with its keywords and their defaults, without the types."""
    try:
        params = list(inspect.signature(function).parameters.values())
    except (TypeError, ValueError):
        return name
    out, star = [], False
    for k, p in enumerate(params):
        if k == 0 and p.name in ("self", "cls"):
            continue
        if p.kind is p.VAR_POSITIONAL:
            out.append("*" + p.name)
            star = True
        elif p.kind is p.VAR_KEYWORD:
            out.append("**" + p.name)
        else:
            if p.kind is p.KEYWORD_ONLY and not star:
                out.append("*")
                star = True
            out.append(p.name if p.default is p.empty else f"{p.name}={p.default!r}")
    return f"{name}({', '.join(out)})"


def of_class(name: str, cls) -> list[str]:
    out = [f"### `{name}`", "", first(cls), ""]
    if dataclasses.is_dataclass(cls):
        fields = [f for f in dataclasses.fields(cls) if not f.name.startswith("_")]
        out += ["Fields: " + ", ".join(f"`{f.name}`" for f in fields) + ".", ""]
    rows = []
    for key, member in vars(cls).items():
        if key.startswith("_"):
            continue
        if isinstance(member, property):
            rows.append((f"`.{key}`", first(member)))
        elif isinstance(member, (classmethod, staticmethod)):
            rows.append((f"`{call(f'{name}.{key}', member.__func__)}`", first(member.__func__)))
        elif inspect.isfunction(member):
            rows.append((f"`{call(f'.{key}', member)}`", first(member)))
    if rows:
        out += ["| | |", "|---|---|"] + [f"| {a.replace('|', chr(92) + '|')} | {b.replace('|', chr(92) + '|')} |" for a, b in rows] + [""]
    return out


def text() -> str:
    defaults = {f.name: f.default for f in dataclasses.fields(RenderOptions)}
    out = ["# API", "",
           "Written by `python scripts/build_api.py` from the library itself. In Python: `bindview.options()`, "
           "`help(bindview.Diagram)`. How things are done: the [manual](manual.md).", "",
           " · ".join(f"[{t}](#{re.sub(r'[^a-z0-9 -]', '', t.lower()).replace(' ', '-')})"
                      for t in ("What goes in", "Options of render", *(t for t, _ in PARTS), "Functions of modules")), "",
           "## What goes in", "", f"`{call('Diagram.from_structure', bindview.Diagram.from_structure.__func__)}`", ""]
    doc = inspect.getdoc(bindview.Diagram.from_structure) or ""
    out += ["| | |", "|---|---|"]
    for m in re.finditer(r"^([a-z_, ]+):\n((?:    .*\n?|\n)+)", doc, flags=re.M):
        out.append(f"| `{'`, `'.join(n.strip() for n in m.group(1).split(','))}` | {plain(' '.join(m.group(2).split())).replace('|', chr(92) + '|')} |")
    out += ["", "## Options of render", "", f"`{call('Diagram.render', bindview.Diagram.render)}`", "",
            "`style` is `\"detailed\"` or `\"schematic\"`; `theme` is `\"light\"`, `\"dark\"`, `\"auto\"` or a `Palette`. `render_svg(path, ...)` "
            "takes the same and writes the file. `Interface.render` and `Comparison.render` take these options too.", ""]
    for title, rows in GROUPS:
        out += [f"**{title}**", "", "| option | default | |", "|---|---|---|"]
        out += [f"| `{n}` | `{defaults[n]!r}` | {line.replace('|', chr(92) + '|')} |" for n, line in rows] + [""]
    for title, names in PARTS:
        out += [f"## {title}", ""]
        for name in names:
            obj = getattr(bindview, name)
            out += of_class(name, obj) if inspect.isclass(obj) else [f"### `{call(name, obj)}`", "", first(obj), ""]
    out += ["## Functions of modules", ""]
    for module, obj, names in MODULES:
        for name in names:
            out += [f"### `{module}.{call(name, getattr(obj, name))}`", "", first(getattr(obj, name)), ""]
    return "\n".join(out).rstrip() + "\n"


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="compare with the page on disk instead of writing it")
    args = ap.parse_args(argv)
    made = text()
    if args.check:
        same = OUT.exists() and OUT.read_text(encoding="utf-8") == made
        print(f"{OUT.relative_to(ROOT)}: {'as the library says' if same else 'NOT what the library says now'}")
        return 0 if same else 1
    OUT.write_text(made, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({len(made.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
