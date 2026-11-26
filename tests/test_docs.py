"""What is written about the library is true of it: the API page, the help, the README, the manual, the reference page
and the schema of a graph.

The pages are read, not drawn. The calls they print are parsed, and every keyword in them is held to the signature of
the call; every link and picture has to be there; the commands have to be ones the command line takes. The pictures are
drawn again and compared with the files where the stored figures were made (``conftest.on_golden_platform``).
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import inspect
import re
import shlex
import sys
from pathlib import Path

import pytest

from conftest import ROOT, on_golden_platform
import bindview
from bindview import Comparison, Diagram, Ensemble, Graph, Interface
from bindview.cli import build_parser
from bindview.options import GROUPS, options
from bindview.styles.base import RenderOptions

sys.path.insert(0, str(ROOT / "scripts"))

import build_api  # noqa: E402
import build_docs_images  # noqa: E402

PAGES = {name: ROOT / name for name in ("README.md", "docs/manual.md", "docs/reference.md", "docs/api.md",
                                        "docs/graph-schema.md")}
#: The pages that show how the library is called.
TAUGHT = ("README.md", "docs/manual.md")


def page(name: str) -> str:
    return PAGES[name].read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# The help, and the page that is read off it
# ---------------------------------------------------------------------------

def test_the_api_page_is_what_the_library_says():
    assert page("docs/api.md") == build_api.text(), "python scripts/build_api.py writes it"
    listed = {n for _, names in build_api.PARTS for n in names}
    assert listed | set(build_api.LEFT_OUT) == set(bindview.__all__) and not listed & set(build_api.LEFT_OUT)


def test_every_public_name_has_its_line_of_help():
    for name in bindview.__all__:
        obj = getattr(bindview, name)
        if not (inspect.isclass(obj) or inspect.isfunction(obj)):
            continue
        assert (inspect.getdoc(obj) or "").strip(), name
        if name in build_api.LEFT_OUT:
            continue
        for key, member in vars(obj).items() if inspect.isclass(obj) else ():
            if key.startswith("_") or not isinstance(member, (property, classmethod, staticmethod)) and not inspect.isfunction(member):
                continue
            assert (inspect.getdoc(member) or "").strip(), f"{name}.{key}"


def test_every_option_of_render_is_listed_once_with_its_default():
    fields = {f.name: f.default for f in dataclasses.fields(RenderOptions)}
    listed = [name for _, rows in GROUPS for name, _ in rows]
    assert sorted(listed) == sorted(fields)
    text = options()
    for name, default in fields.items():
        assert f"{name}={default!r}" in text and f"| `{name}` | `{default!r}` |" in page("docs/api.md")
        assert f"{name}={default!r}" in (RenderOptions.__doc__ or "")
    assert repr(text) == str(text)                                               # shown as it is at the prompt
    assert "plddt_colors=" in options("pLDDT") and "title=" not in options("pLDDT")
    assert options("zebra") == "no keyword of render mentions 'zebra'"


# ---------------------------------------------------------------------------
# Links, pictures, calls and commands
# ---------------------------------------------------------------------------

def anchors(path: Path) -> set[str]:
    out = set()
    for head in re.findall(r"^#+ (.+)$", path.read_text(encoding="utf-8"), flags=re.M):
        out.add(re.sub(r"[^\w\- ]", "", re.sub(r"`|\*", "", head).lower()).replace(" ", "-"))
    return out


@pytest.mark.parametrize("name", list(PAGES))
def test_every_link_and_picture_is_there(name):
    text = re.sub(r"```.*?```", "", page(name), flags=re.S)
    here = PAGES[name].parent
    targets = re.findall(r"\]\(([^)\s]+)\)", text) + re.findall(r'(?:src|srcset|href)="([^"]+)"', text)
    for target in targets:
        if target.startswith(("http://", "https://")):
            continue
        path, _, anchor = target.removesuffix("?raw=true").partition("#")        # (GitHub shows the file itself then)
        file = (here / path).resolve() if path else PAGES[name]
        assert file.exists(), f"{name}: {target}"
        assert ROOT in file.parents, f"{name}: {target} points out of the repository"
        if anchor and file.suffix == ".md":
            assert anchor in anchors(file), f"{name}: {target}"


def blocks(name: str, language: str) -> list[str]:
    return re.findall(rf"```{language}\n(.*?)```", page(name), flags=re.S)


def keywords(function) -> set[str]:
    return {p.name for p in inspect.signature(function).parameters.values()}


RENDER = {f.name for f in dataclasses.fields(RenderOptions)} | keywords(Diagram.render)
#: What a call on these pages may be given, by the name it is called by.
TAKES = {
    "from_structure": keywords(Diagram.from_structure) | keywords(Ensemble.from_structures),
    "render": RENDER | {"view", "pair", "min_ligands", "thumbnails"}, "render_svg": RENDER | {"path", "view", "pair", "min_ligands"},
    "consensus": keywords(Ensemble.consensus), "diagram": keywords(Ensemble.diagram), "arrange": keywords(Diagram.arrange) | RENDER,
    "Comparison": keywords(Comparison.__init__), "Interface": keywords(Interface.__init__),
    "move": keywords(Graph.move), "rotate": keywords(Graph.rotate), "flip": keywords(Graph.flip),
}


@pytest.mark.parametrize("name", ["README.md", "docs/manual.md", "docs/reference.md"])
def test_the_calls_that_are_printed_are_calls_of_the_library(name):
    found = 0
    for code in blocks(name, "python"):
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            called = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            if called in TAKES:
                found += 1
                for key in node.keywords:
                    assert key.arg is None or key.arg in TAKES[called], f"{name}: {called}({key.arg}=)"
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "graph":
                assert hasattr(Graph, called), f"{name}: graph.{called}"
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == "bindview":
                module = __import__(node.module, fromlist=["_"])
                for alias in node.names:
                    assert hasattr(module, alias.name), f"{name}: from {node.module} import {alias.name}"
    assert found or name not in TAUGHT


@pytest.mark.parametrize("name", ["README.md", "docs/manual.md", "docs/reference.md"])
def test_the_commands_that_are_printed_are_commands_of_bindview(name, monkeypatch):
    def refuse(self, message):
        raise AssertionError(message)
    monkeypatch.setattr(argparse.ArgumentParser, "error", refuse)
    found = 0
    for code in blocks(name, "bash"):
        for line in code.replace("\\\n", " ").splitlines():
            line = line.split(" #")[0].strip()
            if line.startswith("bindview ") and "--help" not in line:
                found += 1
                build_parser().parse_args(shlex.split(line)[1:])
    assert found or name not in TAUGHT


def test_the_front_page_stays_one_to_be_read():
    text = page("README.md")
    assert len(text.splitlines()) < 250 and len(text) < 16_000
    assert all(word in text for word in ("## Install", "## Quick start", "## Figures", "docs/manual.md", "docs/api.md"))


def test_the_pages_name_the_python_the_package_claims():
    tomllib = pytest.importorskip("tomllib")
    claimed = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["requires-python"]
    assert claimed == ">=3.12"
    for name in TAUGHT:
        assert not re.search(r"Python 3\.(?:[0-9]|1[01])\b", page(name)), name


# ---------------------------------------------------------------------------
# The pictures, and the code that draws them
# ---------------------------------------------------------------------------

IMAGES = ROOT / "docs" / "images"


def test_every_picture_of_the_front_page_has_the_code_that_draws_it():
    """The code is run by ``scripts/build_docs_images.py``; here it is counted: one figure to a name, and the files there."""
    shown = set(re.findall(r'src="docs/images/readme/(\w+)', page("README.md")))
    drawn = []
    for names, code in build_docs_images.readme_blocks():
        calls = [node for node in ast.parse(code).body if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                 and not ast.unparse(node.value.func).startswith("sys.")]
        assert len(calls) == len(names), names
        drawn += names
    assert len(set(drawn)) == len(drawn) and set(drawn) == shown
    assert len(blocks("README.md", "python")) == len(build_docs_images.readme_blocks()) + 1          # (and the quick start)
    for name in drawn:
        files = {f.name for f in (IMAGES / "readme").glob(f"{name}.*")}
        assert files in ({f"{name}.svg"}, {f"{name}.light.svg", f"{name}.dark.svg", f"{name}.live.svg"}), name
    assert {f.name.split(".")[0] for f in (IMAGES / "readme").iterdir()} == set(drawn)               # nothing left over


def test_what_a_click_opens_answers_the_pointer_where_github_shows_it():
    """GitHub serves a raw SVG with a policy that takes its style sheet and nothing else: no script, and no picture in
    it. A figure of the library is opened raw and answers the pointer; a sketch with a shaded surface has pictures in
    it, and its link is to the page that shows the file."""
    links = re.findall(r'href="(docs/images/readme/[^"]+)"', page("README.md"))
    assert links
    for link in links:
        svg = (ROOT / link.removesuffix("?raw=true")).read_text(encoding="utf-8")
        assert "<script" not in svg and 'xmlns="http://www.w3.org/2000/svg"' in svg.split(">", 1)[0], link
        if link.endswith("?raw=true"):
            assert "data:image" not in svg, link
            if ".live." in link:
                assert ":hover" in svg, link
        else:
            assert "data:image" in svg, link


def test_the_badges_say_what_the_project_says():
    tomllib = pytest.importorskip("tomllib")
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    made = build_docs_images._badges()
    assert project["version"] == bindview.__version__ and f"bindview {bindview.__version__}" in made["badges/version"]
    assert f"licence {project['license']}" in made["badges/licence"]
    for name, svg in made.items():
        assert (IMAGES / f"{name}.svg").read_text(encoding="utf-8") == svg and f"docs/images/{name}.svg" in page("README.md")
    assert "img.shields.io" not in page("README.md")                        # the page asks no other site for them


@pytest.mark.skipif(not on_golden_platform(), reason="the pictures were drawn on another platform (tests/golden/VERSIONS.json)")
def test_every_picture_is_what_the_code_draws(capsys):
    """``scripts/build_docs_images.py --check``: every picture of the pages drawn again by the code that draws it, and
    no file left that nothing draws."""
    assert build_docs_images.main(["--check"]) == 0, capsys.readouterr().out
