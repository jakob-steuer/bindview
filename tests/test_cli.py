"""The command line interface."""

from __future__ import annotations

import gzip
import importlib.util
import json
import runpy
import shutil
import sys
import xml.etree.ElementTree as ET

import pytest

import bindview.cli
from bindview.cli import main
from bindview.interface import dump_interface_contacts
from conftest import CASE_SPECS, TESTDATA, meta_of, model_of, trimer_contacts, trimer_file

KIT = "kit_imatinib"
RIBOSWITCH = TESTDATA / "nucleic" / "riboswitch_bisguanidine" / "model.pdb"


def render_kit(tmp_path, *more: str) -> "list[str]":
    ligand = meta_of(KIT)["ligand"]
    return ["render", str(model_of(KIT)), "--ligand", ligand["residue_name_in_cif"], "--smiles", ligand["smiles"],
            "--interactions", str(TESTDATA / KIT / "interactions.tsv"), "-o", str(tmp_path / "figure.svg"), *more]


def test_a_model_with_its_table_and_its_meta_json_beside_it(tmp_path, capsys):
    """The table beside the model is found, the SMILES is read from the meta.json beside it, and it is said."""
    out = tmp_path / "out.svg"
    code = main(["render", str(model_of(KIT)), "--ligand", meta_of(KIT)["ligand"]["residue_name_in_cif"],
                 "--smiles-from", "meta.json", "-o", str(out)])
    assert code == 0 and out.read_text().startswith("<svg ")
    err = capsys.readouterr().err
    assert "interactions.tsv (found next to the structure)" in err and "ok: diagram" in err


def test_json_report_and_options(tmp_path, capsys):
    out = tmp_path / "kit.svg"
    code = main(render_kit(tmp_path, "--style", "schematic", "--theme", "dark", "--title", "Imatinib", "--json",
                           "-q", "-o", str(out)))
    assert code == 0
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert captured.err == ""
    assert report["status"] == "ok" and report["style"] == "schematic" and report["theme"] == "dark"
    assert report["n_interactions"] == CASE_SPECS[KIT]["n_interactions"] and report["quality"]["overlaps"] == 0
    assert 'fill="#0f172a"' in out.read_text()


def test_unreadable_exits_3_and_still_writes_the_fallback(tmp_path):
    """A figure 40 mm wide is too small for its letters: the exit status says so, and the summary of the
    residues is written in its place unless that is declined."""
    out = tmp_path / "small.svg"
    assert main(render_kit(tmp_path, "-q", "--width-mm", "40", "-o", str(out))) == 3
    assert "Residue-level summary" in out.read_text()
    out2 = tmp_path / "none.svg"
    assert main(render_kit(tmp_path, "-q", "--width-mm", "40", "--on-unreadable", "none", "-o", str(out2))) == 3
    assert not out2.exists()


def test_empty_complex_exits_0(tmp_path, capsys):
    """A table of no contacts (a header line) is an empty figure, which is no error."""
    table = tmp_path / "none.tsv"
    table.write_text("interaction_type\tresidue_number\tresidue_name\tchain\n")
    out = tmp_path / "empty.svg"
    ligand = meta_of(KIT)["ligand"]
    assert main(["render", str(model_of(KIT)), "--ligand", ligand["residue_name_in_cif"], "--smiles", ligand["smiles"],
                 "--interactions", str(table), "-o", str(out)]) == 0
    assert "empty: empty" in capsys.readouterr().err
    assert "No interactions in the supplied set" in out.read_text()


def test_errors_exit_2_with_a_message(tmp_path, capsys):
    cif, ligand = str(model_of(KIT)), meta_of(KIT)["ligand"]["residue_name_in_cif"]
    assert main(["render", cif, "--ligand", "NOPE", "-o", str(tmp_path / "x.svg")]) == 2
    assert "no residue named 'NOPE'" in capsys.readouterr().err
    assert main(["render", str(tmp_path / "missing.cif"), "--ligand", ligand]) == 2
    assert "cannot read" in capsys.readouterr().err
    assert main(["render", cif, "--ligand", ligand, "--smiles-from", "nothing.json", "-o", str(tmp_path / "y.svg")]) == 2
    assert not (tmp_path / "x.svg").exists()


def test_default_output_name_and_smiles_argument(testdata, tmp_path):
    """Without -o the figure lies beside the model, named after it and the style."""
    work = tmp_path / "case"
    work.mkdir()
    with gzip.open(testdata / "ca2_celecoxib" / "model.cif.gz", "rb") as packed:
        (work / "model.cif").write_bytes(packed.read())
    assert main(["render", str(work / "model.cif"), "--ligand", "ZN", "--smiles", "[Zn+2]", "--distances-only",
                 "-q"]) == 0
    assert (work / "model.detailed.svg").exists()


def test_the_default_output_name_of_a_packed_model_leaves_out_both_suffixes(testdata, tmp_path):
    work = tmp_path / "case"
    work.mkdir()
    shutil.copy(testdata / "ca2_celecoxib" / "model.cif.gz", work / "model.cif.gz")
    assert main(["render", str(work / "model.cif.gz"), "--ligand", "ZN", "--smiles", "[Zn+2]", "--distances-only",
                 "-q"]) == 0
    assert (work / "model.detailed.svg").exists()


@pytest.mark.skipif(importlib.util.find_spec("plip") is not None, reason="PLIP is installed here")
def test_detection_without_plip_is_a_clear_error(tmp_path, capsys):
    code = main(["render", str(model_of(KIT)), "--ligand", meta_of(KIT)["ligand"]["residue_name_in_cif"], "--detect",
                 "-o", str(tmp_path / "d.svg")])
    assert code == 2
    assert "bindview[plip]" in capsys.readouterr().err


def test_help_mentions_the_licence_of_the_detector(capsys):
    with pytest.raises(SystemExit):
        main(["render", "--help"])
    assert "GPL-2.0" in capsys.readouterr().out


def test_the_help_is_short_at_the_top_and_grouped(capsys):
    """A usage line a reader can take in, the options in groups, the ones that each draw one thing more plainly
    last, and examples at the end."""
    with pytest.raises(SystemExit):
        main(["render", "--help"])
    out = capsys.readouterr().out
    assert out.startswith("usage: bindview render STRUCTURE --ligand LIGAND [options]\n")
    groups = [line[:-1] for line in out.splitlines() if line.endswith(":") and not line.startswith(" ")]
    assert groups == ["options", "input", "contacts", "model confidence", "what is drawn", "look",
                      "the settings of a style, one by one", "page and text", "output",
                      "one thing drawn more plainly", "examples"]
    with pytest.raises(SystemExit):
        main(["interface", "--help"])
    out = capsys.readouterr().out
    assert out.startswith("usage: bindview interface STRUCTURE --between A B [options]\n") and "--bare" in out


def test_bare_leaves_the_words_off_and_hands_them_back(tmp_path, capsys):
    """--bare: the drawing and its key; the words it leaves off are in --json and in --notes-file."""
    plain, bare, notes = tmp_path / "plain.svg", tmp_path / "bare.svg", tmp_path / "notes.txt"
    ligand = meta_of(KIT)["ligand"]["residue_name_in_cif"]
    assert main(["render", str(model_of(KIT)), "--ligand", ligand, "--smiles-from", "meta.json",
                 "-o", str(plain), "-q"]) == 0
    capsys.readouterr()
    assert main(["render", str(model_of(KIT)), "--ligand", ligand, "--smiles-from", "meta.json",
                 "-o", str(bare), "--bare", "--notes-file", str(notes), "--json", "-q"]) == 0
    said = json.loads(capsys.readouterr().out)["notes"]
    drawn, left = plain.read_text(), bare.read_text()
    for group in ('class="confidence"', 'class="footnotes"'):
        assert group in drawn and group not in left
    assert 'class="legend"' in left
    assert notes.read_text().splitlines() == said
    assert said[0].startswith("Ligand buried ") and "Distances in Å." in said     # a crystal: no model confidence
    # every word that was left off is a word of the plain figure, and the drawing itself is the same
    assert all(word in drawn for note in said for word in note.split()[:3])
    body = lambda svg: svg[svg.index('<g class="diagram"'):svg.index('<g class="legend"')]
    assert body(drawn).split(">", 1)[1] == body(left).split(">", 1)[1]


def test_basepairs_from_the_command_line(testdata, tmp_path, capsys):
    folder = testdata / "nucleic" / "riboswitch_bisguanidine"
    out, graph = tmp_path / "pairs.svg", tmp_path / "pairs.json"
    assert main(["basepairs", str(folder / "model.pdb"), "--chains", "A", "B",
                 "--pairs", str(folder / "basepairs.rnapolis.json"), "--ligand", "BGD",
                 "--ligand-contacts", str(folder / "interactions.tsv"), "--interactive", "--theme", "auto",
                 "-o", str(out), "--graph", str(graph), "--json", "-q"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert (printed["status"], printed["pairs_across"], printed["pairs_within"], printed["stacks"]) == ("ok", 4, 16, 2)
    assert printed["ligand"]["nucleotides"] == 5 and "lf-live" in out.read_text()
    assert json.loads(graph.read_text())["version"] == "1.4"
    moved = tmp_path / "moved.svg"
    assert main(["basepairs", str(folder / "model.pdb"), "--chains", "A", "B",
                 "--pairs", str(folder / "basepairs.rnapolis.json"), "--ligand", "BGD",
                 "--ligand-contacts", str(folder / "interactions.tsv"), "--from-graph", str(graph),
                 "-o", str(moved), "-q"]) == 0
    assert moved.read_text().startswith("<svg ")
    # No annotation: the plain table's stand-in, across and within each strand, and said.
    plain = tmp_path / "plain.svg"
    assert main(["basepairs", str(folder / "model.pdb"), "--chains", "A", "B", "-o", str(plain)]) == 0
    err = capsys.readouterr().err
    assert "the plain distance table's stand-in" in err and "2 pairs across, 14 within, 2 stacks" in err
    with pytest.raises(SystemExit):
        main(["basepairs", "--help"])
    help_text = capsys.readouterr().out
    assert help_text.startswith("usage: bindview basepairs STRUCTURE --chains A B [options]\n")
    assert "a ligand beside the pairs:" in help_text and "examples:" in help_text


def test_the_settings_of_a_style_are_flags(testdata, tmp_path, capsys):
    """One residue drawn with its atoms and every other as a bubble; the lining of the contacts, and none."""
    out = tmp_path / "f.svg"
    base = render_kit(tmp_path, "--style", "schematic", "-q", "-o", str(out))
    residues = {line.split("\t")[3] + line.split("\t")[1]                          # chain and number, in the table
                for line in (testdata / KIT / "interactions.tsv").read_text().splitlines()[1:] if line.strip()}
    assert main([*base, "--lining", "contacts", "--no-exposure", "--lines", "detailed",
                 "--residues", "Thr670=atoms", "*=bubbles"]) == 0
    svg = out.read_text()
    assert '<g class="lining"' in svg and "Solvent exposure" not in svg and ">Hydrogen bond<" in svg
    assert svg.count('data-form="fragment"') == 1 and svg.count('data-form="bubble"') == len(residues) - 1
    assert main([*base, "--lining", "none"]) == 0 and '<g class="lining"' not in out.read_text()
    for wrong in (["balls"], ["Thr670"], ["Thr670=balls"]):
        assert main([*base, "--residues", *wrong]) == 2
        assert "--residues" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Inputs made here: a few lines of a model, a table, a file that is not what it is said to be
# ---------------------------------------------------------------------------

def glycines(numbers, y: float = 0.0, chain: str = "A", first_atom: int = 1) -> "list[str]":
    """Glycines in a row along x, as the lines of a PDB file."""
    lines, serial = [], first_atom
    for k, number in enumerate(numbers):
        for name, dx, dy in (("N", -1.2, 0.5), ("CA", 0.0, 0.0), ("C", 1.2, 0.5), ("O", 1.3, 1.7)):
            lines.append(f"ATOM  {serial:5d}  {name:<3s} GLY {chain}{number:4d}    {3.8 * k + dx:8.3f}{y + dy:8.3f}{0.0:8.3f}"
                         f"  1.00 20.00          {name[0]:>2s}")
            serial += 1
    return lines


def test_a_domain_may_begin_at_a_negative_residue_number(tmp_path, capsys):
    model = tmp_path / "two_stretches.pdb"
    model.write_text("\n".join(glycines(range(-5, 11)) + glycines(range(20, 31), y=40.0, first_atom=100)) + "\nEND\n")
    for first in ("N=A:-5-10", "N=A:-5--2"):
        assert main(["report", str(model), "--look", "--domain", first, "--domain", "Z=A:20-30"]) == 0
        assert "N, Z: the parts that were named do not touch" in capsys.readouterr().err
    for wrong in ("N=A:5", "N=A:5-", "N=A:-5", "N=A:a-b", "N=A:1-2-3", "N=:1-2", "=A:1-2"):
        assert main(["report", str(model), "--look", "--domain", wrong, "--domain", "Z=A:20-30"]) == 2
        assert f"--domain takes NAME=CHAIN:FROM-TO, not {wrong!r}" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# A wrong input is a message and exit status 2, whatever raises it
# ---------------------------------------------------------------------------

def not_text(tmp_path, name: str) -> str:
    (tmp_path / name).write_bytes(b"\x00\x01\x02\xff\xfe binary\n")
    return str(tmp_path / name)


def refused(argv: "list[str]", capsys) -> str:
    """The message of a command that ends with status 2: one line on standard error, and nothing else there."""
    assert main(argv) == 2
    captured = capsys.readouterr()
    lines = captured.err.splitlines()
    assert lines and lines[-1].startswith("bindview: error: ") and "Traceback" not in captured.err
    assert captured.out == ""
    return lines[-1]


def test_a_file_that_is_not_text_is_named(tmp_path, capsys):
    ligand = meta_of(KIT)["ligand"]["residue_name_in_cif"]
    smiles = not_text(tmp_path, "smiles.json")
    said = refused(["render", str(model_of(KIT)), "--ligand", ligand, "--smiles-from", smiles], capsys)
    assert smiles in said and "UTF-8" in said
    table = not_text(tmp_path, "contacts.tsv")
    said = refused(["render", str(model_of(KIT)), "--ligand", ligand, "--interactions", table], capsys)
    assert table in said
    said = refused(["interface", str(trimer_file("model")), "--between", "A", "B", "--contacts", table], capsys)
    assert table in said and "UTF-8" in said


def test_a_file_that_is_not_there_is_named(tmp_path, capsys):
    missing = str(tmp_path / "missing.tsv")
    assert missing in refused(["interface", str(trimer_file("model")), "--between", "A", "B", "--contacts", missing], capsys)
    pairs = ["basepairs", str(RIBOSWITCH), "--chains", "A", "B", "-o", str(tmp_path / "pairs.svg")]
    missing = str(tmp_path / "missing.json")
    assert missing in refused([*pairs, "--pairs", missing], capsys)
    assert missing in refused([*pairs, "--stacks", missing], capsys)
    garbage = tmp_path / "garbage.json"
    garbage.write_text("{not json")
    assert str(garbage) in refused([*pairs, "--pairs", str(garbage)], capsys)
    assert not (tmp_path / "pairs.svg").exists()


def test_a_folder_that_is_not_there_is_said_before_anything_is_drawn(tmp_path, capsys):
    nowhere = tmp_path / "nowhere"
    notes = str(nowhere / "notes.txt")
    assert notes in refused(render_kit(tmp_path, "--notes-file", notes), capsys)
    assert not (tmp_path / "figure.svg").exists()
    pairs = tmp_path / "pairs.svg"
    said = refused(["basepairs", str(RIBOSWITCH), "--chains", "A", "B", "-o", str(pairs), "--notes-file", notes], capsys)
    assert notes in said and not pairs.exists()
    model = tmp_path / "strand.pdb"
    model.write_text("\n".join(glycines(range(1, 13))) + "\nEND\n")
    page = str(nowhere / "report.html")
    assert page in refused(["report", str(model), "--one-view", "-q", "-o", page], capsys)
    assert not nowhere.exists()


def test_a_file_that_cannot_be_written_is_named(tmp_path, capsys):
    nowhere = tmp_path / "nowhere"
    page = str(nowhere / "arrange.html")
    assert page in refused(render_kit(tmp_path, "-q", "--arrange-page", page), capsys)
    contacts = tmp_path / "contacts.tsv"
    contacts.write_text(dump_interface_contacts(trimer_contacts()), encoding="utf-8")
    figure = str(nowhere / "interface.svg")
    said = refused(["interface", str(trimer_file("model")), "--between", "A", "B", "--contacts", str(contacts), "-q",
                    "-o", figure], capsys)
    assert figure in said and not nowhere.exists()


def test_what_the_library_refuses_is_a_message_too(tmp_path, capsys):
    # A layout of the other style, and a small molecule asked for as a chain: a ValueError each.
    graph = tmp_path / "schematic.json"
    assert main(render_kit(tmp_path, "-q", "--style", "schematic", "--graph", str(graph))) == 0 and graph.exists()
    assert "'schematic' style, not 'detailed'" in refused(render_kit(tmp_path, "-q", "--layout", str(graph)), capsys)
    assert "cannot be drawn as a chain" in refused(render_kit(tmp_path, "-q", "--ligand-as", "chain"), capsys)


@pytest.mark.parametrize("argv, option", [
    (["render", "model.cif", "--ligand", "LIG", "--width-mm", "nan"], "--width-mm"),
    (["render", "model.cif", "--ligand", "LIG", "--width-mm", "0"], "--width-mm"),
    (["render", "model.cif", "--ligand", "LIG", "--bond-length", "-3"], "--bond-length"),
    (["render", "model.cif", "--ligand", "LIG", "--bond-length", "nan"], "--bond-length"),
    (["render", "model.cif", "--ligand", "LIG", "--bond-length", "0"], "--bond-length"),
    (["render", "model.cif", "--ligand", "LIG", "--min-font-pt", "-1"], "--min-font-pt"),
    (["interface", "model.cif", "--between", "A", "B", "--max-residues", "0"], "--max-residues"),
    (["interface", "model.cif", "--between", "A", "B", "--max-residues", "-3"], "--max-residues"),
    (["basepairs", "model.pdb", "--chains", "A", "B", "--step", "0"], "--step"),
    (["basepairs", "model.pdb", "--chains", "A", "B", "--step", "-5"], "--step"),
    (["compare", "a", "b", "--width-mm", "inf"], "--width-mm"),
])
def test_a_number_that_cannot_be_one_is_refused_where_it_is_given(argv, option, capsys):
    with pytest.raises(SystemExit) as left:
        main(argv)
    said = capsys.readouterr().err
    assert left.value.code == 2 and f"argument {option}: {argv[-1]!r} is not a" in said and "Traceback" not in said


def test_a_mistake_of_the_program_is_not_dressed_as_one_of_the_input(monkeypatch):
    def broken(args):
        raise TypeError("a mistake in the program")
    monkeypatch.setattr(bindview.cli, "_render", broken)
    with pytest.raises(TypeError, match="a mistake in the program"):
        main(["render", "model.cif", "--ligand", "LIG"])


def test_base_pairs_that_are_not_legible_at_the_width_asked_for_exit_3(tmp_path, capsys):
    """As render and compare do: the figure is written, the status says that it is not legible, and so does the exit."""
    out = tmp_path / "pairs.svg"
    pairs = ["basepairs", str(RIBOSWITCH), "--chains", "A", "B", "--json", "-q", "-o", str(out)]
    assert main([*pairs, "--width-mm", "20"]) == 3
    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == "unreadable" and out.read_text().startswith("<svg ")
    assert any("its smallest text would be under 5 pt" in w for w in printed["warnings"])
    assert main(pairs) == 0 and json.loads(capsys.readouterr().out)["status"] == "ok"
    with pytest.raises(SystemExit):
        main(["basepairs", "--help"])
    assert "exit status: 0 a figure was written; 3 it is not readable" in capsys.readouterr().out


def test_what_the_help_says_of_no_caption_is_what_becomes_of_the_line(tmp_path, capsys):
    """--no-caption leaves the line of the pairs' source off the figure. Where the line then is, and is not, is
    what the help says: in the file's description; in --notes-file and --json only if the help says so."""
    notes, out = tmp_path / "notes.txt", tmp_path / "pairs.svg"
    pairs = ["basepairs", str(RIBOSWITCH), "--chains", "A", "B", "-q", "-o", str(out), "--notes-file", str(notes), "--json"]

    def described() -> str:
        return ET.fromstring(out.read_text()).find("{http://www.w3.org/2000/svg}desc").text

    assert main(pairs) == 0
    (line,) = notes.read_text().splitlines()
    assert line.startswith("Pairs and stacks: ") and described() == line and json.loads(capsys.readouterr().out)["notes"] == [line]
    assert main([*pairs, "--no-caption"]) == 0
    in_notes, in_json = notes.read_text().splitlines(), json.loads(capsys.readouterr().out)["notes"]
    assert in_notes == in_json and in_notes in ([], [line])
    with pytest.raises(SystemExit):
        main(["basepairs", "--help"])
    said = " ".join(capsys.readouterr().out.split())
    entry = said[said.index("--no-caption"):said.index("--step PX")]
    assert described() == line and "it stays in the file's description" in entry
    assert ("is not in --json or --notes-file" in entry) == (in_notes == [])
    assert ("and in --notes-file" in entry) == (in_notes == [line])


def test_python_m_bindview_is_the_command_line(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(sys, "argv", ["bindview", "--version"])
    with pytest.raises(SystemExit) as left:
        runpy.run_module("bindview", run_name="__main__")
    assert left.value.code == 0 and capsys.readouterr().out.startswith("bindview ")
    monkeypatch.setattr(sys, "argv", ["bindview", "render", str(tmp_path / "no_model.cif"), "--ligand", "LIG"])
    with pytest.raises(SystemExit) as left:
        runpy.run_module("bindview", run_name="__main__")
    assert left.value.code == 2 and "bindview: error: cannot read" in capsys.readouterr().err


def test_a_member_of_a_comparison_is_a_folder_as_the_archive_packs_it(tmp_path):
    """A folder of a comparison was looked for as model.cif only, and the committed cases hold model.cif.gz."""
    out = tmp_path / "pair.svg"
    assert main(["compare", str(TESTDATA / "ar_r1881"), str(TESTDATA / "ar_bicalutamide"), "-q", "-o", str(out)]) == 0
    assert out.read_text().startswith("<svg")
