"""``bindview report``: what an input holds, and the figures that follow from it.

Nobody tells the report what kind of input it has. Its rules are in the docstring of ``bindview.report``;
these tests hold them to the cases of the repository, whose answers are known, and to the one thing a report
promises: the call it prints is the call that drew the figure.
"""

from __future__ import annotations

import ast
import csv
import gzip
import json
import shutil
import subprocess
import sys
from collections import Counter

import pytest

from conftest import CASES, MULTIMER_KEPT, ROOT, TESTDATA, fetched, meta_of, model_of
from bindview import report
from bindview.cli import main
from bindview.report import survey
from bindview.structure import parse_pdb

RIBOSWITCH = TESTDATA / "nucleic" / "riboswitch_bisguanidine"
TRIMER = TESTDATA / "interface_multimer" / "tnfa_trimer"
BUNDLE = TESTDATA / "ensemble" / "dhfr_mtx" / "structure"
TECTB = TESTDATA / "interface_pae" / "afdb_tectb"
OPEN = TESTDATA / "ca2_celecoxib"            # an entry of the PDB: one chain, celecoxib (CEL) and a zinc ion
_NO_ATOMS = "interaction_type\tresidue_number\tresidue_name\tchain\n"
def kinds_in_table(folder) -> Counter:
    """The rows of a committed table by their kind, read without bindview."""
    with (folder / "interactions.tsv").open(newline="", encoding="utf-8") as handle:
        return Counter(row["interaction_type"] for row in csv.DictReader(handle, delimiter="\t"))


def with_its_chain(folder, into):
    """A copy of an AlphaFold database model's folder whose meta.json also names the chain its parts are of, as
    the report asks: the committed meta.json names the parts alone."""
    shutil.copytree(folder, into)
    meta = json.loads((into / "meta.json").read_text())
    (into / "meta.json").write_text(json.dumps({**meta, "chain": "A"}))
    return into


# ---------------------------------------------------------------------------
# What an input holds
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", CASES)
def test_the_ligand_of_an_entry_is_found_with_its_table_and_its_smiles(case):
    """The ligand that the meta.json beside the model names is drawn from the table beside it, which names
    its atoms, and with the SMILES that the meta.json gives."""
    ligand = meta_of(case)["ligand"]
    found = survey(TESTDATA / case)
    (track,) = [t for t in found.tracks if t.title == f"{ligand['component']} in its pocket"]
    assert track.kind == "ligand"
    assert 'load_interactions("interactions.tsv")' in track.code and "which names its atoms" in track.why
    assert f'smiles="{ligand["smiles"]}"' in track.code and "SMILES: from meta.json beside the model" in track.why


def test_a_ligand_in_rna_and_the_interface_between_the_two_hairpins():
    found = survey(RIBOSWITCH)
    assert [(c.name, c.kind, c.residues, c.oligo) for c in found.found.chains] == [
        ("A", "nucleic", 18, True), ("B", "nucleic", 18, True), ("C", "", 0, False)]
    (bgd,) = found.found.candidates
    assert (bgd.name, bgd.kind, bgd.heavy, bgd.bonded, bgd.formula) == ("BGD", "small molecule", 12, False, "C6N6")
    ligand, interface, pairs = found.tracks
    assert ligand.kind == "ligand" and "smiles=" in ligand.code and "meta.json" in ligand.why
    assert (interface.kind, interface.title) == ("interface", "The interface of chains A and B")
    assert "A (nucleic, 18 nucleotides) and B (nucleic, 18 nucleotides) touch" in interface.why
    # The plain table with ring stacking: 16 hydrogen bonds and the two stacks where the loops meet.
    assert "with ring stacking, has 18 rows" in interface.why and ', stacking=True)' in interface.code
    # The two hairpins again as their base pairs, the ligand beside them with the table beside the model.
    assert (pairs.kind, pairs.title) == ("basepairs", "The base pairs of chains A and B")
    assert "read_rnapolis_pairs(annotation)" in pairs.code and "RNApolis's annotation" in pairs.why
    assert 'ligand="C:BGD:19"' in pairs.code and "interactions.tsv" in pairs.code
    assert not found.left                    # no protein here: a hairpin is the ligand of nothing


def test_two_strands_are_drawn_as_their_base_pairs_from_the_best_annotation_there_is(tmp_path):
    """The PDB's annotation, else RNApolis's JSON beside the model, else the plain table's stand-in; a small
    molecule within 4 A of a strand beside the pairs; one view, the pointer."""
    folder = tmp_path / "rna"
    shutil.copytree(RIBOSWITCH, folder)
    found = survey(folder)
    found.draw()
    pairs = found.tracks[2]
    assert pairs.status == "ok" and "lf-live" not in pairs.svg and 'class="ligand"' in pairs.svg
    assert "RNApolis's annotation basepairs.rnapolis.json" in pairs.why and "read_rnapolis_stacks(annotation)" in pairs.code
    (view,) = pairs.views
    assert view.line == "figure = pairs.render(interactive=True)" and "lf-live" in view.svg
    (folder / "basepairs.rnapolis.json").unlink()                     # no annotation: the stand-in, and said
    plain = survey(folder)
    plain.draw()
    assert plain.tracks[2].status == "ok" and "stand-in" in plain.tracks[2].why
    assert "pairs_from_contacts(contacts)[0]" in plain.tracks[2].code


def test_a_trimer_has_three_interfaces_each_with_its_predicted_aligned_errors():
    found = survey(TRIMER)
    assert len(found.models) >= MULTIMER_KEPT                       # the draws in the repository, and any held beside
    assert [t.kind for t in found.tracks] == ["interface"] * 3
    assert {t.title for t in found.tracks} == {f"The interface of chains {a} and {b}"
                                               for a, b in (("A", "B"), ("A", "C"), ("B", "C"))}
    first = found.models[0].name.rsplit(".", 1)[0]
    assert all(f"pae_{first}.npz" in t.code and "from bindview.distance_table import interface_contacts" in t.code
               for t in found.tracks)
    assert found.left == [(f"{len(found.models)} models", "an ensemble is counted for a small molecule, and there is none")]


def test_the_parts_of_one_chain_that_a_file_beside_the_model_names_are_an_interface(tmp_path):
    """On the AlphaFold database's model of TECTB, its meta.json naming the chain and its two domains."""
    found = survey(with_its_chain(TECTB, tmp_path / "tectb"))
    (track,) = found.tracks
    assert (track.kind, track.title) == ("parts", "The interface of N-terminal domain and ZP domain")
    assert '.split("A", {"X": (28, 124), "Y": (126, 269)})' in track.code
    assert "named in meta.json" in track.why and not found.left
    found.draw(views=False)
    assert track.status == "ok"


def test_the_predicted_aligned_error_of_the_database_beside_the_model_goes_in(tmp_path):
    (track,) = survey(with_its_chain(TECTB, tmp_path / "tectb")).tracks
    assert "predicted_aligned_error.json.gz" in track.code and "pae=" in track.code


def test_one_chain_and_nothing_beside_it_has_no_figure_until_somebody_names_its_parts(tmp_path):
    shutil.copy(TECTB / "model.cif.gz", tmp_path / "model.cif.gz")
    found = survey(tmp_path)
    ((what, why),) = found.left
    assert not found.tracks and what == "chain A alone" and "--domain" in why
    page = found.html()
    assert "<h2>Found, and not drawn</h2>" in page and "chain A alone" in page and "<h2>The figures (0)</h2>" in page
    (named,) = survey(tmp_path, domains=[("N", "A", 28, 124), ("ZP", "A", 126, 269)]).tracks
    assert named.kind == "parts" and "named by the caller" in named.why and "pae=" not in named.code
    apart = survey(tmp_path, domains=[("first", "A", 28, 30), ("last", "A", 265, 269)])
    assert not apart.tracks and ("first, last", "the parts that were named do not touch") in apart.left


@pytest.mark.parametrize("parts, said", (
    ([("N", "A", 800, 700), ("C", "A", 4, 10)], "part 'N' runs from 800 to 700: the first number comes first"),
    ([("N", "A", 4, 130), ("C", "Q", 1, 100)], "part 'C': chain 'Q' has no residue from 1 to 100"),
    ([("N", "A", 4, 130), ("C", "A", 100, 260)], "parts 'N' (4 to 130) and 'C' (100 to 260) overlap on chain 'A'"),
    ([("N", "A", 4, 130)], "one part was named ('N'): an interface is between two"),
))
def test_a_part_that_cannot_be_cut_out_is_refused_under_the_name_its_caller_gave_it(parts, said, tmp_path, capsys):
    """The parts get chain names of their own for the cut, X and Y; a refusal names a part by the name its
    caller gave it. One part alone is refused: an interface is between two."""
    with pytest.raises(report.BindviewError) as refused:
        survey(OPEN, domains=parts)
    assert said in str(refused.value) and "'X'" not in str(refused.value) and "'Y'" not in str(refused.value)
    if len(parts) > 1:                                   # (the command refuses one --domain by itself)
        named = [arg for name, chain, first, last in parts for arg in ("--domain", f"{name}={chain}:{first}-{last}")]
        assert main(["report", str(OPEN), "-o", str(tmp_path / "page.html"), *named]) == 2
        assert said in capsys.readouterr().err and not (tmp_path / "page.html").exists()


@pytest.mark.parametrize("named, what, why", (
    ({"chain": "A", "participants": [{"name": "N", "residues": ["a", "b"]}, {"name": "C", "residues": [131, 260]}]},
     "part 'N' of meta.json", "its residues are ['a', 'b'], not two residue numbers"),
    ({"chain": "Q", "participants": [{"name": "N", "residues": [1, 100]}, {"name": "C", "residues": [101, 200]}]},
     "N, C (parts that meta.json names)", "no interface of them is drawn: part 'N': chain 'Q' has no residue from 1 to 100"),
    ({"chain": "A", "participants": [{"name": "N", "residues": [4, 130]}, {"name": "C", "residues": [100, 260]}]},
     "N, C (parts that meta.json names)",
     "no interface of them is drawn: parts 'N' (4 to 130) and 'C' (100 to 260) overlap on chain 'A'"),
))
def test_parts_that_a_file_beside_the_model_names_wrongly_are_a_sentence_and_the_report_is_made(named, what, why, tmp_path):
    """A meta.json is found, not handed in: what is wrong in it (words for residue numbers, a chain that is not
    there, parts that overlap) is a line of the page, and the figure of the ligand is drawn."""
    work = tmp_path / "complex"
    shutil.copytree(OPEN, work)
    meta = json.loads((work / "meta.json").read_text())
    (work / "meta.json").write_text(json.dumps({**meta, **named}))
    found = survey(work)
    (track,) = found.tracks
    assert track.kind == "ligand" and "from meta.json beside the model" in track.why
    assert (what, why) in found.left
    page = found.html()
    assert report.html.escape(what) in page and report.html.escape(why) in page
    # What is right in the same file is drawn as before.
    (work / "meta.json").write_text(json.dumps({**meta, "chain": "A", "participants": [
        {"name": "N", "residues": [4, 130]}, {"name": "C", "residues": [131, 260]}]}))
    assert [t.title for t in survey(work).tracks] == ["CEL in its pocket", "The interface of N and C"]


#: Two residues of a chain, a residue of which one atom is modelled, and a ligand with a glutamate's atom
#: names (as methotrexate has them).
_NAMED_LIKE_A_BACKBONE = """\
ATOM      1  N   GLY A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  GLY A   1       1.460   0.000   0.000  1.00  0.00           C
ATOM      3  C   GLY A   1       2.000   1.400   0.000  1.00  0.00           C
ATOM      4  O   GLY A   1       1.300   2.400   0.000  1.00  0.00           O
ATOM      5  N   ALA A   2       3.330   1.500   0.000  1.00  0.00           N
ATOM      6  CA  ALA A   2       4.000   2.800   0.000  1.00  0.00           C
ATOM      7  C   ALA A   2       5.500   2.700   0.000  1.00  0.00           C
ATOM      8  O   ALA A   2       6.100   1.600   0.000  1.00  0.00           O
ATOM      9  CB  ALA A   2       3.600   3.600   1.200  1.00  0.00           C
ATOM     10  N   PRO A   3       6.100   3.900   0.000  1.00  0.00           N
HETATM   11  N   MTX A 170      20.000  20.000  20.000  1.00  0.00           N
HETATM   12  CA  MTX A 170      21.460  20.000  20.000  1.00  0.00           C
HETATM   13  C   MTX A 170      22.000  21.400  20.000  1.00  0.00           C
HETATM   14  O   MTX A 170      21.300  22.400  20.000  1.00  0.00           O
HETATM   15  C1  MTX A 170      22.100  19.000  21.000  1.00  0.00           C
END
"""


def test_a_ligand_with_backbone_atom_names_is_a_ligand_and_half_a_residue_is_no_ion():
    found = report.look(parse_pdb(_NAMED_LIKE_A_BACKBONE))
    assert [(c.name, c.kind, c.residues) for c in found.chains] == [("A", "protein", 3)]
    (mtx,) = found.candidates
    assert (mtx.name, mtx.kind, mtx.heavy, mtx.bonded) == ("MTX", "small molecule", 5, False)


def _pdb_line(record: str, serial: int, name: str, resname: str, chain: str, number: int, xyz, element: str) -> str:
    return (f"{record:<6s}{serial:5d} {name if len(name) == 4 else ' ' + name:<4s} {resname:>3s} {chain}{number:4d}    "
            f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00          {element:>2s}")


def _capped_stretch(residues: int = 61) -> str:
    """A stretch of a chain of the PDB as a PDB file, with the two caps of a synthetic peptide: an acetyl group where
    the residue before it was, and an amide nitrogen (NH2: one atom) where the nitrogen of the residue after was."""
    from bindview import read_structure

    model = read_structure(model_of("ca2_celecoxib"))
    keys = [k for k in model.residues() if k.chain == "A" and not any(a.hetero for a in model.residue_atoms(k))]
    start = next(i for i in range(1, len(keys) - residues)
                 if keys[i + residues].number - keys[i - 1].number == residues + 1)        # no gap in the numbering
    before, after = keys[start - 1], keys[start + residues]
    lines = [_pdb_line("HETATM", n + 1, cap, "ACE", "A", before.number, model.atom_by_name(before, name).xyz, name[0])
             for n, (cap, name) in enumerate((("C", "C"), ("O", "O"), ("CH3", "CA")))]
    for key in keys[start:start + residues]:
        lines += [_pdb_line("ATOM", len(lines) + 1, a.name, a.resname, "A", key.number, a.xyz, a.element)
                  for a in model.residue_atoms(key) if a.element not in ("H", "D")]
    lines.append(_pdb_line("HETATM", len(lines) + 1, "N", "NH2", "A", after.number, model.atom_by_name(after, "N").xyz, "N"))
    return "\n".join(lines) + "\nEND\n"


def test_the_amide_at_the_end_of_a_chain_is_a_cap_and_no_ion(tmp_path):
    """NH2 is one heavy atom, as an ion is, and it is bonded to the chain, as a cap is. It is the cap."""
    (tmp_path / "capped.pdb").write_text(_capped_stretch())
    found = survey(tmp_path)
    assert [(c.name, c.kind, c.residues) for c in found.found.chains] == [("A", "protein", 61)]
    assert {(c.name, c.kind, c.heavy, c.bonded) for c in found.found.candidates} == {("ACE", "cap", 3, True),
                                                                                    ("NH2", "cap", 1, True)}
    assert not found.tracks
    assert [(what.split(" ")[0], why.split(";")[0]) for what, why in found.left] == [("ACE", "the end of a chain"),
                                                                                    ("NH2", "the end of a chain")]
    # One atom that is bonded to nothing is an ion, whatever it is called.
    alone = "\n".join(line for line in _capped_stretch().splitlines() if " NH2 " not in line)
    alone += "\n" + _pdb_line("HETATM", 9000, "N", "NH2", "A", 900, (99.0, 99.0, 99.0), "N") + "\nEND\n"
    (kind,) = [c.kind for c in report.look(parse_pdb(alone)).candidates if c.name == "NH2"]
    assert kind == "ion"


@fetched("ensemble/dhfr_mtx/structure/1ao8.cif.gz")
def test_the_models_of_a_file_are_an_ensemble_of_its_ligand(monkeypatch):
    monkeypatch.setattr(report, "plip_here", lambda: False)
    found = survey(BUNDLE)
    assert found.models_in_file == 21
    assert [(c.name, c.kind, c.residues) for c in found.found.chains] == [("A", "protein", 162)]
    assert [(c.name, c.kind, c.heavy) for c in found.found.candidates] == [("MTX", "small molecule", 33)]
    ligand, ensemble = found.tracks
    assert (ligand.kind, ensemble.kind) == ("ligand", "ensemble")
    assert ensemble.title == "MTX over 21 models" and "the table of heavy atoms, model by model" in ensemble.why
    assert "tables = [ligand_contacts(m, " in ensemble.code
    found.draw()                                                    # 21 tables and two figures, without PLIP
    assert ensemble.status in ("ok", "degraded") and "occupancy" in ensemble.svg
    monkeypatch.setattr(report, "plip_here", lambda: True)          # with PLIP it is PLIP's; nothing is run here
    assert ["PLIP finds them" in t.why for t in survey(BUNDLE).tracks] == [True, True]


def test_a_peptide_is_the_ligand_and_two_long_chains_are_an_interface():
    found = survey(TESTDATA / "mhc_peptide")
    assert [(c.name, c.peptide) for c in found.found.chains] == [("A", False), ("B", False), ("P", True)]
    peptide, interface = found.tracks
    assert peptide.title == "The peptide of chain P" and '"chain:P"' in peptide.code
    assert "from bindview.distance_table import contact_table" in peptide.code
    assert interface.title == "The interface of chains A and B"
    # The light chain of thrombin, 28 residues, is bonded to the heavy chain: a part of it, and no peptide.
    thrombin = survey(TESTDATA / "thrombin_hirudin")
    assert [(c.name, c.peptide, c.joined) for c in thrombin.found.chains] == [("A", False, "B"), ("B", False, ""),
                                                                              ("H", True, "")]
    assert [t.title for t in thrombin.tracks] == ["The peptide of chain H"]
    assert [what.split(" ")[0] for what, _ in thrombin.left[:2]] == ["NA", "CA"]  # ions beside a ligand: listed
    what, why = thrombin.left[2]
    assert what == "chain A (protein, 28 residues)" and "bonded to chain B: a part of that protein" in why


def test_without_a_table_and_without_plip_a_small_molecule_is_drawn_from_its_heavy_atoms(tmp_path, monkeypatch):
    for name in ("model.cif.gz", "meta.json"):
        shutil.copy(TESTDATA / "kit_imatinib" / name, tmp_path / name)
    monkeypatch.setattr(report, "plip_here", lambda: False)
    found = survey(tmp_path)
    (track,) = found.tracks
    assert "the table of heavy atoms" in track.why and "no water bridge" in track.why
    assert "from bindview.distance_table import ligand_contacts" in track.code and "detect" not in track.code
    found.draw()
    assert track.status == "ok" and 'class="interaction pi_stacking"' in track.svg
    monkeypatch.setattr(report, "plip_here", lambda: True)          # where PLIP is installed it is asked first
    (track,) = survey(tmp_path).tracks
    assert "PLIP finds them" in track.why and "from bindview.detect import detect_interactions" in track.code


def test_a_model_is_drawn_from_the_table_that_is_named_after_it(tmp_path):
    """Two models in one folder, each with a table named after it, and one compound in both: every atom name of
    the first table is an atom of the second model's ligand too, and each model is drawn from its own."""
    rows = (OPEN / "interactions.tsv").read_text().splitlines()
    for stem, kept in (("p1", rows), ("p2", rows[:3])):
        shutil.copy(OPEN / "model.cif.gz", tmp_path / f"{stem}.cif.gz")
        (tmp_path / f"{stem}.interactions.tsv").write_text("\n".join(kept) + "\n")
    for stem, other in (("p1", "p2"), ("p2", "p1")):
        found = survey(tmp_path / f"{stem}.cif.gz")
        (track,) = found.tracks
        assert f"{stem}.interactions.tsv" in track.code and f"{other}.interactions.tsv" not in track.code
        assert f"the table {stem}.interactions.tsv beside the model, which names its atoms" in track.why
        assert [path.name for path, *_ in found.beside.tables] == [f"{stem}.interactions.tsv"]
    # A table that is named after no model is anybody's whose atoms it names, as before.
    (tmp_path / "p2.interactions.tsv").rename(tmp_path / "contacts.tsv")
    assert "contacts.tsv" in survey(tmp_path / "p2.cif.gz").tracks[0].code
    assert "p1.interactions.tsv" in survey(tmp_path / "p1.cif.gz").tracks[0].code


def test_a_table_that_names_no_atom_of_a_ligand_does_not_come_before_one_that_does(tmp_path, monkeypatch):
    """The rule "every ligand atom it names is an atom of the candidate" is true of a table that names none. Such
    a table (a header alone, or two contacts of other residues) does not come before the one that names the
    ligand's atoms, whatever its name."""
    monkeypatch.setattr(report, "plip_here", lambda: False)
    for name in ("model.cif.gz", "interactions.tsv"):
        shutil.copy(OPEN / name, tmp_path / name)
    (tmp_path / "a_header.tsv").write_text(_NO_ATOMS)
    (tmp_path / "a_other.tsv").write_text(_NO_ATOMS + "hydrophobic\t5\tALA\tA\nhbond\t9\tSER\tA\n")
    (track,) = survey(tmp_path).tracks
    assert 'load_interactions("interactions.tsv")' in track.code
    assert "the table interactions.tsv beside the model, which names its atoms" in track.why
    # Without the table that names its atoms, neither of the others says whose it is: the model has two ligands.
    (tmp_path / "interactions.tsv").unlink()
    (track,) = survey(tmp_path).tracks
    assert "load_interactions" not in track.code and "no table beside the model" in track.why
    # Where the model has one ligand, the one table beside it is that ligand's, and the page says what it is.
    lone = tmp_path / "lone"
    lone.mkdir()
    (lone / "model.pdb").write_text(_NAMED_LIKE_A_BACKBONE)
    (lone / "contacts.tsv").write_text(_NO_ATOMS)
    (track,) = [t for t in survey(lone).tracks if t.kind == "ligand"]              # (its three residues are a peptide)
    assert "contacts.tsv" in track.code and "which names its atoms" not in track.why
    assert "the table contacts.tsv beside the model, which names no atom of a ligand" in track.why


def test_a_table_beside_the_model_that_cannot_be_read_is_named_and_the_report_is_made(tmp_path):
    """A spreadsheet's export in another encoding, and a table with a word where a residue number belongs, beside a
    model with its own table: each is named on the page with what is wrong in it, and the report is made."""
    work = tmp_path / "complex"
    shutil.copytree(OPEN, work)
    (work / "export.csv").write_bytes("interaction_type,residue_number\nhbond,199\n".encode("utf-16"))
    (work / "words.tsv").write_text(_NO_ATOMS + "hbond\tfive\tALA\tA\n")
    found = survey(work)
    (track,) = found.tracks
    assert 'load_interactions("interactions.tsv")' in track.code
    assert [path.name for path, *_ in found.beside.tables] == ["interactions.tsv"]
    (export, why), (words, said) = found.beside.not_tables
    assert (export.name, why) == ("export.csv", "it is not UTF-8 text (it has the byte 0xff); save it as UTF-8")
    assert words.name == "words.tsv" and said.startswith("line 2: ") and "five" in said and str(tmp_path) not in said
    found.draw(views=False)
    page = found.html()
    assert track.status == "ok" and track.svg in page
    assert ("not read as a table of contacts: export.csv (it is not UTF-8 text (it has the byte 0xff); save it as "
            "UTF-8); words.tsv (line 2: ") in page
    assert tmp_path.name not in page


def test_a_smiles_that_is_handed_in_goes_into_the_call(tmp_path):
    for name in ("model.cif.gz", "interactions.tsv"):
        shutil.copy(OPEN / name, tmp_path / name)
    smiles = meta_of("ca2_celecoxib")["ligand"]["smiles"]
    (plain,) = survey(tmp_path).tracks
    (given,) = survey(tmp_path, {"CEL": smiles}).tracks
    assert "handed in" in given.why and f'smiles="{smiles}"' in given.code and given.code != plain.code


# ---------------------------------------------------------------------------
# A set of complexes: a series of ligands
# ---------------------------------------------------------------------------

#: Three entries of the glucocorticoid receptor with one small molecule each (the others hold a detergent too).
SERIES = ("gr_azd5423", "gr_azd9567", "gr_mometasone")


def test_several_complexes_of_one_protein_with_a_ligand_each_are_a_series(tmp_path):
    """Three crystal structures of one receptor, each with its ligand and its table, are a series."""
    for slug in SERIES:
        shutil.copytree(TESTDATA / slug, tmp_path / slug)
    found = survey(tmp_path)
    assert found.kind == "series" and [m.name for m in found.members] == list(SERIES)
    assert all(m.table is not None and m.table.name == "interactions.tsv" for m in found.members)
    assert [m.ligand.name for m in found.members] == [meta_of(slug)["ligand"]["component"] for slug in SERIES]
    assert len({m.ligand.formula for m in found.members}) == 3
    (track,) = found.tracks
    assert (track.kind, track.title) == ("series", "3 ligands of one pocket")
    assert track.code.count("Diagram.from_structure(") == 3 and track.code.endswith("figure = Comparison(diagrams).render()")
    found.draw(views=False)
    assert track.status == "ok"
    page = found.html()
    assert "3 complexes of one protein" in page and f"<td>{SERIES[0]}</td>" in page and track.svg in page


def test_two_ligands_are_compared_for_what_they_do_differently_and_a_member_needs_no_table(tmp_path):
    """The two ligands of the androgen receptor, the table beside one of them only."""
    for slug in ("ar_r1881", "ar_bicalutamide"):
        (tmp_path / slug).mkdir()
        for name in ("model.cif.gz", "meta.json"):
            shutil.copy(TESTDATA / slug / name, tmp_path / slug / name)
    shutil.copy(TESTDATA / "ar_r1881" / "interactions.tsv", tmp_path / "ar_r1881" / "interactions.tsv")
    found = survey(tmp_path)
    assert found.kind == "series" and [m.name for m in found.members] == ["ar_bicalutamide", "ar_r1881"]
    assert [m.table is not None for m in found.members] == [False, True]
    (track,) = found.tracks
    assert "what the two do differently" in track.why and "the table of heavy atoms for the others" in track.why
    assert "def member(path, ligand, name, table=None, smiles=None):" in track.code and "ligand_contacts" in track.code
    found.draw(views=False)
    assert track.status == "ok"


def test_complexes_that_are_neither_one_complex_nor_one_series_are_refused(tmp_path):
    """Carbonic anhydrase with celecoxib and c-Kit with imatinib: two proteins."""
    for slug in ("ca2_celecoxib", "kit_imatinib"):
        (tmp_path / slug).mkdir()
        shutil.copy(model_of(slug), tmp_path / slug / "model.cif.gz")
    with pytest.raises(report.BindviewError, match="neither one complex nor a series of ligands on one protein"):
        survey(tmp_path)


def _as_pdb(structure, chain: str, new: str, start: int) -> "list[str]":
    """The heavy atoms of one chain as lines of a PDB file, under another chain name."""
    lines = []
    for a in structure.atoms:
        if a.chain != chain or a.element in ("H", "D") or a.hetero:
            continue
        lines.append(f"ATOM  {start + len(lines):5d} {a.name:<4s} {a.resname:>3s} {new}{a.resnum:4d}    "
                     f"{a.x:8.3f}{a.y:8.3f}{a.z:8.3f}  1.00  0.00          {a.element:>2s}")
    return lines


def test_a_short_chain_of_rna_beside_a_protein_is_drawn_as_the_ligand(tmp_path):
    from bindview import read_structure

    haemoglobin = read_structure(model_of("hb_carbon_monoxide"))
    protein = _as_pdb(haemoglobin, "A", "P", 1)
    rna = _as_pdb(read_structure(RIBOSWITCH / "model.pdb"), "B", "R", len(protein) + 1)
    (tmp_path / "made_up.pdb").write_text("\n".join(protein + rna) + "\nEND\n")
    found = survey(tmp_path)
    residues = len({a.resnum for a in haemoglobin.atoms if a.chain == "A" and not a.hetero})     # of the file
    assert [(c.name, c.kind, c.residues, c.oligo) for c in found.found.chains] == [("P", "protein", residues, False),
                                                                                  ("R", "nucleic", 18, True)]
    (track,) = found.tracks
    assert (track.kind, track.title) == ("oligonucleotide", "The oligonucleotide of chain R")
    assert '"chain:R"' in track.code and "from bindview.distance_table import contact_table" in track.code
    assert 'contact_table(model, "R", partners=["P"], stacking=True)' in track.code


# ---------------------------------------------------------------------------
# The page, and the call it prints
# ---------------------------------------------------------------------------

def test_the_call_a_report_prints_draws_the_figure_it_shows():
    found = survey(RIBOSWITCH)
    found.draw()
    track, interface, pairs = found.tracks
    assert track.status == "ok" and track.svg.startswith("<svg") and interface.status == "ok"
    assert interface.svg.count('class="interaction pi_stacking"') == 2
    for shown in (track, pairs):                         # (a call is written for the folder of the input)
        alone = subprocess.run([sys.executable, "-c", shown.code + "\nimport sys\nsys.stdout.write(figure.svg)"],
                               cwd=RIBOSWITCH, capture_output=True, text=True, check=True)
        assert alone.stdout == shown.svg
    page = found.html()
    assert track.svg in page and interface.svg in page and "from bindview import Diagram, load_interactions" in page
    assert pairs.svg in page and "from bindview import BasePairs" in page
    assert "<h2>The figures (3)</h2>" in page and "<td>an oligonucleotide</td>" in page


STAND_IN = ("class Figure:\n"
            "    def __init__(self, style):\n"
            "        self.svg, self.status = f'<svg>{style}</svg>', 'unreadable' if style == 'detailed' else 'ok'\n"
            "        self.warnings = (['a chain diagram is not readable: 4 residue(s) could not be placed; '\n"
            "                          'a per-residue summary was drawn instead'] if style == 'detailed' else ['as said'])\n\n\n"
            "class Dia:\n"
            "    def render(self, style, **options):\n"
            "        return Figure(style)\n\n\n"
            "dia = Dia()\n")


def test_a_figure_not_readable_in_the_detailed_style_is_drawn_in_the_schematic_one(tmp_path, monkeypatch, capsys):
    """With a stand-in for the library whose detailed figure is not readable: the call runs again in the
    schematic style, and the page prints that call and says why."""
    found = survey(OPEN)
    (track,) = found.tracks
    track.code = STAND_IN + 'figure = dia.render(style="detailed")'
    monkeypatch.setattr(report, "survey", lambda *args: found)
    out = tmp_path / "report.html"
    assert main(["report", str(OPEN), "-o", str(out)]) == 0
    assert (track.svg, track.status, track.schematic) == ("<svg>schematic</svg>", "ok", True)
    assert track.code == STAND_IN + 'figure = dia.render(style="schematic")'
    assert track.said == ["drawn in the schematic style: in the detailed style, a chain diagram is not readable: "
                          "4 residue(s) could not be placed", "as said"]
    assert "CEL in its pocket: ok, in the schematic style: the detailed one is not readable" in capsys.readouterr().err
    page = out.read_text(encoding="utf-8")
    assert "<svg>schematic</svg>" in page and "<svg>detailed</svg>" not in page and track.said[0] in page


@pytest.mark.parametrize("schematic, shown", (("degraded", True), ("empty", False)))
def test_a_schematic_figure_that_is_drawn_with_a_remark_is_shown_as_well(schematic, shown):
    """A schematic figure that comes back degraded (a ligand without a SMILES, say) can be read: it is shown in
    place of the detailed one. One that comes back empty is not, and the detailed figure stays."""
    stand_in = STAND_IN.replace("'unreadable' if style == 'detailed' else 'ok'",
                                f"'unreadable' if style == 'detailed' else '{schematic}'")
    assert stand_in != STAND_IN
    found = survey(OPEN)
    (track,) = found.tracks
    track.code = stand_in + 'figure = dia.render(style="detailed")'
    found.draw(views=False)
    if shown:
        assert (track.svg, track.status, track.schematic) == ("<svg>schematic</svg>", "degraded", True)
        assert track.code == stand_in + 'figure = dia.render(style="schematic")'
        assert track.said == ["drawn in the schematic style: in the detailed style, a chain diagram is not readable: "
                              "4 residue(s) could not be placed", "as said"]
    else:
        assert (track.svg, track.status, track.schematic) == ("<svg>detailed</svg>", "unreadable", False)
        assert track.said[-1] == "the schematic style was tried and came back empty"


def test_where_the_schematic_style_is_not_readable_either_the_detailed_figure_stays():
    """Celecoxib with its five contacts, drawn 50 mm wide: too small for the letters in either style."""
    found = survey(OPEN)
    (track,) = found.tracks
    track.code = track.code.replace('render(style="detailed")', 'render(style="detailed", figure_width_mm=50.0)')
    asked = track.code
    found.draw()
    assert track.status == "unreadable" and not track.schematic and track.code == asked
    assert "Residue-level summary" in track.svg
    assert any(s.startswith("an atom-level diagram is not readable") and s.endswith("a per-residue summary was drawn instead")
               for s in track.said)
    assert track.said[-1] == "the schematic style was tried and is not readable either"


def test_the_page_is_the_same_wherever_the_report_is_run_and_names_nothing_above_the_input(tmp_path, monkeypatch):
    """The calls of a page are written for the folder of the input and run there: wherever the report is run, the
    page is the same, and no folder above the input is named in it."""
    import os
    from pathlib import Path

    work = tmp_path / "somewhere" / "complex"
    shutil.copytree(OPEN, work)
    pages = []
    for cwd, given in ((tmp_path, Path("somewhere") / "complex"), (work, work), (ROOT, work / "model.cif.gz")):
        monkeypatch.chdir(cwd)
        found = survey(given)
        found.draw(views=False)
        assert Path(os.getcwd()) == Path(cwd).resolve()                        # drawn there, and back
        (track,) = found.tracks
        assert track.status == "ok" and "/" not in track.code
        assert 'Diagram.from_structure("model.cif.gz", ligand="A:CEL:701"' in track.code
        assert 'interactions=load_interactions("interactions.tsv"))' in track.code
        page = found.html()
        assert "somewhere" not in page and tmp_path.name not in page and "run it there" in page
        pages.append(page.replace("What model.cif.gz holds", "What complex holds").replace(
            "bindview report: model.cif.gz", "bindview report: complex"))
    assert pages[0] == pages[1] == pages[2]
    # The members of a set are named from the folder that holds them.
    group = tmp_path / "set"
    for name in ("first", "second"):
        shutil.copytree(OPEN, group / name)
    monkeypatch.chdir(ROOT)
    both = survey(group)
    assert both.kind == "ensemble" and [t.kind for t in both.tracks] == ["ligand", "ensemble"]
    assert 'models = ["first/model.cif.gz", "second/model.cif.gz"]' in both.tracks[1].code
    assert 'tables = ["first/interactions.tsv", "second/interactions.tsv"]' in both.tracks[1].code
    both.draw(views=False)
    assert [t.status for t in both.tracks] == ["ok", "ok"] and tmp_path.name not in both.html()


def test_the_same_input_gives_the_same_page():
    pages = []
    for _ in range(2):
        found = survey(OPEN)
        found.draw()
        pages.append(found.html())
    assert pages[0] == pages[1] and "ok</span>" in pages[0]


@pytest.mark.parametrize("words", ['A"); import os; os.system("x', "back\\slash", "new\nline", "µ and 'quotes'", "/C=C\\C"])
def test_whatever_a_file_calls_a_thing_stays_inside_the_quotes(words):
    assert ast.literal_eval(report._q(words)) == words


# ---------------------------------------------------------------------------
# The command
# ---------------------------------------------------------------------------

def test_the_command_writes_one_page(tmp_path, capsys):
    out = tmp_path / "report.html"
    assert main(["report", str(OPEN), "-o", str(out)]) == 0
    said = capsys.readouterr().err
    assert "ligand: CEL in its pocket: ok" in said and f"report -> {out}" in said
    page = out.read_text(encoding="utf-8")
    assert page.startswith("<!doctype html>") and page.count("<article>") == 1 and "<svg" in page
    again = survey(OPEN)
    again.draw()
    assert page == again.html()


def test_the_page_lies_beside_the_input_unless_it_is_told_where(tmp_path):
    work = tmp_path / "complex"
    shutil.copytree(OPEN, work)
    assert main(["report", str(work), "-q"]) == 0
    assert (tmp_path / "complex.report.html").exists()
    assert main(["report", str(work / "model.cif.gz"), "-q"]) == 0
    assert (work / "model.report.html").exists()


def test_looking_draws_nothing_and_writes_nothing(tmp_path, capsys):
    work = tmp_path / "complex"
    shutil.copytree(OPEN, work)
    assert main(["report", str(work), "--look"]) == 0
    said = capsys.readouterr().err
    with gzip.open(OPEN / "model.cif.gz", "rt") as handle:                          # the residues of chain A, in the file
        residues = len({line.split()[16] for line in handle if line.startswith("ATOM ") and line.split()[18] == "A"})
    assert f"chain A: protein of {residues}" in said and "CEL x1 (small molecule, 26 heavy atoms)" in said
    assert "ligand: CEL in its pocket: to draw" in said
    assert not list(tmp_path.glob("*.html")) and not list(work.glob("*.html"))


def test_what_is_no_input_is_an_error_with_a_reason(tmp_path, capsys):
    assert main(["report", str(tmp_path / "nothing")]) == 2
    assert "no such file or folder" in capsys.readouterr().err
    assert main(["report", str(tmp_path)]) == 2
    assert "no model here" in capsys.readouterr().err
    assert main(["report", str(OPEN), "--smiles", "CEL"]) == 2
    assert "--smiles takes NAME=SMILES" in capsys.readouterr().err
    two = tmp_path / "two"
    two.mkdir()
    shutil.copy(OPEN / "model.cif.gz", two / "a.cif.gz")
    shutil.copy(RIBOSWITCH / "model.pdb", two / "b.pdb")
    assert main(["report", str(two)]) == 2
    assert ("2 models that are neither one complex nor a series of ligands on one protein (a.cif.gz, b.pdb): name one "
            "of them") in capsys.readouterr().err
    assert main(["report", str(OPEN), "--domain", "N=A:1-100"]) == 2
    assert "one --domain names one part" in capsys.readouterr().err
    assert main(["report", str(OPEN), "--domain", "N=A", "--domain", "C=A:5-9"]) == 2
    assert "--domain takes NAME=CHAIN:FROM-TO" in capsys.readouterr().err


def test_a_figure_has_its_views_under_it(monkeypatch, tmp_path):
    """Under a figure, the same object drawn again by a last line of its own: in the other style, with
    pLDDT in AlphaFold's colours where the file declares a pLDDT, and answering the pointer. The first figure
    is what it was, and views=False draws it alone. A crystal structure declares no pLDDT."""
    found = survey(TESTDATA / "kit_imatinib")
    assert not found.found.plddt
    found.draw()
    (track,) = found.tracks
    assert [v.line for v in track.views] == ['figure = dia.render(style="schematic")',
                                             'figure = dia.render(style="detailed", interactive=True)']
    assert all(v.status == "ok" and v.svg for v in track.views)
    assert "lf-live" in track.views[1].svg and "lf-live" not in track.svg
    alone = survey(TESTDATA / "kit_imatinib")
    alone.draw(views=False)
    (first,) = alone.tracks
    assert first.svg == track.svg and first.views == []
    page = found.html()
    assert "<h4>Views (2)</h4>" in page and page.count("<svg") == 3 and alone.html().count("<svg") == 1
    # A view is what its printed line draws where the printed call ran: in the folder of the input.
    names: dict = {}
    monkeypatch.chdir(TESTDATA / "kit_imatinib")
    exec(track.code, names)
    exec(track.views[0].line, names)
    assert names["figure"].svg == track.views[0].svg
    # A model of the AlphaFold database declares its pLDDT: a view in its colours, between the other two.
    tectb = survey(with_its_chain(TECTB, tmp_path / "tectb"))
    assert tectb.found.plddt
    tectb.draw()
    assert [v.line for t in tectb.tracks for v in t.views] == [
        'figure = face.render(style="schematic")', 'figure = face.render(style="detailed", plddt_colors=True)',
        'figure = face.render(style="detailed", interactive=True)']


def test_each_input_of_a_folder_gets_a_page_and_a_row(tmp_path, capsys):
    """--each draws every input of a folder as an input of its own, writes a page for each, and an index
    as a page and a CSV: one row an input, with its contacts by kind and what the library refused in its words."""
    screen = tmp_path / "screen"
    for name in ("kit_imatinib", "ca2_celecoxib"):
        shutil.copytree(TESTDATA / name, screen / name)
    # Without --each such a folder is refused, and the refusal names it.
    assert main(["report", str(screen), "-q"]) == 2
    assert "ask for a page of each (--each)" in capsys.readouterr().err
    (screen / "broken").mkdir()
    (screen / "broken" / "model.cif").write_text("data_x\n_atom_site.id 1\n")
    index = tmp_path / "out" / "screen.html"
    assert main(["report", str(screen), "--each", "--one-view", "-o", str(index), "-q"]) == 0
    rows = (tmp_path / "out" / "screen.csv").read_text().splitlines()
    head = rows[0].split(",")
    table = {r.split(",")[0]: dict(zip(head, r.split(","))) for r in rows[1:]}
    assert list(table) == ["broken", "ca2_celecoxib", "kit_imatinib"]
    kinds = ("hbond", "saltbridge", "pi_stacking", "pi_cation", "halogen", "metal", "water_bridge", "hydrophobic")
    for name in ("ca2_celecoxib", "kit_imatinib"):
        row, counted = table[name], kinds_in_table(TESTDATA / name)
        assert row["status"] == "ok" and {k: row[k] for k in kinds} == {k: str(counted[k]) for k in kinds}
        assert row["pocket_plddt"] == ""                                       # a crystal structure has none
    broken = table["broken"]
    assert broken["note"].startswith("could not be read: broken/model.cif: ") and str(tmp_path) not in rows[1]
    page = index.read_text()
    for name in ("ca2_celecoxib", "kit_imatinib"):
        member = tmp_path / "out" / "screen" / f"{name}.html"
        component = meta_of(name)["ligand"]["component"]
        assert f"<a href='screen/{name}.html'>{name}</a>" in page and f"{component} in its pocket" in member.read_text()
    assert not (tmp_path / "out" / "screen" / "broken.html").exists() and "broken/model.cif" in page
    # Each page is the report of that input alone, as without --each.
    alone = survey(screen / "ca2_celecoxib" / "model.cif.gz")
    alone.draw(views=False)
    assert (tmp_path / "out" / "screen" / "ca2_celecoxib.html").read_text() == alone.html()


def test_each_member_of_a_folder_has_a_page_of_its_own_whatever_the_members_are_called(tmp_path, monkeypatch):
    """A member is named after its file, and the second of a name gets a number that no other member has:
    ``a.pdb.gz`` beside ``a.pdb`` and ``a-2.pdb`` is ``a-3``, and each has a page of its own."""
    import gzip

    monkeypatch.setattr(report, "plip_here", lambda: False)
    screen = tmp_path / "screen"
    screen.mkdir()
    (screen / "a-2.pdb").write_text(_NAMED_LIKE_A_BACKBONE)
    (screen / "a.pdb").write_text(_NAMED_LIKE_A_BACKBONE.replace("MTX", "LIG"))
    with gzip.open(screen / "a.pdb.gz", "wt") as packed:
        packed.write(_NAMED_LIKE_A_BACKBONE.replace("MTX", "OTH"))
    rows, index = report.write_each(screen, tmp_path / "out" / "index.html", views=False)
    assert [(row.name, row.model.name) for row in rows] == [("a-2", "a-2.pdb"), ("a", "a.pdb"), ("a-3", "a.pdb.gz")]
    assert sorted(p.name for p in (tmp_path / "out" / "index").iterdir()) == ["a-2.html", "a-3.html", "a.html"]
    for row, ligand in zip(rows, ("MTX", "LIG", "OTH")):
        assert f"{ligand} in its pocket" in (index.parent / row.page).read_text()


def test_the_links_of_an_index_lead_to_its_pages_whatever_the_members_are_called(tmp_path, monkeypatch):
    """A link is an address, and ``#``, ``?`` and ``%`` mean something in one: each link of the index leads to
    the page of its member, whatever the member is called."""
    import html
    import re
    from urllib.parse import unquote, urlsplit

    monkeypatch.setattr(report, "plip_here", lambda: False)
    screen = tmp_path / "screen"
    screen.mkdir()
    names = ("a#1", "b?x", "c%20d", "e f", "g'h")
    for name in names:
        (screen / f"{name}.pdb").write_text(_NAMED_LIKE_A_BACKBONE)
    rows, index = report.write_each(screen, tmp_path / "out" / "index.html", views=False)
    assert [row.name for row in rows] == list(names)
    links = re.findall(r"<a href='([^']*)'>([^<]*)</a>", index.read_text())
    assert [html.unescape(text) for _, text in links] == list(names)
    for (href, _), name in zip(links, names):
        asked = urlsplit(html.unescape(href))            # what a browser makes of the link
        assert (asked.query, asked.fragment) == ("", "")
        assert unquote(asked.path) == f"index/{name}.html" and (index.parent / unquote(asked.path)).is_file()
    # The CSV has the page as a path: it is a table, and no address.
    assert [line.split(",")[1] for line in index.with_suffix(".csv").read_text().splitlines()[1:3]] == [
        "index/a#1.html", "index/b?x.html"]


def test_a_member_that_cannot_be_read_is_a_row_that_says_so_and_the_index_is_written(tmp_path, monkeypatch):
    """One member of three is a torn file, one is no text at all (the reader refuses it as a model it cannot
    read): each is a row with the reader's words, the third has its page, and the index is written."""
    from pathlib import Path

    from bindview.errors import StructureError

    monkeypatch.setattr(report, "plip_here", lambda: False)
    screen = tmp_path / "screen"
    screen.mkdir()
    (screen / "good.pdb").write_text(_NAMED_LIKE_A_BACKBONE)
    (screen / "torn.cif").write_text("data_x\n_atom_site.id 1\n")
    (screen / "bytes.cif").write_bytes(bytes(range(256)) * 8)
    read = report.read_structure

    def reader(path, *args):
        if Path(path).name == "bytes.cif":
            raise StructureError(f"{path}: not a text file")
        return read(path, *args)

    monkeypatch.setattr(report, "read_structure", reader)
    rows, index = report.write_each(screen, tmp_path / "out" / "index.html", views=False)
    assert [(row.name, bool(row.page)) for row in rows] == [("bytes", False), ("good", True), ("torn", False)]
    assert rows[0].note == "could not be read: bytes.cif: not a text file"
    assert rows[2].note.startswith("could not be read: torn.cif: ") and not rows[1].note
    assert (index.parent / rows[1].page).is_file()
    page = index.read_text()
    assert page.count("<span class='badge refused'>could not be read</span>") == 2
    assert "bytes.cif: not a text file" in page and tmp_path.name not in page
    notes = [line.rsplit(",", 1)[1] for line in index.with_suffix(".csv").read_text().splitlines()[1:]]
    assert notes[0] == rows[0].note and notes[1] == "" and notes[2] == rows[2].note


def test_a_subfolder_with_several_models_is_no_member_of_a_set_and_is_named(tmp_path, monkeypatch):
    """The members of a set are the subfolders that hold one model. A subfolder with two is no member, and the
    page and the index say so."""
    monkeypatch.setattr(report, "plip_here", lambda: False)
    group = tmp_path / "set"
    for folder, names in (("one", ("model.pdb",)), ("two", ("a.pdb", "b.pdb"))):
        (group / folder).mkdir(parents=True)
        for name in names:
            (group / folder / name).write_text(_NAMED_LIKE_A_BACKBONE)
    found = survey(group)
    assert [model.parent.name for model in found.models] == ["one"] and found.kind == "complex"
    passed = ("folder two (2 models: a.pdb, b.pdb)",
              "a member of a set is a folder with one model: name this folder by itself to have it read")
    assert passed in found.left and passed[0] in found.html() and passed[1] in found.html()
    rows, index = report.write_each(group, tmp_path / "out" / "index.html", views=False)
    assert [(row.name, bool(row.page)) for row in rows] == [("one", True), ("two", False)]
    assert rows[1].note == ("not read: 2 models in one folder (a.pdb, b.pdb), and a member of a set is a folder with "
                            "one model: name this folder by itself to have it read")
    assert "<span class='badge refused'>not read</span>" in index.read_text()
    assert index.with_suffix(".csv").read_text().splitlines()[2].startswith("two,,")
    # Where no subfolder is a member, the refusal names those that are passed over.
    (group / "one" / "model.pdb").unlink()
    with pytest.raises(report.BindviewError, match="no model here") as refused:
        survey(group)
    assert "folder two (2 models: a.pdb, b.pdb)" in str(refused.value)


def test_of_two_copies_of_a_ligand_the_table_s_is_found_where_it_names_no_chain(tmp_path):
    """Of several copies of a ligand, the table is about the one near the first residue it names. ``NA`` in
    the column of the chain is no chain here, and the residue is found by its number alone."""
    import dataclasses

    import numpy as np

    from bindview.interactions import parse_interactions
    from bindview.report import Beside, _table_of, look
    from bindview.structure import Structure, read_structure

    st = read_structure(model_of("gr_dexamethasone"))             # two receptors, A and D, each with its DEX
    first = next(c for c in look(st).candidates if c.name == "DEX")

    def apart(cand, chain, number=564):
        at = np.array([a.xyz for a in st.residue_atoms(st.find_residue(chain, number))], dtype=float)
        return float(np.linalg.norm(cand.xyz[:, None, :] - at[None, :, :], axis=2).min())

    far = max("AD", key=lambda chain: apart(first, chain))         # the receptor of the other copy
    assert apart(first, far) > 6.0
    # That receptor numbered on, so that a number names one residue of the model.
    moved = Structure(atoms=[dataclasses.replace(a, resnum=a.resnum + 1000) if a.chain == far and not a.hetero else a
                             for a in st.atoms], source=st.source, chem_comp_smiles=st.chem_comp_smiles)
    found = look(moved)
    cand = next(c for c in found.candidates if c.name == "DEX")
    rows = parse_interactions("interaction_type\tresidue_number\tresidue_name\tchain\tprotein_atom\tligand_atom\n"
                              "hbond\t1564\tASN\tNA\tND2\tO5\n")
    table = tmp_path / "interactions.tsv"
    about, path = _table_of(cand, found, moved, Beside(tables=[(table, rows, {"O5"}, False)]))
    assert path == table and about is not cand and about in found.copies["DEX"]
    assert float(np.abs(about.xyz - cand.xyz).max()) > 6.0


@pytest.mark.parametrize("name", ["SOL", "TIP3", "D2O", "hoh"])
def test_a_water_is_a_water_under_any_of_its_names(name):
    """One set of names for every module that asks (``exposure.WATER_NAMES``): the water of a simulation (SOL,
    TIP3) is a water to the report as to the rest, and no residue of its chain, no candidate for the ligand."""
    import dataclasses

    from bindview.report import look
    from bindview.structure import Structure, read_structure

    st = read_structure(model_of("ca2_celecoxib"))
    renamed = Structure(atoms=[dataclasses.replace(a, resname=name) if a.resname == "HOH" else a for a in st.atoms],
                        source=st.source, chem_comp_smiles=st.chem_comp_smiles)
    seen, same = look(st), look(renamed)
    assert seen.waters > 0 and same.waters == seen.waters
    assert [(c.name, c.copies) for c in same.candidates] == [(c.name, c.copies) for c in seen.candidates]
    assert renamed.polymer_residues("A") == st.polymer_residues("A")
    assert renamed._nucleotides("A") == [] and renamed._by_label("A") == st._by_label("A")

