"""A protein on DNA and a protein on RNA, as deposited: a chain of nucleotides with a protein beside it.

The lambda repressor on its operator (``repressor_dna``, PDB 1LMB) and U1A on an RNA hairpin (``u1a_rna``,
PDB 1URN), as the archive gives them. Chain names are digits in the first; in the second a nucleotide is
missing from two of the three hairpins, and of three more only a few atoms are modelled. The contacts the
papers on these complexes describe are in the plain distance table.
"""

from __future__ import annotations

import subprocess
import sys
from collections import Counter

import pytest

from conftest import model_of
from bindview import Diagram, Interface, read_structure
from bindview.backends import Polymer
from bindview.distance_table import contact_table, interface_contacts
from bindview.report import look, survey

REPRESSOR = model_of("repressor_dna")
U1A = model_of("u1a_rna")


def bonds(contacts) -> "set[tuple[str, int, str, str, int, str]]":
    """The hydrogen bonds of an interface as (residue, number, atom, nucleotide, number, atom)."""
    return {(c.resname_a, c.number_a, c.atoms_a[0], c.resname_b, c.number_b, c.atoms_b[0])
            for c in contacts if c.type.value == "hbond"}


def test_two_strands_of_dna_and_two_chains_of_protein_under_names_that_are_digits():
    model = read_structure(REPRESSOR)
    assert [len(model.polymer_residues(c)) for c in "1234"] == [20, 20, 87, 92]
    assert [len(model._nucleotides(c)) for c in "1234"] == [20, 20, 0, 0]
    dia = Diagram.from_structure(model, "chain:1", interactions=contact_table(model, "1", partners=["3", "4"]))
    assert isinstance(dia.backend, Polymer)
    assert dia.backend.decision.reason == "it is a chain of 20 nucleotides in the structure file"
    assert dia.backend.peptide.sequence == "AATACCACTGGCGGTGATAT" and not dia.backend.peptide.breaks
    assert dia.ligand.bond_source == "template"                      # DA, DC, DG and DT are components it knows
    assert dia.n_interactions == 23 and not dia.unresolved
    figure = dia.render(style="schematic")
    assert figure.status == "ok" and figure.svg.count('class="unit"') == 20


def test_the_plain_table_has_what_is_known_of_the_repressor_on_its_operator():
    """The contacts the papers on this complex name: a glutamine on an adenine with two bonds, a lysine of
    the arm and an asparagine on one guanine, a serine on a guanine, a glutamine on a phosphate."""
    model = read_structure(REPRESSOR)
    first, second = bonds(interface_contacts(model, ["3"], ["1"])), bonds(interface_contacts(model, ["4"], ["1"]))
    assert {("GLN", 44, "NE2", "DA", 4, "N7"), ("GLN", 44, "OE1", "DA", 4, "N6"),
            ("GLN", 33, "NE2", "DA", 4, "OP2"), ("GLN", 33, "N", "DT", 3, "OP2")} <= first
    assert {("LYS", 4, "NZ", "DG", 14, "O6"), ("ASN", 55, "ND2", "DG", 14, "N7"), ("SER", 45, "OG", "DG", 16, "O6")} <= second
    # The same glutamine on the adenine of the other half of the operator, by the other repressor.
    assert {("GLN", 44, "NE2", "DA", 24, "N7"), ("GLN", 44, "OE1", "DA", 24, "N6")} <= bonds(interface_contacts(model, ["4"], ["2"]))
    # The two strands with each other: the hydrogen bonds of their base pairs, and nothing else.
    assert Counter(c.type.value for c in interface_contacts(model, ["1"], ["2"])) == {"hbond": 58}


def test_a_lysine_on_a_phosphate_is_a_salt_bridge_to_its_two_oxygens():
    model = read_structure(REPRESSOR)
    rows = contact_table(model, "1", partners=["3", "4"])
    assert Counter(str(r["interaction_type"]) for r in rows) == {"hbond": 20, "saltbridge": 3}
    salt = {(r["residue_name"], r["residue_number"], r["chain"]): r for r in rows if r["interaction_type"] == "saltbridge"}
    assert set(salt) == {("LYS", 19, "3"), ("LYS", 26, "3"), ("LYS", 5, "4")}
    assert salt["LYS", 19, "3"]["ligand_atoms"] == "4/OP1,4/OP2" and salt["LYS", 19, "3"]["protein_atoms"] == "NZ"
    assert all(str(r["ligand_atoms"]).count("/OP") == 2 for r in salt.values())          # never the phosphorus


def test_the_interface_between_a_protein_and_a_strand_is_drawn_from_either_side():
    model = read_structure(REPRESSOR)
    contacts = interface_contacts(model, ["3"], ["1"])
    assert Counter(c.type.value for c in contacts) == {"hbond": 8, "saltbridge": 2}
    for spine in ("3", "1"):
        face = Interface(model, ["3", "1"], contacts, spine=spine)
        for style in ("detailed", "schematic"):
            figure = face.render(style=style)
            assert figure.status == "ok" and not figure.warnings, (spine, style)
    assert 'data-class="nucleotide"' in Interface(model, ["3", "1"], contacts, spine="3").render().svg


def test_a_hairpin_with_a_nucleotide_missing_is_a_chain_with_a_break():
    model = read_structure(U1A)
    assert [len(model.polymer_residues(c)) for c in "PQRABC"] == [20, 21, 20, 96, 96, 96]
    # Of 13, 15 and 21 of chain P a phosphate or one oxygen is modelled: bound to the chain, and no nucleotides.
    assert [len(model._nucleotides(c)) for c in "PQR"] == [17, 21, 18]
    rows = contact_table(model, "P", partners=["A"])
    assert Counter(str(r["interaction_type"]) for r in rows) == {"hbond": 25, "saltbridge": 2, "hydrophobic": 2}
    dia = Diagram.from_structure(model, "chain:P", interactions=rows)
    chain = dia.backend.peptide
    assert dia.backend.decision.reason == "it is a chain of 17 nucleotides in the structure file"
    assert chain.sequence == "AAUCCAUUGCACGGAUU" and chain.breaks == (12,) and len(chain.appendages) == 3
    assert [r.number for r in chain.residues] == [str(n) for n in (*range(1, 13), *range(16, 21))]
    figure = dia.render(style="schematic")
    assert figure.status == "ok" and figure.svg.count('class="unit"') == 17
    # The copy that is modelled in full has the 21 of the sequence, joined from the first to the last.
    assert [k.number for k in model._nucleotides("Q")] == list(range(1, 22))


def test_with_stacking_the_two_rings_that_hold_the_loop_are_there():
    """What ``stacking=True`` was made for: the tyrosine under C10 and the phenylalanine under A11, in each of
    the three copies. Without it the tyrosine is a hydrophobic contact and the phenylalanine is not there."""
    model = read_structure(U1A)
    for rna, protein in (("P", "A"), ("Q", "B"), ("R", "C")):
        plain = contact_table(model, rna, partners=[protein])
        rich = contact_table(model, rna, partners=[protein], stacking=True)
        found = {(r["residue_name"], r["residue_number"], str(r["ligand_atoms"]).split("/")[0]): r
                 for r in rich if r["interaction_type"] == "pi_stacking"}
        assert set(found) == {("TYR", 13, "10"), ("PHE", 56, "11")}, rna
        assert all(float(str(r["distance"])) < 3.8 and float(str(r["angle"])) < 11.0 for r in found.values())
        assert [(r["interaction_type"], r["residue_name"], r["residue_number"]) for r in plain if r not in rich] == [
            ("hydrophobic", "TYR", 13)]
    # The repressor reads its operator with hydrogen bonds; the two strands stack within each strand only.
    operator = read_structure(REPRESSOR)
    assert not [c for c in interface_contacts(operator, ["3", "4"], ["1", "2"], stacking=True) if c.type.value == "pi_stacking"]
    assert not [c for c in interface_contacts(operator, ["1"], ["2"], stacking=True) if c.type.value == "pi_stacking"]


def test_the_report_takes_a_partly_modelled_nucleotide_for_a_residue_of_its_chain():
    found = look(read_structure(U1A))
    assert [(c.name, c.kind, c.residues) for c in found.chains] == [
        ("P", "nucleic", 20), ("Q", "nucleic", 21), ("R", "nucleic", 20),
        ("A", "protein", 96), ("B", "protein", 96), ("C", "protein", 96)]
    # Glycerol and chloride are what else the file holds: no "U" of four atoms, no "C" of one.
    assert [(c.name, c.kind, c.copies) for c in found.candidates] == [("GOL", "additive", 4), ("CL", "ion", 1)]
    result = survey(U1A)
    assert [t.kind for t in result.tracks] == ["oligonucleotide"] * 3 + ["interface"] * 3
    assert [t.title[-1] for t in result.tracks[:3]] == ["P", "R", "Q"]               # the shortest first


@pytest.mark.slow
def test_the_hairpin_whose_detailed_figure_is_not_readable_is_drawn_in_the_schematic_style():
    """The report draws a track in the schematic style where the detailed one is unreadable, and its printed
    call draws the same figure by itself. Around hairpin R lie 18 residues, the phenylalanine that stacks
    among them, and the detailed layout leaves 4 of them unplaced (a regression value of the layout). The
    detailed layout and the second interpreter take long, hence slow."""
    result = survey(U1A)
    result.tracks = [t for t in result.tracks if t.title == "The oligonucleotide of chain R"]
    result.draw()
    (track,) = result.tracks
    assert track.status == "ok" and track.schematic and track.code.endswith('figure = dia.render(style="schematic")')
    assert track.said[:2] == [
        "drawn in the schematic style: in the detailed style, a chain diagram is not readable: 18 ligand residues "
        "(376 atoms), 18 residues, 30 interactions; 4 residue(s) could not be placed, 0 pair(s) overlap and 0 "
        "interaction line(s) cross another residue", "read as a chain of 18 residues"]
    assert track.svg.count('class="unit"') == 18 and track.svg.count('class="interaction pi_stacking"') == 2
    # The call names the structure by its file name, as a page beside the file has it: run where the file is.
    alone = subprocess.run([sys.executable, "-c", track.code + "\nimport sys\nsys.stdout.write(figure.svg)"],
                           cwd=U1A.parent, capture_output=True, text=True, check=True)
    assert alone.stdout == track.svg


def test_the_report_draws_a_strand_with_the_protein_and_the_duplex_as_an_interface():
    result = survey(REPRESSOR)
    assert [(c.name, c.what) for c in result.found.chains] == [
        ("1", "an oligonucleotide"), ("2", "an oligonucleotide"), ("3", "protein"), ("4", "protein")]
    assert [t.title for t in result.tracks] == [
        "The oligonucleotide of chain 1", "The oligonucleotide of chain 2",
        "The interface of chains 1 and 2", "The base pairs of chains 1 and 2", "The interface of chains 3 and 4"]
    # The two strands again as their base pairs, from the PDB's own annotation in the entry: 19 pairs
    # between chains 1 and 2 in its _ndb_struct_na_base_pair.
    assert "the PDB's own annotation in model.cif.gz (19 pairs of these chains)" in result.tracks[3].why
    assert not result.left and not result.found.candidates
    strand = result.tracks[0]
    # The partner strand is no partner of the figure: its 58 bonds are the interface of the two strands.
    assert 'contact_table(model, "1", partners=["3", "4"], stacking=True)' in strand.code
    assert "what the chain pairs with in another nucleic chain is in their interface" in strand.why
