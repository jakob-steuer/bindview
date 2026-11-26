"""Ring stacking in the plain chain table, asked for with ``stacking=True``.

A stack is found by the criterion the table of a small molecule uses (PLIP's published defaults): one row for
a pair of residues, and no hydrophobic contact between two rings that stack. Without ``stacking=True`` the
table has no stacking rows. The two hairpins of the riboswitch stack where their loops meet, and two tyrosines
of the TNF trimer stack edge on; U1A on its RNA hairpin is in ``test_nucleic_deposited.py``.
"""

from __future__ import annotations

from collections import Counter

from conftest import TESTDATA, trimer_structure
from bindview import Diagram, Interface, load_interactions, read_structure
from bindview.distance_table import as_tsv, contact_table, interface_contacts

RIBOSWITCH = TESTDATA / "nucleic" / "riboswitch_bisguanidine" / "model.pdb"


def stacks(rows) -> "dict[tuple[str, int, str], dict]":
    """The stacking rows of a table by (partner chain, partner residue, residue of the chain)."""
    return {(str(r["chain"]), int(str(r["residue_number"])), str(r["ligand_atoms"]).split("/")[0]): r
            for r in rows if r["interaction_type"] == "pi_stacking"}


def test_without_it_there_is_no_stacking():
    model = read_structure(RIBOSWITCH)
    assert Counter(str(r["interaction_type"]) for r in contact_table(model, "B")) == {"hbond": 16}
    assert Counter(c.type.value for c in interface_contacts(model, ["A"], ["B"])) == {"hbond": 16}
    assert all(c.angle is None for c in interface_contacts(model, ["A"], ["B"]))


def test_the_loops_of_the_two_hairpins_stack_where_they_meet():
    model = read_structure(RIBOSWITCH)
    found = stacks(contact_table(model, "B", stacking=True))
    assert set(found) == {("A", 10, "10"), ("A", 11, "11")}                    # G10 on G10, A11 on A11
    g, a = found["A", 10, "10"], found["A", 11, "11"]
    assert (g["residue_name"], g["distance"], a["residue_name"], a["distance"]) == ("G", 3.73, "A", 3.36)
    assert float(str(g["angle"])) < 30.0 and float(str(a["angle"])) < 30.0     # face to face
    # Of each pair of purines the six-membered rings, the file's names on one side and bindview's on the other.
    assert g["protein_atoms"] == "C4,C5,C6,N1,C2,N3" and g["ligand_atoms"] == "10/C4,10/C5,10/C6,10/N1,10/C2,10/N3"
    assert set(stacks(contact_table(model, "A", stacking=True))) == {("B", 10, "10"), ("B", 11, "11")}
    contacts = interface_contacts(model, ["A"], ["B"], stacking=True)
    assert Counter(c.type.value for c in contacts) == {"hbond": 16, "pi_stacking": 2}
    assert all(c.angle is not None for c in contacts if c.type.value == "pi_stacking")


def test_a_stack_is_drawn_from_either_side():
    model = read_structure(RIBOSWITCH)
    face = Interface(model, ["A", "B"], interface_contacts(model, ["A"], ["B"], stacking=True))
    figure = face.render()
    assert figure.status == "ok" and figure.svg.count('class="interaction pi_stacking"') == 2
    hairpin = Diagram.from_structure(model, "chain:B", interactions=contact_table(model, "B", stacking=True))
    assert hairpin.n_interactions == 18 and not hairpin.unresolved
    figure = hairpin.render(style="schematic")
    assert figure.status == "ok" and figure.svg.count('class="interaction pi_stacking"') == 2


def test_two_rings_that_stack_have_no_hydrophobic_contact_between_them():
    """A tyrosine of one chain of the TNF trimer edge-on to the same tyrosine of the next: the stack takes
    the place of the hydrophobic contact between the two rings, and nothing else changes."""
    model = trimer_structure(0)
    plain, rich = contact_table(model, "A", partners=["B"]), contact_table(model, "A", partners=["B"], stacking=True)
    (stack,) = [r for r in rich if r["interaction_type"] == "pi_stacking"]
    assert (stack["residue_name"], stack["residue_number"], stack["chain"]) == ("TYR", 119, "B")
    assert str(stack["ligand_atoms"]).startswith("119/CG,") and stack["distance"] == 4.95
    assert 60.0 <= float(str(stack["angle"])) <= 90.0                         # edge on
    gone = [(r["interaction_type"], r["residue_number"], r["protein_atom"], r["ligand_atom"]) for r in plain if r not in rich]
    assert gone == [("hydrophobic", 119, "CE2", "119/CE2")]
    assert len(rich) == len(plain)


def test_a_table_written_to_a_file_keeps_its_stacks_and_their_angles(tmp_path):
    """``as_tsv`` writes every row of the table, a stack with the angle between its two rings, and a table
    read back from that file has them; without ``stacking=True`` the file has no stack."""
    model = read_structure(RIBOSWITCH)
    rich = contact_table(model, "B", stacking=True)
    (tmp_path / "rich.tsv").write_text(as_tsv(rich), encoding="utf-8")
    (tmp_path / "plain.tsv").write_text(as_tsv(contact_table(model, "B")), encoding="utf-8")
    back = load_interactions(tmp_path / "rich.tsv")
    assert Counter(r.type.value for r in back) == Counter(str(r["interaction_type"]) for r in rich) == {
        "hbond": 16, "pi_stacking": 2}
    written = sorted(float(str(r["angle"])) for r in rich if r["interaction_type"] == "pi_stacking")
    assert sorted(r.angle for r in back if r.type.value == "pi_stacking") == written
    assert Counter(r.type.value for r in load_interactions(tmp_path / "plain.tsv")) == {"hbond": 16}
