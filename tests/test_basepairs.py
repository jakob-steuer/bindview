"""Two chains of RNA and their base pairs as secondary structures facing each other (``bindview.basepairs``).

The riboswitch frame, with its stems from a dot-bracket and its pairs across from the plain table, has 2 pairs
across, 14 within and 2 stacks, and nothing drawn on anything; every pair given between the two chains is drawn
once, and one naming a nucleotide the chains do not have is left out and said; the three themes each give a
figure; the same input gives the same file. Two deposited entries carry the PDB's own annotation of their
pairs (``_ndb_struct_na_base_pair``): the kissing complex of HIV-1's dimerisation site (``rna_kissing``, 1XPF),
6 pairs across and 14 within, and the operator DNA of ``repressor_dna`` (1LMB), 19 pairs across and none
within.
"""

from __future__ import annotations

import re

import pytest

from conftest import RIBOSWITCH_HAIRPIN, TESTDATA, model_of, nucleic_basepairs, nucleic_basepairs_rnapolis
from bindview import (BasePair, BasePairs, BaseStack, BindviewError, Graph, pairs_from_contacts,
                      pairs_from_dot_bracket, pairs_within_from_table, read_fr3d_pairs, read_fr3d_stacks,
                      read_mmcif_pairs, read_rnapolis_pairs, read_rnapolis_stacks, read_structure)
from bindview.distance_table import interface_contacts

RIBOSWITCH = TESTDATA / "nucleic" / "riboswitch_bisguanidine" / "model.pdb"
RNAPOLIS = TESTDATA / "nucleic" / "riboswitch_bisguanidine" / "basepairs.rnapolis.json"
CONTACTS = TESTDATA / "nucleic" / "riboswitch_bisguanidine" / "interactions.tsv"
BGD = "NC(=[NH2+])NCCCCNC(N)=[NH2+]"


@pytest.fixture(scope="module")
def model():
    return read_structure(RIBOSWITCH)


@pytest.fixture(scope="module")
def figure():
    return nucleic_basepairs()


# -- reading -----------------------------------------------------------------------------------------------

FR3D = '''"1XPF|1|A|G|10","cWW","1XPF|1|B|C|15"
"1XPF|1|B|C|15","cWW","1XPF|1|A|G|10"
"3ZD5|1|A|A|31","tHW","3ZD5|1|B|U|13"
"3ZD5|1|B|U|13","tWH","3ZD5|1|A|A|31"
"1XPF|1|A|G|12","ncWW","1XPF|1|B|A|14"
"1XPF|2|A|G|1","cWW","1XPF|2|A|C|23"
"5NOM|1|A|C|9||||6_555","cWW","5NOM|1|A|G|10"
'''

FR3D_STACKS = '''"1XPF|1|A|G|10","s35","1XPF|1|B|A|16"
"1XPF|1|B|A|16","s53","1XPF|1|A|G|10"
"1XPF|1|A|G|12","ns55","1XPF|1|B|A|14"
'''


def test_fr3d_pairs_are_read_once_each_with_the_first_edge_first():
    pairs = read_fr3d_pairs(FR3D)
    got = {(p.chain_a, p.number_a, p.chain_b, p.number_b): p for p in pairs}
    assert len(pairs) == 3                            # each once; the near pair and the second model left out
    assert got[("A", 10, "B", 15)].family == "cWW"
    assert got[("A", 31, "B", 13)].family == "tHW"   # A31 uses its Hoogsteen edge, U13 its Watson-Crick edge
    assert got[("A", 9, "A", 10)].symmetry == "6_555"
    assert all(p.source == "fr3d" and p.ordered for p in pairs)


def test_fr3d_stacks_are_read_once_each():
    stacks = read_fr3d_stacks(FR3D_STACKS)
    assert [(s.chain_a, s.number_a, s.chain_b, s.number_b) for s in stacks] == [("A", 10, "B", 16)]


def test_a_stack_with_the_identity_written_out_is_a_stack_of_the_file():
    """A unit id may carry its symmetry operator. ``1_555`` is the identity: the unit as the file has it.
    Any other is a crystal mate, which no figure of the file's chains has."""
    written = ('"1XPF|1|A|G|10||||1_555","s35","1XPF|1|B|A|16||||1_555"\n'
               '"1XPF|1|A|G|11||||1_555","s35","1XPF|1|B|A|15"\n'
               '"1XPF|1|A|G|12","s35","1XPF|1|B|A|14||||6_555"\n')
    assert [(s.chain_a, s.number_a, s.chain_b, s.number_b) for s in read_fr3d_stacks(written)] == [
        ("A", 10, "B", 16), ("A", 11, "B", 15)]
    # The pairs of the same units, for comparison: read, the mate with its operator.
    pairs = read_fr3d_pairs(written.replace("s35", "cWW"))
    assert [(p.number_a, p.number_b, p.symmetry) for p in pairs] == [(10, 16, ""), (11, 15, ""), (12, 14, "6_555")]


MMCIF = '''data_TEST
loop_
_ndb_struct_na_base_pair.model_number
_ndb_struct_na_base_pair.i_label_asym_id
_ndb_struct_na_base_pair.i_label_comp_id
_ndb_struct_na_base_pair.i_label_seq_id
_ndb_struct_na_base_pair.i_symmetry
_ndb_struct_na_base_pair.j_label_asym_id
_ndb_struct_na_base_pair.j_label_comp_id
_ndb_struct_na_base_pair.j_label_seq_id
_ndb_struct_na_base_pair.j_symmetry
_ndb_struct_na_base_pair.pair_number
_ndb_struct_na_base_pair.i_auth_asym_id
_ndb_struct_na_base_pair.i_auth_seq_id
_ndb_struct_na_base_pair.i_PDB_ins_code
_ndb_struct_na_base_pair.j_auth_asym_id
_ndb_struct_na_base_pair.j_auth_seq_id
_ndb_struct_na_base_pair.j_PDB_ins_code
_ndb_struct_na_base_pair.hbond_type_28
_ndb_struct_na_base_pair.hbond_type_12
1 A G 1 1_555 B C 18 1_555 1 A 1 ? B 18 ? 19 1
1 A U 6 1_555 A A 28 1_555 2 A 6 ? A 28 ? 24 4
1 A G 9 1_555 B A 16 1_555 3 A 9 ? B 16 ? 8 ?
1 A C 9 1_555 A G 10 6_555 4 A 9 ? A 10 ? 19 1
'''


def test_the_pdbs_own_annotation_is_read_with_what_it_does_not_say():
    pairs = read_mmcif_pairs(MMCIF)
    assert [(p.chain_a, p.number_a, p.chain_b, p.number_b, p.family) for p in pairs] == [
        ("A", 1, "B", 18, "cWW"), ("A", 6, "A", 28, "tWH"), ("A", 9, "B", 16, None), ("A", 9, "A", 10, "cWW")]
    assert [p.ordered for p in pairs] == [True, False, True, True]      # tWH: which base uses which edge?
    assert [p.symmetry for p in pairs] == ["", "", "", "6_555"]
    assert read_mmcif_pairs("data_EMPTY\n_entry.id EMPTY\n") == []


def test_the_pdbs_annotation_is_read_for_the_first_model_and_by_label_where_there_is_no_author_s_name():
    """An entry of several models lists the pairs of each: those of the first are the pairs of the model
    that is read. A table without the authors' chain and number names its nucleotides by the label ids,
    as the atoms of such a file are named."""
    head = MMCIF.split("1 A G 1 1_555")[0]
    row = "{m} A G {n} 1_555 B C {j} 1_555 {n} A {n} ? B {j} ? 19 1\n"
    two_models = head + "".join(row.format(m=m, n=n, j=19 - n) for m in (1, 2) for n in (1, 2, 3))
    assert [(p.number_a, p.number_b) for p in read_mmcif_pairs(two_models)] == [(1, 18), (2, 17), (3, 16)]
    # ... the first that the table lists, whatever its number.
    later = two_models.replace("\n1 A G", "\n7 A G").replace("\n2 A G", "\n8 A G")
    assert len(read_mmcif_pairs(later)) == 3
    by_label = """data_TEST
loop_
_ndb_struct_na_base_pair.i_label_asym_id
_ndb_struct_na_base_pair.i_label_seq_id
_ndb_struct_na_base_pair.j_label_asym_id
_ndb_struct_na_base_pair.j_label_seq_id
_ndb_struct_na_base_pair.hbond_type_12
A 1 B 18 1
A 2 B 17 4
"""
    assert [(p.chain_a, p.number_a, p.chain_b, p.number_b, p.family) for p in read_mmcif_pairs(by_label)] == [
        ("A", 1, "B", 18, "cWW"), ("A", 2, "B", 17, "tWH")]
    with pytest.raises(BindviewError, match="_ndb_struct_na_base_pair names no nucleotide"):
        read_mmcif_pairs("data_TEST\nloop_\n_ndb_struct_na_base_pair.hbond_type_12\n"
                         "_ndb_struct_na_base_pair.pair_number\n1 1\n4 2\n")


def test_a_dot_bracket_gives_the_stem_of_a_hairpin(model):
    pairs = pairs_from_dot_bracket(model, "A", RIBOSWITCH_HAIRPIN, family="cWW")
    assert sorted((p.number_a, p.number_b) for p in pairs) == [(k, 19 - k) for k in range(1, 8)]
    assert {p.family for p in pairs} == {"cWW"}
    assert {p.family for p in pairs_from_dot_bracket(model, "A", RIBOSWITCH_HAIRPIN)} == {None}
    with pytest.raises(BindviewError, match="18 nucleotides"):
        pairs_from_dot_bracket(model, "A", "(((....)))")
    with pytest.raises(BindviewError, match="does not close"):
        pairs_from_dot_bracket(model, "A", "((((((((....)))))))"[1:].replace(")", ".", 1))


def test_a_letter_in_a_dot_bracket_is_refused_by_name(model):
    """Some notations write a further level of pseudoknots with letters, ``A`` closing with ``a``. Read
    as unpaired, the pairs they stand for would be lost without a word: a letter is an error."""
    with pytest.raises(BindviewError, match=r"the dot-bracket has the letter 'A' at 5.*\(\) \[\] \{\} <>"):
        pairs_from_dot_bracket(model, "A", "((..AA....))....aa")
    with pytest.raises(BindviewError, match="the letter 'x' at 9"):
        pairs_from_dot_bracket(model, "A", "(((((((.x..)))))))")
    # The same pairs with a second kind of bracket are read, and what is no letter and no bracket is unpaired.
    knot = pairs_from_dot_bracket(model, "A", "((..[[....))....]]")
    assert sorted((p.number_a, p.number_b) for p in knot) == [(1, 12), (2, 11), (5, 18), (6, 17)]
    assert len(pairs_from_dot_bracket(model, "A", "(((((((.-,:)))))))")) == 7


def test_the_plain_table_stands_in_without_a_family(model):
    pairs, stacks = pairs_from_contacts(interface_contacts(model, ["A"], ["B"], stacking=True))
    assert sorted((p.number_a, p.number_b) for p in pairs) == [(9, 10), (10, 9)]
    assert {p.family for p in pairs} == {None}
    assert sorted((s.number_a, s.number_b) for s in stacks) == [(10, 10), (11, 11)]   # A11/A11 a stack, no pair


def bond(chain_a: str, number_a: int, atom_a: str, chain_b: str, number_b: int, atom_b: str, kind: str = "hbond"):
    """A row of the plain table between two nucleotides, written from the side of the first."""
    from bindview import InterfaceContact

    return InterfaceContact(kind, chain_a, number_a, "G", (atom_a,), chain_b, number_b, "C", (atom_b,))


def test_a_pair_from_the_table_does_not_depend_on_the_side_a_row_is_written_from():
    there = [bond("A", 1, "N1", "B", 9, "N3"), bond("A", 1, "N2", "B", 9, "O2")]
    assert [(p.key_a, p.key_b) for p in pairs_from_contacts(there)[0]] == [(("A", 1, ""), ("B", 9, ""))]
    # Two bonds from base to base are a pair whichever side each is written from ...
    mixed = [there[0], there[1].flipped()]
    assert [(p.key_a, p.key_b) for p in pairs_from_contacts(mixed)[0]] == [(("A", 1, ""), ("B", 9, ""))]
    # ... and one bond written from both sides is still one bond, and no pair.
    assert pairs_from_contacts([there[0], there[0].flipped()])[0] == []
    # A pair and a stack exclude each other, from whichever side the stack is written.
    stack = bond("A", 1, "N1", "B", 9, "N3", kind="pi_stacking")
    for row in (stack, stack.flipped()):
        pairs, stacks = pairs_from_contacts(there + [row])
        assert pairs == [] and len(stacks) == 1
    assert len(pairs_from_contacts([stack, stack.flipped()])[1]) == 1
    # The sugar's hydroxyl is no atom of the base, under the name old files give it either.
    for name in ("O2'", "O2*"):
        assert pairs_from_contacts([bond("A", 1, name, "B", 9, "N3"), bond("A", 1, "N2", "B", 9, name)])[0] == []


def test_another_hydrogen_bond_across_is_drawn_once_from_whichever_side_it_is_written(model):
    """Bonds with a sugar or phosphate atom between two nucleotides that are no pair: a line for each two
    nucleotides, with the number of their bonds, though the table has them from both sides."""
    table = interface_contacts(model, ["A"], ["B"], stacking=True)
    sugar = [c for c in table if c.type.value == "hbond" and {c.atoms_a[0], c.atoms_b[0]} & {"O2'", "O4'", "OP1", "OP2"}]
    one = BasePairs(model, ["A", "B"], [], contacts=sugar)
    assert len(sugar) == 4 and len(one.hbonds) == 4 and all(n == 1 for _, _, n in one.hbonds)
    both = BasePairs(model, ["A", "B"], [], contacts=sugar + [c.flipped() for c in sugar])
    assert both.hbonds == one.hbonds
    assert both.render().svg == one.render().svg and one.render().svg.count('class="hbond"') == 4
    # From the other side alone: the same lines, each from the first strand to the second.
    other = BasePairs(model, ["A", "B"], [], contacts=[c.flipped() for c in sugar])
    assert other.hbonds == one.hbonds and all(a[0] == "A" and b[0] == "B" for a, b, _ in other.hbonds)
    # A second bond of the same two nucleotides is counted, and an old file's O2* is the sugar's O2'.
    first = sugar[0]
    more = BasePairs(model, ["A", "B"], [], contacts=sugar + [bond("A", first.number_a, "N3", "B", first.number_b, "O2*")])
    assert sorted(n for _, _, n in more.hbonds) == [1, 1, 1, 2]


def test_rnapolis_output_is_read_with_its_families():
    pairs = read_rnapolis_pairs(RNAPOLIS)
    got = {(p.chain_a, p.number_a, p.chain_b, p.number_b): p.family for p in pairs}
    assert len(pairs) == 20 and all(p.source == "rnapolis" and p.ordered for p in pairs)
    assert got[("A", 9, "B", 10)] == "cWW" and got[("A", 10, "B", 9)] == "cWW"
    assert got[("A", 8, "B", 12)] == "cWS" and got[("A", 12, "B", 8)] == "cSW"     # A8 by its Watson-Crick edge
    assert got[("A", 11, "A", 12)] == "cSH"
    assert sum(f == "cWW" for f in got.values()) == 16
    stacks = read_rnapolis_stacks(RNAPOLIS)
    assert sorted((s.chain_a, s.number_a, s.chain_b, s.number_b) for s in stacks if s.chain_a != s.chain_b) == [
        ("A", 10, "B", 10), ("A", 11, "B", 11)]


def test_rnapolis_output_as_a_path_a_text_or_an_object():
    text = RNAPOLIS.read_text()
    import json as _json
    assert (len(read_rnapolis_pairs(str(RNAPOLIS))) == len(read_rnapolis_pairs(text))
            == len(read_rnapolis_pairs(_json.loads(text))) == 20)
    with pytest.raises(BindviewError, match="not the JSON of RNApolis"):
        read_rnapolis_pairs('{"base_pairs": 3}')


def test_a_text_of_one_line_is_read_as_text_and_a_file_that_is_not_there_is_said(tmp_path):
    """A reader takes a path or the text. A table of one row, or a JSON on one line, has no line break
    and is no file name; a file name that names no file is an error of bindview's, with the name."""
    import gzip

    row = '"1XPF|1|A|G|10","cWW","1XPF|1|B|C|15"'
    assert [(p.chain_a, p.number_a, p.chain_b, p.number_b) for p in read_fr3d_pairs(row)] == [("A", 10, "B", 15)]
    assert [(s.number_a, s.number_b) for s in read_fr3d_stacks('"1XPF|1|A|G|10","s35","1XPF|1|B|A|16"')] == [(10, 16)]
    with pytest.raises(BindviewError, match="not the JSON of RNApolis"):
        read_rnapolis_pairs("[1, 2]")
    # A file is still read by its name, as a str or a Path, gzipped or not.
    (tmp_path / "pairs.csv").write_text(row)
    with gzip.open(tmp_path / "pairs.csv.gz", "wt", encoding="utf-8") as handle:
        handle.write(row)
    for given in (tmp_path / "pairs.csv", str(tmp_path / "pairs.csv"), tmp_path / "pairs.csv.gz"):
        assert [(p.number_a, p.number_b) for p in read_fr3d_pairs(given)] == [(10, 15)]
    for reader, name in ((read_fr3d_pairs, "nothing_here.csv"), (read_fr3d_stacks, "nothing_here.csv"),
                         (read_mmcif_pairs, "nothing_here.cif"), (read_rnapolis_pairs, "nothing_here.json"),
                         (read_rnapolis_stacks, "nothing_here.json")):
        with pytest.raises(BindviewError, match=f"cannot read .*{name}"):
            reader(str(tmp_path / name))


MMCIF_HEAD = """data_TEST
loop_
_ndb_struct_na_base_pair.model_number
_ndb_struct_na_base_pair.i_auth_asym_id
_ndb_struct_na_base_pair.i_auth_seq_id
_ndb_struct_na_base_pair.j_auth_asym_id
_ndb_struct_na_base_pair.j_auth_seq_id
_ndb_struct_na_base_pair.hbond_type_12
"""


def nucleotide(chain: str, number: int) -> dict:
    """A residue as RNApolis's JSON writes it."""
    return {"label": None, "auth": {"chain": chain, "number": number, "icode": None, "name": "G"}}


def test_a_table_that_is_not_what_it_should_be_is_refused_with_the_row_or_the_key():
    with pytest.raises(BindviewError, match=r"line 2 of the FR3D table: not an FR3D unit id: '1XPF\|1\|A\|G\|ten'"):
        read_fr3d_pairs('"1XPF|1|A|G|9","cWW","1XPF|1|B|C|16"\n"1XPF|1|A|G|ten","cWW","1XPF|1|B|C|15"\n')
    with pytest.raises(BindviewError, match=r"line 1 of the FR3D table: not an FR3D unit id: 'unit1'"):
        read_fr3d_stacks("unit1,kind,unit2\n")
    with pytest.raises(BindviewError, match=r"_ndb_struct_na_base_pair, row 2: i_auth_seq_id is '\?', which is no "
                                          r"residue number"):
        read_mmcif_pairs(MMCIF_HEAD + "1 A 1 B 18 1\n1 A ? B 17 1\n")
    with pytest.raises(BindviewError, match=r"_ndb_struct_na_base_pair: .*a row is short"):
        read_mmcif_pairs(MMCIF_HEAD + "1 A 1 B 18 1\n1 A 2 B 17\n")
    with pytest.raises(BindviewError, match=r"base_pairs, entry 2, of RNApolis's JSON has no 'nt2'"):
        read_rnapolis_pairs({"base_pairs": [{"nt1": nucleotide("A", 1), "nt2": nucleotide("B", 9)},
                                            {"nt1": nucleotide("A", 2)}]})
    with pytest.raises(BindviewError, match=r"stackings, entry 1, of RNApolis's JSON has no 'nt1'"):
        read_rnapolis_stacks({"stackings": [{"nt2": nucleotide("B", 9)}]})
    with pytest.raises(BindviewError, match=r"base_pairs, entry 1, of RNApolis's JSON is no object"):
        read_rnapolis_pairs({"base_pairs": [1, 2]})
    with pytest.raises(BindviewError, match="not the JSON of RNApolis's annotator: it is no JSON"):
        read_rnapolis_pairs('{"base_pairs": [}')


def test_a_json_object_that_is_not_rnapolis_s_gives_no_empty_list():
    """An object without ``base_pairs`` and without ``stackings`` is another program's, and not an
    annotation that found nothing."""
    for other in ({"something": 1}, '{"basePairs": [1, 2]}'):
        for reader in (read_rnapolis_pairs, read_rnapolis_stacks):
            with pytest.raises(BindviewError, match="not the JSON of RNApolis's annotator: it has neither"):
                reader(other)
    # An annotation that found none of one kind has the other, and gives none.
    assert read_rnapolis_stacks({"base_pairs": []}) == [] and read_rnapolis_pairs({"stackings": []}) == []
    assert read_rnapolis_pairs({"base_pairs": [], "stackings": []}) == []


def test_a_contact_without_atoms_is_refused_where_its_atoms_are_asked_for(model):
    """A table written by hand may leave the atoms of a contact out. Whether a hydrogen bond is from
    base to base cannot be told then."""
    from bindview import InterfaceContact

    bare = InterfaceContact("hbond", "A", 9, "G", (), "B", 10, "C", ())
    with pytest.raises(BindviewError, match=r"the hydrogen bond of A:9 with B:10 names no atom"):
        pairs_from_contacts([bare])
    with pytest.raises(BindviewError, match=r"the hydrogen bond of A:9 with B:10 names no atom"):
        BasePairs(model, ["A", "B"], [], contacts=[bare])


def test_the_riboswitch_with_rnapolis_every_pair_has_a_family():
    fig = nucleic_basepairs_rnapolis()
    r = fig.render()
    assert r.status == "ok"
    assert (r.data["pairs_across"], r.data["pairs_within"], r.data["stacks"]) == (4, 16, 2)
    assert all(p.family for p in fig.pairs)
    assert r.quality.overlaps == 0 and r.quality.line_crossings == 0 and r.data["names_on_names_or_marks"] == 0
    assert r.notes[0].startswith("Pairs and stacks: RNApolis's annotator (MIT), run on the model.")
    # A11 and C12 are neighbours of a strand: their pair is a small arc out of it, not a line between them.
    pair = re.search(r'<g class="pair"[^>]*>\s*<title>A 11, chain A – C 12, chain A: cSH</title>\s*<(\w+)', r.svg)
    assert pair and pair.group(1) == "polyline"


# -- the figure ----------------------------------------------------------------------------------------------

def test_the_riboswitch_has_its_stems_and_two_pairs_across(figure):
    r = figure.render()
    assert r.status == "ok" and r.figure == "diagram" and r.svg
    assert (r.data["pairs_across"], r.data["pairs_within"], r.data["stacks"]) == (2, 14, 2)
    assert r.data["layout"] == "each strand alone, placed by its pairs across"
    assert r.quality.overlaps == 0 and r.quality.line_crossings == 0 and r.quality.link_hits == 0
    assert r.data["names_on_names_or_marks"] == 0
    assert r.quality.readable
    assert r.svg.count('class="nucleotide"') == 36
    assert r.svg.count('class="pair"') == 16


def test_the_key_says_what_is_drawn_and_the_source_line_where_it_came_from(figure):
    r = figure.render()
    assert r.data["kinds"] == ["across-none", "hbond", "stack", "strands", "within"]
    assert "pair across, its family not annotated" in r.svg
    assert "Watson–Crick (G–C, A–U)" not in r.svg          # no annotated pair across: not in the key
    assert r.notes == ["Pairs: a secondary structure given as a dot-bracket; bindview's plain distance table, "
                       "without a family. Stacks: bindview's plain distance table. Other hydrogen bonds across: "
                       "bindview's plain distance table."]


def test_a_bare_figure_keeps_the_line_of_its_source_in_its_notes(figure):
    """Drawn bare, the drawing alone without its key and without the line of its source (``legend=False,
    caption=False``), the figure says nothing of where its pairs come from. The result keeps that line in
    its notes, for the caption of a page that shows the figure, and the file keeps it in its description."""
    bare = figure.render(legend=False, caption=False)
    assert 'class="key"' not in bare.svg and not re.search(r"<text[^>]*>Pairs: ", bare.svg)
    assert bare.notes == [figure.source_line()] == figure.render().notes
    assert f"<desc>{figure.source_line()}</desc>" in bare.svg


def test_a_figure_without_its_caption_keeps_the_line_in_its_notes(figure):
    """``caption=False`` leaves the line of the source off the figure. The result keeps it: a page that
    shows the figure without its caption takes the words into its own."""
    with_it, without = figure.render(), figure.render(caption=False)
    assert with_it.notes == without.notes == [figure.source_line()]
    assert re.search(r"<text[^>]*>Pairs: ", with_it.svg) and not re.search(r"<text[^>]*>Pairs: ", without.svg)


def test_the_source_line_of_records_without_a_source_says_that_they_have_none(model):
    """A record made by hand names no source unless its maker gives one. The line under the key says
    that, and does not end in an empty list (``Pairs: .``)."""
    by_hand = BasePair("A", 9, "B", 10, "cWW")
    assert BasePairs(model, ["A", "B"], [by_hand]).source_line() == "Pairs: given without a source."
    assert "<desc>Pairs: given without a source.</desc>" in BasePairs(model, ["A", "B"], [by_hand]).render().svg
    noted = BasePair("A", 10, "B", 9, "cWW", source="my own notes")
    stacked = BaseStack("A", 10, "B", 10)
    assert BasePairs(model, ["A", "B"], [noted, by_hand], stacks=[stacked]).source_line() == (
        "Pairs: my own notes; some given without a source. Stacks: given without a source.")
    assert BasePairs(model, ["A", "B"], [by_hand], stacks=[stacked]).source_line() == (
        "Pairs and stacks: given without a source.")
    assert BasePairs(model, ["A", "B"], [noted]).source_line() == "Pairs: my own notes."


def test_the_key_shows_the_symbol_of_a_family_at_the_size_of_the_figure(model):
    """A key shows a thing the way the figure draws it. ``step`` sets the size of a figure, and of the
    symbol on a pair with it: the symbol in the key follows, at any step and not at the default one."""
    from bindview.scene import Circle, Text
    from scenetools import groups

    fig = BasePairs(model, ["A", "B"], [BasePair("A", 9, "B", 10, "tWW", source="by hand"),
                                        BasePair("A", 10, "B", 9, "tWH", source="by hand")])
    for step in (24.0, 48.0, 96.0):
        res = fig.render(step=step)
        (pair,) = [g for g in groups(res, "pair") if g.data["family"] == "tWW"]
        (on_pair,) = [item.r for item in pair.items if isinstance(item, Circle)]
        assert on_pair == pytest.approx(step * 0.21)
        (key,) = groups(res, "key")
        circles = [item for item in key.items if isinstance(item, Circle)]
        texts = [item for item in key.items if isinstance(item, Text)]
        # The two sample nucleotides, the circle of tWH (its Watson-Crick edge) and that of tWW.
        assert [c.r for c in circles[:2]] == [8, 8] and [c.r for c in circles[2:]] == [pytest.approx(on_pair)] * 2
        # ... each in a row of its own, clear of the row above and of the words of its row.
        assert [t.content[:3] for t in texts] == ["str", "pal", "tWH", "tWW"]
        assert max(c.cx + c.r for c in circles) < min(t.x for t in texts)
        assert all(below.cy - above.cy >= above.r + below.r for above, below in zip(circles[1:], circles[2:]))


def test_three_themes(figure):
    light, dark, auto = (figure.render(theme=t) for t in ("light", "dark", "auto"))
    assert "#2f63b0" in light.svg and "#7aa7e6" not in light.svg
    assert "#7aa7e6" in dark.svg and "#2f63b0" not in dark.svg
    assert "--lf-strand-1:#2f63b0" in auto.svg and "--lf-strand-1:#7aa7e6" in auto.svg
    assert (light.theme, dark.theme, auto.theme) == ("light", "dark", "auto")


def test_the_same_input_gives_the_same_file():
    assert nucleic_basepairs().render().svg == nucleic_basepairs().render().svg


def test_the_layout_follows_the_pairs_of_the_figure(model):
    """``pairs`` is a list, and a list can be added to. A figure drawn before and after is laid out by
    the pairs it has when it is drawn, as a figure that was made with those pairs is."""
    stems = [p for c in "AB" for p in pairs_from_dot_bracket(model, c, RIBOSWITCH_HAIRPIN, family="cWW")]
    across = [BasePair("A", 9, "B", 10, "cWW", source="dot-bracket"), BasePair("A", 10, "B", 9, "cWW", source="dot-bracket")]
    fig = BasePairs(model, ["A", "B"], stems)
    before = fig.render()
    assert fig._layout() is fig._layout()                       # laid out once while nothing changes
    fig.pairs.extend(across)
    after = fig.render()
    made_with_them = BasePairs(model, ["A", "B"], stems + across).render()
    assert after.svg == made_with_them.svg != before.svg
    assert (after.data["pairs_across"], after.status) == (2, "ok")


def test_pairs_it_cannot_draw_are_left_out_and_said(model):
    stems = pairs_from_dot_bracket(model, "A", RIBOSWITCH_HAIRPIN, family="cWW")
    extra = [BasePair("A", 9, "B", 10, "cWW"), BasePair("A", 9, "B", 10, "cWW"),     # given twice: drawn once
             BasePair("A", 99, "B", 10, "cWW"), BasePair("A", 9, "C", 1, "cWW"),
             BasePair("A", 9, "B", 10, "cWW", symmetry="6_555")]
    fig = BasePairs(model, ["A", "B"], stems + extra, stacks=[BaseStack("A", 10, "B", 10)])
    r = fig.render()
    assert r.data["pairs_across"] == 1 and r.data["pairs_within"] == 7 and r.data["stacks"] == 1
    assert any("1 pair(s) naming a nucleotide" in w for w in r.warnings)
    assert any("1 pair(s) of other chains" in w for w in r.warnings)
    assert any("1 pair(s) with a crystal mate" in w for w in r.warnings)


def test_a_figure_with_nothing_across_is_empty(model):
    stems = [p for c in "AB" for p in pairs_from_dot_bracket(model, c, RIBOSWITCH_HAIRPIN, family="cWW")]
    r = BasePairs(model, ["A", "B"], stems, stacks=[]).render()
    assert r.status == "empty" and r.svg
    assert any("no pair, stack or hydrogen bond between chains A and B" in w for w in r.warnings)


def test_a_figure_too_small_to_read_on_its_page_is_drawn_and_says_unreadable(figure, model):
    """The figure is drawn at any size. Where its smallest text would be under the minimum on the page
    that was asked for, the result says ``unreadable``, as that of every other figure does, and the
    figure is the same file."""
    r = figure.render()
    assert r.status == "ok" and r.ok and r.quality.readable and "smallest text" not in " ".join(r.warnings)
    small = figure.render(figure_width_mm=60)
    assert not small.quality.readable and (small.status, small.figure, small.ok) == ("unreadable", "diagram", False)
    assert small.svg == r.svg
    assert [w for w in small.warnings if w.startswith("at 60 × 240 mm its smallest text would be under 5 pt")]
    # A larger minimum on the usual page is the same case, and a smaller one on the narrow page is none.
    assert figure.render(min_font_pt=9.0).status == "unreadable"
    assert figure.render(figure_width_mm=60, min_font_pt=3.0).status == "ok"
    # A figure with nothing between its strands says that first: it is empty, and its warnings say both.
    nothing = BasePairs(model, ["A", "B"], []).render(figure_width_mm=60)
    assert nothing.status == "empty" and "smallest text would be under" in " ".join(nothing.warnings)


def test_two_chains_and_nucleotides_only(model):
    with pytest.raises(BindviewError, match="two chains"):
        BasePairs(model, ["A", "A"], [])
    with pytest.raises(BindviewError, match="no nucleotides"):
        BasePairs(model, ["A", "Z"], [])


def test_a_strand_of_one_nucleotide_is_refused_as_what_it_is(model):
    """One nucleotide is no strand: a chain is nucleotides joined to one another. The refusal says that
    there is one, and not that there is none."""
    from dataclasses import replace

    from bindview.structure import Structure

    kept = [a for a in model.atoms if a.chain != "B" or a.resnum == 10]
    one = Structure(atoms=[replace(a, index=i) for i, a in enumerate(kept)], source="the riboswitch, cut")
    with pytest.raises(BindviewError) as told:
        BasePairs(one, ["A", "B"], [BasePair("A", 9, "B", 10, "cWW")])
    said = str(told.value)
    assert "chain 'B' has one nucleotide in the riboswitch, cut (G 10)" in said and "joined" in said
    assert "has no nucleotides" not in said
    with pytest.raises(BindviewError, match="chain 'Z' has no nucleotides in"):
        BasePairs(model, ["A", "Z"], [])


# -- deposited entries with the PDB's own annotation -----------------------------------------------------

#: The two strands of each entry.
STRANDS = {"rna_kissing": ("A", "B"), "repressor_dna": ("1", "2")}


def _deposited(slug: str):
    path = model_of(slug)
    st = read_structure(path)
    a, b = STRANDS[slug]
    return BasePairs(st, [a, b], read_mmcif_pairs(path), contacts=interface_contacts(st, [a], [b]))


def test_the_kissing_complex_by_the_pdbs_annotation():
    """The entry's table of base pairs has 20 rows: 6 between chains A and B, 7 within each hairpin."""
    fig = _deposited("rna_kissing")
    r = fig.render()
    assert r.status == "ok" and r.data["pairs_across"] == 6 and r.data["pairs_within"] == 14
    assert {p.family for p in fig.across} == {"cWW"}
    assert r.data["layout"] == "each strand alone, placed by its pairs across"
    assert r.quality.overlaps == 0 and r.quality.line_crossings == 0


def test_two_strands_of_a_duplex_are_one_structure():
    """The operator DNA of the repressor: the entry's table of base pairs has 19 rows, all between strands
    1 and 2, and none within a strand. Paired along their length, the two strands are drawn as one."""
    r = _deposited("repressor_dna").render()
    assert r.status == "ok" and r.data["pairs_across"] == 19 and r.data["pairs_within"] == 0
    assert r.data["layout"] == "one structure of the joined strands"
    assert r.quality.overlaps == 0 and r.quality.line_crossings == 0


def test_a_family_whose_order_the_source_does_not_give_is_said_in_the_key(model):
    """A family of two different edges, such as tWH, names which base uses which edge only where the source
    says so (FR3D does, the PDB's annotation does not). The key says which of the two it is."""
    stems = [p for c in "AB" for p in pairs_from_dot_bracket(model, c, RIBOSWITCH_HAIRPIN, family="cWW")]
    for ordered, said in ((False, "which base uses which edge, the source does not say"),
                          (True, "each symbol at its own base")):
        across = [BasePair("A", 9, "B", 10, "tWH", ordered=ordered, source="by hand"),
                  BasePair("A", 10, "B", 9, "cWW", source="by hand")]
        r = BasePairs(model, ["A", "B"], stems + across).render()
        assert r.status == "ok" and said in r.svg


# -- the stand-in within a strand --------------------------------------------------------------------------

def test_the_plain_table_finds_the_stems_of_both_hairpins(model):
    # The seven stem pairs of each hairpin, as the dot-bracket has them: one pair for a nucleotide, and not
    # the diagonal bonds between neighbouring pairs of the helix.
    for chain in "AB":
        got = pairs_within_from_table(model, chain)
        want = pairs_from_dot_bracket(model, chain, RIBOSWITCH_HAIRPIN)
        assert sorted((p.number_a, p.number_b) for p in got) == sorted((p.number_a, p.number_b) for p in want)
        assert all(p.family is None and p.source == "plain-table" for p in got)


def test_the_plain_table_within_a_strand_against_the_pdbs_annotation():
    """The stand-in finds no pair within a strand that the PDB's annotation does not have, and every one
    that the annotation gives a family: of the kissing complex's 14 it misses only U6-A18 of chain B, the
    row whose hydrogen-bond types the entry leaves unknown. The strands of DNA have none, and it finds none."""
    for slug in STRANDS:
        path = model_of(slug)
        st, ann = read_structure(path), read_mmcif_pairs(path)
        for chain in STRANDS[slug]:
            mine = {(p.number_a, p.number_b) for p in pairs_within_from_table(st, chain)}
            within = [p for p in ann if p.chain_a == chain == p.chain_b and not p.symmetry]
            pdb = {tuple(sorted((p.number_a, p.number_b))) for p in within}
            with_family = {tuple(sorted((p.number_a, p.number_b))) for p in within if p.family}
            assert mine <= pdb and with_family <= mine, (slug, chain)
            assert pdb - with_family == ({(6, 18)} if (slug, chain) == ("rna_kissing", "B") else set())


# -- a ligand beside the pairs -------------------------------------------------------------------------------

def _with_ligand(**options):
    return BasePairs(read_structure(RIBOSWITCH), ["A", "B"], read_rnapolis_pairs(RNAPOLIS),
                     stacks=read_rnapolis_stacks(RNAPOLIS),
                     contacts=interface_contacts(read_structure(RIBOSWITCH), ["A"], ["B"], stacking=True),
                     ligand="BGD", **options)


def test_a_ligand_stands_between_the_nucleotides_it_touches():
    r = _with_ligand(ligand_contacts=CONTACTS).render()
    lig = r.data["ligand"]
    assert (lig["name"], lig["residue"], lig["nucleotides"]) == ("BGD", "C:19", 5)
    assert lig["lines"] == {"hbond": 3, "saltbridge": 3, "pi_cation": 2}       # a line per kind per nucleotide
    assert r.svg.count('class="ligand"') == 1 and r.svg.count('class="ligand-contact"') == 5
    assert r.notes[0].endswith("Contacts of BGD: interactions.tsv.")
    assert "salt bridge with the ligand" in r.svg and "π–cation with the ligand" in r.svg
    assert r.quality.overlaps == 0 and r.data["names_on_names_or_marks"] == 0 and not r.warnings
    plain = _with_ligand(ligand_smiles=BGD).render()                           # the plain table's, the same ones
    assert plain.data["ligand"]["nucleotides"] == 5 and plain.data["ligand"]["lines"] == lig["lines"]
    assert plain.notes[0].endswith("Contacts of BGD: bindview's plain distance table.")


def test_a_ligands_contacts_outside_the_strands_are_left_out_and_said():
    rows = [{"interaction_type": "hbond", "residue_number": 8, "residue_name": "A", "chain": "B", "distance": 2.9},
            {"interaction_type": "hbond", "residue_number": 99, "residue_name": "LYS", "chain": "Z", "distance": 3.0}]
    r = _with_ligand(ligand_contacts=rows, ligand_name="bis-guanidine").render()
    assert r.data["ligand"]["nucleotides"] == 1 and r.data["ligand"]["name"] == "bis-guanidine"
    assert any("1 contact(s) of the ligand bis-guanidine with residues outside" in w for w in r.warnings)
    assert r.notes[0].endswith("Contacts of bis-guanidine: as given.")
    nothing = _with_ligand(ligand_contacts=rows[1:]).render()
    assert "ligand" not in nothing.data and any("touches no nucleotide" in w for w in nothing.warnings)
    with pytest.raises(BindviewError, match="nucleotide of chain A"):
        BasePairs(read_structure(RIBOSWITCH), ["A", "B"], [], ligand="A:8")


# -- the graph, and a figure that answers the pointer ------------------------------------------------------

def test_the_figure_as_a_graph_and_drawn_again_from_it():
    fig = _with_ligand(ligand_contacts=CONTACTS)
    r = fig.render()
    g = r.graph
    assert g.to_dict()["version"] == "1.4" and g.view == "basepairs"
    assert len(g.of_kind("residue")) == 36 and [n.id for n in g.of_kind("ligand_group")] == ["ligand:C:19"]
    kinds = {c.type for c in g.contacts}
    assert {"base_pair", "base_stack", "hbond", "saltbridge", "pi_cation"} <= kinds
    pairs = [c for c in g.contacts if c.type == "base_pair"]
    assert len(pairs) == r.data["pairs_across"] + r.data["pairs_within"]
    assert {c.attrs["family"] for c in pairs} >= {"cWW", "cWS"}
    assert g.meta["basepairs"]["chains"] == ["A", "B"]
    again = Graph.from_json(g.to_json())
    assert fig.render(graph=again).svg == r.svg                     # unmoved: the same file
    again.move("ligand:C:19", 0.0, 60.0)
    again.move("A:1", -24.0, 0.0)
    moved = fig.render(graph=again)
    assert moved.data["ligand"]["at"][1] == pytest.approx(r.data["ligand"]["at"][1] + 60.0)
    assert moved.svg != r.svg
    with pytest.raises(BindviewError, match="not one of this figure"):
        nucleic_basepairs().render(graph=Graph(view="basepairs", bond_length=24.0))


def test_a_figure_that_answers_the_pointer():
    fig = _with_ligand(ligand_contacts=CONTACTS)
    live, still = fig.render(interactive=True), fig.render()
    assert 'class="bindview lf-live"' in live.svg and "lf-live" not in still.svg
    assert live.svg.count('tabindex="0"') == 37                     # every nucleotide and the ligand
    assert live.svg.count('tabindex="-1"') == len(live.graph.contacts) - sum(
        1 for c in live.graph.contacts if c.ligand == "ligand:C:19") + 5          # pairs, stacks, bonds; a line a nucleotide
    assert ".lf-t-g0:hover" in live.svg and ".lf-t-n0:hover" in live.svg
