"""The interaction record and its loaders."""

from __future__ import annotations

import pytest

from bindview import Interaction, InteractionType, load_interactions, parse_interactions
from bindview.errors import InteractionError
from bindview.interactions import coerce_interactions
from conftest import CASE_SPECS, CASES


def test_record_normalises_its_fields():
    r = Interaction("H-Bond", "712", "gln", chain="A", protein_atom="NE2", ligand_atom="N7",
                    distance="4.02", angle="141.02", ligand_is_donor="False", beyond_contact_set="True")
    assert r.type is InteractionType.HBOND
    assert (r.residue_number, r.residue_name, r.distance, r.angle) == (712, "GLN", 4.02, 141.02)
    assert r.ligand_is_donor is False and r.beyond_contact_set is True
    assert r.residue_label == "Gln712"


def test_empty_cells_mean_unknown_not_empty_string():
    r = Interaction("pi_stacking", 765, "PHE", protein_atom="", ligand_atom=None, distance="", ligand_is_donor="")
    assert r.protein_atom is None and r.ligand_atom is None and r.distance is None and r.ligand_is_donor is None


@pytest.mark.parametrize("alias,expected", [
    ("salt_bridge", "saltbridge"), ("pistacking", "pi_stacking"), ("cation-pi", "pi_cation"),
    ("halogen bond", "halogen"), ("metal_complex", "metal"), ("waterbridge", "water_bridge"),
    ("Hydrophobic", "hydrophobic"),
])
def test_type_aliases(alias, expected):
    assert Interaction(alias, 1, "ALA").type.value == expected


def test_malformed_records_are_rejected():
    with pytest.raises(InteractionError, match="unknown interaction type"):
        Interaction("stapling", 1, "ALA")
    with pytest.raises(InteractionError, match="residue_number"):
        Interaction("hbond", "twelve", "ALA")
    with pytest.raises(InteractionError, match="residue_name"):
        Interaction("hbond", 12, "")
    with pytest.raises(InteractionError, match="distance"):
        Interaction("hbond", 12, "ALA", distance="near")
    with pytest.raises(InteractionError, match="negative"):
        Interaction("hbond", 12, "ALA", distance=-1.0)


def test_rows_from_a_database_are_accepted_as_mappings():
    rows = [{"interaction_type": "hbond", "residue_number": 712, "residue_name": "GLN", "chain": "A",
             "protein_atom": "NE2", "ligand_atom": "N7", "distance": 4.02, "ligand_is_donor": False},
            {"type": "saltbridge", "resnr": 21, "restype": "LYS", "reschain": "A", "dist": 3.92,
             "ligand_atoms": "P2,O6,O7", "protein_atoms": ["NZ"]}]
    recs = coerce_interactions(rows)
    assert recs[0].ligand_atom == "N7"
    assert recs[1].ligand_atoms == ("P2", "O6", "O7") and recs[1].protein_atoms == ("NZ",)
    with pytest.raises(InteractionError, match="interaction #2"):
        coerce_interactions([rows[0], {"type": "hbond"}])


def test_header_only_table_is_an_empty_result_not_an_error(testdata, tmp_path):
    header = (testdata / "kit_imatinib" / "interactions.tsv").read_text().splitlines()[0]
    (tmp_path / "none.tsv").write_text(header + "\n")
    assert load_interactions(tmp_path / "none.tsv") == []
    assert parse_interactions("interaction_type\tresidue_number\tresidue_name\n") == []


def test_a_missing_or_blank_table_is_an_error(tmp_path):
    with pytest.raises(InteractionError, match="cannot read"):
        load_interactions(tmp_path / "nope.tsv")
    with pytest.raises(InteractionError, match="no header"):
        parse_interactions("")
    with pytest.raises(InteractionError, match="'type' or 'interaction_type'"):
        parse_interactions("a,b\n1,2\n")


def test_bad_row_reports_its_line(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("type,residue_number,residue_name\nhbond,7,ALA\nwibble,8,GLY\n")
    with pytest.raises(InteractionError, match="line 3"):
        load_interactions(p)


def test_supplied_tables_load(testdata):
    """Every row of a committed table is one record: as many as the file has lines after its header, and as many
    as ``cases.json`` gives for the case."""
    for case in CASES:
        path = testdata / case / "interactions.tsv"
        rows = [line for line in path.read_text().splitlines()[1:] if line.strip()]
        assert len(load_interactions(path)) == len(rows) == CASE_SPECS[case]["n_interactions"]


_TABLE = "type,residue_number,residue_name,chain,protein_atom,ligand_atom,distance\nhbond,12,ASP,A,OD1,N1,2.9\n"


def test_a_table_with_a_byte_order_mark_is_read(tmp_path):
    """What a spreadsheet program saves as "CSV UTF-8" begins with a byte-order mark, before the first column's name."""
    path = tmp_path / "from_a_spreadsheet.csv"
    path.write_bytes(b"\xef\xbb\xbf" + _TABLE.encode())
    (rec,) = load_interactions(path)
    assert (rec.type.value, rec.residue_number, rec.chain) == ("hbond", 12, "A")
    assert (rec.protein_atom, rec.ligand_atom, rec.distance) == ("OD1", "N1", 2.9)
    assert parse_interactions("\ufeff" + _TABLE) == [rec]


def test_a_table_is_read_with_tabs_commas_or_semicolons_and_with_the_separator_that_is_named():
    """The separator is found in the header row: a tab, a comma, or the semicolon a spreadsheet program writes
    where that is its list separator. Any other is named by the caller."""
    (rec,) = parse_interactions(_TABLE)
    for separator in ("\t", ";"):
        assert parse_interactions(_TABLE.replace(",", separator)) == [rec]
    piped = _TABLE.replace(",", "|")
    assert parse_interactions(piped, delimiter="|") == [rec]
    with pytest.raises(InteractionError, match="needs a 'type'"):
        parse_interactions(piped)
    # A separator that is named is used, whatever the header looks like: here every row is one field.
    with pytest.raises(InteractionError, match="needs a 'type'"):
        parse_interactions(_TABLE, delimiter=";")
    with pytest.raises(InteractionError, match="one character"):
        parse_interactions(_TABLE, delimiter=", ")


def test_a_number_with_a_decimal_comma_is_refused_with_its_line():
    """Semicolons often come with decimal commas. Such a number is not guessed at: the row is named."""
    table = _TABLE.replace(",", ";").replace("2.9", "2,9")
    with pytest.raises(InteractionError, match="line 2"):
        parse_interactions(table)


_COLUMNS = ("type", "residue_number", "residue_name", "chain", "insertion_code", "protein_atom", "ligand_atom",
            "distance", "angle", "ligand_is_donor", "protein_atoms", "ligand_atoms", "water")


def _one_row(*cells: str):
    (rec,) = parse_interactions("\t".join(_COLUMNS) + "\n" + "\t".join(cells) + "\n")
    return rec


def test_a_word_for_a_value_that_is_not_given_is_understood_in_every_column():
    """R writes NA for a missing value, a data frame nan, mmCIF a question mark or a dot: in every column,
    the two that hold the atoms of a group among them."""
    bare = Interaction("pi_stacking", 765, "PHE")
    assert (bare.chain, bare.protein_atoms, bare.ligand_atoms) == ("", (), ())
    for word in ("", "?", ".", "nan", "None"):
        assert _one_row("pi_stacking", "765", "PHE", *[word] * 10) == bare
    # NA can also be a name. Where a name can stand it is kept, for the structure to say which it is
    # (bindview.resolve); in every other column it is a value that is not given.
    rec = _one_row("pi_stacking", "765", "PHE", *["NA"] * 10)
    assert (rec.chain, rec.protein_atom, rec.ligand_atom, rec.protein_atoms, rec.ligand_atoms) == \
        ("NA", "NA", "NA", ("NA",), ("NA",))
    assert rec.with_(chain="", protein_atom=None, ligand_atom=None, protein_atoms=(), ligand_atoms=()) == bare
    # Among several names it is a name, and no word for something else.
    assert Interaction("metal", 500, "HEM", protein_atoms="NA,NB, NC;ND").protein_atoms == ("NA", "NB", "NC", "ND")
    # The row of a data frame says the same with a number that is none.
    row = {"type": "pi_stacking", "residue_number": 765, "residue_name": "PHE", "chain": float("nan"),
           "protein_atoms": float("nan"), "ligand_atoms": float("nan"), "distance": float("nan")}
    assert Interaction.from_mapping(row) == bare


def test_a_table_that_is_not_utf8_is_an_interaction_error(tmp_path):
    path = tmp_path / "latin1.csv"
    path.write_bytes(b"# distances in \xc5\n" + _TABLE.encode())
    said = r"cannot read .*latin1.csv: it is not UTF-8 text \(it has the byte 0xc5\)"
    with pytest.raises(InteractionError, match=said):
        load_interactions(path)
