"""Structure reading and ligand selection."""

from __future__ import annotations

import gzip

import pytest

from bindview.errors import LigandNotFoundError, StructureError
from bindview.interactions import load_interactions
from bindview.structure import ResidueKey, parse_mmcif, parse_pdb, read_models, read_structure
from conftest import TESTDATA, meta_of, model_of

#: A model of the AlphaFold database: one chain, a pLDDT for every atom, declared as a model declares it.
AFDB_MODEL = TESTDATA / "interface_pae" / "afdb_tectb" / "model.cif.gz"

MINI_CIF = """data_test
loop_
_chem_comp.id
_chem_comp.pdbx_smiles
ALA "C[C@@H](C(=O)O)N"
NUC "C1OC(CO)C1"
#
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.auth_seq_id
_atom_site.auth_asym_id
_atom_site.pdbx_PDB_model_num
ATOM   1 N N     . ALA A 1 1.0 2.0 3.0 7 A 1
ATOM   2 C CA    A ALA A 1 2.0 2.0 3.0 7 A 1
ATOM   3 C CA    B ALA A 1 2.1 2.0 3.0 7 A 1
HETATM 4 O "O5'" . NUC B . 5.0 5.0 5.0 1 B 1
HETATM 5 C 'C1"' . NUC B . 6.0 5.0 5.0 1 B 1
HETATM 6 ZN ZN   . ZN  C . 9.0 9.0 9.0 1 C 1
ATOM   7 N N     . ALA A 1 1.0 2.0 3.0 7 A 2
"""


def test_mmcif_reads_columns_by_name_and_quoted_atom_names():
    st = parse_mmcif(MINI_CIF)
    assert [a.name for a in st.atoms] == ["N", "CA", "O5'", 'C1"', "ZN"]     # alt-loc B and model 2 dropped
    assert st.atoms[0].resnum == 7 and st.atoms[0].chain == "A"              # auth numbering wins
    assert st.atoms[2].hetero and not st.atoms[0].hetero
    assert st.atoms[4].element == "Zn"
    assert st.chem_comp_smiles["NUC"] == "C1OC(CO)C1"
    assert st.coords.shape == (5, 3)


def test_ligand_selection_forms():
    st = parse_mmcif(MINI_CIF)
    key = ResidueKey("B", 1)
    assert st.select_ligand("NUC") == key
    assert st.select_ligand("B:1") == key
    assert st.select_ligand("B:NUC:1") == key
    assert st.select_ligand(("B", 1)) == key
    assert st.select_ligand(key) == key
    assert st.select_ligand("C") == ResidueKey("C", 1)          # a chain holding one hetero residue


def test_ligand_selection_errors_are_specific():
    st = parse_mmcif(MINI_CIF)
    with pytest.raises(LigandNotFoundError, match="hetero residues present: NUC, ZN"):
        st.select_ligand("XYZ")
    with pytest.raises(LigandNotFoundError):
        st.select_ligand("B:99")
    with pytest.raises(LigandNotFoundError, match="cannot parse"):
        st.select_ligand("B:1:2:3")


def test_ambiguous_ligand_name_is_an_error_not_a_silent_pick():
    doubled = MINI_CIF.replace("ATOM   7 N N     . ALA A 1 1.0 2.0 3.0 7 A 2",
                               "HETATM 7 ZN ZN   . ZN  D . 1.0 1.0 1.0 1 D 1")
    st = parse_mmcif(doubled)
    with pytest.raises(LigandNotFoundError, match="2 residues are named 'ZN'"):
        st.select_ligand("ZN")


def test_missing_atom_site_is_a_structure_error():
    with pytest.raises(StructureError):
        parse_mmcif("data_x\n_entry.id x\n")


def test_pdb_parsing_and_element_guessing():
    text = (
        "ATOM      1  N   ALA A   7       1.000   2.000   3.000  1.00  0.00           N\n"
        "ATOM      2  CA  ALA A   7       2.000   2.000   3.000  1.00  0.00\n"
        "HETATM    3 FE   HEM B   1       5.000   5.000   5.000  1.00  0.00\n"
        "HETATM    4  O1  LIG B   2       6.000   5.000   5.000  1.00  0.00           O\n"
        "END\n")
    st = parse_pdb(text)
    assert [(a.name, a.element) for a in st.atoms] == [("N", "N"), ("CA", "C"), ("FE", "Fe"), ("O1", "O")]
    assert st.select_ligand("LIG") == ResidueKey("B", 2)


def test_an_entry_of_the_archive_parses(testdata):
    """An entry of the PDB as the archive packs it: the ligand has as many atoms as the archive's record of its
    component, each named once, and every residue that the contact table names is there under that name."""
    st = read_structure(model_of("kit_imatinib"))
    names = [a.name for a in st.residue_atoms(st.select_ligand("STI"))]
    assert len(names) == len(set(names)) == meta_of("kit_imatinib")["ligand"]["heavy_atoms"]
    table = load_interactions(testdata / "kit_imatinib" / "interactions.tsv")
    assert table
    for row in table:
        key = st.find_residue(row.chain, row.residue_number)
        assert key is not None and st.residue_name(key) == row.residue_name


# ---------------------------------------------------------------------------
# A file with several models
# ---------------------------------------------------------------------------

def _pdb_atom(serial, name, resname, chain, number, x, element, hetero=False):
    return (f"{'HETATM' if hetero else 'ATOM  '}{serial:5d} {name:<4s} {resname:>3s} {chain}{number:4d}    "
            f"{x:8.3f}{2.0:8.3f}{3.0:8.3f}  1.00  0.00          {element:>2s}")


def _pdb_model(shift, record="MODEL     %4d", number=1):
    atoms = [_pdb_atom(1, "N", "ALA", "A", 7, 1.0 + shift, "N"), _pdb_atom(2, "CA", "ALA", "A", 7, 2.0 + shift, "C"),
             _pdb_atom(3, "ZN", "ZN", "C", 1, 9.0 + shift, "ZN", hetero=True)]
    head = (record % number) if "%" in record else record
    return "\n".join([head, *atoms, "ENDMDL"]) + "\n"


def test_mmcif_models_are_read_by_the_number_the_file_gives_them():
    first = parse_mmcif(MINI_CIF)
    assert first.model is None and len(first.atoms) == 5                     # the first, unnamed
    one, two = parse_mmcif(MINI_CIF, model=1), parse_mmcif(MINI_CIF, model=2)
    assert one.model == 1 and [(a.name, a.xyz) for a in one.atoms] == [(a.name, a.xyz) for a in first.atoms]
    assert two.model == 2 and [a.name for a in two.atoms] == ["N"]
    assert two.chem_comp_smiles == first.chem_comp_smiles                    # what the file says once holds for each
    with pytest.raises(StructureError, match="there is no model 3: the file has 2 models, numbered 1, 2"):
        parse_mmcif(MINI_CIF, model=3)
    # A file that numbers no model is one model, number 1.
    plain = MINI_CIF.replace("_atom_site.pdbx_PDB_model_num\n", "")
    plain = "\n".join(ln.rsplit(" ", 1)[0] if ln.startswith(("ATOM", "HETATM")) else ln for ln in plain.splitlines()[:-1])
    assert parse_mmcif(plain, model=1).model == 1 and len(parse_mmcif(plain, model=1).atoms) == 5
    with pytest.raises(StructureError, match="there is no model 2: the file has one model, number 1"):
        parse_mmcif(plain, model=2)


def test_pdb_models_are_read_by_number_and_by_order_where_the_file_does_not_number_them():
    text = _pdb_model(0.0, number=1) + _pdb_model(0.5, number=2) + _pdb_model(2.0, number=7) + "END\n"
    first = parse_pdb(text)
    assert first.model is None and first.atoms[0].x == 1.0 and len(first.atoms) == 3     # down to the first ENDMDL
    assert parse_pdb(text, model=2).atoms[0].x == 1.5 and parse_pdb(text, model=7).atoms[2].x == 11.0
    assert parse_pdb(text, model=7).model == 7 and parse_pdb(text, model=7).atoms[2].hetero
    with pytest.raises(StructureError, match="there is no model 3: the file has 3 models, numbered 1, 2, 7"):
        parse_pdb(text, model=3)
    # Frames that every carry the number 1, or none: they are told apart by their order.
    for record in ("MODEL        1", "MODEL"):
        frames = _pdb_model(0.0, record) + _pdb_model(0.5, record) + "END\n"
        assert parse_pdb(frames, model=2).atoms[0].x == 1.5 and parse_pdb(frames, model=2).model == 2
        assert parse_pdb(frames).atoms[0].x == 1.0
    # No MODEL record at all: one model, number 1.
    bare = "\n".join(_pdb_model(0.0).splitlines()[1:-1]) + "\nEND\n"
    assert parse_pdb(bare, model=1).model == 1 and len(parse_pdb(bare).atoms) == 3
    with pytest.raises(StructureError, match="there is no model 2: the file has one model, number 1"):
        parse_pdb(bare, model=2)
    with pytest.raises(StructureError, match="no ATOM/HETATM records in model 2"):
        parse_pdb(_pdb_model(0.0, number=1) + "MODEL        2\nENDMDL\nEND\n", model=2)


def test_every_model_of_a_file_gzipped_or_not(tmp_path):
    pdb = _pdb_model(0.0, number=1) + _pdb_model(0.5, number=2) + _pdb_model(2.0, number=3) + "END\n"
    (tmp_path / "bundle.pdb").write_text(pdb)
    (tmp_path / "bundle.cif").write_text(MINI_CIF)
    with gzip.open(tmp_path / "bundle.pdb.gz", "wt") as handle:
        handle.write(pdb)
    with gzip.open(tmp_path / "bundle.cif.gz", "wt") as handle:
        handle.write(MINI_CIF)
    for name in ("bundle.pdb", "bundle.pdb.gz"):
        models = read_models(tmp_path / name)
        assert [m.model for m in models] == [1, 2, 3] and [m.atoms[0].x for m in models] == [1.0, 1.5, 3.0]
        assert all(m.source == str(tmp_path / name) for m in models)
        assert read_structure(tmp_path / name).atoms[0].x == 1.0 and read_structure(tmp_path / name).model is None
        assert read_structure(tmp_path / name, model=3).atoms[0].x == 3.0
    for name in ("bundle.cif", "bundle.cif.gz"):
        models = read_models(tmp_path / name)
        assert [m.model for m in models] == [1, 2] and [len(m.atoms) for m in models] == [5, 1]
        assert len(read_structure(tmp_path / name).atoms) == 5
    with pytest.raises(StructureError, match="cannot read"):
        read_models(tmp_path / "nothing.pdb")
    # A model of the AlphaFold database is one model: read as the one model of its file, it has the atoms and
    # the declared confidence that it has when it is read alone.
    (only,) = read_models(AFDB_MODEL)
    alone = read_structure(AFDB_MODEL)
    assert only.model == 1 and alone.model is None
    assert [(a.name, a.chain, a.resnum, a.xyz, a.b_iso) for a in only.atoms] == \
        [(a.name, a.chain, a.resnum, a.xyz, a.b_iso) for a in alone.atoms]
    assert only.declares_plddt and only.plddt_scale == alone.plddt_scale and only.qa_metrics == alone.qa_metrics


def test_stretches_of_a_chain_get_chain_names_of_their_own():
    model = read_structure(AFDB_MODEL)
    numbers = {k.number for k in model.residues() if k.chain == "A"}
    cut = model.split("A", {"X": (28, 124), "Y": (126, 269)})
    assert sorted({a.chain for a in cut.atoms}) == ["A", "X", "Y"] and len(cut.atoms) == len(model.atoms)
    assert {k.number for k in cut.residues() if k.chain == "X"} == {n for n in numbers if 28 <= n <= 124}
    assert {k.number for k in cut.residues() if k.chain == "Y"} == {n for n in numbers if 126 <= n <= 269}
    assert {k.number for k in cut.residues() if k.chain == "A"} == numbers - set(range(28, 125)) - set(range(126, 270))
    assert 125 in numbers and max(numbers) > 269                              # (so each part of A is there)
    assert sorted({a.chain for a in model.atoms}) == ["A"]                    # the model itself is as it was
    for parts, said in (({"A": (1, 5)}, "the structure has a chain of that name"),
                        ({"X": (9, 5)}, "the first number comes first"),
                        ({"X": (1, 50), "Y": (40, 90)}, "overlaps another part"),
                        ({"X": (900, 950)}, "has no residue from 900 to 950")):
        with pytest.raises(StructureError, match=said):
            model.split("A", parts)


# ---------------------------------------------------------------------------
# Files that are legal and unusual, and files that are cut or damaged
# ---------------------------------------------------------------------------

def _bare_pdb_atom(serial, field, resname, number, x, record="ATOM"):
    """An atom record that ends after the B-factor: no element columns, the name as its four columns have it."""
    return f"{record:<6s}{serial:5d} {field} {resname:>3s} A{number:4d}    {x:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 10.00"


def test_a_hydrogen_whose_name_fills_its_four_columns_is_no_mercury():
    """Without the element columns the element is read from the name. The hydrogens on the gamma carbons of
    valine, isoleucine and threonine begin in the first column, as a mercury ion does, and go on with two digits."""
    hydrogens = ["HG11", "HG12", "HG13", "HG21", "HG22", "HG23"]
    lines = [_bare_pdb_atom(1, " N  ", "VAL", 1, 0.0), _bare_pdb_atom(2, " CA ", "VAL", 1, 1.5),
             _bare_pdb_atom(3, " CG1", "VAL", 1, 2.5)]
    lines += [_bare_pdb_atom(4 + k, name, "VAL", 1, 3.5 + k) for k, name in enumerate(hydrogens)]
    lines += [_bare_pdb_atom(10, "HG  ", " HG", 201, 20.0, "HETATM"),
              _bare_pdb_atom(11, "CA  ", " CA", 202, 25.0, "HETATM")]
    st = parse_pdb("\n".join(lines) + "\nEND\n")
    assert [(a.name, a.element) for a in st.atoms] == (
        [("N", "N"), ("CA", "C"), ("CG1", "C")] + [(name, "H") for name in hydrogens] + [("HG", "Hg"), ("CA", "Ca")])
    cif = ("data_x\nloop_\n_atom_site.label_atom_id\n_atom_site.label_comp_id\n_atom_site.label_asym_id\n"
           "_atom_site.label_seq_id\n_atom_site.Cartn_x\n_atom_site.Cartn_y\n_atom_site.Cartn_z\n"
           "CG2 VAL A 1 0.0 0.0 0.0\nHG21 VAL A 1 1.0 0.0 0.0\nHG22 VAL A 1 0.0 1.0 0.0\n")
    assert [(a.name, a.element) for a in parse_mmcif(cif).atoms] == [("CG2", "C"), ("HG21", "H"), ("HG22", "H")]


def _label_only_cif(columns, rows):
    return "data_x\nloop_\n" + "".join(f"_atom_site.{c}\n" for c in columns) + "\n".join(rows) + "\n"


def _residues(st):
    return [(str(k), st.residue_name(k), [a.name for a in st.residue_atoms(k)]) for k in st.residues()]


def test_residues_that_the_file_gives_no_number_stay_apart():
    """Without auth_seq_id a ligand or a water has no number: its label_seq_id is a dot. Each is a residue of
    its own all the same, numbered after the last numbered residue of its chain, and none of its atoms is lost."""
    columns = ["group_PDB", "type_symbol", "label_atom_id", "label_comp_id", "label_asym_id", "label_seq_id",
               "Cartn_x", "Cartn_y", "Cartn_z"]
    glycine = [f"ATOM {el} {name} GLY A 7 {x} 0.0 0.0" for el, name, x in (("N", "N", 0.0), ("C", "CA", 1.5))]
    waters = [f"HETATM O O HOH B . {x} 0.0 0.0" for x in (10.0, 20.0, 30.0)]
    st = parse_mmcif(_label_only_cif(columns, glycine + waters))
    assert _residues(st) == [("A:7", "GLY", ["N", "CA"]), ("B:1", "HOH", ["O"]), ("B:2", "HOH", ["O"]),
                             ("B:3", "HOH", ["O"])]
    assert [a.x for a in st.atoms[2:]] == [10.0, 20.0, 30.0]
    # Two ligands that the author's chain name puts into the chain of the protein, and no author's number.
    columns = columns[:6] + ["auth_asym_id", "label_alt_id"] + columns[6:]
    glycine = [f"ATOM {el} {name} GLY A 7 A . {x} 0.0 0.0" for el, name, x in (("N", "N", 0.0), ("C", "CA", 1.5))]
    ligands = ["HETATM C C1 AAA B . A . 0.0 5.0 0.0", "HETATM O O1 AAA B . A . 1.4 5.0 0.0",
               "HETATM C C1 BBB C . A A 9.0 5.0 0.0", "HETATM C C1 BBB C . A B 9.5 5.0 0.0",
               "HETATM N N1 BBB C . A . 10.4 5.0 0.0"]
    st = parse_mmcif(_label_only_cif(columns, ligands[:2] + glycine + ligands[2:]))
    assert _residues(st) == [("A:8", "AAA", ["C1", "O1"]), ("A:7", "GLY", ["N", "CA"]), ("A:9", "BBB", ["C1", "N1"])]
    assert st.atom_by_name(ResidueKey("A", 9), "C1").x == 9.0            # of two alternate locations, the first
    assert st.select_ligand("BBB") == ResidueKey("A", 9)


_ALT_COLUMNS = ["group_PDB", "type_symbol", "label_atom_id", "label_alt_id", "label_comp_id", "label_asym_id",
                "label_seq_id", "Cartn_x", "Cartn_y", "Cartn_z"]


def _alt_cif(atoms):
    """mmCIF text of atoms of chain A, each as (atom name, alternate location, residue name, number, x)."""
    return _label_only_cif(_ALT_COLUMNS, [f"ATOM {name[0]} {name} {alt} {resname} A {number} {x} 0.0 0.0"
                                          for name, alt, resname, number, x in atoms])


def _alt_pdb(atoms):
    """The same atoms as PDB text: the alternate location stands in the column after the atom name."""
    lines = []
    for serial, (name, alt, resname, number, x) in enumerate(atoms, start=1):
        line = _pdb_atom(serial, name, resname, "A", number, x, name[0])
        lines.append(line[:16] + alt.replace(".", " ") + line[17:])
    return "\n".join(lines) + "\nEND\n"


def _named(st):
    return [(str(a.residue), a.resname, a.name, a.x) for a in st.atoms]


@pytest.mark.parametrize("reader, written", [(parse_mmcif, _alt_cif), (parse_pdb, _alt_pdb)], ids=["mmcif", "pdb"])
def test_a_position_of_two_residue_types_is_read_as_one_residue_of_the_first(reader, written):
    """A deposited structure writes two residue types at one position as alternate locations of each other:
    serine as A, threonine as B. The residue that is read has the atoms of one of them, the first in the file."""
    glycine = [("N", ".", "GLY", 9, 0.0), ("CA", ".", "GLY", 9, 1.0)]
    serine = [("N", "A", "SER", 10, 2.0), ("CA", "A", "SER", 10, 3.0), ("CB", "A", "SER", 10, 4.0),
              ("OG", "A", "SER", 10, 5.0)]
    threonine = [("N", "B", "THR", 10, 2.1), ("CA", "B", "THR", 10, 3.1), ("CB", "B", "THR", 10, 4.1),
                 ("OG1", "B", "THR", 10, 5.1), ("CG2", "B", "THR", 10, 6.1)]
    # One residue type in two conformations, the second with an atom the first lacks: of each atom name the first.
    lysine = [("N", ".", "LYS", 11, 7.0), ("CB", "A", "LYS", 11, 8.0), ("CB", "B", "LYS", 11, 8.1),
              ("CG", "B", "LYS", 11, 9.1)]
    as_one = [("A:9", "GLY", "N", 0.0), ("A:9", "GLY", "CA", 1.0)]
    as_serine = [("A:10", "SER", name, x) for name, _, _, _, x in serine]
    as_threonine = [("A:10", "THR", name, x) for name, _, _, _, x in threonine]
    as_lysine = [("A:11", "LYS", "N", 7.0), ("A:11", "LYS", "CB", 8.0), ("A:11", "LYS", "CG", 9.1)]

    st = reader(written(glycine + serine + threonine + lysine))
    assert _named(st) == as_one + as_serine + as_lysine
    assert [a.index for a in st.atoms] == list(range(len(st.atoms)))
    assert st.residue_name(ResidueKey("A", 10)) == "SER" and st.atom_by_name(ResidueKey("A", 10), "CG2") is None
    # The order of the file decides, not the letter, and not whether the two are written one after the other.
    assert _named(reader(written(glycine + threonine + serine + lysine))) == as_one + as_threonine + as_lysine
    mingled = [atom for pair in zip(serine, threonine) for atom in pair] + threonine[4:]
    assert _named(reader(written(glycine + mingled + lysine))) == as_one + as_serine + as_lysine
    # A third alternate location, of the first type again: the residue is one alternate location, the first.
    # An atom without an alternate location belongs to each of them.
    shared = [("C", ".", "SER", 10, 1.5)]
    third = [("CB", "C", "SER", 10, 4.2), ("OXT", "C", "SER", 10, 6.2)]
    st = reader(written(glycine + shared + serine + threonine + third + lysine))
    assert _named(st) == as_one + [("A:10", "SER", "C", 1.5)] + as_serine + as_lysine


@pytest.mark.parametrize("reader, written", [(parse_mmcif, _alt_cif), (parse_pdb, _alt_pdb)], ids=["mmcif", "pdb"])
def test_a_number_given_twice_is_no_alternate_location(reader, written):
    """A docked ligand that its program numbers like a residue of the protein's chain has no alternate-location
    id: no atom of either is left out."""
    glycine = [("N", ".", "GLY", 1, 0.0), ("CA", ".", "GLY", 1, 1.0)]
    ligand = [("C1", ".", "UNL", 1, 5.0), ("O1", ".", "UNL", 1, 6.0)]
    assert _named(reader(written(glycine + ligand))) == [
        ("A:1", "GLY", "N", 0.0), ("A:1", "GLY", "CA", 1.0), ("A:1", "UNL", "C1", 5.0), ("A:1", "UNL", "O1", 6.0)]


def test_values_after_the_semicolon_that_closes_a_text_field_are_read():
    """A text field ends with a semicolon at the start of a line; what follows on that line are further values."""
    text = ("data_x\nloop_\n_chem_comp.id\n_chem_comp.pdbx_smiles\n"
            "LIG\n;CCO\n; ALA 'N[C@@H](C)C(=O)O'\n"
            "GLY\n;NCC(=O)O\n;\n"
            + _label_only_cif(["label_atom_id", "label_comp_id", "label_asym_id", "label_seq_id", "Cartn_x", "Cartn_y",
                               "Cartn_z"], ["N GLY A 1 0.0 0.0 0.0"]).replace("data_x\n", ""))
    st = parse_mmcif(text)
    assert st.chem_comp_smiles == {"LIG": "CCO", "ALA": "N[C@@H](C)C(=O)O", "GLY": "NCC(=O)O"}
    assert [(a.name, a.resname) for a in st.atoms] == [("N", "GLY")]


def test_a_cut_or_incomplete_atom_site_loop_is_a_structure_error():
    whole = MINI_CIF.splitlines()
    assert len(parse_mmcif("\n".join(whole)).atoms) == 5
    # The file ends inside its last row: inside the row of an atom it has read before, and inside that of a new one.
    for last in (whole[-1], "HETATM 8 O O . HOH D . 4.0 4.0 4.0 1 D 1"):
        cut = "\n".join(whole[:-1] + [" ".join(last.split()[:10])]) + "\n"
        with pytest.raises(StructureError, match="_atom_site loop ends inside a row: its last row has 10 of 14 values"):
            parse_mmcif(cut)
    no_y = _label_only_cif(["label_atom_id", "label_comp_id", "label_asym_id", "label_seq_id", "Cartn_x"],
                           ["N GLY A 1 0.0"])
    with pytest.raises(StructureError, match="_atom_site has no Cartn_y, Cartn_z"):
        parse_mmcif(no_y, source="half.cif")


def test_a_file_with_a_byte_that_is_not_utf8_is_a_structure_error(tmp_path):
    """An old PDB file with the ångström sign of Latin-1 in a remark."""
    raw = b"REMARK   2 RESOLUTION.    1.80 \xc5NGSTROMS.\n" + _pdb_model(0.0).encode() + b"END\n"
    (tmp_path / "latin1.pdb").write_bytes(raw)
    with gzip.open(tmp_path / "latin1.pdb.gz", "wb") as handle:
        handle.write(raw)
    for name in ("latin1.pdb", "latin1.pdb.gz"):
        said = rf"cannot read .*{name}: it is not UTF-8 text \(it has the byte 0xc5\)"
        with pytest.raises(StructureError, match=said):
            read_structure(tmp_path / name)
        with pytest.raises(StructureError, match="not UTF-8"):
            read_models(tmp_path / name)
    (tmp_path / "utf8.pdb").write_bytes(raw.replace(b"\xc5", "Å".encode()))
    assert len(read_structure(tmp_path / "utf8.pdb").atoms) == 3


def test_a_model_number_given_as_text_is_that_number(tmp_path):
    pdb = _pdb_model(0.0, number=1) + _pdb_model(0.5, number=2) + "END\n"
    (tmp_path / "bundle.pdb").write_text(pdb)
    second = parse_mmcif(MINI_CIF, model="2")
    assert second.model == 2 and [a.name for a in second.atoms] == ["N"]
    assert parse_pdb(pdb, model=" 2 ").atoms[0].x == 1.5
    assert parse_pdb(pdb, model=2.0).model == 2                              # a whole number out of a table
    assert read_structure(tmp_path / "bundle.pdb", model="1").model == 1
    with pytest.raises(StructureError, match="there is no model 3: the file has 2 models, numbered 1, 2"):
        read_structure(tmp_path / "bundle.pdb", model="3")
    # Anything else is no number of a model, and is not taken for one.
    for reader, text in ((parse_mmcif, MINI_CIF), (parse_pdb, pdb)):
        for wrong in ("first", "all", 1.5, True):
            with pytest.raises(StructureError, match="model is the number of one model of the file"):
                reader(text, model=wrong)
    with pytest.raises(StructureError, match="bundle.pdb: model is the number of one model of the file, not 'all'"):
        read_structure(tmp_path / "bundle.pdb", model="all")


def test_a_pair_selects_a_residue_by_its_number_with_or_without_an_insertion_code():
    lines = [_pdb_atom(1, "N", "GLY", "A", 698, 0.0, "N"),
             _pdb_atom(2, "N", "ALA", "A", 698, 3.0, "N").replace(" 698 ", " 698A"),
             _pdb_atom(3, "C1", "LIG", "A", -2, 9.0, "C", hetero=True)]
    st = parse_pdb("\n".join(lines) + "\nEND\n")
    assert st.select_ligand(("A", "698A")) == ResidueKey("A", 698, "A") == st.select_ligand("A:698A")
    assert st.select_ligand(("A", 698)) == ResidueKey("A", 698) == st.select_ligand(("A", "698"))
    assert st.select_ligand(("A", "-2")) == ResidueKey("A", -2) == st.select_ligand(("A", -2))
    assert st.select_ligand(("A", 698.0)) == ResidueKey("A", 698)            # the number as a table gives it
    with pytest.raises(LigandNotFoundError, match="no residue A:698B"):
        st.select_ligand(("A", "698B"))
    for wrong in (("A", "the first"), ("A", 698, "A"), ("A",)):
        with pytest.raises(LigandNotFoundError, match="cannot parse ligand selection"):
            st.select_ligand(wrong)
