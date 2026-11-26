"""A peptide ligand that is a polymer chain of its own in the structure file.

Two kinds of test. The first takes a stretch of ten residues of a protein chain of a committed model
(human DHFR, ``dhfr_methotrexate``) and gives it a chain name of its own, which makes it what a deposited
peptide ligand is -- residues with names, numbers and ATOM records. The second runs on two deposited
peptides: ten residues of hirudin on thrombin (``thrombin_hirudin``, chain H, with a sulfated tyrosine)
and a nonamer in the groove of HLA-A2 (``mhc_peptide``, chain P).
"""

from __future__ import annotations

import math
import sys
from dataclasses import replace

import pytest
from rdkit import Chem

from conftest import ONE_LETTER, needs_plip, one_residue_peptide, structure_of
from bindview import Diagram, Polymer, SmallMolecule
from bindview.chains import KNOWN_COMPONENTS, chain_smiles
from bindview.distance_table import contact_table, ligand_contacts
from bindview.errors import DetectionUnavailableError, LigandNotFoundError
from bindview.structure import Atom, Structure
from scenetools import groups, words

# ---------------------------------------------------------------------------
# Without a deposited peptide: a stretch of DHFR as a chain of its own
# ---------------------------------------------------------------------------

SOURCE = "dhfr_methotrexate"
#: Residues 27 to 36 of chain A: Leu27 first, Arg28, Glu30, Tyr33 and Arg36 among them; Pro26 before and
#: Met37 after, all numbered without a gap or an insertion code.
FIRST, LAST = 27, 36


def stretch(first: int = FIRST, last: int = LAST, label: str = "", water_in_chain: bool = True) -> Structure:
    """The DHFR model with residues ``first``..``last`` of chain A as chain P."""
    st = structure_of(SOURCE)
    atoms = [replace(a, chain="P", label_chain=label) if a.chain == "A" and first <= a.resnum <= last and not a.hetero
             else a for a in st.atoms]
    if water_in_chain:
        # A deposited file puts a water under the chain it lies next to.
        ref = next(a for a in atoms if a.chain == "P")
        atoms.append(Atom(index=len(atoms), name="O", element="O", resname="HOH", chain="P", resnum=900, icode="",
                          x=ref.x + 30.0, y=ref.y, z=ref.z, hetero=True))
    return Structure(atoms=atoms, source="DHFR with a stretch as chain P")


def in_the_file(first: int = FIRST, last: int = LAST) -> list[tuple[int, str]]:
    """``(number, name)`` of the residues ``first``..``last`` of chain A, as the file has them."""
    seen: dict[int, str] = {}
    for a in structure_of(SOURCE).atoms:
        if a.chain == "A" and not a.hetero and first <= a.resnum <= last:
            seen.setdefault(a.resnum, a.resname)
    return sorted(seen.items())


def test_a_chain_is_selected_by_its_name_and_becomes_one_ligand():
    st = stretch()
    new, key, given, notes = st.resolve_ligand("P")
    assert notes == [] and str(key) == f"P:{FIRST}"
    assert [f"{r.name}{r.number}" for r in given] == [f"{name}{n}" for n, name in in_the_file()]
    assert [r.name for r in given][:2] == ["LEU", "ARG"] and len(given) == 10
    lig_atoms = new.residue_atoms(key)
    assert len(lig_atoms) == sum(len(r.atoms) for r in given)                 # the water is not part of it
    assert len({a.name for a in lig_atoms}) == len(lig_atoms) and "28/NE" in {a.name for a in lig_atoms}
    assert len(new.atoms) == len(st.atoms)                                    # nothing else changed
    assert st.resolve_ligand("chain:P")[1] == key
    # A residue name names one residue, which is no chain.
    same, lig, nothing, _ = structure_of(SOURCE).resolve_ligand("MTX")
    assert nothing == () and same.residue_name(lig) == "MTX"


def test_the_protein_is_not_a_ligand_and_the_chain_is_never_guessed():
    st = structure_of(SOURCE)
    with pytest.raises(LigandNotFoundError, match="one side of an interface"):
        st.resolve_ligand("A")
    with pytest.raises(LigandNotFoundError):
        st.resolve_ligand("Z")


def test_a_chain_known_only_by_its_label_is_found_and_said():
    st = stretch(label="X")
    new, key, given, notes = st.resolve_ligand("X")
    assert str(key) == f"P:{FIRST}" and len(given) == 10
    assert notes == ["chain 'X' is the file's label_asym_id; its authors call the chain 'P', and so does the figure"]


def test_two_heavy_atoms_cannot_be_an_angstrom_apart():
    st = stretch()
    cg, cd = (next(a for a in st.atoms if a.chain == "P" and a.resnum == FIRST and a.name == n) for n in ("CG", "CD1"))
    # 0.92 A beyond CD1, on the line from CG through it: close to CD1 only.
    away = [(q - p) * 0.92 / math.dist(cg.xyz, cd.xyz) for p, q in zip(cg.xyz, cd.xyz)]
    bogus = replace(cd, index=len(st.atoms), name="CX", x=cd.x + away[0], y=cd.y + away[1], z=cd.z + away[2])
    st2 = Structure(atoms=st.atoms + [bogus], source="with an atom that cannot be")
    new, key, given, notes = st2.resolve_ligand("P")
    assert len(notes) == 1 and notes[0].startswith(f"atom CX of LEU {FIRST} lies 0.9") and "was left out" in notes[0]
    assert f"{FIRST}/CX" not in given[0].atoms and len(new.atoms) == len(st2.atoms) - 1


def test_bond_orders_come_from_the_names_of_the_residues():
    st = stretch()
    new, key, given, _ = st.resolve_ligand("P")
    smiles, notes = chain_smiles(new.residue_atoms(key), given)
    assert notes == [] and smiles is not None
    mol = Chem.MolFromSmiles(smiles)
    for group in ("c1ccc(O)cc1", "NC(=N)N", "CC(=O)[OH]"):                    # Tyr33's phenol, an arginine, Glu30
        assert mol.HasSubstructMatch(Chem.MolFromSmarts(group)), group
    dia = Diagram.from_structure(st, "P", interactions=[])
    assert dia.ligand.bond_source == "template" and dia.ligand.given == given and dia.ligand.chain == "P"
    assert not dia.ligand.stereo_mismatches
    # A component whose bonds are not known here is said, and no bond order is invented.
    odd = tuple(replace(r, name="ZZZ") if r.number == "30" else r for r in given)
    smiles, notes = chain_smiles(new.residue_atoms(key), odd)
    assert smiles is None and notes[0].startswith("ZZZ 30 is not a component whose bonds are known here")
    assert {"PTR", "TYS", "SEP", "ACE", "NH2"} <= set(KNOWN_COMPONENTS)


#: Two contacts made up for the stretch: the carbonyl of Pro26 on the nitrogen of Leu27, and Met37 on Arg36.
RECORDS = [
    {"interaction_type": "hbond", "residue_number": FIRST - 1, "residue_name": "PRO", "chain": "A", "protein_atom": "O",
     "ligand_atom": f"{FIRST}/N", "distance": 2.25},
    {"interaction_type": "hydrophobic", "residue_number": LAST + 1, "residue_name": "MET", "chain": "A",
     "protein_atom": "CA", "ligand_atom": f"{LAST}/CG", "distance": 3.9},
]


def test_a_given_chain_is_drawn_with_the_names_and_numbers_of_the_file():
    dia = Diagram.from_structure(stretch(), "P", interactions=RECORDS)
    backend = dia.backend
    assert isinstance(backend, Polymer) and backend.deposited
    assert backend.decision.reason == "it is a chain of 10 residues in the structure file, 10 of them proteinogenic"
    pep = backend.peptide
    assert pep.numbering == "deposited" and not pep.appendages and not pep.breaks
    assert pep.sequence == "".join(ONE_LETTER[name] for _, name in in_the_file())
    assert [r.shown for r in pep.residues] == [str(n) for n in range(FIRST, LAST + 1)]
    assert [r.position for r in pep.residues] == list(range(1, 11))
    assert all(r.stereo in ("L", None) for r in pep.residues)
    res = dia.render(style="schematic")
    assert res.status == "ok" and res.figure == "diagram"
    units = groups(res, "unit")
    assert [g.title for g in units][:2] == ["Leu 27 of the ligand (numbered as in the structure file)",
                                           "Arg 28 of the ligand (numbered as in the structure file)"]
    text = words(res)
    assert "Residue of the ligand, numbered as in the structure file" in text
    assert "Residues of the ligand carry the names and numbers of the structure file (chain P)." in text
    assert "counted from its N-terminus" not in text and "own N-terminus" not in text
    graph = res.graph
    nodes = [n for n in graph.nodes if n.attrs.get("form") == "chain_residue"]
    assert [n.label for n in nodes][:3] == ["Leu27", "Arg28", "Asn29"]
    assert [(n.attrs["position"], n.attrs["number"]) for n in nodes][:2] == [(1, "27"), (2, "28")]
    assert graph.meta["chain"]["numbering_origin"] == "deposited"
    assert sorted(c.ligand_residue for c in graph.contacts) == ["Arg36", "Leu27"]
    assert res.data["ligand"]["numbering"] == "as deposited in the structure file"


def test_a_perceived_chain_still_says_that_its_numbers_are_a_count():
    """A peptide written as one hetero residue (the nonamer of ``mhc_peptide`` with its SMILES) has no
    numbers of its own: its residues are counted from its N-terminus, and the figure says so."""
    st, smiles = one_residue_peptide("mhc_peptide")
    dia = Diagram.from_structure(st, "LIG", smiles=smiles, interactions=ligand_contacts(st, "LIG", smiles))
    res = dia.render(style="schematic")
    assert res.graph.meta["chain"]["numbering_origin"] == "derived"
    assert "numbered from its own N-terminus" in words(res)
    assert all("number" not in n.attrs for n in res.graph.nodes if n.attrs.get("form") == "chain_residue")


def test_a_short_chain_is_drawn_as_atoms_and_a_gap_is_not_bridged():
    short = Diagram.from_structure(stretch(FIRST, FIRST + 4), "P", interactions=RECORDS[:1])
    assert isinstance(short.backend, SmallMolecule)
    assert short.backend.decision.reason == "its chain has 5 residues in the structure file, fewer than 8"
    assert short.render().status == "ok"
    forced = Diagram.from_structure(stretch(FIRST, FIRST + 4), "P", interactions=RECORDS[:1], ligand_as="chain")
    assert isinstance(forced.backend, Polymer) and forced.backend.deposited
    # Residues missing from the model: the chain goes on, the backbone is not drawn across the gap.
    st = stretch(FIRST, FIRST + 11)
    gapped = Structure(atoms=[replace(a, index=i) for i, a in enumerate(
        a for a in st.atoms if not (a.chain == "P" and a.resnum in (FIRST + 5, FIRST + 6)))], source="with a gap")
    dia = Diagram.from_structure(gapped, "P", interactions=RECORDS[:1])
    pep = dia.backend.peptide
    assert pep.breaks == (5,) and pep.runs == (5, 5) and len(pep) == 10
    dep = dia.backend.depict(__import__("bindview").theme.Metrics())
    assert sum(1 for b in dep.bonds if b.kind == "chain") == 8


def test_without_a_table_and_without_plip_a_chain_is_refused_in_words(monkeypatch):
    """Without a table the contacts of a chain are PLIP's (its peptide mode). Where PLIP cannot be imported,
    the error says that it is what is missing."""
    for name in {n for n in sys.modules if n == "plip" or n.startswith("plip.")} | {"plip", "plip.basic",
                                                                                     "plip.structure.preparation"}:
        monkeypatch.setitem(sys.modules, name, None)
    with pytest.raises(DetectionUnavailableError, match="needs PLIP"):
        Diagram.from_structure(stretch(), "P")


@needs_plip
def test_without_a_table_plip_finds_the_contacts_of_a_chain():
    dia = Diagram.from_structure(stretch(), "P")
    assert dia.source == "detected" and dia.interactions and dia.backend.kind == "chain"


def test_command_line_takes_a_chain(tmp_path, capsys):
    from bindview.cli import main as cli

    st = stretch()
    lines = ["data_x", "loop_"] + ["_atom_site." + c for c in (
        "group_PDB", "type_symbol", "label_atom_id", "label_comp_id", "label_asym_id", "label_seq_id",
        "Cartn_x", "Cartn_y", "Cartn_z", "pdbx_PDB_model_num")]
    for a in st.atoms:
        if a.element in ("H", "D"):
            continue
        lines.append(f"{'HETATM' if a.hetero else 'ATOM'} {a.element} {a.name} {a.resname} {a.chain} {a.resnum} "
                     f"{a.x:.3f} {a.y:.3f} {a.z:.3f} 1")
    cif = tmp_path / "model.cif"
    cif.write_text("\n".join(lines) + "\n")
    tsv = tmp_path / "contacts.tsv"
    tsv.write_text("interaction_type\tresidue_number\tresidue_name\tchain\tprotein_atom\tligand_atom\tdistance\n"
                   f"hbond\t{FIRST - 1}\tPRO\tA\tO\t{FIRST}/N\t2.25\n")
    out = tmp_path / "out.svg"
    assert cli(["render", str(cif), "--ligand", "P", "--interactions", str(tsv), "-o", str(out),
                "--style", "schematic"]) == 0
    assert "numbered as in the structure file" in out.read_text()


# ---------------------------------------------------------------------------
# Two deposited peptides
# ---------------------------------------------------------------------------

EXPECTED = {
    # case: (the authors' chain, the file's label_asym_id of the peptide, its residues as the file names them)
    "mhc_peptide": ("P", "C", "Gly1 Leu2 Cys3 Thr4 Leu5 Val6 Ala7 Met8 Leu9"),
    "thrombin_hirudin": ("H", "C", "Asp55 Phe56 Glu57 Glu58 Ile59 Pro60 Glu61 Glu62 Tys63 Leu64"),
}


def deposited(slug: str, name: str | None = None) -> tuple[Diagram, list]:
    st = structure_of(slug)
    rows = contact_table(st, EXPECTED[slug][0])
    return Diagram.from_structure(st, name or EXPECTED[slug][0], interactions=rows), rows


@pytest.mark.parametrize("slug", sorted(EXPECTED))
def test_a_deposited_peptide_is_a_chain_with_the_names_and_numbers_of_its_file(slug):
    chain, label, residues = EXPECTED[slug]
    dia, rows = deposited(slug)
    assert dia.backend.kind == "chain" and dia.ligand.bond_source == "template" and dia.ligand.chain == chain
    assert rows and len(rows) == len(dia.interactions) and not dia.unresolved
    pep = dia.backend.peptide
    assert " ".join(f"{r.code3}{r.shown}" for r in pep.residues) == residues
    assert pep.numbering == "deposited" and not pep.appendages
    res = dia.render(style="schematic")
    assert res.status == "ok" and res.figure == "diagram" and res.quality.valid
    assert "numbered as in the structure file" in words(res)
    # Named as the mmCIF's label names the chain: its authors call it otherwise, and the figure follows them.
    by_label, _ = deposited(slug, label)
    assert by_label.ligand.chain == chain and any("label_asym_id" in w for w in by_label.warnings)


def test_the_other_chains_are_the_environment():
    """Thrombin's light chain (chain A, 28 residues) is short, and no ligand: the peptide's contacts are with
    the heavy chain. Beta-2 microglobulin is no second ligand of the nonamer either."""
    dia, rows = deposited("thrombin_hirudin")
    # 94: the heavy atoms of the ten residues of chain H in the file, the sulfate of Tys63 among them.
    assert {r["chain"] for r in rows} == {"B"} and dia.ligand.n_atoms == 94
    mhc, rows = deposited("mhc_peptide")
    assert {r["chain"] for r in rows} <= {"A", "B"} and "A" in {r["chain"] for r in rows}
    assert mhc.ligand.chain == "P" and len(mhc.backend.peptide) == 9


def test_a_sulfotyrosine_is_a_residue_of_the_chain():
    dia, _ = deposited("thrombin_hirudin")
    tys = dia.backend.peptide.residues[8]
    # 16: the heavy atoms of TYS 63 in the file, a tyrosine's twelve and the sulfate's four.
    assert (tys.name, tys.shown, tys.proteinogenic, len(tys.atoms)) == ("TYS", "63", False, 16)
    res = dia.render()
    assert res.status == "ok"
    assert "Tys 63 of the ligand (numbered as in the structure file)" in [g.title for g in groups(res, "unit")]


def test_a_partner_chain_that_is_not_there_is_refused():
    """The table of a chain against a chain the model does not have came back without a row, which reads as:
    the two do not touch."""
    st = structure_of("mhc_peptide")
    assert contact_table(st, "P", partners=["A"])
    with pytest.raises(LigandNotFoundError, match="no chain 'Z' in .* to be a partner of chain 'P': its chains are 'A'"):
        contact_table(st, "P", partners=["A", "Z"])


def test_an_atom_too_close_to_two_others_is_left_out_once():
    """An atom less than an angstrom from two atoms was left out twice, with a note for each."""
    st = stretch()
    cg, cd1 = (next(a for a in st.atoms if a.chain == "P" and a.resnum == FIRST and a.name == n) for n in ("CG", "CD1"))
    middle = [(p + q) / 2.0 for p, q in zip(cg.xyz, cd1.xyz)]      # (half a bond from each)
    assert math.dist(middle, cg.xyz) < 1.05 and math.dist(middle, cd1.xyz) < 1.05
    bogus = replace(cd1, index=len(st.atoms), name="CX", x=middle[0], y=middle[1], z=middle[2])
    new, key, given, notes = Structure(atoms=st.atoms + [bogus], source="with an atom between two").resolve_ligand("P")
    assert len(notes) == 1 and notes[0].startswith(f"atom CX of LEU {FIRST}") and len(new.atoms) == len(st.atoms)

