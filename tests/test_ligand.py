"""Building the named-atom ligand from the atoms of a structure and a SMILES."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pytest
from rdkit import Chem

from conftest import AP5A, BIOEMU, CASES, diagram_of, meta_of, structure_of
from bindview.depict import depict_ligand, layout_score
from bindview.errors import LigandBuildError
from bindview.interactions import load_interactions
from bindview.ligand import build_ligand
from bindview.structure import Atom, read_structure
from bindview.theme import Metrics


@pytest.mark.parametrize("case", CASES)
def test_every_named_ligand_atom_resolves(case, testdata):
    """The whole point of building from the structure file: the names a table gives land on atoms, and the
    ligand has as many heavy atoms as the archive's record of its component."""
    lig = diagram_of(case).ligand
    named = [r.ligand_atom for r in load_interactions(testdata / case / "interactions.tsv") if r.ligand_atom]
    missing = [n for n in named if lig.index_of(n) is None]
    assert named and not missing
    assert lig.n_atoms == meta_of(case)["ligand"]["heavy_atoms"]


@pytest.mark.parametrize("case", CASES)
def test_built_molecule_is_the_template_molecule(case):
    """Graph identity with the SMILES, ignoring the display-only ionisation and stereo."""
    lig = diagram_of(case).ligand
    assert lig.bond_source == "template"
    built = Chem.RWMol(lig.mol)
    for i in lig.ionized:                       # undo the display ionisation before comparing
        a = built.GetAtomWithIdx(i)
        a.SetFormalCharge(0)
        a.SetNumExplicitHs(1)
    Chem.SanitizeMol(built)
    want = Chem.MolFromSmiles(meta_of(case)["ligand"]["smiles"])
    assert Chem.MolToSmiles(built, isomericSmiles=False) == Chem.MolToSmiles(want, isomericSmiles=False)


def test_atom_names_survive_on_the_molecule():
    lig = diagram_of("ar_bicalutamide").ligand
    for i, name in enumerate(lig.names):
        assert lig.mol.GetAtomWithIdx(i).GetProp("atomName") == name
    assert lig.element(lig.index_of("F7B")) == "F"


@lru_cache(maxsize=1)
def _ap5a_atoms() -> "tuple[Atom, ...]":
    """The atoms of AP5A in the first of the twenty BioEmu samples."""
    model = read_structure(BIOEMU)
    return tuple(model.residue_atoms(model.select_ligand("AP5")))


def test_clash_bonds_from_the_model_are_dropped_and_reported():
    """The first BioEmu sample puts two phosphorus atoms of AP5A, PD and PE, 2.48 Å apart: at the length of a
    bond, and not bonded in the SMILES. The pair is no bond of the molecule, and the ligand says so."""
    at = {a.name: np.array(a.xyz) for a in _ap5a_atoms()}
    assert np.linalg.norm(at["PD"] - at["PE"]) == pytest.approx(2.48, abs=0.005)       # read from the model
    lig = build_ligand(list(_ap5a_atoms()), AP5A)
    pairs = {tuple(sorted((lig.names[i], lig.names[j]))) for i, j in lig.artefact_bonds}
    assert pairs == {("PD", "PE")}
    for i, j in lig.artefact_bonds:
        assert lig.mol.GetBondBetweenAtoms(i, j) is None
    assert any("clash" in d and "PD" in d and "PE" in d for d in lig.diagnostics)


#: Protoporphyrin IX and an iron(II) ion apart from it: the haem as a SMILES that bonds nothing to its metal.
_PORPHYRIN_AND_IRON = ("Cc1c(CCC(=O)O)c2cc3[nH]c(cc4nc(cc5[nH]c(cc1n2)c(C)c5C=C)c(C)c4C=C)c(C)c3CCC(=O)O"
                       ".[Fe+2]")


def test_metal_coordination_is_kept_apart_from_covalent_bonds():
    """The haem of haemoglobin with a SMILES that has its iron as an ion of its own: the iron has no bond, and
    the four nitrogens of the porphyrin that hold it are its coordination."""
    st = structure_of("hb_carbon_monoxide")
    lig = build_ligand(st.residue_atoms(st.select_ligand("A:142")), _PORPHYRIN_AND_IRON)
    assert lig.bond_source == "template" and lig.n_atoms == 43
    fe = lig.index_of("FE")
    assert lig.mol.GetAtomWithIdx(fe).GetDegree() == 0
    donors = sorted(lig.names[d] for m, d in lig.coordination if m == fe)
    assert donors == ["NA", "NB", "NC", "ND"]                     # the haem's four nitrogens, by their names
    assert all(lig.element(d) == "N" for _, d in lig.coordination)


def test_stereo_differences_between_model_and_smiles_are_reported():
    """Bicalutamide has one stereocentre. With the SMILES of its mirror image the model and the SMILES differ
    there, and the ligand says so; with its own SMILES they agree."""
    dia = diagram_of("ar_bicalutamide")
    assert dia.ligand.stereo_mismatches == []
    smiles = meta_of("ar_bicalutamide")["ligand"]["smiles"]
    mirror = smiles.replace("[C@]", "[C@@]")
    assert mirror != smiles
    lig = build_ligand(dia.structure.residue_atoms(dia.ligand_key), mirror)
    assert lig.bond_source == "template" and len(lig.stereo_mismatches) == 1
    assert any("stereocentre" in d for d in lig.diagnostics)


def test_phosphates_carry_no_pseudo_stereo():
    lig = build_ligand(list(_ap5a_atoms()), AP5A)
    phosphorus = [a for a in lig.mol.GetAtoms() if a.GetSymbol() == "P"]
    assert len(phosphorus) == 5
    for a in phosphorus:
        assert a.GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED


def test_acids_are_ionised_for_display_and_only_acids():
    """Methotrexate has two carboxylic acids: one oxygen of each is drawn charged. Imatinib's amines are bases,
    and are left as the SMILES has them."""
    lig = diagram_of("dhfr_methotrexate").ligand
    charged = [a for a in lig.mol.GetAtoms() if a.GetFormalCharge() != 0]
    assert len(charged) == 2 and all(a.GetSymbol() == "O" and a.GetFormalCharge() == -1 for a in charged)
    assert sorted(lig.ionized) == sorted(a.GetIdx() for a in charged)
    assert all(a.GetFormalCharge() == 0 for a in diagram_of("kit_imatinib").ligand.mol.GetAtoms())


def test_ionisation_can_be_switched_off():
    st = structure_of("dhfr_methotrexate")
    lig = build_ligand(st.residue_atoms(st.select_ligand("MTX")), meta_of("dhfr_methotrexate")["ligand"]["smiles"],
                       ionize_acids=False)
    assert all(a.GetFormalCharge() == 0 for a in lig.mol.GetAtoms()) and lig.ionized == []


def test_single_atom_ligand_takes_its_charge_from_the_smiles():
    """The zinc of carbonic anhydrase II: with a SMILES it has that charge, without one it claims none."""
    st = structure_of("ca2_celecoxib")
    atoms = st.residue_atoms(st.select_ligand("ZN"))
    lig = build_ligand(atoms, "[Zn+2]")
    assert lig.bond_source == "single-atom" and lig.n_atoms == 1
    assert lig.mol.GetAtomWithIdx(0).GetFormalCharge() == 2
    assert build_ligand(atoms).mol.GetAtomWithIdx(0).GetFormalCharge() == 0


def _benzene_atoms(spacing: float = 1.39) -> list[Atom]:
    out = []
    for k in range(6):
        ang = np.pi / 3 * k
        out.append(Atom(k, f"C{k + 1}", "C", "BNZ", "L", 1, "", spacing * np.cos(ang), spacing * np.sin(ang), 0.0, True))
    return out


def _benzene_hydrogens() -> list[Atom]:
    return [Atom(6 + k, f"H{k + 1}", "H", "BNZ", "L", 1, "", 2.48 * np.cos(np.pi / 3 * k),
                 2.48 * np.sin(np.pi / 3 * k), 0.0, True) for k in range(6)]


def test_without_template_or_hydrogens_only_connectivity_is_claimed():
    """Valence-based bond orders on heavy atoms alone are wrong (benzene gets triple bonds),
    so they are not attempted: the molecule is drawn as connectivity and flagged."""
    lig = build_ligand(_benzene_atoms())
    assert lig.bond_source == "connectivity"
    assert all(b.GetBondTypeAsDouble() == 1.0 for b in lig.mol.GetBonds()) and lig.mol.GetNumBonds() == 6
    assert all(a.GetTotalNumHs() == 0 for a in lig.mol.GetAtoms())
    assert any("only the connectivity is known" in d for d in lig.diagnostics)


def test_without_template_but_with_hydrogens_bond_orders_come_from_geometry():
    lig = build_ligand(_benzene_atoms() + _benzene_hydrogens())
    assert lig.bond_source == "geometry" and lig.n_atoms == 6
    assert Chem.MolToSmiles(lig.mol) == "c1ccccc1"
    assert lig.names == [f"C{k}" for k in range(1, 7)]
    assert any("inferred from the 3D geometry" in d for d in lig.diagnostics)


def test_hydrogens_in_the_structure_are_ignored_when_a_template_is_given():
    lig = build_ligand(_benzene_atoms() + _benzene_hydrogens(), "c1ccccc1")
    assert lig.bond_source == "template" and lig.n_atoms == 6
    assert all(a.GetTotalNumHs() == 1 for a in lig.mol.GetAtoms())


def test_a_template_for_a_different_molecule_is_not_silently_used():
    lig = build_ligand(_benzene_atoms(), "CCO")
    assert lig.bond_source != "template"
    assert any("does not match" in d for d in lig.diagnostics)


def test_counter_ions_in_the_smiles_are_ignored():
    lig = build_ligand(_benzene_atoms(), "c1ccccc1.[Na+].[Cl-]")
    assert lig.bond_source == "template" and lig.n_atoms == 6
    assert any("not present in the structure" in d for d in lig.diagnostics)


def test_a_smiles_with_atom_map_numbers_is_the_molecule_without_them():
    """A SMILES out of a reaction or a matched pair numbers its atoms (``[CH3:1]``). The numbers say which atom is
    which there, and nothing about the molecule."""
    plain = build_ligand(_benzene_atoms(), "c1ccccc1")
    mapped = build_ligand(_benzene_atoms(), "[cH:1]1[cH:2]ccc[cH:6]1")
    assert mapped.bond_source == "template" and mapped.diagnostics == []
    assert Chem.MolToSmiles(mapped.mol) == Chem.MolToSmiles(plain.mol) == "c1ccccc1"
    # A ligand of the archive: imatinib, with its two methyl groups and an amide nitrogen numbered.
    dia = diagram_of("kit_imatinib")
    smiles = meta_of("kit_imatinib")["ligand"]["smiles"]
    numbered = "[CH3:1]" + smiles[1:-1].replace(")NC(=O)c4", ")[NH:7]C(=O)c4") + "[CH3:2]"
    atoms = dia.structure.residue_atoms(dia.ligand_key)
    mapped = build_ligand(atoms, numbered)
    assert mapped.bond_source == "template" and mapped.names == dia.ligand.names
    assert Chem.MolToSmiles(mapped.mol) == Chem.MolToSmiles(dia.ligand.mol)
    assert all(a.GetAtomMapNum() == 0 for a in mapped.mol.GetAtoms())


#: 4-hydroxycyclohexanone. Its graph without bond orders has a mirror that the molecule does not have: the
#: SMILES fits the atoms with the ketone on either oxygen.
_KETOL = "O=C1CCC(O)CC1"


def _ketol_atoms(reverse: bool = False, both: "float | None" = None, hydrogens: bool = False) -> list[Atom]:
    """The ketol as a force field leaves it (C=O 1.23 Å, C-OH 1.42 Å), its oxygens named for what they are in
    the coordinates: ``OK`` the ketone's, ``OH`` the hydroxyl's. ``reverse``: the atoms in the opposite order,
    as another file may have them. ``both``: the two oxygens at this distance from their carbons, so that the
    bond lengths say nothing. ``hydrogens``: with the hydrogens, as a file with hydrogens has them."""
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles(_KETOL))
    assert AllChem.EmbedMolecule(mol, randomSeed=11) == 0
    AllChem.MMFFOptimizeMolecule(mol)
    xyz = np.array(mol.GetConformer().GetPositions())
    names = {}
    for a in mol.GetAtoms():
        if a.GetSymbol() == "O":
            carbon = next(n.GetIdx() for n in a.GetNeighbors() if n.GetSymbol() == "C")
            names[a.GetIdx()] = "OH" if a.GetDegree() == 2 else "OK"
            if both is not None:
                away = xyz[a.GetIdx()] - xyz[carbon]
                xyz[a.GetIdx()] = xyz[carbon] + away / np.linalg.norm(away) * both
        else:
            names[a.GetIdx()] = f"{a.GetSymbol()}{a.GetIdx()}"
    order = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() != "H"][::-1 if reverse else 1]
    order += [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "H"] if hydrogens else []
    return [Atom(k, names[i], mol.GetAtomWithIdx(i).GetSymbol(), "LIG", "L", 1, "", *map(float, xyz[i]), True)
            for k, i in enumerate(order)]


def _ketol_oxygens(lig) -> dict[str, tuple[str, int]]:
    """Of each oxygen of the ketol as it was built: the order of its bond, and its hydrogens."""
    atoms = {name: lig.mol.GetAtomWithIdx(lig.index_of(name)) for name in ("OK", "OH")}
    return {name: (str(a.GetBonds()[0].GetBondType()), a.GetTotalNumHs()) for name, a in atoms.items()}


@pytest.mark.parametrize("reverse", [False, True])
def test_a_smiles_lies_on_the_atoms_as_the_bond_lengths_of_the_model_say(reverse):
    """Of the two ways the SMILES of the ketol fits its atoms, the model's own bonds say which is the molecule
    (1.23 Å against 1.42 Å); the order of the atoms in the file says nothing."""
    lig = build_ligand(_ketol_atoms(reverse), _KETOL)
    assert lig.bond_source == "template" and lig.diagnostics == []
    assert _ketol_oxygens(lig) == {"OK": ("DOUBLE", 0), "OH": ("SINGLE", 1)}


@pytest.mark.parametrize("reverse", [False, True])
def test_the_hydrogens_of_the_file_say_how_a_smiles_lies_where_the_bond_lengths_do_not(reverse):
    lig = build_ligand(_ketol_atoms(reverse, both=1.33, hydrogens=True), _KETOL)
    assert lig.n_atoms == 8 and lig.diagnostics == []
    assert _ketol_oxygens(lig) == {"OK": ("DOUBLE", 0), "OH": ("SINGLE", 1)}


def test_a_smiles_that_fits_two_ways_which_the_model_does_not_tell_apart_is_said():
    """Both oxygens 1.33 Å from their carbons and no hydrogen in the file: which is the ketone is not in the
    model. One of the two is drawn, and the ligand says so and names the atoms."""
    for reverse in (False, True):
        lig = build_ligand(_ketol_atoms(reverse, both=1.33), _KETOL)
        said = [d for d in lig.diagnostics if "fits the model's atoms in more than one way" in d]
        assert len(lig.diagnostics) == 1 and len(said) == 1
        assert "OK" in said[0] and "OH" in said[0]
        assert sorted(_ketol_oxygens(lig).values()) == [("DOUBLE", 0), ("SINGLE", 1)]


def _acetic_acid(first: float, second: float, hydrogen_on: "str | None" = None) -> list[Atom]:
    """Acetic acid in a plane, its two oxygens ``O1`` and ``O2`` at these distances from the carbon; with
    ``hydrogen_on``, the hydrogen of the hydroxyl on that oxygen, as a file with hydrogens has it."""
    reach = {"O1": (first, 0.866), "O2": (second, -0.866)}
    spot = {"C1": (-1.5, 0.0), "C2": (0.0, 0.0), **{name: (r * 0.5, r * up) for name, (r, up) in reach.items()}}
    if hydrogen_on:
        r, up = reach[hydrogen_on]
        spot["H1"] = ((r + 0.97) * 0.5, (r + 0.97) * up)
    return [Atom(k, name, name[0], "ACY", "L", 1, "", x, y, 0.0, True) for k, (name, (x, y)) in enumerate(spot.items())]


def _with_the_double_bond(lig) -> str:
    return next(lig.names[b.GetOtherAtomIdx(lig.index_of("C2"))] for b in lig.mol.GetBonds()
                if b.GetBondTypeAsDouble() == 2.0)


def test_the_ends_of_one_group_stay_where_they_are_whatever_their_bond_lengths():
    """The two oxygens of a carboxylate are one group. A model has one of them nearer its carbon than the other
    by a few hundredths of an ångström, a rough model by more than a tenth, and that says nothing: the SMILES
    lies as it does on two equal bonds, and nothing is said."""
    built = [build_ligand(_acetic_acid(first, second), "CC(=O)O")
             for first, second in ((1.25, 1.25), (1.24, 1.27), (1.27, 1.24), (1.19, 1.34), (1.34, 1.19))]
    assert len({_with_the_double_bond(lig) for lig in built}) == 1
    assert all(lig.bond_source == "template" and lig.diagnostics == [] for lig in built)


@pytest.mark.parametrize("hydroxyl, carbonyl", [("O1", "O2"), ("O2", "O1")])
def test_the_hydrogen_of_the_file_says_which_end_of_a_group_has_it(hydroxyl, carbonyl):
    lig = build_ligand(_acetic_acid(1.25, 1.25, hydrogen_on=hydroxyl), "CC(=O)O", ionize_acids=False)
    assert lig.n_atoms == 4 and lig.diagnostics == []
    assert _with_the_double_bond(lig) == carbonyl
    assert lig.mol.GetAtomWithIdx(lig.index_of(hydroxyl)).GetTotalNumHs() == 1


def test_bad_input_raises_build_errors():
    with pytest.raises(LigandBuildError, match="no heavy atoms"):
        build_ligand([])
    dup = _benzene_atoms()
    dup[1] = Atom(1, "C1", "C", "BNZ", "L", 1, "", 9.0, 9.0, 9.0, True)
    with pytest.raises(LigandBuildError, match="not unique"):
        build_ligand(dup)
    with pytest.raises(LigandBuildError, match="does not parse"):
        build_ligand(_benzene_atoms(), "this is not smiles")


@pytest.mark.parametrize("case", CASES)
def test_ligand_depiction_has_no_clashes_or_crossed_bonds(case):
    """No two atoms of a ligand's drawing lie on one another, and no two of its bonds cross."""
    lig = diagram_of(case).ligand
    dep = depict_ligand(lig, Metrics())
    xy = np.array([[a.x, a.y] for a in dep.atoms[: dep.n_heavy]]) / dep.metrics.bond_length
    assert layout_score(lig.mol, xy) == (0, 0)


def test_depiction_keeps_names_and_adds_only_stereo_hydrogens():
    lig = diagram_of("ar_r1881").ligand
    dep = depict_ligand(lig, Metrics())
    assert [a.name for a in dep.atoms[: dep.n_heavy]] == lig.names
    extra = dep.atoms[dep.n_heavy:]
    assert extra and all(a.element == "H" and a.name is None for a in extra)


def test_caller_supplied_coordinates_are_used():
    lig = build_ligand(_benzene_atoms(), "c1ccccc1")
    coords = {f"C{k + 1}": (float(np.cos(np.pi / 3 * k)) * 5, float(np.sin(np.pi / 3 * k)) * 5) for k in range(6)}
    dep = depict_ligand(lig, Metrics(), coords=coords)
    assert dep.coord_source == "supplied"
    c1 = dep.atoms[0]
    assert (c1.x, c1.y) == pytest.approx((Metrics().bond_length, 0.0), abs=1e-6)     # rescaled to one bond length
    with pytest.raises(ValueError, match="lack atoms"):
        depict_ligand(lig, Metrics(), coords={"C1": (0, 0)})


def test_the_package_is_imported_without_rdkit_s_drawing_module():
    """RDKit's wheel links its drawing module against libXrender and libexpat, which a minimal container image
    does not have: the module is loaded only where a ligand is drawn. (In a process of its own: here the module
    is long loaded.)"""
    import os
    import subprocess
    import sys

    code = "import sys, bindview; print(sorted(m for m in sys.modules if m.startswith('rdkit.Chem.Draw')))"
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=600)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "[]"


def test_a_system_library_that_is_missing_is_named_where_a_ligand_is_drawn(monkeypatch):
    """The loader's message alone does not say what to do; the error names the two libraries and their packages."""
    from conftest import fresh_diagram

    from bindview import depict
    from bindview.errors import BindviewError

    def missing():
        raise ImportError("libXrender.so.1: cannot open shared object file: No such file or directory")

    monkeypatch.setattr(depict, "_load_rdkit_drawing", missing)
    with pytest.raises(BindviewError, match="libXrender.so.1: cannot open shared object file.*libxrender1 libexpat1"):
        fresh_diagram("kit_imatinib").render()


#: The haem as the PDB's dictionary of components writes it (HEM): the iron with four bonds like any other.
_HAEM = "Cc1c2n3c(c1CCC(=O)O)C=C4C(=C(C5=[N]4[Fe]36[N]7=C(C=C8N6C(=C5)C(=C8C)C=C)C(=C(C7=C2)C)C=C)C)CCC(=O)O"


def test_a_smiles_that_bonds_a_metal_like_any_atom_is_refused_with_what_to_do():
    """Two of the haem's nitrogens get a valence of four from that SMILES. The refusal names the metal and what
    draws the ligand instead; without a SMILES it is drawn from its connectivity, and the figure says so."""
    from bindview import Diagram

    st = structure_of("hb_carbon_monoxide")
    with pytest.raises(LigandBuildError, match=r"holds a metal \(Fe\): a SMILES that writes the bonds to a metal.*"
                                               r"Without a SMILES the ligand is drawn from its connectivity alone"):
        Diagram.from_structure(st, "A:142", smiles=_HAEM, interactions="distances")
    drawn = Diagram.from_structure(st, "A:142", interactions="distances")
    figure = drawn.render()
    assert drawn.ligand.bond_source == "connectivity" and figure.status == "degraded"
    assert any("only the connectivity is known" in w for w in figure.warnings)

