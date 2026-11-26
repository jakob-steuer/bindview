"""Every interaction attaches to the thing that makes it: an atom, a ring, a charged group or a metal, and never
to somewhere on the ligand."""

from __future__ import annotations

import collections
import csv

import pytest

from conftest import CASES, TESTDATA, diagram_of, structure_of
from bindview import Diagram, Interaction, InteractionType
from bindview.features import perceive_charged_groups
from bindview.resolve import resolve_interactions

T = InteractionType

#: Protoporphyrin IX and an iron(II) ion apart from it: the haem as a SMILES that bonds nothing to its metal.
_PORPHYRIN_AND_IRON = ("Cc1c(CCC(=O)O)c2cc3[nH]c(cc4nc(cc5[nH]c(cc1n2)c(C)c5C=C)c(C)c4C=C)c(C)c3CCC(=O)O"
                       ".[Fe+2]")


def by_type(case: str, kind: InteractionType):
    return [r for r in diagram_of(case).interactions if r.type is kind]


def table_rows(case: str) -> "list[dict[str, str]]":
    """The rows of a case's committed table as the file has them, read without bindview."""
    with (TESTDATA / case / "interactions.tsv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def without_atoms(r, lig, structure):
    """The record of a resolved interaction resolved again without the atoms it names: by its distance alone."""
    (found,) = resolve_interactions([r.record.with_(ligand_atoms=(), protein_atoms=())], lig, structure)
    return found


@pytest.mark.parametrize("case", CASES)
def test_every_record_resolves(case):
    """A row that names its ligand atom is placed on that atom, a row that names the atoms of a group (a ring, a
    charged group) on those atoms; nothing is left over."""
    dia = diagram_of(case)
    assert dia.unresolved == []
    for r in dia.interactions:
        assert r.method == ("supplied" if r.record.ligand_atoms else "named")


def test_counts_across_the_bundle():
    """As many rows are placed by name as the tables have rows with a ligand atom, and as many by the atoms of
    their group as have those (both counted in the files themselves)."""
    methods = collections.Counter(r.method for c in CASES for r in diagram_of(c).interactions)
    rows = [row for c in CASES for row in table_rows(c)]
    assert methods == {"named": sum(bool(row["ligand_atom"]) for row in rows),
                       "supplied": sum(bool(row["ligand_atoms"]) for row in rows)}
    assert sum(methods.values()) == len(rows)


# --- the chemistry of single cases -----------------------------------------------------------------

def test_bicalutamide_halogen_bond_is_on_a_fluorine_of_the_cf3():
    lig = diagram_of("ar_bicalutamide").ligand
    (hal,) = by_type("ar_bicalutamide", T.HALOGEN)
    assert hal.record.residue_label == "Met742" and hal.ligand.kind == "atom"
    f = lig.mol.GetAtomWithIdx(hal.ligand.atom)
    assert f.GetSymbol() == "F"
    (carbon,) = f.GetNeighbors()
    assert sorted(n.GetSymbol() for n in carbon.GetNeighbors()) == ["C", "F", "F", "F"]       # a CF3 carbon
    assert not f.IsInRing()


def test_imatinib_hbond_from_the_hinge_is_on_the_pyridine_nitrogen():
    """The backbone nitrogen of Cys673 donates to imatinib's pyridine: an aromatic nitrogen without a hydrogen,
    in a ring of six with no other nitrogen."""
    lig = diagram_of("kit_imatinib").ligand
    (hb,) = [r for r in by_type("kit_imatinib", T.HBOND) if r.record.residue_label == "Cys673"]
    n = lig.mol.GetAtomWithIdx(hb.ligand.atom)
    assert n.GetSymbol() == "N" and n.GetIsAromatic() and n.GetTotalNumHs() == 0
    rings = [r for r in lig.mol.GetRingInfo().AtomRings() if n.GetIdx() in r]
    assert len(rings) == 1 and len(rings[0]) == 6
    assert sum(lig.element(i) == "N" for i in rings[0]) == 1


def test_imatinib_pi_stacking_is_on_a_ring_not_an_atom():
    """Phe811 stacks on imatinib's pyrimidine: a ring of aromatic atoms on each side. Found from the distance
    alone, without the atoms the table names, it is the same pair of rings."""
    dia = diagram_of("kit_imatinib")
    lig = dia.ligand
    (pi,) = by_type("kit_imatinib", T.PI_STACKING)
    assert pi.record.residue_label == "Phe811"
    assert pi.ligand.kind == "ring" and len(pi.ligand.atoms) == 6
    assert all(lig.mol.GetAtomWithIdx(i).GetIsAromatic() for i in pi.ligand.atoms)
    assert pi.protein.kind == "ring" and set(pi.protein.atoms) == {"CG", "CD1", "CD2", "CE1", "CE2", "CZ"}
    found = without_atoms(pi, lig, dia.structure)
    assert found.method == "distance-match" and found.ligand.kind == "ring" and found.protein.kind == "ring"
    assert set(found.ligand.atoms) == set(pi.ligand.atoms) and set(found.protein.atoms) == set(pi.protein.atoms)


def test_imatinib_pi_cation_found_by_its_distance_is_on_the_basic_amine():
    """His790's ring over imatinib's piperazine. Without the atom the table names, the cation is found from the
    distance and the chemistry: the amine nitrogen that the table names."""
    dia = diagram_of("kit_imatinib")
    (pc,) = [r for r in by_type("kit_imatinib", T.PI_CATION) if r.record.residue_label == "His790"]
    found = without_atoms(pc, dia.ligand, dia.structure)
    assert found.method == "distance-match" and found.ligand.atoms == pc.ligand.atoms
    assert [dia.ligand.names[i] for i in found.ligand.atoms] == list(pc.record.ligand_atoms)
    assert found.ligand.group_kind == "amine" and found.protein.kind == "ring"


def test_r1881_two_hbonds_converge_on_the_same_keto_oxygen():
    """Gln711 and Arg752 both name O83 in the table: one oxygen, double-bonded to a carbon of the ring system."""
    lig = diagram_of("ar_r1881").ligand
    hb = {r.record.residue_label: r for r in by_type("ar_r1881", T.HBOND)}
    assert hb["Arg752"].ligand.atoms == hb["Gln711"].ligand.atoms
    o = lig.mol.GetAtomWithIdx(hb["Arg752"].ligand.atom)
    assert o.GetSymbol() == "O" and o.GetDegree() == 1
    (c,) = o.GetNeighbors()
    assert lig.mol.GetBondBetweenAtoms(o.GetIdx(), c.GetIdx()).GetBondTypeAsDouble() == 2.0 and c.IsInRing()


def test_r1881_steroid_rings_are_one_fused_system():
    lig = diagram_of("ar_r1881").ligand
    rings = [set(r) for r in lig.mol.GetRingInfo().AtomRings()]
    # Union-find over rings that share a bond (two atoms).
    parent = list(range(len(rings)))

    def find(i):
        while parent[i] != i:
            i = parent[i]
        return i
    for i in range(len(rings)):
        for j in range(i + 1, len(rings)):
            if len(rings[i] & rings[j]) >= 2:
                parent[find(i)] = find(j)
    sizes = collections.Counter(find(i) for i in range(len(rings)))
    assert len(rings) == 4 and max(sizes.values()) == 4          # A, B, C and D


def test_pi_interactions_are_all_on_rings():
    """A stacking is between a ring of each side; a cation over a ring has a ring on one side. A ring of the
    ligand is of aromatic atoms."""
    seen = 0
    for case in CASES:
        lig = diagram_of(case).ligand
        for r in by_type(case, T.PI_STACKING) + by_type(case, T.PI_CATION):
            seen += 1
            rings = [side for side, site in (("ligand", r.ligand), ("protein", r.protein)) if site.kind == "ring"]
            if r.type is T.PI_STACKING:
                assert rings == ["ligand", "protein"]
            else:
                assert len(rings) == 1
            if r.ligand.kind == "ring":
                assert all(lig.mol.GetAtomWithIdx(i).GetIsAromatic() for i in r.ligand.atoms)
    assert seen == sum(row["interaction_type"] in ("pi_stacking", "pi_cation") for c in CASES for row in table_rows(c))


def test_salt_bridges_found_by_their_distance_are_on_charged_groups_of_opposite_charge():
    """Without the atoms the table names, a salt bridge is placed on a charged group of the ligand from the
    distance alone: the group that holds the atoms the table names, its charge opposite to the residue's.
    Methotrexate meets Arg70 with a carboxylate and Glu30 with its diaminopteridine, a cation of the kind
    that is named after the guanidinium."""
    bridges = 0
    for case in CASES:
        dia = diagram_of(case)
        charges = {frozenset(g.atoms): g.charge for g in perceive_charged_groups(dia.ligand)}
        for r in by_type(case, T.SALTBRIDGE):
            bridges += 1
            found = without_atoms(r, dia.ligand, dia.structure)
            assert found.method == "distance-match" and set(r.ligand.atoms) <= set(found.ligand.atoms)
            assert found.protein.charge != 0 and charges[frozenset(found.ligand.atoms)] == -found.protein.charge
    assert bridges == sum(row["interaction_type"] == "saltbridge" for c in CASES for row in table_rows(c))
    dia = diagram_of("dhfr_methotrexate")
    kinds = {r.record.residue_label: without_atoms(r, dia.ligand, dia.structure).ligand.group_kind
             for r in by_type("dhfr_methotrexate", T.SALTBRIDGE)}
    assert kinds == {"Arg70": "carboxylate", "Glu30": "guanidinium"}


# --- metals -----------------------------------------------------------------------------------------

def test_a_metal_ion_as_the_ligand_has_its_partners_identified():
    """The zinc of carbonic anhydrase II as the ligand, with the plain distance table: every contact is on the
    zinc, and they are the three histidine nitrogens that the file's own record of its links names."""
    dia = Diagram.from_structure(structure_of("ca2_celecoxib"), "ZN", interactions="distances")
    metal = [r for r in dia.interactions if r.type is T.METAL]
    assert len(metal) == len(dia.interactions) == 3 and all(r.ligand.atoms == (0,) for r in metal)
    partners = sorted((r.record.residue_label, r.protein.atoms[0]) for r in metal)
    assert partners == [("His119", "ND1"), ("His94", "NE2"), ("His96", "NE2")]     # _struct_conn, metalc1-3
    assert dia.ligand.element(0) == "Zn"


def test_the_coordination_of_a_metal_by_its_own_ligand_is_no_protein_contact():
    """The haem with its iron as an ion of its own in the SMILES: the four nitrogens of the porphyrin hold the
    iron, and none of that is a contact with the protein. The one metal contact is His87's, the residue that
    the file's own record of its links puts on the iron (_struct_conn, metalc1)."""
    dia = Diagram.from_structure(structure_of("hb_carbon_monoxide"), "A:142", _PORPHYRIN_AND_IRON, "distances")
    fe = dia.ligand.index_of("FE")
    assert sorted(dia.ligand.names[d] for m, d in dia.ligand.coordination if m == fe) == ["NA", "NB", "NC", "ND"]
    metal = [(r.record.residue_label, r.protein.atoms, r.ligand.atoms) for r in dia.interactions if r.type is T.METAL]
    assert metal == [("His87", ("NE2",), (fe,))]
    assert all(fe not in r.ligand.atoms for r in dia.interactions if r.type is not T.METAL)


# --- behaviour on records the library cannot place --------------------------

def test_unknown_ligand_atom_is_unresolved_never_guessed():
    dia = diagram_of("kit_imatinib")
    bogus = Interaction("hbond", 670, "THR", chain="A", protein_atom="OG1", ligand_atom="N99", distance=3.0)
    (res,) = resolve_interactions([bogus], dia.ligand, dia.structure)
    assert res.ligand is None and res.method == "unresolved"
    assert "N99" in res.notes[0]


def test_unreproducible_distance_is_not_drawn():
    dia = diagram_of("kit_imatinib")
    wrong = Interaction("pi_stacking", 811, "PHE", chain="A", distance=1.50)        # no ring pair is that close
    (res,) = resolve_interactions([wrong], dia.ligand, dia.structure)
    assert res.ligand is None and "not drawn" in res.notes[-1]


def test_near_miss_distance_is_attached_but_flagged_approximate():
    """The table has Phe811's stacking at 4.84 Å; a record of it at 4.95 Å is placed on the ring and called
    approximate."""
    dia = diagram_of("kit_imatinib")
    (stack,) = [r.record for r in dia.interactions if r.type is T.PI_STACKING]
    assert stack.distance == 4.84
    near = Interaction("pi_stacking", 811, "PHE", chain="A", distance=4.95)
    (res,) = resolve_interactions([near], dia.ligand, dia.structure)
    assert res.ligand is not None and res.ligand.kind == "ring" and res.method == "approximate"


def test_explicit_group_atoms_bypass_inference_and_need_no_structure():
    dia = diagram_of("kit_imatinib")
    ring = tuple(dia.ligand.names[i] for i in by_type("kit_imatinib", T.PI_STACKING)[0].ligand.atoms)
    rec = Interaction("pi_stacking", 811, "PHE", chain="A", ligand_atoms=ring,
                      protein_atoms=("CG", "CD1", "CD2", "CE1", "CE2", "CZ"))
    (res,) = resolve_interactions([rec], dia.ligand, None)
    assert res.method == "supplied" and res.ligand.kind == "ring" and res.protein.kind == "ring"


def test_group_interaction_without_a_structure_is_reported_unresolved():
    dia = diagram_of("kit_imatinib")
    (res,) = resolve_interactions([Interaction("pi_stacking", 811, "PHE", distance=4.84)], dia.ligand, None)
    assert res.ligand is None and "no structure was supplied" in res.notes[0]


def test_residue_name_mismatch_is_noted():
    """The table has Leu595's contact with C6; a record that calls the residue a valine is placed, and noted."""
    dia = diagram_of("kit_imatinib")
    rec = Interaction("hydrophobic", 595, "VAL", chain="A", protein_atom="CD1", ligand_atom="C6", distance=3.77)
    (res,) = resolve_interactions([rec], dia.ligand, dia.structure)
    assert res.ligand is not None and any("structure has LEU" in n for n in res.notes)


def test_an_atom_called_na_is_named_and_not_missing():
    """A heme has a nitrogen called NA. A table can also write NA for a name that is not given: which it is, the
    ligand says."""
    from dataclasses import replace

    from bindview.structure import Structure
    from conftest import load_interactions, meta_of

    base, case, ligand = diagram_of("kit_imatinib"), "kit_imatinib", meta_of("kit_imatinib")["ligand"]
    table = load_interactions(TESTDATA / case / "interactions.tsv")
    (hbond,) = [r for r in table if r.type is T.HBOND and r.residue_number == 670]
    assert (hbond.ligand_atom, hbond.protein_atom) == ("N13", "OG1")

    def named(st, rec):
        dia = Diagram.from_structure(st, ligand["residue_name_in_cif"], ligand["smiles"], [rec])
        (res,) = dia.interactions
        return dia.ligand.names[res.ligand.atom], res.method, res.record.ligand_atom, res.record.protein_atom
    renamed = Structure(atoms=[replace(a, name="NA") if a.residue == base.ligand_key and a.name == "N13" else a
                               for a in base.structure.atoms], source=base.structure.source)
    # The ligand has an atom of that name: it is that atom, with or without a distance.
    assert named(renamed, hbond.with_(ligand_atom="NA")) == ("NA", "named", "NA", "OG1")
    assert named(renamed, hbond.with_(ligand_atom="NA", distance=None)) == ("NA", "named", "NA", "OG1")
    # It has none: the name is not given, and the atom is found by the distance.
    for word in ("NA", "nan", ""):
        assert named(base.structure, hbond.with_(ligand_atom=word)) == ("N13", "distance-match", None, "OG1")
    # The same of the residue's atom: threonine has none called NA.
    assert named(base.structure, hbond.with_(protein_atom="NA")) == ("N13", "named", "N13", None)
    # A residue can be called so: sodium.
    sodium = Interaction("metal", 301, "NA", chain="A", protein_atom="NA", ligand_atom="N13", distance=2.4)
    assert (sodium.residue_name, sodium.protein_atom) == ("NA", "NA")
    (res,) = resolve_interactions([sodium], base.ligand, None)
    assert res.ligand is not None and res.protein.atoms == ("NA",)


# --- tables and structures that are unusual, on the open systems and on a few atoms ----------------

def _structure(*rows):
    """A structure of a few atoms: ``(chain, number, residue name, atom name, element, x, y, z)``."""
    from bindview.structure import Atom, Structure

    return Structure(atoms=[Atom(i, name, element, resname, chain, number, "", x, y, z, resname not in ("ASP", "GLY"))
                            for i, (chain, number, resname, name, element, x, y, z) in enumerate(rows)])


def _methanol(structure, chain="L", number=1):
    from bindview.ligand import build_ligand
    from bindview.structure import ResidueKey

    return build_ligand(structure.residue_atoms(ResidueKey(chain, number)), "CO")


def _what(resolved):
    return [(r.method, r.ligand, r.protein, r.residue, r.record) for r in resolved]


@pytest.mark.parametrize("word", ["NA", "?", "."])
def test_a_table_with_a_word_in_every_empty_cell_resolves_as_the_table_itself(word):
    """R writes NA into every cell without a value, the two columns of a group's atoms among them."""
    from bindview import parse_interactions
    from conftest import TESTDATA

    dia = diagram_of("kit_imatinib")
    text = (TESTDATA / "kit_imatinib" / "interactions.tsv").read_text()
    filled = "".join("\t".join(cell or word for cell in line.split("\t")) + "\n" for line in text.splitlines())
    plain = resolve_interactions(parse_interactions(text), dia.ligand, dia.structure)
    assert len(plain) == 23 and all(r.ligand is not None for r in plain)
    assert {r.method for r in plain} == {"named", "supplied"}
    assert _what(resolve_interactions(parse_interactions(filled), dia.ligand, dia.structure)) == _what(plain)


def test_na_for_the_atoms_of_a_group_is_an_atom_where_the_ligand_has_one():
    from dataclasses import replace

    from bindview.structure import Structure

    dia = diagram_of("kit_imatinib")
    (bridge,) = [r.record for r in dia.interactions if r.type is T.SALTBRIDGE and r.record.ligand_atoms == ("N48",)]
    renamed = Structure(atoms=[replace(a, name="NA") if a.residue == dia.ligand_key and a.name == "N48" else a
                               for a in dia.structure.atoms])
    from bindview.ligand import build_ligand

    ligand = build_ligand(renamed.residue_atoms(dia.ligand_key), dia.ligand.smiles)
    (res,) = resolve_interactions([bridge.with_(ligand_atoms="NA")], ligand, renamed)
    assert res.method == "supplied" and [ligand.names[i] for i in res.ligand.atoms] == ["NA"]
    assert res.record.ligand_atoms == ("NA",)
    # The ligand of the file has no atom of that name: the atoms are not given, and are found by the distance.
    (res,) = resolve_interactions([bridge.with_(ligand_atoms="NA", protein_atoms="NA")], dia.ligand, dia.structure)
    assert res.method == "distance-match" and [dia.ligand.names[i] for i in res.ligand.atoms] == ["N48"]
    assert res.record.ligand_atoms == () and res.record.protein_atoms == () and res.protein.atoms == ("OD1", "OD2")


def test_a_chain_called_na_is_a_chain_where_the_structure_has_one():
    """A large structure names its chains with two letters, NA among them. A table can also write NA for a chain
    that is not given: which it is, the structure says."""
    from bindview.structure import ResidueKey

    ligand = [("L", 1, "MOH", "C1", "C", 0.0, 0.0, 0.0), ("L", 1, "MOH", "O1", "O", 1.4, 0.0, 0.0)]

    def aspartate(chain, x):
        return [(chain, 12, "ASP", "CG", "C", x, 3.0, 0.0), (chain, 12, "ASP", "OD1", "O", x, 2.0, 0.0)]

    rec = Interaction("hbond", 12, "ASP", chain="NA", protein_atom="OD1", ligand_atom="O1", distance=2.0)
    assert rec.chain == "NA"
    two = _structure(*ligand, *aspartate("NA", 1.4), *aspartate("NB", 40.0))
    (res,) = resolve_interactions([rec], _methanol(two), two)
    assert res.residue == ResidueKey("NA", 12) and res.record.chain == "NA" and res.ligand is not None
    # No chain of that name: the chain is not given, and the residue is the one with that number.
    one = _structure(*ligand, *aspartate("A", 1.4))
    (res,) = resolve_interactions([rec], _methanol(one), one)
    assert res.residue == ResidueKey("A", 12) and res.record.chain == ""
    (res,) = resolve_interactions([rec], _methanol(one), None)
    assert res.residue is None and res.record.chain == ""


def test_a_record_without_a_chain_says_so_where_two_chains_have_its_residue():
    """A homodimer: both chains have an Asp12. A row that names no chain cannot be placed, and is told why."""
    ligand = [("L", 1, "MOH", "C1", "C", 0.0, 0.0, 0.0), ("L", 1, "MOH", "O1", "O", 1.4, 0.0, 0.0)]
    dimer = _structure(*ligand, *[(chain, 12, "ASP", name, element, x, y, 0.0) for chain, x in (("A", 1.4), ("B", 40.0))
                                  for name, element, y in (("CG", "C", 4.0), ("OD1", "O", 3.0))])
    notes = {}
    for number in (12, 99):
        (res,) = resolve_interactions([Interaction("saltbridge", number, "ASP", distance=3.0)], _methanol(dimer), dimer)
        assert res.method == "unresolved" and res.residue is None
        (notes[number],) = res.notes
    assert notes[12] == ("cannot resolve the saltbridge partner: residue 12 is in 2 chains (A, B) "
                         "and the record names no chain")
    assert notes[99] == "cannot resolve the saltbridge partner: residue ?:99 is not in the structure"


def test_a_metal_row_finds_the_metal_inside_a_larger_residue():
    """The iron of a haem is one atom of a residue that is called HEM: a metal by its element, whatever the
    residue is called. And a carbon called CA is none in a residue whose name begins with those letters."""
    from bindview.ligand import build_ligand
    from bindview.structure import ResidueKey

    imidazole = [("L", 1, "IMD", "N1", "N", 0.0, 0.0, 2.1), ("L", 1, "IMD", "C2", "C", 1.1, 0.0, 2.9),
                 ("L", 1, "IMD", "N3", "N", 0.7, 0.0, 4.2), ("L", 1, "IMD", "C4", "C", -0.7, 0.0, 4.2),
                 ("L", 1, "IMD", "C5", "C", -1.1, 0.0, 2.9)]
    held = {"HEM": [("A", 500, "HEM", "FE", "Fe", 0.0, 0.0, 0.0), ("A", 500, "HEM", "NA", "N", 2.0, 0.0, 0.0),
                    ("A", 500, "HEM", "NB", "N", 0.0, 2.0, 0.0), ("A", 500, "HEM", "NC", "N", -2.0, 0.0, 0.0),
                    ("A", 500, "HEM", "ND", "N", 0.0, -2.0, 0.0)],
            "ZN": [("A", 500, "ZN", "ZN", "Zn", 0.0, 0.0, 0.0)],
            "CAS": [("A", 500, "CAS", "CA", "C", 0.0, 0.0, 0.0), ("A", 500, "CAS", "AS", "As", 0.0, 2.0, 0.0)]}
    found = {}
    for name, atoms in held.items():
        st = _structure(*imidazole, *atoms)
        ligand = build_ligand(st.residue_atoms(ResidueKey("L", 1)), "c1c[nH]cn1")
        (res,) = resolve_interactions([Interaction("metal", 500, name, chain="A", distance=2.1)], ligand, st)
        found[name] = (res.method, res.ligand and ligand.names[res.ligand.atom], res.protein.atoms)
    assert found["ZN"] == ("distance-match", "N1", ("ZN",))
    assert found["HEM"] == ("distance-match", "N1", ("FE",))
    assert found["CAS"] == ("unresolved", None, ())


def test_many_rows_without_atom_names_on_one_residue_are_placed_without_trying_every_combination():
    """The ten nearest pairs of atoms between Phe811 and imatinib, each as a row with its distance and no atom
    name. The rows of one residue are placed together, so that no pair is used twice: each on its own pair, and
    in a time that does not grow with the number of combinations of them."""
    import time

    dia = diagram_of("kit_imatinib")
    pairs = [(3.41, "CZ", "C12"), (3.62, "CZ", "C11"), (3.68, "CE1", "C11"), (3.70, "CE1", "C12"),
             (3.82, "CE2", "C11"), (3.91, "CE2", "C12"), (3.95, "CD1", "C11"), (4.08, "CD2", "C11"),
             (4.15, "CG", "C11"), (4.42, "CD1", "C12")]
    rows = [Interaction("hydrophobic", 811, "PHE", chain="A", distance=distance) for distance, _, _ in pairs]
    for k in (4, 8, 10):
        started = time.process_time()
        resolved = resolve_interactions(rows[:k], dia.ligand, dia.structure)
        spent = time.process_time() - started
        assert [(r.method, r.ligand and dia.ligand.names[r.ligand.atom], r.protein.atoms) for r in resolved] == \
            [("distance-match", ligand_atom, (atom,)) for _, atom, ligand_atom in pairs[:k]]
        assert spent < 5.0, f"{k} rows took {spent:.1f} s of processor time"
    # More rows than there are pairs to choose from, the same row eight times (imatinib has four aromatic rings,
    # the residue one): each takes the best pair there is.
    (stack,) = [r.record for r in dia.interactions if r.type is T.PI_STACKING]
    assert stack.residue_number == 811 and len(stack.ligand_atoms) == 6
    resolved = resolve_interactions([stack.with_(ligand_atoms=(), protein_atoms=())] * 8, dia.ligand, dia.structure)
    assert [(r.method, r.ligand and sorted(dia.ligand.names[i] for i in r.ligand.atoms)) for r in resolved] == \
        [("distance-match", sorted(stack.ligand_atoms))] * 8


def test_rows_on_one_residue_that_name_its_atom_each_keep_that_atom():
    """Three ring carbons of Phe811 lie against one carbon of imatinib. Each row names the residue's atom and its
    distance, and leaves the ligand's atom to be found: on its own atom of the residue, whatever the rows' order."""
    dia = diagram_of("kit_imatinib")
    rows = [Interaction("hydrophobic", 811, "PHE", chain="A", protein_atom=atom, distance=distance)
            for atom, distance in (("CD1", 3.95), ("CE1", 3.68), ("CE2", 3.82))]
    for order in (rows, rows[::-1]):
        resolved = resolve_interactions(order, dia.ligand, dia.structure)
        assert [(r.method, r.ligand and dia.ligand.names[r.ligand.atom], r.protein.atoms) for r in resolved] == \
            [("distance-match", "C11", (rec.protein_atom,)) for rec in order]
