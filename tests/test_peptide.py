"""A peptide ligand is drawn as a chain of its residues: perceived, decided, laid out, attached to.

A peptide that a model writes as one hetero component with one residue record has residues that are
*perceived* from the bonds: here the nonamer of ``mhc_peptide`` rewritten so, with its SMILES. A peptide
deposited as a chain of its own has residues with the names and numbers of its file: ten residues of hirudin
on thrombin (``thrombin_hirudin``, chain H, with a sulfated tyrosine), and semaglutide in the GLP-1 receptor
(``glp1r_semaglutide``, chain P: 30 residues with an Aib, and a group bonded to Lys26 whose bonds are not
known, so that its figure is ``degraded``). The tests follow that order -- what is seen in a molecule, when
that is drawn as a chain, what the chain looks like, and where a contact lands on it.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace

import numpy as np
import pytest
from rdkit import Chem

from conftest import (CASE_SPECS, ONE_LETTER, chain_diagram, diagram_of, invented_plddt, model_of,
                      one_residue_peptide, render_of, structure_of)
from bindview import Diagram, Graph, Polymer, SmallMolecule, perceive_peptide
from bindview.backends import NUMBERING_NOTE, choose_backend
from bindview.cli import main as cli
from bindview.confidence import band_of
from bindview.distance_table import as_tsv, contact_table, ligand_contacts
from bindview.peptide import MIN_RECOGNISED, MIN_RESIDUES, decide
from bindview.polymer import depict_polymer
from bindview.ligand import Ligand
from bindview.scene import Line, Polyline, Rect, Text
from bindview.structure import Structure
from bindview.styles.polymer_draw import draw_polymer
from bindview.theme import LIGHT, Metrics
from scenetools import groups, walk, words

#: The peptides drawn as chains: the nonamer written as one residue, and two deposited chains.
NONAMER, HIRUDIN, SEMAGLUTIDE = "nonamer", "thrombin_hirudin", "glp1r_semaglutide"
CHAINS = (NONAMER, HIRUDIN, SEMAGLUTIDE)
#: The figures of them that are drawn. The nonamer's detailed figure is too crowded to be drawn (below).
DRAWN = [(NONAMER, "schematic"), (HIRUDIN, "detailed"), (HIRUDIN, "schematic"), (SEMAGLUTIDE, "detailed"),
         (SEMAGLUTIDE, "schematic")]

_diagrams: dict[str, Diagram] = {}
_figures: dict[tuple[str, str], object] = {}


def peptide_diagram(case: str) -> Diagram:
    """The diagram of one of :data:`CHAINS`, built once for the module: do not change it."""
    if case not in _diagrams:
        if case == NONAMER:
            st, smiles = one_residue_peptide("mhc_peptide")
            _diagrams[case] = Diagram.from_structure(st, "LIG", smiles=smiles,
                                                     interactions=ligand_contacts(st, "LIG", smiles))
        else:
            _diagrams[case] = chain_diagram(case)
    return _diagrams[case]


def figure(case: str, style: str = "detailed"):
    if (case, style) not in _figures:
        _figures[case, style] = peptide_diagram(case).render(style=style)
    return _figures[case, style]


def peptide_of(case: str):
    backend = peptide_diagram(case).backend
    assert isinstance(backend, Polymer)
    return backend.peptide


def helm(body: str, links: str = "") -> Chem.Mol:
    return Chem.MolFromHELM("PEPTIDE1{" + body + "}" + "$" + links + "$" * 3)


def unit_groups(res) -> list:
    return groups(res, "unit")


def heavy_atoms(slug: str, chain: str, resnum: "int | None" = None) -> list:
    """The heavy atoms of a chain (or of one residue of it) in a case's file, without its waters."""
    return [a for a in structure_of(slug).atoms if a.chain == chain and a.element not in ("H", "D")
            and a.resname != "HOH" and (resnum is None or a.resnum == resnum)]


# ---------------------------------------------------------------------------
# Perception
# ---------------------------------------------------------------------------

def test_a_peptide_written_as_one_residue_is_read_from_its_bonds():
    """62 atoms in a single residue record; the sequence comes out of the bonds, and is chain P's of the file."""
    pep = peptide_of(NONAMER)
    file_sequence = "".join(ONE_LETTER[a.resname] for a in heavy_atoms("mhc_peptide", "P") if a.name == "CA")
    assert pep.sequence == file_sequence == "GLCTLVAML" and len(pep) == 9
    assert not pep.cyclic and not pep.crosslinks and pep.appendages == () and pep.runs == (9,)
    # Every atom is in a residue, and in exactly one of them.
    lig = peptide_diagram(NONAMER).ligand
    owned = [a for r in pep.residues for a in r.atoms]
    assert sorted(owned) == list(range(lig.n_atoms))
    assert pep.in_chain == lig.n_atoms == len(heavy_atoms("mhc_peptide", "P"))


def test_a_group_bonded_to_a_residue_is_an_appendage_that_knows_where_it_hangs():
    """Semaglutide's group on Lys26 (WF1 101 in the file) is no residue of the chain: it hangs off the
    lysine's NZ, with every atom the file gives it. Aib8 is a residue, and not a proteinogenic one."""
    pep = peptide_of(SEMAGLUTIDE)
    lig = peptide_diagram(SEMAGLUTIDE).ligand
    (group,) = pep.appendages
    in_file = heavy_atoms(SEMAGLUTIDE, "P", 101)
    assert len(group.atoms) == len(in_file) and pep.in_chain == lig.n_atoms - len(in_file)
    ((position, holder, first),) = group.attachments
    lysine = pep.residue(position)
    assert (lysine.name, lysine.shown, lysine.name_of(holder)) == ("LYS", "26", "NZ")
    assert lig.mol.GetBondBetweenAtoms(holder, first) is not None
    assert Counter(lig.element(a) for a in group.atoms) == Counter(a.element for a in in_file)
    assert pep.residue_of(first) is None and pep.appendage_of(first) is group
    aib = pep.residues[1]
    assert (aib.name, aib.shown, aib.proteinogenic) == ("AIB", "8", False)


def test_d_and_l_come_from_the_smiles_and_not_from_the_model():
    """The nonamer with a SMILES that makes its Leu2 a D-leucine: the model has an L-leucine there. The
    chain is read from the SMILES, and the model's centre is said to disagree."""
    st, _ = one_residue_peptide("mhc_peptide")
    smiles = Chem.MolToSmiles(helm("G.[dL].C.T.L.V.A.M.L"))
    dia = Diagram.from_structure(st, "LIG", smiles=smiles, interactions=[])
    pep = dia.backend.peptide
    assert pep.sequence == "GlCTLVAML" and pep.residues[1].code3 == "D-Leu" and pep.residues[1].stereo == "D"
    assert [r.stereo for r in pep.residues].count("D") == 1
    assert len(dia.ligand.stereo_mismatches) == 1                 # the model disagrees with the SMILES there ...
    assert any("differ between the model and the SMILES" in w for w in dia.warnings)
    as_modelled = perceive_peptide(dia.ligand.mol)
    assert [r.stereo for r in as_modelled.residues].count("D") == 0   # ... and read from the model, there would be none
    assert all(r.stereo is None for r in pep.residues if r.name == "GLY")


def test_atoms_get_their_names_within_the_residue():
    pep = peptide_of(NONAMER)
    met = pep.residue(8)
    assert met.name == "MET" and sorted(n for _, n in met.atom_names) == sorted(
        ["N", "CA", "C", "O", "CB", "CG", "SD", "CE"])
    assert met.is_backbone(met.n) and met.is_backbone(met.o) and not met.is_backbone(dict((n, a) for a, n in met.atom_names)["SD"])
    last = pep.residue(9)
    assert "OXT" in [n for _, n in last.atom_names]    # both oxygens of the C-terminal acid are the last residue's


BUILT = {
    "cyclic": (helm("A.L.K.F.G.S.V.D.T.W", "PEPTIDE1,PEPTIDE1,10:R2-1:R1"), "ALKFGSVDTW"),
    "oxytocin": (helm("C.Y.I.Q.N.C.P.L.G.[am]", "PEPTIDE1,PEPTIDE1,1:R3-6:R3"), "CYIQNCPLG"),
    "d": (helm("[dF].P.R.[dA].G.[dW].K.L.[dE].Y"), "fPRaGwKLeY"),
    "capped": (helm("[ac].A.K.L.F.G.S.V.D.T.[am]"), "AKLFGSVDT"),
}


def test_rings_crosslinks_caps_and_d_residues():
    cyclic = perceive_peptide(BUILT["cyclic"][0])
    assert cyclic.cyclic and cyclic.sequence == "ALKFGSVDTW" and not cyclic.appendages
    # A ring has no terminus: the start is chosen from the sequence, so the order of the atoms does not matter.
    order = list(range(BUILT["cyclic"][0].GetNumAtoms()))
    shuffled = Chem.RenumberAtoms(BUILT["cyclic"][0], order[37:] + order[:37][::-1])
    assert perceive_peptide(shuffled).sequence == cyclic.sequence

    oxytocin = perceive_peptide(BUILT["oxytocin"][0])
    assert oxytocin.sequence == "CYIQNCPLG" and not oxytocin.cyclic
    (bridge,) = oxytocin.crosslinks
    assert (bridge.a[0], bridge.b[0], bridge.kind) == (1, 6, "disulfide")
    (amide,) = oxytocin.appendages                     # the C-terminal amide nitrogen
    assert len(amide.atoms) == 1 and amide.attachments[0][0] == 9

    assert perceive_peptide(BUILT["d"][0]).sequence == "fPRaGwKLeY"
    capped = perceive_peptide(BUILT["capped"][0])
    assert capped.sequence == "AKLFGSVDT" and sorted(len(a.atoms) for a in capped.appendages) == [1, 3]
    assert {a.attachments[0][0] for a in capped.appendages} == {1, 9}


def test_a_side_chain_is_recognised_exactly_or_not_at_all():
    """More may hang off a heteroatom of a side chain; nothing may hang off one of its carbons."""
    phospho = perceive_peptide(Chem.MolFromSmiles(
        "N[C@@H](C)C(=O)N[C@@H](COP(=O)(O)O)C(=O)N[C@@H](CC(C)C)C(=O)O"))
    assert [r.name for r in phospho.residues] == ["ALA", "SER", "LEU"]
    (phosphate,) = phospho.appendages
    assert len(phosphate.atoms) == 4 and phospho.residue(2).name_of(phosphate.attachments[0][1]) == "OG"

    chloro = perceive_peptide(Chem.MolFromSmiles("N[C@@H](Cc1ccc(Cl)cc1)C(=O)N[C@@H](C)C(=O)O"))
    assert [r.name for r in chloro.residues] == ["XAA", "ALA"] and not chloro.appendages
    assert len(chloro.residue(1).atoms) == 4 + 8        # an unrecognised residue keeps its whole side chain

    others = perceive_peptide(Chem.MolFromSmiles(
        "N[C@@H](C)C(=O)N(C)[C@@H](CC(C)C)C(=O)NC(C)(C)C(=O)N[C@@H](CCCC)C(=O)N[C@@H](CCCN)C(=O)NCC(=O)O"))
    assert [r.code3 for r in others.residues] == ["Ala", "MeLeu", "Aib", "Nle", "Orn", "Gly"]
    assert [r.proteinogenic for r in others.residues] == [True, True, False, False, False, True]
    # cyclohexylalanine is not phenylalanine: the carbons are the same in number and not in kind
    assert perceive_peptide(Chem.MolFromSmiles("N[C@@H](CC1CCCCC1)C(=O)NCC(=O)O")).residues[0].name == "CHA"


# ---------------------------------------------------------------------------
# The decision: by what the molecule is, not by how big it is
# ---------------------------------------------------------------------------

def test_the_rule_and_its_two_numbers():
    assert (MIN_RESIDUES, MIN_RECOGNISED) == (8, 0.5)
    for n in range(2, 13):
        mol = Chem.MolFromSequence(("ACDEFGHIKLMNPQRSTVWY" * 2)[:n])
        verdict = decide(perceive_peptide(mol))
        assert verdict.polymer == (n >= 8) and verdict.longest_run == n
    assert not decide(None).polymer and not decide(perceive_peptide(Chem.MolFromSmiles("CCO"))).polymer


TRAPS = {
    "carfilzomib": "CC(C)C[C@H](NC(=O)[C@H](CCc1ccccc1)NC(=O)CN1CCOCC1)C(=O)N[C@@H](Cc1ccccc1)C(=O)"
                   "N[C@@H](CC(C)C)C(=O)[C@@]1(C)CO1",
    "glutathione": "N[C@@H](CCC(=O)N[C@@H](CS)C(=O)NCC(=O)O)C(=O)O",
    "peptoid octamer": "NCC(=O)" + "N(CCc1ccccc1)CC(=O)" * 7 + "O",
    "beta-peptide octamer": "NCCC(=O)" + "NCCC(=O)" * 7 + "O",
    "depsipeptide": "N[C@@H](C)C(=O)O[C@@H](C)C(=O)" * 5 + "O",
    "all-4-chloro-Phe octamer": "N[C@@H](Cc1ccc(Cl)cc1)C(=O)" * 8 + "O",
    "nylon-6": "NCCCCCC(=O)" * 6 + "O",
}


@pytest.mark.parametrize("name", sorted(TRAPS))
def test_what_only_looks_like_a_peptide_is_drawn_as_atoms(name):
    pep = perceive_peptide(Chem.MolFromSmiles(TRAPS[name]))
    verdict = decide(pep)
    assert not verdict.polymer
    if name in ("peptoid octamer", "all-4-chloro-Phe octamer"):
        # Eight units in a row, and still not a chain: almost none of them is an amino acid one could name.
        assert len(pep) == 8 and pep.recognised <= 1 and "proteinogenic" in verdict.reason
    if name == "carfilzomib":
        assert len(pep) == 5 and "fewer than 8" in verdict.reason


def test_the_backend_follows_what_the_ligand_is():
    """A haem of 43 atoms around its iron, a macrocycle, is no peptide; nine residues are, whether written as
    one residue or deposited as a chain."""
    haem = Diagram.from_structure(structure_of("hb_carbon_monoxide"), ligand="A:142", smiles=None,
                                  interactions="distances")
    assert isinstance(haem.backend, SmallMolecule) and haem.backend.kind == "atoms"
    assert haem.ligand.n_atoms == len(heavy_atoms("hb_carbon_monoxide", "A", 142)) == 43
    assert haem.backend.decision.longest_run == 0
    for case in CHAINS:
        assert peptide_diagram(case).backend.kind == "chain", case
    for case in CASE_SPECS:
        assert diagram_of(case).backend.kind == "atoms", case
    # Pinned to atoms, the nonamer is drawn as atoms and says so; left alone it is a chain.
    st, smiles = one_residue_peptide("mhc_peptide")
    pinned = Diagram.from_structure(st, "LIG", smiles=smiles, interactions=[], ligand_as="atoms").backend
    assert pinned.kind == "atoms" and pinned.decision.longest_run == 9 and "asked" in pinned.decision.reason
    assert choose_backend(peptide_diagram(NONAMER).ligand).kind == "chain"


def test_the_caller_can_overrule_and_cannot_ask_for_the_impossible():
    lig = peptide_diagram(NONAMER).ligand
    assert choose_backend(lig, "atoms").kind == "atoms" and choose_backend(lig, "chain").kind == "chain"
    short = diagram_of("dhfr_methotrexate")
    assert short.backend.kind == "atoms" and short.backend.decision.longest_run == 0
    with pytest.raises(ValueError, match="cannot be drawn as a chain"):
        choose_backend(short.ligand, "chain")
    with pytest.raises(ValueError, match="ligand_as"):
        choose_backend(lig, "beads")


# ---------------------------------------------------------------------------
# The depiction: residues on a column, the rest as atoms
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", CHAINS)
def test_a_chain_is_a_column_and_every_atom_of_a_residue_sits_on_it(case):
    dia = peptide_diagram(case)
    pep = peptide_of(case)
    dep = dia.backend.depict(Metrics())
    assert len(dep.units) == len(pep) and dep.n_heavy == dia.ligand.n_atoms
    xs = {u.x for u in dep.units}
    steps = {round(b.y - a.y, 6) for a, b in zip(dep.units, dep.units[1:])}
    assert len(xs) == 1 and len(steps) == 1 and steps.pop() >= 22.0       # one column, one pitch, top to bottom
    assert len({(u.box[2] - u.box[0], u.box[3] - u.box[1]) for u in dep.units}) == 1
    for unit, res in zip(dep.units, pep.residues):
        assert unit.position == res.position and unit.atoms == res.atoms and unit.text == res.code3
        assert all(dep.pos(a) == unit.pos for a in unit.atoms)
        assert dep.unit_of(res.ca) is unit and dep.unit_for(res.atoms) is unit
    chain = [b for b in dep.bonds if b.kind == "chain"]
    assert len(chain) == len(pep) - 1
    outside = [a for a in dep.atoms[:dep.n_heavy] if dep.unit_of(a.idx) is None]
    assert len(outside) == dia.ligand.n_atoms - pep.in_chain


def test_the_appendage_keeps_its_atoms_and_grows_out_of_its_lysine():
    """Semaglutide's group on Lys26: every atom has a place of its own, every bond of it is drawn, and one
    bond holds it to the lysine. How far it reaches (more than ten bonds) is a regression value of the layout."""
    dia = peptide_diagram(SEMAGLUTIDE)
    pep = peptide_of(SEMAGLUTIDE)
    dep = dia.backend.depict(Metrics())
    (group,) = pep.appendages
    lysine = dep.units[pep.residue_of(group.attachments[0][1]).position - 1]
    assert lysine.text == "Lys"
    positions = {dep.pos(a) for a in group.atoms}
    assert len(positions) == len(group.atoms)                     # every atom of it has a place of its own
    inner = [b for b in dep.bonds if b.kind == "covalent" and b.a in group.atoms and b.b in group.atoms]
    in_molecule = [b for b in dia.ligand.mol.GetBonds()
                   if b.GetBeginAtomIdx() in group.atoms and b.GetEndAtomIdx() in group.atoms]
    assert len(inner) == len(in_molecule) == len(group.atoms) - 1   # its bonds: a tree, as the file has it
    holding = [b for b in dep.bonds if b.kind == "covalent" and (b.a in lysine.atoms) != (b.b in lysine.atoms)]
    assert len(holding) == 1                                      # and one bond that holds it to the lysine
    assert max(x for x, _ in positions) - lysine.box[2] > 10 * Metrics().bond_length
    assert dia.backend.depict(Metrics()).atoms == dep.atoms       # and it comes out the same every time


def test_no_atom_of_the_appendage_stands_on_the_column():
    """A group that hangs off a residue grows away from the column of boxes: none of its atoms stands on a
    box, in the depiction or in the figure."""
    dia = peptide_diagram(SEMAGLUTIDE)
    pep = peptide_of(SEMAGLUTIDE)
    dep = dia.backend.depict(Metrics())
    (group,) = pep.appendages
    on_a_box = [(dia.ligand.names[a], u.text + str(u.position)) for a in group.atoms for u in dep.units
                if u.box[0] <= dep.pos(a)[0] <= u.box[2] and u.box[1] <= dep.pos(a)[1] <= u.box[3]]
    res = figure(SEMAGLUTIDE, "schematic")
    boxes = [(r.x, r.y, r.x + r.w, r.y + r.h) for g in unit_groups(res) for r in g.items if isinstance(r, Rect)]
    names = {dia.ligand.names[a] for a in group.atoms}
    drawn_on = [n.label for n in res.graph.nodes if n.kind == "ligand_atom" and n.label in names
                and any(b[0] <= n.x <= b[2] and b[1] <= n.y <= b[3] for b in boxes)]
    assert on_a_box == [] and drawn_on == []


def test_a_stored_layout_is_redrawn_where_it_was():
    dia = peptide_diagram(SEMAGLUTIDE)
    pep = peptide_of(SEMAGLUTIDE)
    dep = dia.backend.depict(Metrics())
    moved = {u.position: (u.x + (40.0 if u.position > 15 else 0.0), u.y) for u in dep.units}
    exact = {dep.atoms[a].name: dep.pos(a) for app in pep.appendages for a in app.atoms}
    again = depict_polymer(dia.ligand, pep, Metrics(), unit_xy=moved, exact=exact)
    assert [u.pos for u in again.units] == [moved[u.position] for u in dep.units]
    assert all(again.pos(a) == dep.pos(a) for app in pep.appendages for a in app.atoms)


def _without_a_structure(mol: Chem.Mol) -> Ligand:
    """A ligand that never was in a structure file: enough for what needs no coordinates."""
    mol = Chem.Mol(mol)
    names = [f"{a.GetSymbol()}{a.GetIdx() + 1}" for a in mol.GetAtoms()]
    for atom, name in zip(mol.GetAtoms(), names):
        atom.SetProp("atomName", name)
    return Ligand(mol=mol, names=names, coords3d=np.zeros((mol.GetNumAtoms(), 3)), bond_source="template")


@pytest.mark.parametrize("key, closed", [("oxytocin", False), ("cyclic", True)])
def test_a_disulfide_and_a_ring_are_brackets_beside_the_column(key, closed):
    """Drawn on molecules built for the test, without a protein around them."""
    lig = _without_a_structure(BUILT[key][0])
    pep = perceive_peptide(lig.mol)
    dep = depict_polymer(lig, pep, Metrics())
    kinds = [b.kind for b in dep.bonds]
    assert kinds.count("chain") == len(pep) - 1 and kinds.count("closure" if closed else "crosslink") == 1
    drawn = draw_polymer(dep, LIGHT, Metrics())
    by = {g.cls: g for g in drawn.items}
    (bracket,) = [i for i in by["backbone"].items if isinstance(i, Polyline)]
    left = min(u.box[0] for u in dep.units)
    assert min(x for x, _ in bracket.points) < left and max(x for x, _ in bracket.points) <= left
    ys = sorted({y for _, y in bracket.points})
    ends = (dep.units[0].y, dep.units[-1].y) if closed else (dep.units[0].y, dep.units[5].y)   # Cys1 to Cys6
    assert (ys[0], ys[-1]) == ends
    assert bool(bracket.paint.dash) != closed          # the bond that closes a ring is a peptide bond: solid
    letters = [t.content for t in by["units"].items if isinstance(t, Text)]
    # a ring has no terminus; oxytocin has an N-terminus, and its C-terminus carries the amide, drawn as an atom
    assert letters == ([] if closed else ["N"])
    assert len(by["atoms"].items) == (0 if closed else 1)


# ---------------------------------------------------------------------------
# The figure
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case, style", DRAWN)
def test_a_chain_is_drawn_in_its_figures(case, style):
    """Every figure of a chain is drawn whole, with as many residues around it as its table names. The
    figure of semaglutide is ``degraded``: the bonds of its group on Lys26 are not known, and it says so."""
    res = figure(case, style)
    dia = peptide_diagram(case)
    assert res.status == ("degraded" if case == SEMAGLUTIDE else "ok") and res.figure == "diagram"
    if case == SEMAGLUTIDE:
        assert any(w.startswith("WF1 101 is not a component whose bonds are known here") for w in res.warnings)
    q = res.quality
    assert q.valid and q.readable and q.overlaps == 0 and q.unplaced == 0 and q.link_hits == 0
    assert q.n_interactions == len(dia.interactions)
    assert q.n_residues == len({(i.record.chain, i.record.residue_number) for i in dia.interactions})
    units = unit_groups(res)
    assert len(units) == len(peptide_of(case))
    assert [int(g.data["position"]) for g in units] == list(range(1, len(units) + 1))
    assert res.data["ligand"]["drawn_as"] == "chain" and res.data["ligand"]["residues"] == len(units)


@pytest.mark.parametrize("case, style", [(NONAMER, "schematic"), (HIRUDIN, "detailed"), (SEMAGLUTIDE, "schematic")])
def test_every_contact_lands_on_the_residue_whose_atom_makes_it(case, style):
    dia = peptide_diagram(case)
    pep = peptide_of(case)
    lig = dia.ligand
    graph = figure(case, style).graph
    seen: list[int] = []
    for contact in graph.contacts:
        node = graph.node(contact.ligand)
        for index in contact.records:
            atoms = dia.interactions[index].ligand.atoms
            owners = {pep.residue_of(a) for a in atoms}
            (owner,) = owners                                      # no contact straddles two residues here
            if owner is None:
                assert node.kind in ("ligand_atom", "ligand_group", "ring_centroid") and "position" not in node.attrs
                assert contact.ligand_residue is None
            else:
                assert node.kind == "ligand_group" and node.attrs["position"] == owner.position
                assert contact.ligand_residue == node.label
                assert set(lig.names[a] for a in atoms) <= set(contact.attrs["ligand_atoms"]) <= set(node.attrs["atoms"])
                made_by_backbone = all(owner.is_backbone(lig.index_of(n)) for n in contact.attrs["ligand_atoms"])
                part = "backbone" if made_by_backbone else "side chain"
                assert contact.attrs["ligand_part"] == part
            seen.append(index)
    assert sorted(seen) == list(range(len(dia.interactions)))


def test_where_the_contacts_of_a_chain_fall_is_where_its_table_says():
    """The contacts of hirudin made by a backbone and by a side chain, counted in the figure, are those of
    the table: a row is the backbone's where every atom of the ligand it names is N, CA, C, O or OXT."""
    rows = contact_table(structure_of(HIRUDIN), "H")
    table = Counter()
    for r in rows:
        names = [n.split("/")[1] for n in str(r["ligand_atom"] or r["ligand_atoms"]).split(",")]
        table["backbone" if set(names) <= {"N", "CA", "C", "O", "OXT"} else "side chain"] += 1
    drawn = Counter()
    for c in figure(HIRUDIN).graph.contacts:
        drawn[c.attrs.get("ligand_part", "outside the chain")] += len(c.records)
    assert drawn == table and table["backbone"] and table["side chain"]


def test_the_figure_says_whose_numbers_these_are():
    """A position in the ligand's own chain must not pass for a sequence position."""
    flat = words(figure(NONAMER, "schematic"))
    assert "numbered by their position in its own chain" in flat and "not positions in any sequence database" in flat
    assert "Residue of the ligand, numbered from its own N-terminus" in flat
    assert "N-terminus" in NUMBERING_NOTE
    # In a box, the code is bold and the position smaller and paler: not one string like "Cys3".
    (unit,) = [g for g in unit_groups(figure(NONAMER, "schematic")) if g.data["position"] == "3"]
    (label,) = [i for i in unit.items if isinstance(i, Text)]
    code, position = label.spans
    assert code.text == "Cys" and code.bold and position.text.strip() == "3"
    assert position.size < label.size and position.fill == "text_muted"
    assert "counted from its N-terminus" in unit.title
    # a ligand drawn as atoms carries none of this
    assert "N-terminus" not in words(render_of("kit_imatinib"))


def test_the_appendage_is_drawn_as_atoms_and_attached():
    res = figure(SEMAGLUTIDE, "schematic")
    pep = peptide_of(SEMAGLUTIDE)
    lig = peptide_diagram(SEMAGLUTIDE).ligand
    (ligand,) = groups(res, "ligand")
    assert ligand.data["drawn-as"] == "chain"
    labelled = {g.data["atom"] for g in groups(res, "atom") if g.data.get("atom")
                and any(p is ligand for f in walk(res.scene) for p in f.path if g in f.path)}
    group = {lig.names[a] for a in pep.appendages[0].atoms}
    hetero = {lig.names[a] for a in pep.appendages[0].atoms if lig.element(a) != "C"}
    assert hetero and hetero <= labelled <= group                 # its heteroatoms are written out, and only its atoms
    (bonds,) = [g for g in ligand.items if getattr(g, "cls", "") == "bonds"]
    lines = [i for i in bonds.items if isinstance(i, Line)]
    assert len(lines) == len(pep.appendages[0].atoms)             # its bonds, and the one that holds it
    # That one starts on the outline of the lysine's box and ends on the atom bonded to the lysine's NZ.
    lysine = next(n for n in res.graph.nodes if n.kind == "ligand_group" and n.label == "Lys26")
    first = next(n for n in res.graph.nodes if n.label == lig.names[pep.appendages[0].attachments[0][2]])
    half_w, half_h = lysine.attrs["width"] / 2.0, lysine.attrs["height"] / 2.0

    def on_outline(x: float, y: float) -> bool:
        dx, dy = abs(x - lysine.x), abs(y - lysine.y)
        return (abs(dx - half_w) < 0.6 and dy <= half_h + 0.6) or (abs(dy - half_h) < 0.6 and dx <= half_w + 0.6)

    holding = [ln for ln in lines for (x, y), (u, v) in (((ln.x1, ln.y1), (ln.x2, ln.y2)), ((ln.x2, ln.y2), (ln.x1, ln.y1)))
               if on_outline(x, y) and (u, v) == pytest.approx((first.x, first.y))]
    assert len(holding) == 1


def test_a_contact_of_the_appendage_is_drawn_on_its_atom():
    """A hydrophobic contact made up for the test, of a carbon of semaglutide's group on Lys26 with the
    nearest carbon of the receptor: it is drawn on that atom and on no residue of the chain, and without a
    line, as a hydrophobic contact of any ligand drawn as atoms is."""
    st = structure_of(SEMAGLUTIDE)
    group = [a for a in heavy_atoms(SEMAGLUTIDE, "P", 101) if a.element == "C"]
    receptor = [a for a in st.atoms if a.chain == "R" and a.element == "C"]
    mine, theirs = min(((a, b) for a in group for b in receptor), key=lambda ab: np.linalg.norm(
        np.subtract(ab[0].xyz, ab[1].xyz)))
    made = {"interaction_type": "hydrophobic", "residue_number": theirs.resnum, "residue_name": theirs.resname,
            "chain": "R", "protein_atom": theirs.name, "ligand_atom": f"101/{mine.name}",
            "distance": round(float(np.linalg.norm(np.subtract(mine.xyz, theirs.xyz))), 2)}
    dia = Diagram.from_structure(st, "chain:P", interactions=contact_table(st, "P") + [made])
    assert not dia.unresolved
    res = dia.render(style="schematic")
    (on_it,) = [c for c in res.graph.contacts if c.ligand_residue is None]
    node = res.graph.node(on_it.ligand)
    assert node.kind == "ligand_atom" and node.label == f"101/{mine.name}" and "position" not in node.attrs
    assert on_it.type == "hydrophobic" and not on_it.drawn


def test_a_pure_peptide_needs_no_atoms_at_all():
    """Hirudin's ten residues, the sulfated tyrosine among them, are all residues: boxes on a backbone and
    nothing drawn as atoms."""
    res = figure(HIRUDIN, "schematic")
    (ligand,) = groups(res, "ligand")
    by = {g.cls: g for g in ligand.items if hasattr(g, "cls")}
    assert not by["bonds"].items and not by["atoms"].items and len(by["backbone"].items) == len(peptide_of(HIRUDIN)) - 1
    boxes = [i for g in by["units"].items if hasattr(g, "items") for i in g.items if isinstance(i, Rect)]
    assert len(boxes) == len(peptide_of(HIRUDIN))
    ends = [t.content for t in by["units"].items if isinstance(t, Text)]
    assert ends == ["N", "C"]                                      # where the chain begins and where it ends


def test_confidence_reaches_the_boxes():
    """A residue of the ligand is drawn at the mean pLDDT of its atoms: its box in that band, and each band
    with a stroke of its own. The pLDDT is made up for the test (:func:`conftest.invented_plddt`)."""
    st = invented_plddt(structure_of(HIRUDIN))
    res = chain_diagram(HIRUDIN, structure=st).render()
    mean = {}
    for a in st.atoms:
        if a.chain == "H" and a.resname != "HOH" and a.element not in ("H", "D"):
            mean.setdefault(str(a.resnum), []).append(a.b_iso)
    units = unit_groups(res)
    numbers = {g.data["position"]: r.shown for g, r in zip(units, peptide_of(HIRUDIN).residues)}
    bands = {g.data["position"]: g.data.get("confidence") for g in units}
    dashes = {g.data["position"]: next(i for i in g.items if isinstance(i, Rect)).paint.dash for g in units}
    assert bands == {p: band_of(float(np.mean(mean[numbers[p]]))) for p in bands}
    assert len(set(bands.values())) >= 2                          # more than one band in this chain ...
    by_band = {b: {dashes[p] for p in bands if bands[p] == b} for b in set(bands.values())}
    assert all(len(d) == 1 for d in by_band.values())             # ... each drawn one way,
    assert len({next(iter(d)) for d in by_band.values()}) == len(by_band)   # and no two bands alike
    for node in res.graph.nodes:
        if node.kind == "ligand_group" and "position" in node.attrs:
            assert node.confidence == pytest.approx(float(np.mean(mean[node.attrs["number"]])))


def test_where_the_confidence_of_the_appendage_is_very_low_its_bonds_fade():
    """Semaglutide with a pLDDT made up for the test, very low on its group on Lys26: the bonds of that group
    are drawn in a group of their own, faded."""
    st = structure_of(SEMAGLUTIDE)
    low = Structure(atoms=[replace(a, b_iso=90.0) if a.chain == "P" and a.resnum == 101 else a for a in st.atoms],
                    source=st.source, chem_comp_smiles=dict(st.chem_comp_smiles), links=st.links)
    res = chain_diagram(SEMAGLUTIDE, structure=invented_plddt(low)).render(style="schematic")
    (ligand,) = groups(res, "ligand")
    (bonds,) = [g for g in ligand.items if getattr(g, "cls", "") == "bonds"]
    faded = [g for g in bonds.items if getattr(g, "cls", "").startswith("bonds-outside-")]
    assert faded and all(g.opacity < 1.0 for g in faded)
    assert {g.cls for g in faded} == {"bonds-outside-very_low"}


# ---------------------------------------------------------------------------
# The graph
# ---------------------------------------------------------------------------

def test_the_graph_of_a_chain():
    res = figure(SEMAGLUTIDE, "schematic")
    graph = res.graph
    data = graph.to_dict()
    assert data["version"] == "1.1" and render_of("kit_imatinib").graph.to_dict()["version"] == "1.0"
    pep = peptide_of(SEMAGLUTIDE)
    residues = [n for n in graph.nodes if n.kind == "ligand_group" and "position" in n.attrs]
    assert [n.attrs["position"] for n in residues] == list(range(1, len(pep) + 1)) and residues[19].label == "Lys26"
    assert all(n.attrs["form"] == "chain_residue" and not n.members for n in residues)
    atoms = [n for n in graph.nodes if n.kind == "ligand_atom" and not n.attrs.get("drawing_only")]
    assert len(atoms) == len(pep.appendages[0].atoms)
    # No node kind was added for this: a 1.0 reader meets nothing it has no name for.
    assert {n.kind for n in graph.nodes} <= {"ligand_atom", "ligand_group", "ring_centroid", "residue", "legend"}
    # The chain is not one rigid body. Only the lysine and the group on it are held together.
    (group,) = graph.groups
    assert group.id == "ligand:at20" and "L:res:20" in group.members and len(group.members) == len(atoms) + 1
    assert all(n.group is None for n in residues if n.id != "L:res:20")
    assert graph.meta["chain"]["sequence"] == pep.sequence and graph.meta["chain"]["drawn_as"] == "chain"
    assert graph.meta["chain"]["numbering"] == "as deposited in the structure file"

    edited = Graph.from_json(graph.to_json())                      # a copy: the result above is shared between tests
    alone = edited.move("L:res:5", 12.0, 0.0)
    assert alone == ["L:res:5"]                                    # a residue moves alone ...
    together = edited.move("L:res:20", 0.0, 3.0)
    assert set(together) == set(group.members)                     # ... except the one that carries the group


@pytest.mark.parametrize("style", ("detailed", "schematic"))
def test_a_chain_figure_is_redrawn_from_its_graph_to_the_byte(style):
    for case in (HIRUDIN, SEMAGLUTIDE):
        res = figure(case, style)
        again = peptide_diagram(case).render(style=style, graph=Graph.from_json(res.graph.to_json()))
        assert again.svg == res.svg and again.status == res.status


def test_a_moved_residue_is_drawn_where_it_was_moved_to():
    res = figure(HIRUDIN, "schematic")
    graph = Graph.from_json(res.graph.to_json())
    graph.move("L:res:10", 30.0, 0.0)
    moved = peptide_diagram(HIRUDIN).render(style="schematic", graph=graph, on_unreadable="render")
    (unit,) = [g for g in unit_groups(moved) if g.data["position"] == "10"]
    (before,) = [g for g in unit_groups(res) if g.data["position"] == "10"]
    box, old = (next(i for i in g.items if isinstance(i, Rect)) for g in (unit, before))
    assert box.x == pytest.approx(old.x + 30.0) and box.y == pytest.approx(old.y)


# ---------------------------------------------------------------------------
# What cannot be drawn legibly
# ---------------------------------------------------------------------------

def test_a_chain_that_cannot_be_drawn_legibly_is_refused():
    """A chain that cannot be drawn legibly is not drawn: a table is, and it says a chain was what failed.
    Hirudin's 10 residues and 94 atoms (the heavy atoms of chain H in the file) on a page of 60 by 60 mm."""
    res = peptide_diagram(HIRUDIN).render(figure_width_mm=60.0, figure_height_mm=60.0)
    assert res.status == "unreadable" and res.figure == "summary"
    assert len(heavy_atoms(HIRUDIN, "H")) == 94
    (message,) = [w for w in res.warnings if "not readable" in w]
    assert message.startswith("a chain diagram is not readable: 10 ligand residues (94 atoms)")
    assert peptide_diagram(HIRUDIN).render(figure_width_mm=60.0, figure_height_mm=60.0,
                                           on_unreadable="none").svg is None
    # The nonamer pinned to atoms is refused in the words of a ligand drawn as atoms.
    st, smiles = one_residue_peptide("mhc_peptide")
    atoms = Diagram.from_structure(st, "LIG", smiles=smiles, interactions=ligand_contacts(st, "LIG", smiles),
                                   ligand_as="atoms").render(figure_width_mm=60.0, figure_height_mm=60.0)
    assert atoms.status == "unreadable" and atoms.figure == "summary"
    assert any(w.startswith("an atom-level diagram is not readable: 62 ligand atoms") for w in atoms.warnings)
    assert atoms.data["ligand"]["drawn_as"] == "atoms"


def test_a_crowded_chain_is_given_up_early_with_effort_quick():
    """The nonamer written as one residue, with its 21 residues around it, has no detailed figure in which
    every residue has a place; ``effort="quick"`` finds that out from the roomiest layout the page allows."""
    res = peptide_diagram(NONAMER).render(style="detailed", effort="quick")
    assert res.status == "unreadable" and res.figure == "summary"
    assert any(w.startswith("a chain diagram is not readable: 9 ligand residues (62 atoms)") for w in res.warnings)


def test_command_line(tmp_path, capsys):
    st = structure_of(HIRUDIN)
    tsv, out = tmp_path / "contacts.tsv", tmp_path / "chain.svg"
    tsv.write_text(as_tsv(contact_table(st, "H")), encoding="utf-8")
    args = ["render", str(model_of(HIRUDIN)), "--ligand", "H", "--interactions", str(tsv), "--style", "schematic",
            "-o", str(out), "--json", "-q"]
    assert cli(args) == 0
    said = json.loads(capsys.readouterr().out)
    assert said["status"] == "ok" and said["ligand"]["drawn_as"] == "chain" and said["ligand"]["residues"] == 10
    assert said["ligand"]["sequence"] == peptide_of(HIRUDIN).sequence
    assert 'data-drawn-as="chain"' in out.read_text(encoding="utf-8")
    as_atoms = [a for a in args if a != str(out)]
    as_atoms.insert(as_atoms.index("-o") + 1, str(tmp_path / "atoms.svg"))
    assert cli(as_atoms + ["--ligand-as", "atoms"]) == 0
    said = json.loads(capsys.readouterr().out)
    assert said["ligand"]["drawn_as"] == "atoms" and 'data-drawn-as="chain"' not in (tmp_path / "atoms.svg").read_text()


# ---------------------------------------------------------------------------
# Every contact is a line, and a line says one thing
# ---------------------------------------------------------------------------

def _enters(line: Line, box: tuple[float, float, float, float]) -> bool:
    """Does the segment run through the inside of the box?"""
    t0, t1 = 0.0, 1.0
    for start, d, lo, hi in ((line.x1, line.x2 - line.x1, box[0], box[2]), (line.y1, line.y2 - line.y1, box[1], box[3])):
        if abs(d) < 1e-12:
            if not lo < start < hi:
                return False
            continue
        near, far = sorted(((lo - start) / d, (hi - start) / d))
        t0, t1 = max(t0, near), min(t1, far)
    return t1 - t0 > 1e-9


@pytest.mark.parametrize("case, style", DRAWN)
def test_every_contact_of_a_chain_figure_is_a_line(case, style):
    """Beside a column of boxes a contact without a line is read off the nearest box, which is not
    always its own: so a hydrophobic contact gets a line too, and one for each pair of residues."""
    res = figure(case, style)
    graph = res.graph
    on_the_chain = [c for c in graph.contacts if "position" in graph.node(c.ligand).attrs]
    assert on_the_chain and all(c.drawn for c in on_the_chain)
    greasy = [c for c in on_the_chain if c.type == "hydrophobic"]
    assert greasy and all(c.label is None for c in greasy)
    pairs = [(c.residue, c.ligand) for c in greasy]
    assert len(pairs) == len(set(pairs))
    drawn = groups(res, "interaction hydrophobic")
    assert len(drawn) == len(greasy)
    assert all(any(isinstance(item, Line) for item in g.items) for g in drawn)
    # What is not a residue is drawn as atoms; none of these contacts is with such an atom.
    assert all("position" in graph.node(c.ligand).attrs for c in graph.contacts if c.type == "hydrophobic")


@pytest.mark.parametrize("case, style", DRAWN)
def test_no_line_runs_across_a_box_it_does_not_end_on(case, style):
    res = figure(case, style)
    boxes = []
    for g in unit_groups(res):
        (rect,) = [item for item in g.items if isinstance(item, Rect)]
        boxes.append((rect.x, rect.y, rect.x + rect.w, rect.y + rect.h))
    assert len(boxes) == len(peptide_of(case))
    lines = [item for g in groups(res, "interaction") for item in g.items if isinstance(item, Line)]
    assert len(lines) >= sum(1 for c in res.graph.contacts if c.drawn)

    def away(line: Line, box) -> float:
        """How far the ligand end of the line is from the box."""
        dx = max(box[0] - line.x2, 0.0, line.x2 - box[2])
        dy = max(box[1] - line.y2, 0.0, line.y2 - box[3])
        return float(np.hypot(dx, dy))

    crossed = []
    for line in lines:
        for box in boxes:
            if away(line, box) < 1.5:
                continue                # the box it ends on: it may touch that one, at a rounded corner
            if _enters(line, (box[0] + 0.5, box[1] + 0.5, box[2] - 0.5, box[3] - 0.5)):
                crossed.append((round(line.x1), round(line.y1), round(line.x2), round(line.y2), box))
    assert not crossed


def test_the_field_knows_a_line_across_a_box():
    from bindview.layout.field import LigandField

    dep = peptide_diagram(HIRUDIN).backend.depict(Metrics())
    field = LigandField(dep, reach=60.0)
    first, third = dep.units[0], dep.units[2]
    own = np.array(third.pos)
    from_above = np.array([[first.x + 2.0, first.box[1] - 30.0]])
    from_the_side = np.array([[third.box[2] + 40.0, third.y + 1.0]])
    assert field.boxes_crossed(from_above, own[None, :], own).tolist() == [True]       # through the first two
    assert field.boxes_crossed(from_the_side, own[None, :], own).tolist() == [False]   # into its own only
    atoms = diagram_of("kit_imatinib").backend.depict(Metrics())
    assert not LigandField(atoms, reach=60.0).boxes_crossed(from_above, own[None, :], own).any()


def test_a_residue_that_touches_both_ends_is_drawn_at_both():
    """Arg73 of thrombin makes a salt bridge with Asp55, the first residue of hirudin's peptide; a contact
    with Leu64, the last, is made up for the test. The column is straight, so one drawing of Arg73 cannot
    stand beside both: it is drawn twice, and the figure says so."""
    st = structure_of(HIRUDIN)
    rows = contact_table(st, "H")
    (bridge,) = [r for r in rows if r["interaction_type"] == "saltbridge"]
    assert (bridge["residue_name"], bridge["residue_number"], str(bridge["ligand_atoms"]).split("/")[0]) == ("ARG", 73, "55")
    made = dict(bridge, interaction_type="hydrophobic", ligand_atom="64/CD1", ligand_atoms="", protein_atom="CZ",
                protein_atoms="", distance=4.0)
    dia = Diagram.from_structure(st, "chain:H", interactions=rows + [made])
    for style in ("detailed", "schematic"):
        res = dia.render(style=style)
        graph = res.graph
        twice = sorted((n for n in graph.nodes if n.kind == "residue" and "drawings" in n.attrs), key=lambda n: n.id)
        assert [n.label for n in twice] == ["Arg73", "Arg73"]
        assert [n.id[-2:] for n in twice] == ["~1", "~2"] and twice[0].id[:-2] == twice[1].id[:-2]
        assert [(n.attrs["drawing"], n.attrs["drawings"], n.attrs["number"]) for n in twice] == [(1, 2, 73), (2, 2, 73)]
        touched = [sorted(graph.node(c.ligand).attrs["position"] for c in graph.contacts if c.residue == n.id)
                   for n in twice]
        assert touched == [[1], [10]]
        ys = {n.attrs["position"]: n.y for n in graph.nodes if "position" in n.attrs}
        assert all(abs(n.y - ys[p[0]]) < abs(n.y - ys[11 - p[0]]) for n, p in zip(twice, touched))
        assert "Drawn more than once, each time beside the stretch of the ligand it touches there: Arg73." in words(res)
        marked = [g for g in groups(res, "residue") if "drawing" in g.data]
        assert sorted(g.data["drawing"] for g in marked) == ["1 of 2", "2 of 2"]
        assert len({g.data["residue"] for g in marked}) == 1
        # a residue drawn twice is still one residue
        assert res.quality.n_residues == len({(i.record.chain, i.record.residue_number) for i in dia.interactions})
    assert "Drawn more than once" not in words(figure(HIRUDIN))


def test_where_along_the_chain_an_atom_outside_it_is():
    from bindview.styles.base import SPLIT_GAP, chain_index

    dia = peptide_diagram(SEMAGLUTIDE)
    pep = peptide_of(SEMAGLUTIDE)
    dep = dia.backend.depict(Metrics())
    (group,) = pep.appendages
    assert {chain_index(dep, [a]) for a in group.atoms} == {19}            # all of it hangs on Lys26, the 20th
    assert chain_index(dep, list(pep.residues[4].atoms)) == 4
    assert SPLIT_GAP == 8


def test_a_chain_figure_does_not_depend_on_hash_randomisation():
    """Perception, the split of residues and the lanes go through sets and dictionaries; the bytes must not."""
    import os
    import subprocess
    import sys

    from conftest import ROOT

    code = (
        "import hashlib, sys\n"
        "sys.path.insert(0, 'tests')\n"
        "from conftest import chain_diagram, one_residue_peptide\n"
        "from bindview import Diagram\n"
        "from bindview.distance_table import ligand_contacts\n"
        "st, smiles = one_residue_peptide('mhc_peptide')\n"
        "made = [Diagram.from_structure(st, 'LIG', smiles=smiles, interactions=ligand_contacts(st, 'LIG', smiles))]\n"
        "made += [chain_diagram(slug) for slug in ('thrombin_hirudin', 'glp1r_semaglutide')]\n"
        "out = ''\n"
        "for dia in made:\n"
        "    res = dia.render(style='schematic')\n"
        "    out += res.svg + res.graph.to_json()\n"
        "print(hashlib.sha256(out.encode()).hexdigest())\n")
    digests = set()
    for seed in ("0", "1234"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        run = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, check=True)
        digests.add(run.stdout.strip().splitlines()[-1])
    assert len(digests) == 1
