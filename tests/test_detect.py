"""The optional PLIP adapter. The tests that run PLIP are skipped where it is not installed; the others run."""

from __future__ import annotations

import collections
from dataclasses import replace

import pytest

from bindview import Diagram, load_interactions
from bindview.detect import structure_to_pdb
from bindview.structure import Structure
from conftest import CASES, meta_of, model_of, needs_plip, structure_of


def test_pdb_rendering_keeps_names_and_renames_the_long_component():
    """The zinc of carbonic anhydrase II under a name of five characters, as a co-folding model names a ligand:
    written as LIG, with its atom name in the columns of a two-letter element, and found again by its serial."""
    st = structure_of("ca2_celecoxib")
    long = Structure(atoms=[replace(a, resname="ZN_B") if a.resname == "ZN" else a for a in st.atoms],
                     source="the zinc under a long name")
    text, by_serial, chain = structure_to_pdb(long, long.select_ligand("ZN_B"))
    het = [line for line in text.splitlines() if line[17:20] == "LIG"]
    assert len(het) == 1 and het[0].startswith("HETATM") and het[0][12:16] == "ZN  " and het[0][76:78] == "ZN"
    assert chain == "A" and by_serial[int(het[0][6:11])].name == "ZN"
    assert all(len(line) == 78 for line in text.splitlines() if line[:4] in ("ATOM", "HETA"))


_KINDS = ("hydrophobic_contacts", "hbonds_ldon", "hbonds_pdon", "halogen_bonds", "saltbridge_lneg", "saltbridge_pneg",
          "pistacking", "pication_laro", "pication_paro", "metal_complexes", "water_bridges")


def _stand_in_for_plip(found):
    """What the adapter asks of PLIP, without PLIP: a complex that takes the PDB text, knows the ligand that is
    written as LIG, and reports for it the contacts that ``found`` holds by their kind."""
    from types import SimpleNamespace

    class Complex:
        def load_pdb(self, text, as_string=False):
            line = next(ln for ln in text.splitlines() if ln[17:20] == "LIG")
            self.ligands = [SimpleNamespace(hetid="LIG", chain=line[21], position=int(line[22:26]))]
            self.interaction_sets = {}

        def characterize_complex(self, ligand):
            site = SimpleNamespace(**{kind: found.get(kind, []) for kind in _KINDS})
            self.interaction_sets[f"{ligand.hetid}:{ligand.chain}:{ligand.position}"] = site

    return SimpleNamespace(NOFIXFILE=False, DNARECEPTOR=False), Complex


def test_a_contact_has_the_chain_of_its_atom_where_chains_have_long_names(monkeypatch):
    """A PDB file has one letter for a chain, so chains AA and AB are written for PLIP as A and B, and PLIP
    reports a contact on B. Both chains have a Gly10: the contact is on the chain whose atom it names."""
    from types import SimpleNamespace as found

    from bindview import detect
    from bindview.structure import Atom, ResidueKey, Structure

    def atoms(chain, number, resname, x, *names):
        return [(chain, number, resname, name, name[0], x + 1.4 * k) for k, name in enumerate(names)]

    def contacts_of(chains, on):
        """One contact of every kind between the ligand and Gly10 of chain ``on``, as PLIP would report it."""
        rows = [row for c, x in zip(chains, (0.0, 40.0, 80.0)) for row in atoms(c, 10, "GLY", x, "N", "CA", "C", "O")]
        rows += atoms(chains[0], 301, "LIG", 0.0, "C1", "O1") + atoms(on, 501, "HOH", 3.0, "O")
        st = Structure(atoms=[
            Atom(i, name, element, resname, chain, number, "", x, 5.0 * (resname != "GLY"), 0.0, resname != "GLY")
            for i, (chain, number, resname, name, element, x) in enumerate(rows)])
        key = ResidueKey(chains[0], 301)
        text, by_serial, _ = structure_to_pdb(st, key)
        serial = {(a.chain, a.resname, a.name): s for s, a in by_serial.items()}
        ca, o, water = serial[on, "GLY", "CA"], serial[on, "GLY", "O"], serial[on, "HOH", "O"]
        c1, o1 = serial[chains[0], "LIG", "C1"], serial[chains[0], "LIG", "O1"]
        where = {"resnr": 10, "restype": "GLY", "reschain": text.splitlines()[ca - 1][21]}
        group = lambda *serials: found(atoms_orig_idx=list(serials))  # noqa: E731
        plip = _stand_in_for_plip({
            "hydrophobic_contacts": [found(bsatom_orig_idx=ca, ligatom_orig_idx=c1, distance=3.9, **where)],
            "hbonds_pdon": [found(protisdon=True, d_orig_idx=ca, a_orig_idx=o1, distance_ad=2.9, angle=160.0, **where)],
            "hbonds_ldon": [found(protisdon=False, d_orig_idx=o1, a_orig_idx=o, distance_ad=3.0, angle=150.0, **where)],
            "halogen_bonds": [found(acc_orig_idx=o, don_orig_idx=c1, distance=3.2, don_angle=165.0, **where)],
            "saltbridge_lneg": [found(protispos=True, positive=group(ca), negative=group(o1), distance=4.0, **where)],
            "saltbridge_pneg": [found(protispos=False, positive=group(c1), negative=group(o), distance=4.1, **where)],
            "pistacking": [found(proteinring=group(ca, o), ligandring=group(c1, o1), distance=4.5, angle=10.0,
                                 **where)],
            "pication_laro": [found(ring=group(c1, o1), charge=group(ca), protcharged=True, distance=4.2, **where)],
            "pication_paro": [found(ring=group(ca, o), charge=group(c1), protcharged=False, distance=4.3, **where)],
            "metal_complexes": [found(metal_orig_idx=c1, target_orig_idx=o, distance=2.1, **where)],
            "water_bridges": [found(protisdon=True, d_orig_idx=ca, a_orig_idx=o1, water_orig_idx=water, distance_aw=2.8,
                                    distance_dw=2.9, w_angle=110.0, **where)]})
        monkeypatch.setattr(detect, "_load_plip", lambda: plip)
        return where["reschain"], detect.detect_interactions(st, key)

    for chains, on, written in ((("AA", "AB"), "AB", "B"), (("AA", "B", "C"), "B", "B"), (("AA", "B", "C"), "C", "C"),
                                (("AA", "AB"), "AA", "A")):
        letter, records = contacts_of(chains, on)
        assert letter == written and len(records) == len(_KINDS)
        assert {(r.chain, r.residue_number, r.residue_name) for r in records} == {(on, 10, "GLY")}
        assert collections.Counter(r.type.value for r in records) == {
            "hydrophobic": 1, "hbond": 2, "halogen": 1, "saltbridge": 2, "pi_stacking": 1, "pi_cation": 2, "metal": 1,
            "water_bridge": 1}
        assert {r.water for r in records if r.type.value == "water_bridge"} == {f"{on}:501"}
        assert {(r.protein_atoms, r.ligand_atoms) for r in records if r.type.value == "pi_cation"} == \
            {(("CA",), ("C1", "O1")), (("CA", "O"), ("C1",))}


def _key(r):
    return (r.type.value, r.residue_number, r.distance)


@needs_plip
@pytest.mark.parametrize("case", CASES)
def test_detection_reproduces_the_supplied_table(case, testdata):
    """The committed tables are PLIP's (``scripts/case_tables.py``): detection on the model gives them again."""
    from bindview import detect_interactions

    detected = detect_interactions(model_of(case), meta_of(case)["ligand"]["residue_name_in_cif"])
    supplied = load_interactions(testdata / case / "interactions.tsv")
    assert collections.Counter(map(_key, detected)) == collections.Counter(map(_key, supplied))


@needs_plip
def test_detected_groups_arrive_named_so_nothing_is_inferred():
    ligand = meta_of("kit_imatinib")["ligand"]
    dia = Diagram.from_structure(model_of("kit_imatinib"), ligand["residue_name_in_cif"], ligand["smiles"])
    assert dia.source == "detected" and not dia.unresolved
    pi = [r for r in dia.interactions if r.type.value == "pi_stacking"]
    assert pi and all(r.method == "supplied" and r.ligand.kind == "ring" for r in pi)


@needs_plip
def test_intra_ligand_metal_coordination_is_not_reported():
    """The four nitrogens of the haem hold its iron: that is no contact of the haem with the protein."""
    from bindview import detect_interactions

    found = detect_interactions(model_of("hb_carbon_monoxide"), "A:142")
    own = [r for r in found if r.type.value == "metal"
           and ((r.chain, r.residue_number) == ("A", 142) or r.protein_atom in ("NA", "NB", "NC", "ND"))]
    assert found and not own
