"""A plain distance table of the contacts of a chain: of a peptide that is the ligand, or of an interface.

This is NOT an interaction analysis and must not be read as one. bindview analyses no interactions: a table
of contacts comes from the caller, or from PLIP behind :func:`bindview.detect_interactions`. This table is
for a structure whose ligand is a chain of its own, or in which two chains touch, when neither is at hand.
``bindview report`` draws such a chain from it, and the call it prints is this module's. The criteria are
deliberately crude -- distances only, because a crystal structure has no hydrogens:

  hydrogen bond        N or O of the ligand within 3.5 A of N or O of the protein, at least
                       one of the two able to donate (by residue and atom name); none between
                       two atoms that both lie within 3.0 A of one metal atom of the structure:
                       the two hold that metal, which is why they are that near each other
  salt bridge          an atom of a cationic group within 4.0 A of an atom of an anionic group
  hydrophobic contact  two carbons, neither bonded to N or O, within 4.0 A; one per pair of
                       residues, the closest
  ring stacking        only with ``stacking=True``: an aromatic ring of each (Phe, Tyr, Trp, His,
                       a base), centres within 5.5 A, planes within 30 degrees of parallel or of
                       perpendicular, the centre of one within 2.0 A of the other's axis (the
                       criterion of :func:`ligand_contacts`); one per pair of residues, the
                       closest pair of rings. Two rings that stack then have no hydrophobic
                       contact between them, as PLIP has it.

No angles, no waters, and no row for a metal: a metal only takes away the hydrogen bond between two
atoms that hold it. The partners are the polymer residues of every other chain; which chain is the
ligand is said by the caller and never guessed.

    from bindview import Diagram, read_structure
    from bindview.distance_table import contact_table

    model = read_structure("model.cif")
    Diagram.from_structure(model, "chain:P", interactions=contact_table(model, "P"))

A ligand atom is named as bindview names it in a chain ligand: ``63/OE1`` is atom OE1 of residue 63.

:func:`ligand_contacts` is the table for a ligand that is one residue: a small molecule, a cofactor, an
ion. It goes by the positions of the heavy atoms and by what the ligand is (its rings, its charged
groups, which of its atoms carry a hydrogen), and so has what the chain table leaves out: a cation over a
ring, a halogen bond, a metal, and stacking without being asked. It places no hydrogen, so a hydrogen bond has no
angle, and it knows no water bridge. As in the chain table, two atoms that hold one metal (a sulfonamide's
nitrogen and a histidine's on one zinc) have no hydrogen bond between them. It is cruder than PLIP, whose
tables it was measured against, and is there for when PLIP is not: its numbers are in docs/reference.md.

    from bindview import Diagram, read_structure
    from bindview.distance_table import ligand_contacts

    model = read_structure("model.cif")
    Diagram.from_structure(model, "LIG_B", smiles=smiles, interactions=ligand_contacts(model, "LIG_B", smiles))
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .errors import LigandNotFoundError
from .exposure import WATER_NAMES
from .near import distance_matrix, nearest_distance
from .structure import ResidueKey, Structure

__all__ = ["contact_table", "interface_contacts", "ligand_contacts", "as_tsv", "HBOND", "SALT", "GREASY", "COLUMNS",
           "STACK", "STACK_ANGLE", "OFF_AXIS", "CATION", "HALOGEN", "HALOGEN_ANGLE", "HALOGEN_AT_ACCEPTOR", "METAL"]

HBOND, SALT, GREASY = 3.5, 4.0, 4.0
COLUMNS = ("interaction_type", "residue_number", "residue_name", "chain", "insertion_code", "protein_atom",
           "ligand_atom", "distance", "angle", "ligand_is_donor", "protein_atoms", "ligand_atoms")

_DONORS = {"ARG": {"NE", "NH1", "NH2"}, "LYS": {"NZ"}, "HIS": {"ND1", "NE2"}, "ASN": {"ND2"}, "GLN": {"NE2"},
           "TRP": {"NE1"}, "SER": {"OG"}, "THR": {"OG1"}, "TYR": {"OH"}}
_ACCEPTOR_N = {"HIS": {"ND1", "NE2"}}
_CATIONS = {"LYS": ("NZ",), "ARG": ("NE", "NH1", "NH2"), "HIS": ("ND1", "NE2")}
_ANIONS = {"ASP": ("OD1", "OD2"), "GLU": ("OE1", "OE2"), "PTR": ("O1P", "O2P", "O3P"), "SEP": ("O1P", "O2P", "O3P"),
           "TPO": ("O1P", "O2P", "O3P"), "TYS": ("O1", "O2", "O3")}


def _donates(resname: str, atom: str) -> bool:
    from .residues import can_donate, is_nucleotide

    if is_nucleotide(resname):
        return can_donate(resname, atom) is True
    return (atom == "N" and resname != "PRO") or atom in _DONORS.get(resname, ())


def _accepts(resname: str, atom: str, element: str) -> bool:
    from .residues import can_accept, is_nucleotide

    if is_nucleotide(resname):
        return can_accept(resname, atom, element) is True
    return element == "O" or atom in _ACCEPTOR_N.get(resname, ())


def contact_table(structure: Structure, chain: str, partners: "Sequence[str] | None" = None, *,
                  stacking: bool = False) -> list[dict[str, object]]:
    """The table for the polymer of ``chain`` as ligand; rows in a fixed order, ready for bindview.

    ``partners`` restricts the other side to those chains (an interface between two participants);
    by default it is every other chain. The chain may be of any length here. ``stacking`` adds the rings
    that stack, one row for a pair of residues (see the module); left out, the table has no such row.
    Where the structure has a metal, an atom of the chain and an atom of a partner that both lie within
    ``METAL`` (3.0 A) of it have no hydrogen bond between them.
    """
    from .residues import aromatic_rings, canonical_atom, is_nucleotide

    residues = structure.polymer_residues(chain)
    if not residues:
        chains = sorted({a.chain for a in structure.atoms})
        if chain not in chains:
            raise LigandNotFoundError(f"there is no chain {chain!r} in {structure.source or 'the structure'}: its chains "
                                      f"are {', '.join(repr(c) for c in chains) or 'none'}")
        raise LigandNotFoundError(f"chain {chain!r} of {structure.source or 'the structure'} has no residue of a "
                                  f"protein or a nucleic acid: this table is of a chain of residues, and "
                                  f"ligand_contacts is the one of a ligand that is one residue")
    if partners is not None:
        chains = sorted({a.chain for a in structure.atoms})
        absent = [p for p in partners if p not in chains]
        if absent:                                     # (no rows would read as: the two do not touch)
            raise LigandNotFoundError(f"there is no chain {absent[0]!r} in {structure.source or 'the structure'} to be "
                                      f"a partner of chain {chain!r}: its chains are "
                                      f"{', '.join(repr(c) for c in chains)}")
    st, key, given, _ = structure.with_residues_as_ligand(residues)
    comp = {name: res.name.upper() for res in given for name in res.atoms}
    number = {name: res.number for res in given for name in res.atoms}
    lig = [a for a in st.residue_atoms(key) if a.element not in ("H", "D")]
    env = [a for a in st.atoms if a.residue != key and a.element not in ("H", "D") and a.resname.upper() not in WATER_NAMES]
    # The residues of the other side are those of their chains' polymers, as the chain's own are: a residue
    # without N, CA or C, a cap and a modified residue count from either side, and so do the two ends of a chain.
    polymer: set[ResidueKey] = set()
    ends: dict[ResidueKey, str] = {}
    for side in sorted({k.chain for k in st.residues() if k != key and k.chain != chain}):
        if partners is not None and side not in partners:
            continue
        of = st.polymer_residues(side)
        polymer.update(of)
        amino = [k for k in of if {"N", "CA", "C"} <= {a.name for a in st.residue_atoms(k)}]
        if amino:
            if amino[0] == of[0]:                    # (no cap before the first residue)
                ends[amino[0]] = "N"
            if st.atom_by_name(amino[-1], "OXT") is not None:
                ends[amino[-1]] = "C"
    env = [a for a in env if a.residue in polymer]
    if not env:
        return []
    lxyz = np.array([a.xyz for a in lig], dtype=float)
    exyz = np.array([a.xyz for a in env], dtype=float)
    reach = nearest_distance(exyz, lxyz)             # of each atom around: to the nearest atom of the chain
    # Rings: two whose centres are within STACK have atoms within STACK + 3 A of each other, whatever their tilt.
    ringside = [env[i] for i in np.nonzero(reach <= STACK + 3.0)[0]] if stacking else []
    near = np.nonzero(reach < 4.6)[0]
    env = [env[i] for i in near]
    exyz = exyz[near]
    dist = distance_matrix(lxyz, exyz)                                            # (ligand, protein)
    on_one_metal = _on_one_metal(_metals_at(st, lxyz), lxyz, exyz)
    polar = np.array([a.xyz for a in st.atoms if a.element in ("N", "O")], dtype=float)

    def greasy(atom) -> bool:
        return atom.element == "C" and not (np.linalg.norm(polar - np.array(atom.xyz), axis=1) < 1.7).any()

    def local(name: str) -> str:
        return name.split("/", 1)[1]

    first, last = given[0], given[-1]
    capped = any(res.name.upper() == "ACE" for res in given)

    def charge_group(resname: str, atom: str, lig_name: str | None = None,
                     residue: "ResidueKey | None" = None) -> tuple[str, tuple[str, ...]] | None:
        for table, sign in ((_CATIONS, "+"), (_ANIONS, "-")):
            if atom in table.get(resname, ()):
                return sign, table[resname]
        if is_nucleotide(resname) and atom in ("OP1", "OP2", "O1P", "O2P"):
            return "-", ("OP1", "OP2", "O1P", "O2P")          # the phosphate, as the file names its oxygens
        if lig_name is not None:
            if atom in ("O", "OXT") and lig_name in last.atoms and any(local(n) == "OXT" for n in last.atoms):
                return "-", ("O", "OXT")
            if atom == "N" and lig_name in first.atoms and not capped and comp[lig_name] != "PRO":
                return "+", ("N",)
        end = ends.get(residue) if residue is not None else None
        if end == "C" and atom in ("O", "OXT"):
            return "-", ("O", "OXT")
        if end == "N" and atom == "N" and resname != "PRO":
            return "+", ("N",)
        return None

    rows: list[dict[str, object]] = []
    salted: set[tuple[int, int]] = set()
    bridges: dict[tuple, tuple[float, dict[str, object]]] = {}
    for i, a in enumerate(lig):
        ga = charge_group(comp[a.name], local(a.name), a.name)
        if ga is None:
            continue
        for j, b in enumerate(env):
            gb = charge_group(b.resname, b.name, residue=b.residue)
            if gb is None or gb[0] == ga[0] or dist[i, j] > SALT:
                continue
            salted.add((i, j))
            prefix = a.name.split("/", 1)[0] + "/"
            mine = tuple(prefix + n for n in ga[1] if any(x.name == prefix + n for x in lig))
            theirs = tuple(n for n in gb[1] if st.atom_by_name(b.residue, n) is not None)
            ident = (b.residue, mine)
            row = {"interaction_type": "saltbridge", "residue_number": b.resnum, "residue_name": b.resname,
                   "chain": b.chain, "insertion_code": b.icode, "protein_atom": "", "ligand_atom": "",
                   "distance": round(float(dist[i, j]), 2), "ligand_is_donor": "",
                   "protein_atoms": ",".join(theirs), "ligand_atoms": ",".join(mine)}
            if ident not in bridges or dist[i, j] < bridges[ident][0]:
                bridges[ident] = (float(dist[i, j]), row)
    rows.extend(row for _, row in bridges.values())
    for i, a in enumerate(lig):
        if a.element not in ("N", "O"):
            continue
        for j, b in enumerate(env):
            if b.element not in ("N", "O") or dist[i, j] > HBOND or (i, j) in salted or (i, j) in on_one_metal:
                continue
            la, pa = local(a.name), b.name
            lig_d, lig_a = _donates(comp[a.name], la), _accepts(comp[a.name], la, a.element)
            pro_d, pro_a = _donates(b.resname, pa), _accepts(b.resname, pa, b.element)
            one, other = lig_d and pro_a, pro_d and lig_a
            if not (one or other):
                continue
            rows.append({"interaction_type": "hbond", "residue_number": b.resnum, "residue_name": b.resname,
                         "chain": b.chain, "insertion_code": b.icode, "protein_atom": pa, "ligand_atom": a.name,
                         "distance": round(float(dist[i, j]), 2),
                         # No direction where both could donate, or where a histidine ring nitrogen takes
                         # part: which of its two nitrogens carries the hydrogen a crystal structure does not say.
                         "ligand_is_donor": "" if (one and other) or "HIS" in (
                             comp[a.name] if la in ("ND1", "NE2") else "", b.resname if pa in ("ND1", "NE2") else "")
                         else str(bool(one)),
                         "protein_atoms": "", "ligand_atoms": ""})
    # Rings that stack, when asked for: of a pair of residues, the two rings whose centres are closest.
    stacked: dict[tuple[ResidueKey, str], tuple[set[str], set[str]]] = {}     # two residues -> the atoms of their rings
    if stacking:
        by_name = {a.name: a for a in lig}
        chain_rings: list[tuple[str, list[str], np.ndarray, np.ndarray]] = []
        for res in given:
            here = {canonical_atom(res.name.upper(), local(n)): n for n in res.atoms if n in by_name}
            for ring in aromatic_rings(res.name.upper()):
                if all(r in here for r in ring):
                    members = [here[r] for r in ring]
                    centre, normal = _plane(np.array([by_name[n].xyz for n in members], dtype=float))
                    chain_rings.append((res.number, members, centre, normal))
        grouped: dict[ResidueKey, list[Any]] = {}
        for b in ringside:
            grouped.setdefault(b.residue, []).append(b)
        partner_rings: list[tuple[ResidueKey, str, list[str], np.ndarray, np.ndarray]] = []
        for res_key, atoms in grouped.items():
            there = {canonical_atom(atoms[0].resname, b.name): b for b in atoms}
            for ring in aromatic_rings(atoms[0].resname):
                if all(r in there for r in ring):
                    centre, normal = _plane(np.array([there[r].xyz for r in ring], dtype=float))
                    partner_rings.append((res_key, atoms[0].resname, [there[r].name for r in ring], centre, normal))
        best: dict[tuple[ResidueKey, str], tuple[float, dict[str, object]]] = {}
        for res_number, members, c1, n1 in chain_rings:
            for res_key, resname, others, c2, n2 in partner_rings:
                stack = _stacks(c1, n1, c2, n2)
                if stack is None:
                    continue
                both = (res_key, res_number)
                stacked.setdefault(both, (set(), set()))[0].update(members)
                stacked[both][1].update(others)
                row = {"interaction_type": "pi_stacking", "residue_number": res_key.number, "residue_name": resname,
                       "chain": res_key.chain, "insertion_code": res_key.icode, "protein_atom": "", "ligand_atom": "",
                       "distance": round(stack[0], 2), "ligand_is_donor": "", "protein_atoms": ",".join(others),
                       "ligand_atoms": ",".join(members), "angle": round(stack[1], 2)}
                if both not in best or stack[0] < best[both][0]:
                    best[both] = (stack[0], row)
        rows.extend(row for _, row in best.values())
    closest: dict[tuple, tuple[float, dict[str, object]]] = {}
    lig_greasy = [greasy(a) for a in lig]
    env_greasy = [greasy(b) for b in env]
    for i, a in enumerate(lig):
        if not lig_greasy[i]:
            continue
        for j, b in enumerate(env):
            if not env_greasy[j] or dist[i, j] > GREASY:
                continue
            pair = (b.residue, number[a.name])
            if pair in stacked and a.name in stacked[pair][0] and b.name in stacked[pair][1]:
                continue                                      # two rings that stack: no hydrophobic contact between them
            row = {"interaction_type": "hydrophobic", "residue_number": b.resnum, "residue_name": b.resname,
                   "chain": b.chain, "insertion_code": b.icode, "protein_atom": b.name, "ligand_atom": a.name,
                   "distance": round(float(dist[i, j]), 2), "ligand_is_donor": "", "protein_atoms": "",
                   "ligand_atoms": ""}
            if pair not in closest or dist[i, j] < closest[pair][0]:
                closest[pair] = (float(dist[i, j]), row)
    rows.extend(row for _, row in closest.values())
    for row in rows:
        named = str(row["ligand_atom"]) or str(row["ligand_atoms"]).split(",")[0]
        row["_ligand_residue_name"] = comp[named]              # not a column of the table: for interface_contacts
    order = {"saltbridge": 0, "hbond": 1, "pi_stacking": 2, "hydrophobic": 3}
    rows.sort(key=lambda r: (order[str(r["interaction_type"])], str(r["chain"]), int(str(r["residue_number"])),
                             str(r["ligand_atom"]) or str(r["ligand_atoms"]), str(r["protein_atom"])))
    return rows


def interface_contacts(structure: Structure, chains_a: "Sequence[str]", chains_b: "Sequence[str]", *,
                       stacking: bool = False) -> list:
    """The same plain table for an interface between two sets of chains, as
    :class:`bindview.interface.InterfaceContact` records with side ``a`` in ``chains_a``. ``stacking`` as
    for :func:`contact_table`."""
    import re

    from .interactions import InteractionType
    from .interface import InterfaceContact

    def split(name: str) -> tuple[int, str, str]:
        m = re.fullmatch(r"(-?\d+)([A-Za-z]?)/(.+)", name)
        assert m, name
        return int(m.group(1)), m.group(2), m.group(3)

    out = []
    for chain in chains_a:
        for row in contact_table(structure, chain, partners=chains_b, stacking=stacking):
            mine = [split(n) for n in (str(row["ligand_atoms"]).split(",") if row["ligand_atoms"]
                                       else [str(row["ligand_atom"])])]
            theirs = tuple(str(row["protein_atoms"]).split(",")) if row["protein_atoms"] else (str(row["protein_atom"]),)
            donor = {"True": True, "False": False}.get(str(row["ligand_is_donor"]))
            out.append(InterfaceContact(
                type=InteractionType(str(row["interaction_type"])), chain_a=chain, number_a=mine[0][0], icode_a=mine[0][1],
                resname_a=str(row["_ligand_residue_name"]), atoms_a=tuple(m[2] for m in mine),
                chain_b=str(row["chain"]), number_b=int(str(row["residue_number"])),
                icode_b=str(row["insertion_code"]), resname_b=str(row["residue_name"]), atoms_b=theirs,
                distance=float(str(row["distance"])), a_is_donor=donor,
                angle=float(str(row["angle"])) if "angle" in row else None))
    return sorted(out, key=lambda c: c.sort_key)


# ---------------------------------------------------------------------------
# A ligand that is one residue
# ---------------------------------------------------------------------------

#: Ring to ring, the deviation of two planes from parallel or from perpendicular, and how far a centre
#: may lie off the other ring's axis; a cation over a ring; a halogen and its angle at the halogen; a
#: metal. Angstrom and degrees: the published defaults of PLIP (Salentin et al., Nucleic Acids Res. 2015).
STACK, STACK_ANGLE, OFF_AXIS, CATION, HALOGEN, HALOGEN_ANGLE, METAL = 5.5, 30.0, 2.0, 6.0, 4.0, 135.0, 3.0
HALOGEN_AT_ACCEPTOR = (90.0, 150.0)  # the angle at the acceptor, between the halogen and the atom it is bound to
_REACH = 8.0                         # residues with an atom this near are looked at
_ORDER = {"saltbridge": 0, "hbond": 1, "pi_stacking": 2, "pi_cation": 3, "halogen": 4, "metal": 5, "hydrophobic": 6}


def _plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The centre of a ring and the unit normal of its plane."""
    centre = points.mean(axis=0)
    return centre, np.linalg.svd(points - centre)[2][-1]


def _off_axis(point: np.ndarray, centre: np.ndarray, normal: np.ndarray) -> float:
    """How far a point lies from the axis of a ring: the line through its centre along its normal."""
    d = point - centre
    return float(np.sqrt(max(float(d @ d) - float(d @ normal) ** 2, 0.0)))


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """The angle at ``b`` between ``a`` and ``c``, in degrees."""
    u, v = a - b, c - b
    return float(np.degrees(np.arccos(np.clip(float(u @ v) / float(np.linalg.norm(u) * np.linalg.norm(v)), -1.0, 1.0))))


def _tilt(n1: np.ndarray, n2: np.ndarray) -> float:
    """The angle between two planes, 0 to 90 degrees."""
    return float(np.degrees(np.arccos(min(1.0, abs(float(n1 @ n2))))))


def _stacks(c1: np.ndarray, n1: np.ndarray, c2: np.ndarray, n2: np.ndarray) -> "tuple[float, float] | None":
    """Do two rings stack, by their centres and normals? Their distance and the angle between their planes
    if they do: centres within STACK, planes within STACK_ANGLE of parallel or of perpendicular, and the
    centre of one within OFF_AXIS of the other's axis (the smaller of the two). Both tables ask this."""
    apart = float(np.linalg.norm(c1 - c2))
    tilt = _tilt(n1, n2)
    off = min(_off_axis(c1, c2, n2), _off_axis(c2, c1, n1))
    if apart <= STACK and off <= OFF_AXIS and (tilt <= STACK_ANGLE or tilt >= 90.0 - STACK_ANGLE):
        return apart, tilt
    return None


def _metals_at(structure: Structure, xyz: np.ndarray) -> np.ndarray:
    """The positions of the metal atoms of the structure that have one of the atoms at ``xyz`` within METAL."""
    from .chem import is_metal

    at = np.array([a.xyz for a in structure.atoms if is_metal(a.element.capitalize())], dtype=float).reshape(-1, 3)
    return at[[bool((np.linalg.norm(xyz - m, axis=1) <= METAL).any()) for m in at]]


def _on_one_metal(metals: np.ndarray, first: np.ndarray, second: np.ndarray) -> set[tuple[int, int]]:
    """The pairs (an atom of ``first``, an atom of ``second``) that both lie within METAL of one of the metal
    atoms at ``metals``. Two such atoms are the metal's: they are about 3 A apart because it is between them,
    and that is no hydrogen bond. Both tables ask this."""
    pairs: set[tuple[int, int]] = set()
    for m in metals:
        mine = np.nonzero(np.linalg.norm(first - m, axis=1) <= METAL)[0]
        theirs = np.nonzero(np.linalg.norm(second - m, axis=1) <= METAL)[0]
        pairs.update((int(i), int(j)) for i in mine for j in theirs)
    return pairs


def ligand_contacts(structure: Structure, ligand: "str | ResidueKey | tuple[str, int]", smiles: "str | None" = None,
                    ionize_acids: bool = True) -> list[dict[str, object]]:
    """The table for a ligand that is one residue; rows in a fixed order, ready for bindview.

    ``smiles`` gives the ligand its bond orders, as it does for the diagram; left out, the file's own
    SMILES is used where it has one. Without bond orders and without hydrogens in the file the ligand
    has no rings that are known to be aromatic and no hydrogens that are known: every N and O may then
    donate or accept, and the table has no stacking and fewer salt bridges.

    An atom of the ligand and an atom of a residue that both lie within ``METAL`` (3.0 A) of one metal atom
    have no hydrogen bond between them, however near they are: both hold that metal, be it an ion beside the
    ligand, an atom of another residue (a haem's iron) or an atom of the ligand itself.
    """
    from rdkit import Chem

    from .chem import HALOGENS, is_metal
    from .features import _is_amide_like, perceive_charged_groups, perceive_metals, perceive_rings
    from .ligand import build_ligand
    from .residues import (aromatic_rings, can_accept, can_donate, canonical_atom, charged_groups, is_nucleotide,
                           template_for)

    key = ligand if isinstance(ligand, ResidueKey) else structure.select_ligand(ligand)
    if smiles is None:
        smiles = structure.chem_comp_smiles.get(structure.residue_name(key))
    lig = build_ligand(structure.residue_atoms(key), smiles, ionize_acids=ionize_acids)
    mol: Chem.Mol = lig.mol
    xyz = np.asarray(lig.coords3d, dtype=float).reshape(-1, 3)
    known = lig.bond_source != "connectivity"
    symbol = [a.GetSymbol() for a in mol.GetAtoms()]

    # -- the surroundings: residues the library has a template for, and metal ions --------------
    partners: list[tuple[ResidueKey, str, list[Any]]] = []
    for k in structure.residues():
        if k == key:
            continue
        name = structure.residue_name(k)
        heavy = [a for a in structure.residue_atoms(k) if a.element not in ("H", "D")]
        if not heavy or name.upper() in WATER_NAMES:
            continue
        ion = len(heavy) == 1 and is_metal(heavy[0].element.capitalize())
        if template_for(name) is None and not ion:
            continue
        at = np.array([a.xyz for a in heavy], dtype=float)
        if np.linalg.norm(at[:, None, :] - xyz[None, :, :], axis=2).min() <= _REACH:
            partners.append((k, name, heavy))
    polar = np.array([a.xyz for a in structure.atoms if a.element in ("N", "O")], dtype=float).reshape(-1, 3)

    def greasy(atom: Any) -> bool:
        return atom.element == "C" and not (len(polar) and (np.linalg.norm(polar - np.array(atom.xyz), axis=1) < 1.7).any())

    def row(kind: str, k: ResidueKey, name: str, distance: float, **more: object) -> dict[str, object]:
        out: dict[str, object] = {
            "interaction_type": kind, "residue_number": k.number, "residue_name": name, "chain": k.chain,
            "insertion_code": k.icode, "protein_atom": "", "ligand_atom": "", "distance": round(distance, 2),
            "ligand_is_donor": "", "protein_atoms": "", "ligand_atoms": ""}
        out.update(more)
        return out

    def names(indices: Sequence[int]) -> str:
        return ",".join(lig.names[i] for i in indices)

    # -- what the ligand is ------------------------------------------------------------------
    def donates(i: int) -> bool:
        return not known or mol.GetAtomWithIdx(i).GetTotalNumHs() > 0

    def accepts(i: int) -> bool:
        atom = mol.GetAtomWithIdx(i)
        if not known or symbol[i] == "O":
            return True
        if symbol[i] != "N" or atom.GetTotalNumHs() > 0 or atom.GetFormalCharge() > 0:
            return False
        return atom.GetDegree() == 2 if atom.GetIsAromatic() else not _is_amide_like(atom)

    safe = ("carboxylate", "guanidinium", "phosphate", "sulfonate")          # known from the connectivity alone
    groups = [g for g in perceive_charged_groups(lig)
              if g.kind != "metal" and (g.kind == "anion" or g.rank <= 1) and (known or g.kind in safe)]
    # A nitrogen that carries the charge of a guanidinium is a group by itself too: the larger one counts.
    groups = [g for g in groups if not any(set(g.atoms) < set(h.atoms) and g.charge * h.charge > 0 for h in groups)]
    bearing = {id(g): ([i for i in g.atoms if symbol[i] in ("N", "O")] or list(g.atoms)) for g in groups}
    rings = [(g.atoms, *_plane(xyz[list(g.atoms)])) for g in perceive_rings(lig) if g.rank == 0] if known else []
    metals = perceive_metals(lig)
    metals_at = _metals_at(structure, xyz)        # the ligand's own and those beside it, where it lies on one

    rows: list[dict[str, object]] = []
    for k, name, heavy in partners:
        salted: set[tuple[int, int]] = set()          # (atom of the ligand, atom of this residue)
        at = np.array([a.xyz for a in heavy], dtype=float)
        dist = np.linalg.norm(xyz[:, None, :] - at[None, :, :], axis=2)              # (ligand, residue)
        by_name = {canonical_atom(name, a.name): j for j, a in enumerate(heavy)}
        given = {canonical_atom(name, a.name): a.name for a in heavy}
        if len(heavy) == 1 and is_metal(heavy[0].element.capitalize()):               # 7: a metal that is a residue
            near = [i for i in range(len(xyz)) if symbol[i] in ("N", "O", "S") and dist[i, 0] <= METAL]
            rows.extend(row("metal", k, name, float(dist[i, 0]), protein_atom=heavy[0].name, ligand_atom=lig.names[i])
                        for i in near)
            continue
        # 2: salt bridges, an atom of one group to an atom of the other
        res_groups = []
        for sign, wanted in charged_groups(name):
            there = [n for n in wanted if canonical_atom(name, n) in by_name]
            ends = [by_name[canonical_atom(name, n)] for n in there if heavy[by_name[canonical_atom(name, n)]].element in ("N", "O")]
            if ends:
                res_groups.append((sign, list(dict.fromkeys(given[canonical_atom(name, n)] for n in there)),
                                   list(dict.fromkeys(ends))))
        for g in groups:
            for sign, members, ends in res_groups:
                if sign * g.charge >= 0:
                    continue
                sub = dist[np.ix_(bearing[id(g)], ends)]
                if sub.min() > SALT:
                    continue
                salted.update((i, j) for i in bearing[id(g)] for j in ends)
                rows.append(row("saltbridge", k, name, float(sub.min()), protein_atoms=",".join(members),
                                ligand_atoms=names(g.atoms)))
        # 1: hydrogen bonds between N and O, but not between two atoms on one metal
        on_one_metal = _on_one_metal(metals_at, xyz, at)
        for i in range(len(xyz)):
            if symbol[i] not in ("N", "O"):
                continue
            for j, b in enumerate(heavy):
                if b.element not in ("N", "O") or dist[i, j] > HBOND or (i, j) in salted or (i, j) in on_one_metal:
                    continue
                gives, takes = can_donate(name, b.name), can_accept(name, b.name, b.element)
                one = donates(i) and takes is not False                  # the ligand donates
                other = gives is not False and accepts(i)                # the residue donates
                if not (one or other):
                    continue
                rows.append(row("hbond", k, name, float(dist[i, j]), protein_atom=b.name, ligand_atom=lig.names[i],
                                ligand_is_donor="" if one and other else str(bool(one))))
        # 4 and 5: rings
        res_rings = []
        for ring in aromatic_rings(name):
            if all(n in by_name for n in ring):
                res_rings.append(([given[n] for n in ring], *_plane(at[[by_name[n] for n in ring]])))
        for atoms, centre, normal in rings:
            for members, c2, n2 in res_rings:
                stack = _stacks(centre, normal, c2, n2)
                if stack is not None:
                    rows.append(row("pi_stacking", k, name, stack[0], protein_atoms=",".join(members),
                                    ligand_atoms=names(atoms), angle=round(stack[1], 2)))
            for sign, members, ends in res_groups:
                if sign < 0 or is_nucleotide(name):
                    continue
                centre2 = at[ends].mean(axis=0)
                apart = float(np.linalg.norm(centre - centre2))
                if apart <= CATION and _off_axis(centre2, centre, normal) <= OFF_AXIS:
                    rows.append(row("pi_cation", k, name, apart, protein_atoms=",".join(members),
                                    ligand_atoms=names(atoms)))
        for g in groups:
            if g.charge <= 0:
                continue
            centre2 = g.center3d(xyz)
            for members, c2, n2 in res_rings:
                apart = float(np.linalg.norm(centre2 - c2))
                if apart <= CATION and _off_axis(centre2, c2, n2) <= OFF_AXIS:
                    rows.append(row("pi_cation", k, name, apart, protein_atoms=",".join(members),
                                    ligand_atoms=names(g.atoms)))
        # 6: a halogen on a carbon, and what it points at
        for i in range(len(xyz)):
            if symbol[i] not in HALOGENS:
                continue
            carbons = [n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors() if n.GetSymbol() == "C"]
            best: tuple[float, int] | None = None
            for j, b in enumerate(heavy):
                if b.element not in ("N", "O", "S") or dist[i, j] > HALOGEN or not carbons:
                    continue
                if _angle(xyz[carbons[0]], xyz[i], at[j]) < HALOGEN_ANGLE:
                    continue
                bound = [m for m in range(len(heavy)) if m != j and np.linalg.norm(at[m] - at[j]) < 1.9]
                if not any(HALOGEN_AT_ACCEPTOR[0] <= _angle(xyz[i], at[j], at[m]) <= HALOGEN_AT_ACCEPTOR[1] for m in bound):
                    continue
                if best is None or dist[i, j] < best[0]:
                    best = (float(dist[i, j]), j)
            if best is not None:
                rows.append(row("halogen", k, name, best[0], protein_atom=heavy[best[1]].name, ligand_atom=lig.names[i]))
        # 7: a metal of the ligand
        for i in metals:
            rows.extend(row("metal", k, name, float(dist[i, j]), protein_atom=b.name, ligand_atom=lig.names[i])
                        for j, b in enumerate(heavy) if b.element in ("N", "O", "S") and dist[i, j] <= METAL)
        # 3: carbon to carbon, the closest pair of a residue
        mine = [i for i in range(len(xyz)) if symbol[i] == "C"
                and not any(n.GetSymbol() in ("N", "O") for n in mol.GetAtomWithIdx(i).GetNeighbors())]
        theirs = [j for j, b in enumerate(heavy) if greasy(b)]
        if mine and theirs:
            sub = dist[np.ix_(mine, theirs)]
            if sub.min() <= GREASY:
                a, b2 = np.unravel_index(int(sub.argmin()), sub.shape)
                rows.append(row("hydrophobic", k, name, float(sub.min()), protein_atom=heavy[theirs[int(b2)]].name,
                                ligand_atom=lig.names[mine[int(a)]]))
    rows.sort(key=lambda r: (_ORDER[str(r["interaction_type"])], str(r["chain"]), int(str(r["residue_number"])),
                             str(r["ligand_atom"]) or str(r["ligand_atoms"]), str(r["protein_atom"]) or str(r["protein_atoms"])))
    return rows


def as_tsv(rows: list[dict[str, object]]) -> str:
    lines = ["\t".join(COLUMNS)]
    lines += ["\t".join(str(row.get(c, "")) for c in COLUMNS) for row in rows]
    return "\n".join(lines) + "\n"
