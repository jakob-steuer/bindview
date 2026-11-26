"""Optional interaction detection with PLIP.

**Licence boundary.** PLIP is distributed under the GPL-2.0. Nothing else in
this package imports it: the core (structure reading, ligand building, anchor
resolution, layout, rendering) works on interaction records supplied by the
caller and carries no GPL dependency. This module is the only place PLIP is
touched, it imports PLIP when a detection is run and not when it is loaded
itself, and it is only reached when a caller passes ``interactions=None``.
Install it with ``pip install "bindview[plip]"`` (in a clone of the repository,
``pip install ".[plip]"``) if you want detection and accept PLIP's terms.

The adapter hands PLIP an in-memory PDB rendering of the structure; nothing is
written to disk and the structure file is never modified.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from .errors import DetectionUnavailableError, BindviewError, StructureError
from .exposure import WATER_NAMES
from .interactions import Interaction, InteractionType
from .structure import Atom, ResidueKey, Structure, read_structure

__all__ = ["detect_interactions", "structure_to_pdb"]

T = InteractionType
_PLIP_HETID = "LIG"


def _atom_name_field(name: str, element: str) -> str:
    """PDB columns 13-16: one-letter elements start in column 14."""
    if len(name) >= 4:
        return name[:4]
    if len(element) == 2:
        return name.ljust(4)
    return (" " + name).ljust(4)


def structure_to_pdb(structure: Structure, ligand: ResidueKey) -> tuple[str, dict[int, Atom], str]:
    """Render a structure as PDB text for a detector that only reads PDB.

    Returns ``(text, serial -> atom, ligand chain id as written)``. Component
    names longer than three characters (AlphaFold 3 writes ``LIG_B``) do not
    fit the format, so the ligand is written under the name ``LIG``.
    """
    atoms = [a for a in structure.atoms if a.element not in ("H", "D")]
    if len(atoms) > 99999:
        raise StructureError("structure has more than 99,999 heavy atoms and cannot be written as PDB for detection")
    chains: dict[str, str] = {}
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    for a in atoms:
        if a.chain not in chains:
            if len(a.chain) == 1 and a.chain not in chains.values():
                chains[a.chain] = a.chain
            else:
                free = [c for c in alphabet if c not in chains.values()]
                if not free:
                    raise StructureError("too many chains to write as PDB for detection")
                chains[a.chain] = free[0]
    lines: list[str] = []
    by_serial: dict[int, Atom] = {}
    for serial, a in enumerate(atoms, start=1):
        if not -999 <= a.resnum <= 9999:
            raise StructureError(f"residue number {a.resnum} does not fit the PDB format needed for detection")
        is_ligand = a.residue == ligand
        resname = _PLIP_HETID if is_ligand else a.resname[:3]
        record = "HETATM" if (a.hetero or is_ligand) else "ATOM  "
        lines.append(
            f"{record}{serial:5d} {_atom_name_field(a.name, a.element)} {resname:>3s} {chains[a.chain]}"
            f"{a.resnum:4d}{(a.icode or ' ')[:1]}   {a.x:8.3f}{a.y:8.3f}{a.z:8.3f}  1.00  0.00          "
            f"{a.element.upper():>2s}")
        by_serial[serial] = a
    lines.append("END")
    return "\n".join(lines) + "\n", by_serial, chains[ligand.chain]


def _load_plip() -> tuple[Any, Any]:
    try:
        from plip.basic import config
        from plip.structure.preparation import PDBComplex
    except ImportError as exc:
        raise DetectionUnavailableError(
            "interaction detection needs PLIP, which is an optional dependency under the GPL-2.0: "
            "install it with `pip install \"bindview[plip]\"` (in a clone of the repository, `pip install \".[plip]\"`), "
            "pass your own interactions "
            "(interactions=[...]), or have a plain table made from the distances of the heavy atoms "
            "(interactions=\"distances\", --distances-only)") from exc
    return config, PDBComplex


def detect_interactions(structure: "Structure | str | Path", ligand: "ResidueKey | str | tuple[str, int]",
                        receptor: str = "protein") -> list[Interaction]:
    """Detect protein-ligand interactions with PLIP and return them as records.

    ``receptor="nucleic"`` is for a ligand in RNA or DNA: PLIP then takes the nucleic acid for the
    receptor (its ``--dnareceptor``). Left alone it looks at protein only, and finds nothing there.

    Raises :class:`DetectionUnavailableError` when PLIP is not installed. An
    empty list is a real answer: PLIP ran and found nothing.
    """
    if receptor not in ("protein", "nucleic"):
        raise ValueError(f"receptor is 'protein' or 'nucleic', not {receptor!r}")
    config, PDBComplex = _load_plip()
    st = structure if isinstance(structure, Structure) else read_structure(structure)
    key = ligand if isinstance(ligand, ResidueKey) else st.select_ligand(ligand)
    text, by_serial, chain = structure_to_pdb(st, key)

    config.NOFIXFILE = True            # never write PLIP's "fixed" copy of the input
    before = getattr(config, "DNARECEPTOR", False)
    config.DNARECEPTOR = receptor == "nucleic"
    try:
        mol = PDBComplex()
        mol.load_pdb(text, as_string=True)
        target = None
        for lig in mol.ligands:
            if lig.hetid == _PLIP_HETID and lig.chain == chain and int(lig.position) == key.number:
                target = lig
                break
        if target is None:
            raise BindviewError(f"PLIP did not recognise residue {key} as a ligand")
        mol.characterize_complex(target)
        site = mol.interaction_sets[f"{target.hetid}:{target.chain}:{target.position}"]
    except BindviewError:
        raise
    except Exception as exc:  # PLIP / OpenBabel raise assorted exception types
        raise BindviewError(f"PLIP failed on this structure: {exc}") from exc
    finally:
        config.DNARECEPTOR = before        # PLIP keeps it for the process: the next call says what it wants

    def name(serial: int) -> str | None:
        atom = by_serial.get(int(serial))
        return atom.name if atom is not None else None

    def names(serials: Any) -> tuple[str, ...]:
        return tuple(n for n in (name(s) for s in serials) if n)

    def base(kind: InteractionType, c: Any, at: Any) -> dict[str, Any]:
        """What every record has. PLIP has the chain as it was written for it, with one letter; the name the
        structure has for it is that of the atom the contact has on the residue's side (``at``, its serial)."""
        atom = by_serial.get(int(at))
        return {"type": kind, "residue_number": int(c.resnr), "residue_name": str(c.restype),
                "chain": atom.chain if atom is not None else str(c.reschain)}

    def on_ligand(serial: Any) -> bool:
        atom = by_serial.get(int(serial))
        return atom is not None and atom.residue == key

    out: list[Interaction] = []
    for c in site.hydrophobic_contacts:
        out.append(Interaction(**base(T.HYDROPHOBIC, c, c.bsatom_orig_idx), protein_atom=name(c.bsatom_orig_idx),
                               ligand_atom=name(c.ligatom_orig_idx), distance=round(float(c.distance), 2)))
    # A hydroxyl facing a hydroxyl is reported once in each direction; keep the
    # assignment with the straighter donor angle so the pair is one interaction.
    hbonds: dict[tuple[int, int], Interaction] = {}
    for c in list(site.hbonds_ldon) + list(site.hbonds_pdon):
        lig_is_donor = not bool(c.protisdon)
        lig_serial = c.d_orig_idx if lig_is_donor else c.a_orig_idx
        prot_serial = c.a_orig_idx if lig_is_donor else c.d_orig_idx
        rec = Interaction(**base(T.HBOND, c, prot_serial), protein_atom=name(prot_serial),
                          ligand_atom=name(lig_serial), distance=round(float(c.distance_ad), 2),
                          angle=round(float(c.angle), 2), ligand_is_donor=lig_is_donor)
        pair = (int(prot_serial), int(lig_serial))
        if pair not in hbonds or (rec.angle or 0.0) > (hbonds[pair].angle or 0.0):
            hbonds[pair] = rec
    out.extend(hbonds.values())
    for c in site.halogen_bonds:
        out.append(Interaction(**base(T.HALOGEN, c, c.acc_orig_idx), protein_atom=name(c.acc_orig_idx),
                               ligand_atom=name(c.don_orig_idx), distance=round(float(c.distance), 2),
                               angle=round(float(c.don_angle), 2)))
    for c in list(site.saltbridge_lneg) + list(site.saltbridge_pneg):
        prot_group = c.positive if c.protispos else c.negative
        lig_group = c.negative if c.protispos else c.positive
        out.append(Interaction(**base(T.SALTBRIDGE, c, prot_group.atoms_orig_idx[0]),
                               distance=round(float(c.distance), 2),
                               protein_atoms=names(prot_group.atoms_orig_idx),
                               ligand_atoms=names(lig_group.atoms_orig_idx)))
    for c in site.pistacking:
        out.append(Interaction(**base(T.PI_STACKING, c, c.proteinring.atoms_orig_idx[0]),
                               distance=round(float(c.distance), 2), angle=round(float(c.angle), 2),
                               protein_atoms=names(c.proteinring.atoms_orig_idx),
                               ligand_atoms=names(c.ligandring.atoms_orig_idx)))
    for c in list(site.pication_laro) + list(site.pication_paro):
        ring_names, charge_names = names(c.ring.atoms_orig_idx), names(c.charge.atoms_orig_idx)
        prot, lig = (charge_names, ring_names) if c.protcharged else (ring_names, charge_names)
        theirs = c.charge if c.protcharged else c.ring
        out.append(Interaction(**base(T.PI_CATION, c, theirs.atoms_orig_idx[0]), distance=round(float(c.distance), 2),
                               protein_atoms=prot, ligand_atoms=lig))
    for c in site.metal_complexes:
        if on_ligand(c.metal_orig_idx) and on_ligand(c.target_orig_idx):
            continue        # a metal held by its own ligand (cobalamin's Co-N) is not a protein contact
        metal, partner = name(c.metal_orig_idx), name(c.target_orig_idx)
        metal_on_ligand = on_ligand(c.metal_orig_idx)
        out.append(Interaction(**base(T.METAL, c, c.target_orig_idx), distance=round(float(c.distance), 2),
                               ligand_atom=metal if metal_on_ligand else partner,
                               protein_atom=partner if metal_on_ligand else metal))
    for c in site.water_bridges:
        lig_is_donor = not bool(c.protisdon)
        lig_serial = c.d_orig_idx if lig_is_donor else c.a_orig_idx
        prot_serial = c.a_orig_idx if lig_is_donor else c.d_orig_idx
        water = by_serial.get(int(c.water_orig_idx))
        out.append(Interaction(**base(T.WATER_BRIDGE, c, prot_serial), protein_atom=name(prot_serial),
                               ligand_atom=name(lig_serial),
                               distance=round(float(c.distance_aw if lig_is_donor else c.distance_dw), 2),
                               angle=round(float(c.w_angle), 2), ligand_is_donor=lig_is_donor,
                               water=f"{water.chain}:{water.resnum}" if water else None))
    out.sort(key=lambda r: (r.type.value, r.chain, r.residue_number, r.protein_atom or "", r.ligand_atom or "",
                            r.distance if r.distance is not None else 0.0))
    return out


# ---------------------------------------------------------------------------
# Two sets of chains: an interface
# ---------------------------------------------------------------------------

#: PLIP takes a chain as a "peptide ligand" only residue by residue that OpenBabel knows as an
#: amino acid. A modified residue is dropped from it without a word, and its contacts with it (the
#: sulfate of a sulfotyrosine, the phosphate of a phosphotyrosine: the recognition feature). Written
#: under the name of its parent it stays in the chain with all its atoms.
_PARENTS = {"TYS": "TYR", "PTR": "TYR", "SEP": "SER", "TPO": "THR", "MSE": "MET", "CSO": "CYS", "HYP": "PRO",
            "ALY": "LYS", "MLY": "LYS", "M3L": "LYS"}


def _interface_pdb(structure: Structure, peptides: "set[str]") -> tuple[str, dict[int, Atom], dict[str, str]]:
    """The structure as PDB text for PLIP's peptide mode: ``(text, serial -> atom, chain -> chain as written)``.

    Two things are not written as the file has them, because PLIP gets them
    wrong otherwise. Residues are numbered 1, 2, 3 ... within their chain: with
    insertion codes (thrombin's 14A to 14L) PLIP collapses residues into one
    and reports contacts that are not there. And a modified residue of a chain
    that is to be a peptide is written as an ATOM record under its parent's
    name (:data:`_PARENTS`). Every atom is mapped back by its serial number,
    so none of this reaches the result.
    """
    atoms = [a for a in structure.atoms if a.element not in ("H", "D")]
    if len(atoms) > 99999:
        raise StructureError("structure has more than 99,999 heavy atoms and cannot be written as PDB for detection")
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    chains: dict[str, str] = {}
    for a in atoms:
        if a.chain not in chains:
            if len(a.chain) == 1 and a.chain not in chains.values():
                chains[a.chain] = a.chain
            else:
                free = [c for c in alphabet if c not in chains.values()]
                if not free:
                    raise StructureError("too many chains to write as PDB for detection")
                chains[a.chain] = free[0]
    number: dict[Any, int] = {}
    count: dict[str, int] = {}
    lines: list[str] = []
    by_serial: dict[int, Atom] = {}
    for serial, a in enumerate(atoms, start=1):
        if a.residue not in number:
            count[a.chain] = count.get(a.chain, 0) + 1
            number[a.residue] = count[a.chain]
            if count[a.chain] > 9999:
                raise StructureError(f"chain {a.chain} has more than 9,999 residues and cannot be written as PDB")
        inside = a.chain in peptides and a.resname.upper() not in WATER_NAMES
        resname = _PARENTS.get(a.resname, a.resname) if inside else a.resname
        record = "ATOM  " if (inside or not a.hetero) else "HETATM"
        lines.append(
            f"{record}{serial:5d} {_atom_name_field(a.name, a.element)} {resname[:3]:>3s} {chains[a.chain]}"
            f"{number[a.residue]:4d}    {a.x:8.3f}{a.y:8.3f}{a.z:8.3f}  1.00  0.00          {a.element.upper():>2s}")
        by_serial[serial] = a
    lines.append("END")
    return "\n".join(lines) + "\n", by_serial, chains


def detect_interface(structure: "Structure | str | Path", chains_a: "Sequence[str]", chains_b: "Sequence[str]"
                     ) -> "list[Any]":
    """Contacts between two sets of chains, detected with PLIP's peptide mode.

    PLIP is a protein-ligand profiler, but a chain can be named as a "peptide
    ligand", of any length, and then its contacts with every other chain are
    reported with the residue and the atoms on both sides. The smaller of the
    two sets is named so; contacts with chains of neither set are dropped.
    Returns :class:`bindview.interface.InterfaceContact` records with side ``a``
    in ``chains_a``.

    What PLIP does not do here: a cap (``ACE``) is not part of the peptide and
    its contacts are not reported, and a phosphate or sulfate on a residue is
    not seen as a charged group, so its salt bridges come out as hydrogen bonds.
    """
    from .interface import InterfaceContact

    config, PDBComplex = _load_plip()
    st = structure if isinstance(structure, Structure) else read_structure(structure)
    a_set, b_set = set(chains_a), set(chains_b)
    if not a_set or not b_set or a_set & b_set:
        raise BindviewError("an interface is between two different, non-empty sets of chains")
    present = {a.chain for a in st.atoms}
    missing = sorted((a_set | b_set) - present)
    if missing:
        raise BindviewError(f"no chain {', '.join(missing)} in {st.source or 'the structure'}")
    size = {c: 0 for c in a_set | b_set}
    for atom in st.atoms:
        if atom.chain in size:
            size[atom.chain] += 1
    flipped = sum(size[c] for c in b_set) < sum(size[c] for c in a_set)
    peptides, partners = (b_set, a_set) if flipped else (a_set, b_set)
    text, by_serial, written = _interface_pdb(st, peptides)
    before = (config.PEPTIDES, config.NOFIXFILE)
    config.PEPTIDES = sorted(written[c] for c in peptides)
    config.NOFIXFILE = True
    out: list[Any] = []

    def side(serials: Any) -> "tuple[Atom, tuple[str, ...]] | None":
        found = [x for x in (by_serial.get(int(s)) for s in serials) if x is not None]
        return (found[0], tuple(x.name for x in found)) if found else None

    def add(kind: InteractionType, pep: Any, other: Any, **more: Any) -> None:
        if pep is None or other is None or pep[0].chain not in peptides or other[0].chain not in partners:
            return
        one, two = (other, pep) if flipped else (pep, other)
        donor = more.pop("peptide_donates", None)
        out.append(InterfaceContact(
            type=kind, chain_a=one[0].chain, number_a=one[0].resnum, icode_a=one[0].icode, resname_a=one[0].resname,
            atoms_a=one[1], chain_b=two[0].chain, number_b=two[0].resnum, icode_b=two[0].icode,
            resname_b=two[0].resname, atoms_b=two[1],
            a_is_donor=None if donor is None else (donor != flipped), **more))

    try:
        mol = PDBComplex()
        mol.load_pdb(text, as_string=True)
        for lig in mol.ligands:
            if lig.chain not in config.PEPTIDES:
                continue
            mol.characterize_complex(lig)
            site = mol.interaction_sets[f"{lig.hetid}:{lig.chain}:{lig.position}"]
            for c in site.hydrophobic_contacts:
                add(T.HYDROPHOBIC, side([c.ligatom_orig_idx]), side([c.bsatom_orig_idx]),
                    distance=round(float(c.distance), 2))
            for c in list(site.hbonds_ldon) + list(site.hbonds_pdon):
                pep_donates = not bool(c.protisdon)
                pep_serial = c.d_orig_idx if pep_donates else c.a_orig_idx
                other_serial = c.a_orig_idx if pep_donates else c.d_orig_idx
                add(T.HBOND, side([pep_serial]), side([other_serial]), distance=round(float(c.distance_ad), 2),
                    angle=round(float(c.angle), 2), peptide_donates=pep_donates)
            for c in list(site.saltbridge_lneg) + list(site.saltbridge_pneg):
                other_group = c.positive if c.protispos else c.negative
                pep_group = c.negative if c.protispos else c.positive
                add(T.SALTBRIDGE, side(pep_group.atoms_orig_idx), side(other_group.atoms_orig_idx),
                    distance=round(float(c.distance), 2))
            for c in site.pistacking:
                add(T.PI_STACKING, side(c.ligandring.atoms_orig_idx), side(c.proteinring.atoms_orig_idx),
                    distance=round(float(c.distance), 2), angle=round(float(c.angle), 2))
            for c in list(site.pication_laro) + list(site.pication_paro):
                ring, charge = side(c.ring.atoms_orig_idx), side(c.charge.atoms_orig_idx)
                pep, other = (ring, charge) if c.protcharged else (charge, ring)
                add(T.PI_CATION, pep, other, distance=round(float(c.distance), 2))
            for c in site.water_bridges:
                pep_donates = not bool(c.protisdon)
                pep_serial = c.d_orig_idx if pep_donates else c.a_orig_idx
                other_serial = c.a_orig_idx if pep_donates else c.d_orig_idx
                water = by_serial.get(int(c.water_orig_idx))
                add(T.WATER_BRIDGE, side([pep_serial]), side([other_serial]),
                    distance=round(float(c.distance_aw if pep_donates else c.distance_dw), 2),
                    angle=round(float(c.w_angle), 2), peptide_donates=pep_donates,
                    water=f"{water.chain}:{water.resnum}" if water else None)
    except BindviewError:
        raise
    except Exception as exc:  # PLIP / OpenBabel raise assorted exception types
        raise BindviewError(f"PLIP failed on this structure: {exc}") from exc
    finally:
        config.PEPTIDES, config.NOFIXFILE = before
    # A hydroxyl facing a hydroxyl is reported once in each direction: keep the straighter one.
    best: dict[tuple, Any] = {}
    rest: list[Any] = []
    for c in out:
        if c.type is not T.HBOND:
            rest.append(c)
            continue
        ident = (c.residue_a, c.atoms_a, c.residue_b, c.atoms_b)
        if ident not in best or (c.angle or 0.0) > (best[ident].angle or 0.0):
            best[ident] = c
    return sorted(rest + list(best.values()), key=lambda c: c.sort_key)
