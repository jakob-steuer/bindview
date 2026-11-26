"""A peptide ligand that is a polymer chain of its own in the structure file.

A co-folding model writes a peptide ligand as one component with one residue
record, and its residues have to be perceived (:mod:`bindview.peptide`). A
deposited structure writes it as what it is: a chain of residues, each with
its name and its number. That is the easier case and the one a crystal
structure is uploaded in, and both end in the same two things:

* one :class:`~bindview.ligand.Ligand`, a lump of named atoms. The chain's atoms
  get names that are unique in it (``63/OE1``:
  :meth:`bindview.structure.Structure.with_chain_as_ligand`), and since the file
  has no SMILES for a chain, :func:`chain_smiles` writes one from the names of
  the residues and their atoms, so that bond orders are known;
* one :class:`~bindview.peptide.Peptide`, the decomposition into residues. Here
  nothing of it is perceived (:func:`given_peptide`): a residue is what the
  file says it is, with all the atoms the file gives it, under the name and
  the number the file gives it. A component of the chain without a backbone of
  its own, an acetyl cap for one, is not a residue but something that hangs on
  one, exactly as the perceived route has it.

The numbers are then the deposited ones, and the figure says so; for a
perceived chain they are a count, and the figure says that.
"""

from __future__ import annotations

from typing import Sequence

from rdkit import Chem

from .chem import quiet_rdkit
from .peptide import _ONE, _OTHERS, Appendage, Crosslink, Peptide, PeptideResidue, _stereo
from .residues import _NUCLEOTIDE_TEMPLATES, _TEMPLATES, canonical_atom, canonical_resname
from .structure import Atom, ChainResidue

__all__ = ["chain_smiles", "given_peptide", "KNOWN_COMPONENTS"]

#: Components a chain may hold beside the twenty and the residues :mod:`bindview.peptide` can name:
#: the common modified residues, and caps. SMILES atom order as in :mod:`bindview.residues`
#: (N, CA, C, O, then the side chain) for a residue; a cap has no backbone.
_MODIFIED: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("PTR", "NC(C=O)Cc1ccc(OP(=O)(O)O)cc1",
     ("N", "CA", "C", "O", "CB", "CG", "CD1", "CE1", "CZ", "OH", "P", "O1P", "O2P", "O3P", "CE2", "CD2")),
    ("TYS", "NC(C=O)Cc1ccc(OS(=O)(=O)O)cc1",
     ("N", "CA", "C", "O", "CB", "CG", "CD1", "CE1", "CZ", "OH", "S", "O1", "O2", "O3", "CE2", "CD2")),
    ("SEP", "NC(C=O)COP(=O)(O)O", ("N", "CA", "C", "O", "CB", "OG", "P", "O1P", "O2P", "O3P")),
    ("TPO", "NC(C=O)C(C)OP(=O)(O)O", ("N", "CA", "C", "O", "CB", "CG2", "OG1", "P", "O1P", "O2P", "O3P")),
    ("CSO", "NC(C=O)CSO", ("N", "CA", "C", "O", "CB", "SG", "OD")),
    ("ALY", "NC(C=O)CCCCNC(C)=O", ("N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ", "CH", "CH3", "OH")),
    ("MLY", "NC(C=O)CCCCN(C)C", ("N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ", "CH1", "CH2")),
    ("M3L", "NC(C=O)CCCC[N+](C)(C)C", ("N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ", "CM1", "CM2", "CM3")),
    ("ACE", "CC=O", ("CH3", "C", "O")),
    ("NH2", "N", ("N",)),
    ("NME", "NC", ("N", "C")),
)


def _components() -> dict[str, tuple[str, tuple[str, ...]]]:
    out = {t.name: (t.smiles, t.atoms) for t in _TEMPLATES.values()}
    out.update({name: (smiles, atoms) for name, smiles, atoms in _OTHERS})
    out.update({name: (smiles, atoms) for name, smiles, atoms in _MODIFIED})
    out.update({name: (t.smiles, t.atoms) for name, t in _NUCLEOTIDE_TEMPLATES.items()})
    return out


#: Every component whose bonds are known here by the names of its atoms.
KNOWN_COMPONENTS = _components()
_COVALENT = 1.9
_COVALENT_BIG = 2.3            # where sulfur, phosphorus or selenium takes part


def _local(name: str) -> str:
    """``OE1`` from ``63/OE1``: an atom's name within its residue."""
    return name.split("/", 1)[1] if "/" in name else name


def chain_smiles(atoms: Sequence[Atom], residues: Sequence[ChainResidue]) -> tuple[str | None, list[str]]:
    """A SMILES for a chain ligand, written from the names of its residues and atoms.

    ``atoms`` are the atoms of the merged ligand, ``residues`` what the file
    said about them. Within a residue the bonds and their orders are those of
    the component of that name, between the atoms that are present (a side
    chain modelled only to its beta carbon is that much shorter). Between
    residues every pair of atoms at bonding distance is bonded, singly: the
    peptide bond, a disulfide, the bond of a cap.

    Returns ``(None, notes)`` when a residue's component is not known here,
    one of its atoms is not an atom of that component, or the file gives an
    atom no element: bond orders are then not invented, and the ligand is
    built from its connectivity alone.
    """
    notes: list[str] = []
    by_name = {a.name: a for a in atoms if a.element not in ("H", "D")}
    rw = Chem.RWMol()
    index: dict[str, int] = {}
    with quiet_rdkit():
        for res in residues:
            known = KNOWN_COMPONENTS.get(res.name.upper())
            nucleotide = canonical_resname(res.name) in _NUCLEOTIDE_TEMPLATES
            if nucleotide:                      # under whatever name a force field gives it
                known = KNOWN_COMPONENTS[canonical_resname(res.name)]
            if known is None:
                notes.append(f"{res.name} {res.number} is not a component whose bonds are known here; "
                             f"bond orders of the ligand are not known")
                return None, notes
            smiles, names = known
            template = Chem.MolFromSmiles(smiles)
            Chem.Kekulize(template, clearAromaticFlags=True)
            order = {n: k for k, n in enumerate(names)}
            if nucleotide:
                # A file may name the two free oxygens of the phosphate, or the primes of the sugar, its own
                # way. A phosphate that ends a chain has a third oxygen, as a carboxyl end has its OXT.
                given = {canonical_atom(res.name, _local(n)): n for n in res.atoms}
                order = {given[n].split("/", 1)[-1]: k for n, k in order.items() if n in given}
                names = tuple(given[n].split("/", 1)[-1] if n in given else n for n in names)
            present = [n for n in res.atoms if _local(n) in order]
            ends = ("OXT", "OP3", "O3P") if nucleotide else ("OXT",)
            extra = [n for n in res.atoms if _local(n) not in order and _local(n) not in ends]
            if extra:
                notes.append(f"{res.name} {res.number} has an atom {_local(extra[0])} that the component of that "
                             f"name does not have; bond orders of the ligand are not known")
                return None, notes
            for n in res.atoms:
                element = by_name[n].element
                try:
                    index[n] = rw.AddAtom(Chem.Atom(element[:1].upper() + element[1:].lower()))
                except RuntimeError:            # RDKit has no such element: the file does not say what the atom is
                    notes.append(f"{res.name} {res.number} has an atom {_local(n)} of no element that is known "
                                 f"({element!r}); bond orders of the ligand are not known")
                    return None, notes
            for n in present:
                atom = template.GetAtomWithIdx(order[_local(n)])
                rw.GetAtomWithIdx(index[n]).SetFormalCharge(atom.GetFormalCharge())
            prefix = res.atoms[0].split("/", 1)[0] + "/" if "/" in res.atoms[0] else ""
            for bond in template.GetBonds():
                a, b = prefix + names[bond.GetBeginAtomIdx()], prefix + names[bond.GetEndAtomIdx()]
                if a in index and b in index:
                    rw.AddBond(index[a], index[b], bond.GetBondType())
            if prefix + "OXT" in index and prefix + "C" in index:
                rw.AddBond(index[prefix + "OXT"], index[prefix + "C"], Chem.BondType.SINGLE)
            for third in ("OP3", "O3P") if nucleotide else ():
                if prefix + third in index and prefix + "P" in index:
                    rw.AddBond(index[prefix + third], index[prefix + "P"], Chem.BondType.SINGLE)
        owner = {n: k for k, res in enumerate(residues) for n in res.atoms}
        backbone = [{"N", "CA", "C"} <= {_local(n) for n in res.atoms} for res in residues]
        names_in_order = list(index)
        for i, a in enumerate(names_in_order):
            for b in names_in_order[i + 1:]:
                if owner[a] == owner[b]:
                    continue
                p, q = by_name[a], by_name[b]
                pair = {p.element.upper(), q.element.upper()}
                d = ((p.x - q.x) ** 2 + (p.y - q.y) ** 2 + (p.z - q.z) ** 2) ** 0.5
                if backbone[owner[a]] and backbone[owner[b]]:
                    # Two residues are joined by a peptide (or isopeptide) bond, a disulfide or a thioether,
                    # and by nothing else. Atoms of two residues that merely lie on top of each other, as a
                    # predicted model has them where it is unsure, are a clash and not a bond.
                    if not ((pair == {"C", "N"} and d < 1.7) or (pair == {"S"} and d < 2.3)
                            or (pair == {"C", "S"} and d < 1.95)):
                        continue
                elif d >= (_COVALENT_BIG if pair & {"S", "P", "SE"} else _COVALENT):
                    continue                    # a cap or the like: whatever it is bonded to, by distance
                rw.AddBond(index[a], index[b], Chem.BondType.SINGLE)
        try:
            mol = rw.GetMol()
            Chem.SanitizeMol(mol)
        except Exception as exc:                      # a valence that cannot be: the file's atoms do not add up
            notes.append(f"the residues of the chain do not make a molecule ({str(exc).splitlines()[0][:80]}); "
                         f"bond orders of the ligand are not known")
            return None, notes
        return Chem.MolToSmiles(mol), notes


def given_peptide(mol: Chem.Mol, names: Sequence[str], residues: Sequence[ChainResidue]) -> Peptide | None:
    """The decomposition of a chain ligand as the file gives it.

    ``mol`` and ``names`` are the ligand's molecule and atom names. A residue
    of the file that has a backbone (``N``, ``CA``, ``C``) is a residue of the
    chain, with every atom the file gives it; what has none hangs on the
    residue it is bonded to. ``None`` if no residue has a backbone.
    """
    idx = {n: i for i, n in enumerate(names)}
    members = [[idx[n] for n in res.atoms if n in idx] for res in residues]
    found: list[tuple[ChainResidue, list[int], dict[str, int]]] = []
    loose: set[int] = set()
    for res, atoms in zip(residues, members):
        local = {_local(names[a]): a for a in atoms}
        sugar = {n.replace("*", "'"): a for n, a in local.items()}
        if {"N", "CA", "C"} <= set(local):
            found.append((res, atoms, local))
        elif {"C1'", "C3'", "C4'"} <= set(sugar):
            # A nucleotide, by its sugar and not by its name: its atoms under the names of the templates.
            found.append((res, atoms, {canonical_atom(canonical_resname(res.name), n) if canonical_resname(res.name)
                                       in _NUCLEOTIDE_TEMPLATES else n: a for n, a in sugar.items()}))
        else:
            loose.update(atoms)
    # An atom of the ligand that is of no residue of the file: an ion that was taken into the ligand.
    loose.update(set(range(len(names))) - {a for atoms in members for a in atoms})
    if not found:
        return None
    out: list[PeptideResidue] = []
    for k, (res, atoms, local) in enumerate(found, start=1):
        name = res.name.upper()
        if "CA" not in local:
            head = next(local[n] for n in ("P", "O5'", "C5'", "C4'") if n in local)
            tail = local.get("O3'", local["C3'"])
            out.append(PeptideResidue(
                position=k, name=name, atoms=tuple(atoms), atom_names=tuple((a, n) for n, a in local.items()),
                n=head, ca=local["C4'"], c=tail, o=tail, proteinogenic=False, number=res.number, kind="nucleotide"))
            continue
        out.append(PeptideResidue(
            position=k, name=name, atoms=tuple(atoms), atom_names=tuple((a, n) for n, a in local.items()),
            n=local["N"], ca=local["CA"], c=local["C"], o=local.get("O", local["C"]),
            proteinogenic=name in _ONE, stereo=_stereo(mol, local["N"], local["CA"], local["C"]),
            number=res.number))
    owner = {a: r.position for r in out for a in r.atoms}

    def bonded(a: int, b: int) -> bool:
        return mol.GetBondBetweenAtoms(a, b) is not None

    breaks = tuple(r.position for r, nxt in zip(out, out[1:]) if not bonded(r.c, nxt.n))
    cyclic = len(out) > 2 and bonded(out[-1].c, out[0].n)
    crosslinks = []
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        pa, pb = owner.get(a), owner.get(b)
        if pa is None or pb is None or pa == pb:
            continue
        (pa, a), (pb, b) = sorted(((pa, a), (pb, b)))
        peptide_bond = (pb == pa + 1 and a == out[pa - 1].c and b == out[pb - 1].n)
        if not peptide_bond and {(pa, a), (pb, b)} & {(r.position, r.c) for r in out} \
                and {(pa, a), (pb, b)} & {(r.position, r.n) for r in out} and not cyclic:
            continue        # a peptide bond between two residues that are not drawn next to each other
        closing = cyclic and pa == 1 and pb == len(out) and a == out[0].n and b == out[-1].c
        if not peptide_bond and not closing:
            both_s = mol.GetAtomWithIdx(a).GetAtomicNum() == 16 and mol.GetAtomWithIdx(b).GetAtomicNum() == 16
            crosslinks.append(Crosslink((pa, a), (pb, b), "disulfide" if both_s else "bond"))
    # What is no residue: grouped by what is bonded together, each with the bonds that hold it to the chain.
    appendages = []
    left = set(loose)
    while left:
        start = min(left)
        group, queue = {start}, [start]
        while queue:
            for nb in mol.GetAtomWithIdx(queue.pop()).GetNeighbors():
                j = nb.GetIdx()
                if j in left and j not in group:
                    group.add(j)
                    queue.append(j)
        left -= group
        holds = sorted((owner[nb.GetIdx()], nb.GetIdx(), a) for a in group
                       for nb in mol.GetAtomWithIdx(a).GetNeighbors() if nb.GetIdx() in owner)
        appendages.append(Appendage(tuple(sorted(group)), tuple(holds)))
    runs, length = [], 1
    for r in out[:-1]:
        if r.position in breaks:
            runs.append(length)
            length = 1
        else:
            length += 1
    runs.append(length)
    return Peptide(residues=tuple(out), appendages=tuple(appendages), crosslinks=tuple(crosslinks), cyclic=cyclic,
                   runs=tuple(sorted(runs, reverse=True)), n_atoms=mol.GetNumAtoms(), numbering="deposited",
                   breaks=breaks)
