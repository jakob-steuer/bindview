"""An interface between two proteins, and the two of them opened like a book.

A SKETCH, not part of bindview: nothing here is tested or promised.

The book: a plane is laid through the atoms of the interface. Seen from the side, partner A is on the left
of it and partner B on the right, and the longer axis of the interface stands upright. That axis is the
spine. A is turned a quarter turn one way about it and B a quarter turn the other way, so both faces look
at the reader. Two residues that touched across the interface are then at the same height, and mirrored
left to right.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

from bindview import Structure, read_structure  # noqa: E402
from bindview.distance_table import interface_contacts  # noqa: E402
from bindview.pae import IPSAE_CUTOFF, Pae, ipsae_d0  # noqa: E402

from surface import PROBE, radii  # noqa: E402

NEAR = 4.5          # an atom belongs to the interface within this distance of the partner (ångström)
TOUCH = 4.0         # two residues touch when two of their atoms are this close
PER_REGION = 6.0    # a region holds about this many residues of each partner


@dataclass
class Residue:
    side: int
    chain: str
    number: int
    icode: str
    name: str
    atoms: np.ndarray                       # indices into the atom arrays
    near: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))   # those at the interface
    buried: float = 0.0                     # area that the partner covers, Å²
    partners: int = 0
    region: int = -1                        # which part of the interface it belongs to, counted down the spine
    index: int = -1                         # its place in ``Complex.residues``
    ca: int = -1                            # its alpha carbon, as an index into the atom arrays (-1: none)

    @property
    def label(self) -> str:
        return f"{self.name.capitalize()}{self.number}{self.icode}"

    @property
    def key(self) -> str:
        return f"{'ab'[self.side]}{self.chain}{self.number}{self.icode}"


@dataclass
class Whole(Residue):
    """A partner that is one piece and no chain of residues, a ligand: it is named as a whole."""

    shown: str = ""

    @property
    def label(self) -> str:
        return self.shown or self.name


@dataclass
class Contact:
    a: int                                   # index of the residue on side A
    b: int
    kind: str                                # "saltbridge", "hbond", "hydrophobic" or "near"
    distance: float
    atoms_a: "tuple[int, ...]" = ()          # the atoms that make it, as indices into the atom arrays
    atoms_b: "tuple[int, ...]" = ()


@dataclass
class Complex:
    title: str
    names: "tuple[str, str]"
    xyz: np.ndarray
    elements: "list[str]"
    side: np.ndarray                         # 0 or 1 per atom
    residue_of: np.ndarray                   # index into ``residues`` per atom
    residues: "list[Residue]"
    contacts: "list[Contact]"
    n_regions: int = 0
    centre: np.ndarray = field(default_factory=lambda: np.zeros(3))
    axes: np.ndarray = field(default_factory=lambda: np.eye(3))   # rows: across the interface, spine, towards the reader
    # What a model says of itself, per residue (by its index); empty where the file says nothing.
    plddt: "dict[int, float]" = field(default_factory=dict)       # 0 to 100
    pae: "dict[int, float]" = field(default_factory=dict)         # ångström: mean over the residues it touches
    ipsae: "dict[int, float]" = field(default_factory=dict)       # 0 to 1: the partner scored, the model aligned on it
    ipsae_both: "tuple[float, float] | None" = None               # the score itself, first on second and back
    shared: "dict[int, float]" = field(default_factory=dict)      # share of the models of an ensemble it is at the interface in
    models: int = 0                                               # how many models that ensemble has
    shared_says: str = ""                                         # what that share is of, where it is not models
    #: Numbers from outside, by name: ``(what the key says, {index of a residue: value})``.
    given: "dict[str, tuple[str, dict[int, float]]]" = field(default_factory=dict)
    #: The name of every atom, as its file has it (``CA``, ``OE1``): what tells a side chain from the backbone.
    atoms: "list[str]" = field(default_factory=list)

    def find(self, chain: str, number: int, icode: str = "") -> "int | None":
        """The index of a residue, or ``None``."""
        for r in self.residues:
            if (r.chain, r.number, r.icode) == (chain, number, icode):
                return r.index
        return None

    def at_interface(self, side: int) -> "list[int]":
        return [i for i, r in enumerate(self.residues) if r.side == side and len(r.near)]

    # ------------------------------------------------------------------ the two views
    def closed(self) -> np.ndarray:
        """Every atom as the complex stands, seen from the side: A on the left, the spine upright."""
        return (self.xyz - self.centre) @ self.axes.T

    def opened(self, gap: float = 8.0, keep: "np.ndarray | None" = None) -> np.ndarray:
        """Every atom after the book is opened: both faces look at the reader (+z), A on the left page.

        ``keep`` (a mask) says which atoms the pages are set by; the gap between the pages is ``gap``.
        """
        shut = self.closed()
        out = np.empty_like(shut)
        for side, quarter in ((0, -1.0), (1, 1.0)):
            m = self.side == side
            # A quarter turn about the spine (y): x, the way across, becomes the way to the reader.
            out[m, 0] = quarter * shut[m, 2]
            out[m, 1] = shut[m, 1]
            out[m, 2] = -quarter * shut[m, 0]
            shown = m if keep is None else (m & keep)
            if side == 0:
                out[m, 0] -= out[shown, 0].max() + gap / 2.0
            else:
                out[m, 0] -= out[shown, 0].min() - gap / 2.0
        return out


def faced(cx: Complex) -> np.ndarray:
    """Every atom in the frame of A's opened page, with B left where it sits: between A and the reader.

    Across the page as on A's page of the opened book, up the page the spine, towards the reader the way
    from A to B. The one-sheet map and the face with the partner's atoms on it are both drawn in it.
    """
    shut = cx.closed()
    return np.stack([-shut[:, 2], shut[:, 1], shut[:, 0]], axis=1)


def _sphere(n: int) -> np.ndarray:
    i = np.arange(n) + 0.5
    phi, theta = np.arccos(1.0 - 2.0 * i / n), np.pi * (1.0 + 5.0 ** 0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1)


def accessible(which: np.ndarray, xyz: np.ndarray, rad: np.ndarray, among: np.ndarray, n: int = 96) -> np.ndarray:
    """Solvent-accessible area of the atoms ``which`` with the atoms ``among`` (a mask) present (Shrake, Rupley)."""
    points, env, out = _sphere(n), np.where(among)[0], np.zeros(len(which))
    for k, i in enumerate(which):
        reach = rad[i] + PROBE
        d = np.linalg.norm(xyz[env] - xyz[i], axis=1)
        nb = env[(d < reach + rad[env] + PROBE) & (env != i)]
        p = xyz[i] + reach * points
        free = np.ones(n, dtype=bool)
        if len(nb):
            free = (np.linalg.norm(p[:, None, :] - xyz[nb][None, :, :], axis=2) > (rad[nb] + PROBE)[None, :]).all(axis=1)
        out[k] = 4.0 * np.pi * reach * reach * free.mean()
    return out


def split(st: Structure, chain: str, parts: "dict[str, tuple[int, int]]") -> Structure:
    """One chain cut into several: every residue of ``chain`` whose number is in a range of ``parts``
    (name of the new chain: first and last number) gets that name. For two domains of one chain."""
    out = []
    for a in st.atoms:
        name = next((n for n, (lo, hi) in parts.items() if a.chain == chain and lo <= a.resnum <= hi), None)
        if a.chain != chain or name is not None:
            out.append(replace(a, index=len(out), chain=name or a.chain))
    return Structure(out, st.source, dict(st.chem_comp_smiles), st.plddt_scale, st.qa_metrics)


def _near(st: Structure, chains_a, chains_b) -> "set[tuple[int, str, int, str]]":
    """The residues at the interface of a model, as (side, chain, number, insertion code), and no more."""
    sides = []
    for chains in (chains_a, chains_b):
        atoms = [a for c in chains for k in st.polymer_residues(c) for a in st.residue_atoms(k)
                 if a.element.upper() != "H"]
        sides.append((atoms, np.array([a.xyz for a in atoms], dtype=float).reshape(-1, 3)))
    (a, pa), (b, pb) = sides
    if not len(pa) or not len(pb):
        return set()
    d = np.linalg.norm(pa[:, None, :] - pb[None, :, :], axis=2)
    return ({(0, x.chain, x.resnum, x.icode) for x, close in zip(a, d.min(axis=1) <= NEAR) if close}
            | {(1, x.chain, x.resnum, x.icode) for x, close in zip(b, d.min(axis=0) <= NEAR) if close})


def ensemble(cx: Complex, models, chains_a, chains_b) -> Complex:
    """Other models of the same complex (paths or structures): every residue at the interface of ``cx``
    learns in what share of all the models, ``cx`` among them, it is at the interface."""
    count = {r.index: 1 for r in cx.residues if len(r.near)}
    n = 1
    for m in models:
        near = _near(m if isinstance(m, Structure) else read_structure(m), chains_a, chains_b)
        n += 1
        for i in count:
            r = cx.residues[i]
            count[i] += (r.side, r.chain, r.number, r.icode) in near
    cx.shared, cx.models = {i: c / n for i, c in count.items()}, n
    return cx


def load(path: "str | Path | Structure", chains_a, chains_b, names, title: str = "",
         pae: "str | Path | Pae | None" = None) -> Complex:
    """A complex of two partners, each a list of chains. ``pae`` is the model's matrix of predicted
    aligned errors (AlphaFold 3's ``confidences.json``, the AlphaFold database's ``predicted_aligned_error``
    file, Boltz's ``pae_*.npz``), for a model that has one."""
    st = path if isinstance(path, Structure) else read_structure(path)
    xyz, elements, side, residue_of, residues = [], [], [], [], []
    atom_names: "list[str]" = []
    confidence: "dict[int, float]" = {}                         # pLDDT, where the file says its B column is one
    index: "dict[tuple[int, str, int, str], int]" = {}
    atom_at: "dict[tuple[int, str], int]" = {}                  # (residue, atom name) -> atom
    for s, chains in enumerate((chains_a, chains_b)):
        for chain in chains:
            for key in st.polymer_residues(chain):
                atoms = [a for a in st.residue_atoms(key) if a.element.upper() != "H"]
                if not atoms:
                    continue
                first = len(xyz)
                for a in atoms:
                    atom_at[(len(residues), a.name)] = len(xyz)
                    xyz.append(a.xyz)
                    elements.append(a.element)
                    atom_names.append(a.name)
                    side.append(s)
                    residue_of.append(len(residues))
                index[(s, key.chain, key.number, key.icode)] = len(residues)
                ca = next((first + n for n, a in enumerate(atoms) if a.name == "CA"), -1)
                residues.append(Residue(s, key.chain, key.number, key.icode, atoms[0].resname,
                                        np.arange(first, len(xyz)), index=len(residues), ca=ca))
                if st.plddt_scale is not None:
                    values = [a.b_iso for a in atoms if a.b_iso is not None]
                    if values:
                        confidence[len(residues) - 1] = float(np.mean(values)) * st.plddt_scale
    cx = Complex(title, tuple(names), np.array(xyz, dtype=float), elements, np.array(side), np.array(residue_of),
                 residues, [], atoms=atom_names)
    typed = []
    for c in interface_contacts(st, list(chains_a), list(chains_b)):
        ia = index.get((0, c.chain_a, int(c.number_a), c.icode_a))
        ib = index.get((1, c.chain_b, int(c.number_b), c.icode_b))
        if ia is not None and ib is not None:
            typed.append((ia, ib, c.type.value if hasattr(c.type, "value") else str(c.type), c.distance,
                          tuple(atom_at[(ia, n)] for n in c.atoms_a if (ia, n) in atom_at),
                          tuple(atom_at[(ib, n)] for n in c.atoms_b if (ib, n) in atom_at)))
    _complete(cx, typed)
    cx.plddt = confidence
    if pae is not None:
        errors(cx, pae if isinstance(pae, Pae) else Pae.read(pae, st))
    group(cx)
    return cx


def _complete(cx: Complex, typed, several: bool = False, plane: "np.ndarray | None" = None) -> None:
    """What a complex has once its atoms are known: which atoms are at the interface, which residues
    touch and how, the area each loses, and the plane the book opens on.

    ``typed`` are the contacts of a table, each as (residue of the first partner, of the second, kind,
    distance or ``None``, atoms of the first, atoms of the second). Between two chains the strongest
    kind of a pair of residues is kept; ``several`` keeps every contact (a ligand is one piece, and one
    residue can reach it in more than one way). ``plane`` is a mask of the atoms the plane is laid
    through, where that is not all the atoms of the interface.
    """
    residues = cx.residues
    elements = cx.elements
    a, b = np.where(cx.side == 0)[0], np.where(cx.side == 1)[0]

    # Who is near whom, and which residues touch.
    d = np.linalg.norm(cx.xyz[a][:, None, :] - cx.xyz[b][None, :, :], axis=2)
    near = np.zeros(len(cx.xyz), dtype=bool)
    near[a[d.min(axis=1) <= NEAR]] = True
    near[b[d.min(axis=0) <= NEAR]] = True
    for r in residues:
        r.near = r.atoms[near[r.atoms]]
    closest: "dict[tuple[int, int], float]" = {}
    by: "dict[tuple[int, int], tuple[int, int]]" = {}           # the two atoms that are closest
    for i, j in zip(*np.where(d <= TOUCH)):
        pair = (int(cx.residue_of[a[i]]), int(cx.residue_of[b[j]]))
        if float(d[i, j]) < closest.get(pair, 9.9):
            closest[pair], by[pair] = float(d[i, j]), (int(a[i]), int(b[j]))

    if several:
        # Every contact of the table, and of a pair the table does not have, that its atoms are near.
        cx.contacts = [Contact(ia, ib, kind, float(dist) if dist is not None else closest.get((ia, ib), 4.0),
                               tuple(mine), tuple(theirs)) for ia, ib, kind, dist, mine, theirs in typed]
        made = {(c.a, c.b) for c in cx.contacts}
        cx.contacts += [Contact(q[0], q[1], "near", dist, (by[q][0],), (by[q][1],))
                        for q, dist in sorted(closest.items()) if q not in made]
        for ia, ib in made | set(closest):
            residues[ia].partners += 1
            residues[ib].partners += 1
    else:
        # What kind of contact: the plain distance table, strongest kind per pair.
        rank = {"saltbridge": 0, "hbond": 1, "hydrophobic": 2, "near": 3}
        kinds: "dict[tuple[int, int], tuple[str, float]]" = {q: ("near", dist) for q, dist in closest.items()}
        atoms: "dict[tuple[int, int], tuple[tuple[int, ...], tuple[int, ...]]]" = {
            q: ((pair[0],), (pair[1],)) for q, pair in by.items()}
        for ia, ib, kind, dist, mine, theirs in typed:
            if kind not in rank:
                continue
            held = kinds.get((ia, ib))
            if held is None or rank[kind] < rank[held[0]]:
                kinds[(ia, ib)] = (kind, float(dist) if dist is not None else closest.get((ia, ib), 4.0))
                if mine and theirs:
                    atoms[(ia, ib)] = (mine, theirs)
        cx.contacts = [Contact(q[0], q[1], k, dist, *atoms.get(q, ((), ()))) for q, (k, dist) in sorted(kinds.items())]
        for c in cx.contacts:
            residues[c.a].partners += 1
            residues[c.b].partners += 1

    # The area each residue loses to the partner.
    rad = radii(elements)
    for s in (0, 1):
        mine = np.concatenate([residues[i].atoms for i in cx.at_interface(s)] or [np.zeros(0, dtype=int)])
        if not len(mine):
            continue
        lost = accessible(mine, cx.xyz, rad, cx.side == s) - accessible(mine, cx.xyz, rad, np.ones(len(rad), bool))
        for i in cx.at_interface(s):
            residues[i].buried = float(lost[np.isin(mine, residues[i].atoms)].sum())

    # The plane of the interface and the spine of the book.
    p = cx.xyz[near if plane is None else plane]
    if len(p) < 3:                              # an ion has no plane of its own: that of what is around it
        p = cx.xyz[near]
    cx.centre = p.mean(axis=0)
    _, vec = np.linalg.eigh(np.cov((p - cx.centre).T))
    across, short, spine = vec[:, 0], vec[:, 1], vec[:, 2]
    if across @ (cx.xyz[b][near[b]].mean(axis=0) - cx.xyz[a][near[a]].mean(axis=0)) < 0:
        across = -across
    axes = np.stack([across, spine, short])
    if np.linalg.det(axes) < 0:                 # a turn, never a mirror image
        axes[2] = -axes[2]
    cx.axes = axes


def partner_residues(st: Structure, chain: str) -> list:
    """The residues of a chain that a ligand can be in contact with: its polymer as the library tells one
    (a backbone of N, CA and C) and, where the chain has none, its nucleotides (a ribose: C1' and C4'),
    which the library does not count as a polymer."""
    found = list(st.polymer_residues(chain))
    if found:
        return found
    return [k for k in st.residues() if k.chain == chain
            and {"C1'", "C4'"} <= {a.name.replace("*", "'") for a in st.residue_atoms(k)}]


def load_ligand(diagram, names: "tuple[str, str] | None" = None, title: str = "") -> Complex:
    """A ligand in its pocket as a complex of two partners: the protein, and the ligand as one piece.

    ``diagram`` is a bindview ``Diagram`` made from a structure: its ligand, and its contacts as the
    library resolved them to atoms. The plane the book opens on is laid through the ligand alone, which
    for a flat ligand is the ligand's own plane: cut there, each half of the pocket has one face of the
    ligand on it.
    """
    st, key = diagram.structure, diagram.ligand_key
    xyz, elements, side, residue_of, residues = [], [], [], [], []
    atom_names: "list[str]" = []
    confidence: "dict[int, float]" = {}
    index: "dict[tuple[str, int, str], int]" = {}
    atom_at: "dict[tuple[int, str], int]" = {}
    for chain in sorted({x.chain for x in st.atoms}):
        for k in partner_residues(st, chain):
            atoms = [x for x in st.residue_atoms(k) if x.element.upper() != "H"]
            if k == key or not atoms:
                continue
            first = len(xyz)
            for x in atoms:
                atom_at[(len(residues), x.name)] = len(xyz)
                xyz.append(x.xyz)
                elements.append(x.element)
                atom_names.append(x.name)
                side.append(0)
                residue_of.append(len(residues))
            index[(k.chain, k.number, k.icode)] = len(residues)
            ca = next((first + n for n, x in enumerate(atoms) if x.name == "CA"), -1)
            residues.append(Residue(0, k.chain, k.number, k.icode, atoms[0].resname, np.arange(first, len(xyz)),
                                    index=len(residues), ca=ca))
            if st.plddt_scale is not None:
                values = [x.b_iso for x in atoms if x.b_iso is not None]
                if values:
                    confidence[len(residues) - 1] = float(np.mean(values)) * st.plddt_scale
    # The ligand, atom for atom in the order the library has it: its contacts name atoms by that order.
    own = {x.name: x for x in st.residue_atoms(key)}
    first = len(xyz)
    for name in diagram.ligand.names:
        x = own[name]
        xyz.append(x.xyz)
        elements.append(x.element)
        atom_names.append(name)
        side.append(1)
        residue_of.append(len(residues))
    names = tuple(names) if names else ("the protein", diagram.name or st.residue_name(key))
    ligand = Whole(1, key.chain, key.number, key.icode, st.residue_name(key), np.arange(first, len(xyz)),
                   index=len(residues), shown=names[1])
    residues.append(ligand)
    cx = Complex(title, names, np.array(xyz, dtype=float), elements, np.array(side), np.array(residue_of), residues, [],
                 atoms=atom_names)
    typed = []
    for r in diagram.interactions:
        ia = index.get((r.residue.chain, r.residue.number, r.residue.icode)) if r.residue is not None else None
        if ia is None or r.ligand is None:
            continue                                    # a water, a metal, or a contact that found no atom
        theirs = tuple(first + i for i in r.ligand.atoms)
        mine = tuple(atom_at[(ia, n)] for n in r.protein.atoms if (ia, n) in atom_at)
        if not mine:                                    # the residue as a whole: its atom nearest to the ligand's
            d = np.linalg.norm(cx.xyz[residues[ia].atoms][:, None, :] - cx.xyz[list(theirs)][None, :, :], axis=2)
            mine = (int(residues[ia].atoms[int(np.argmin(d.min(axis=1)))]),)
        typed.append((ia, ligand.index, r.type.value, r.record.distance, mine, theirs))
    _complete(cx, typed, several=True, plane=cx.side == 1)
    cx.plddt = confidence
    group(cx)
    return cx


def errors(cx: Complex, pae: Pae, cutoff: float = IPSAE_CUTOFF) -> None:
    """What the matrix of predicted aligned errors says of every residue at the interface.

    ``cx.pae``: the mean, over the residues of the partner it touches, of the error of the pair (the
    smaller of its two directions, as bindview reads a contact's). ``cx.ipsae``: the term of ipSAE for the
    residue, which is the partner scored with the model aligned on this residue (the mean of
    ``1 / (1 + (PAE / d0)**2)`` over the partner's residues with a PAE below the cut-off); ipSAE itself
    is the largest term of a side, and is kept in ``cx.ipsae_both``.
    """
    row = {(c, n): i for i, (c, n) in reversed(list(enumerate(zip(pae.chains, pae.numbers))))}
    one_chain = len(set(pae.chains)) == 1                       # a chain cut into two keeps its numbers
    rows = {}
    for r in cx.residues:
        at = row.get((r.chain, r.number))
        if at is None and one_chain:
            at = next((i for i, n in enumerate(pae.numbers) if n == r.number), None)
        if at is not None:
            rows[r.index] = at
    pairs: "dict[int, list[float]]" = {}
    for c in cx.contacts:
        if c.a in rows and c.b in rows:
            value = float(np.fmin(pae.matrix[rows[c.a], rows[c.b]], pae.matrix[rows[c.b], rows[c.a]]))
            if np.isfinite(value):
                pairs.setdefault(c.a, []).append(value)
                pairs.setdefault(c.b, []).append(value)
    cx.pae = {i: float(np.mean(v)) for i, v in pairs.items()}
    best = []
    for s in (0, 1):
        mine = [r.index for r in cx.residues if r.side == s and r.index in rows]
        theirs = [r.index for r in cx.residues if r.side != s and r.index in rows]
        if not mine or not theirs:
            best.append(0.0)
            continue
        block = pae.matrix[np.ix_([rows[i] for i in mine], [rows[i] for i in theirs])]
        with np.errstate(invalid="ignore"):
            below = block < cutoff
        n = below.sum(axis=1)
        d0 = np.array([ipsae_d0(float(k)) for k in n])
        with np.errstate(invalid="ignore"):
            terms = np.where(below, 1.0 / (1.0 + (block / d0[:, None]) ** 2), 0.0)
        values = np.divide(terms.sum(axis=1), n, out=np.zeros(len(mine)), where=n > 0)
        for i, v in zip(mine, values):
            cx.ipsae[i] = float(v)
        best.append(float(values.max()))
    cx.ipsae_both = (best[0], best[1])


@dataclass
class Grouping:
    """How the regions of an interface come about (:func:`grouping`), step by step. A place is on the plane of
    the interface: along the spine, and across the page. Groups are counted down the spine."""

    ids: "list[int]"                         # the residues that are grouped
    place: "dict[int, np.ndarray]"           # where each of them lies
    mids: np.ndarray                         # per contact, the middle of its two residues
    k: int                                   # how many groups
    starts: "list[int]"                      # the contacts that the grouping starts from, in the order they are taken
    rounds: int                              # how often the points were handed out until nothing changed
    own: np.ndarray                          # per contact, its group
    centre: np.ndarray                       # per group, the middle of its points
    votes: "dict[int, np.ndarray]"           # per residue, how many of its contacts each group holds


def grouping(cx: Complex) -> "Grouping | None":
    """What :func:`group` does, with what it does it by; ``None`` where there is nothing to group."""
    ids = cx.at_interface(0) + cx.at_interface(1)
    if not cx.contacts or not ids:
        return None
    # (A residue of a table's contact need not be at the interface: one that reaches the ligand through a
    # water has no atom within its reach. It is placed by all its atoms.)
    ids = sorted(set(ids) | {i for c in cx.contacts for i in (c.a, c.b)})
    shut = cx.closed()
    place = {i: shut[cx.residues[i].near if len(cx.residues[i].near) else cx.residues[i].atoms][:, 1:3].mean(axis=0)
             for i in ids}                                                        # along the spine, across the page
    mids = np.array([(place[c.a] + place[c.b]) / 2.0 for c in cx.contacts])
    k = min(len(mids), int(np.clip(round(len(ids) / 2.0 / PER_REGION), 2, 6)))
    starts = [int(np.argmax(np.linalg.norm(mids - mids.mean(axis=0), axis=1)))]
    centres = [mids[starts[0]]]
    while len(centres) < k:
        far = np.min([np.linalg.norm(mids - c, axis=1) for c in centres], axis=0)
        starts.append(int(np.argmax(far)))
        centres.append(mids[starts[-1]])
    centre = np.array(centres, dtype=float)
    own, rounds = np.full(len(mids), -1), 0
    for _ in range(100):
        new = np.argmin(np.linalg.norm(mids[:, None, :] - centre[None, :, :], axis=2), axis=1)
        if (new == own).all():
            break
        own, rounds = new, rounds + 1
        for j in range(k):
            if (own == j).any():
                centre[j] = mids[own == j].mean(axis=0)
    order = np.argsort(-centre[:, 0])                         # the highest on the page first
    rank = {int(j): n for n, j in enumerate(order)}
    centre = centre[order]
    votes = {i: np.zeros(k) for i in ids}
    for c, j in zip(cx.contacts, own):
        votes[c.a][rank[int(j)]] += 1.0
        votes[c.b][rank[int(j)]] += 1.0
    return Grouping(ids, place, mids, k, starts, rounds, np.array([rank[int(j)] for j in own]), centre, votes)


def group(cx: Complex) -> None:
    """The interface in regions, so that a reader can match a part of one face to a part of the other.

    The touching pairs are grouped by where they lie on the plane: k-means on the middles of the pairs,
    started from the pairs that lie farthest apart, with one region for about ``PER_REGION`` residues of
    each partner (two at least, six at most). A residue belongs to the region that holds most of its
    pairs; one that touches nobody, to the nearest. Regions are counted down the spine.
    """
    g = grouping(cx)
    if g is None:
        return
    for i in g.ids:
        near = np.linalg.norm(g.centre - g.place[i], axis=1)
        if g.votes[i].sum() > 0:
            best = np.flatnonzero(g.votes[i] == g.votes[i].max())
            cx.residues[i].region = int(best[np.argmin(near[best])])
        else:
            cx.residues[i].region = int(np.argmin(near))
    cx.n_regions = g.k
