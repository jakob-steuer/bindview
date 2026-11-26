"""The public entry point: :class:`Diagram` and :class:`RenderResult`.

``Diagram.from_structure`` does the chemistry once (build the named-atom
ligand, resolve every interaction to what makes it); ``render`` can then be
called for any style and theme. A render never raises for a complex that is
merely awkward: it returns a result whose ``status`` says what happened.

========== ====================================================================
status     meaning
========== ====================================================================
ok         an atom-level diagram, every hard requirement met
empty      analysed, no interactions; the figure says so explicitly
degraded   a diagram was drawn but something was left out or assumed (see
           ``warnings``): a record that could not be attached, unknown bond
           orders, ...
unreadable the complex cannot be laid out legibly at the requested figure size;
           by default a per-residue summary table is drawn instead
========== ====================================================================
"""

from __future__ import annotations

import copy
import math
import numbers
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

import numpy as np

from .backends import Polymer, SmallMolecule, choose_backend, describe as describe_backend
from .clearance import keep_lines_off_words
from .plane import ERROR_MAX, KEPT_MIN, SCALE_BAR
from .polymer import MIN_PITCH as POLYMER_MIN_PITCH, piece_sides, shown_atoms
from .confidence import MARK, Confidence, ModelConfidence, look_of, pocket_atoms
from .covalent import as_bound, covalent_bonds, drop_bonded
from .depict import Depiction
from .errors import GraphError, RenderError
from .exposure import BURIAL_WARNING_BELOW, WATER_NAMES, Burial, burial_warning, structure_burial
from .fragments import check_detail
from .graph import Graph, Node
from .graphbuild import diagram_graph, poses_from
from .interactions import (Interaction, InteractionType, check_confidence_scale, coerce_interactions,
                           load_interactions)
from .interactive import NOTHING_TO_POINT_AT, ROOT_CLASS
from .ions import bridges, holders, seat, with_ions, without_the_bridge
from .layout import LayoutParams, LayoutResult, LigandField, audit_layout, layout_bodies, layout_key
from .ligand import Ligand, build_ligand
from .pae import Pae, PaeDraws, pae_key, read_pae
from .residues import format_residue_label
from .resolve import ResolvedInteraction, resolve_interactions
from .scene import Group, Item, Paint, Rect, Scene, Span, Text, items_bounds
from .solvent import VALIDATION, SolventSite, has_waters, predict_solvent_sites
from .structure import ResidueKey, Structure, read_structure
from .styles.absence import Absence, absence, from_whom, mark_groups
from .styles.closeup import atom_hits, closeups
from .styles.mixed import style_for, unknown_residues
from .styles.base import (Context, LegendEntry, RenderOptions, beyond_any_page, confidence_legend,
                          group_residues, legend_group, page_that_would_do, plddt_legend, split_along_chain,
                          too_small_on_page)
from .styles.summary import summary_table, text_block, wrap_text
from .svg import svg_and_css
from .textmetrics import CAP_HEIGHT, text_width
from .theme import Metrics, Palette, get_palette, scheme

if TYPE_CHECKING:  # pragma: no cover
    from .consensus import ResidueShare
    from .fingerprint import Fingerprint

__all__ = ["Diagram", "RenderResult", "Quality", "STYLES"]

_PX_PER_MM = 96.0 / 25.4
_PT_PER_PX = 0.75
#: Bond lengths tried, largest first, when the caller does not fix one.
_BOND_LENGTHS = (30.0, 26.0, 22.0, 19.0)
#: The longest bond that a figure is drawn with: ten times the longest that is tried where none is asked for. At
#: its own size such a figure has 79 mm to a bond, far more than a page holds of a ligand, and what a layout
#: takes of memory grows with the length of a bond.
_BOND_LENGTH_MAX = 10.0 * _BOND_LENGTHS[0]
#: A ligand drawn as a chain that is too crowded to lay out is tried again with its residues
#: further apart, in these steps (bond lengths from one residue to the next). Shrinking, which
#: is what helps a ligand drawn as atoms, would crowd a chain further.
_CHAIN_PITCHES = (0.9, 1.0, 1.1, 1.2, 1.35, 1.55, 1.8, 2.1, 2.5, 3.0)
#: A short peptide in a closed groove has two or three residues of the protein for each of its own,
#: and each of those needs its place beside the column. When the first layout fails and there are
#: that many, the next one is not the next step but the pitch at which there is about this much
#: room (in bond lengths) for every residue of the protein per residue of the ligand.
_ROOM_PER_RESIDUE = 0.9
#: ... and how many layouts of a chain are tried before the best of them is taken for what there is.
_CHAIN_ATTEMPTS = 6


#: How many a diagram keeps at most of what it keeps of its figures, each kind by itself: the figures as they
#: were drawn, the fields round the ligand, and the placements of the residues. The one that was used longest
#: ago goes first. A figure that is asked for again after it went is drawn again on a placement that stayed,
#: which costs a drawing and no layout.
_KEPT_AT_MOST = {"figure": 32, "field": 32, "layout": 128}

STYLES = ("detailed", "schematic")


@dataclass(frozen=True)
class Quality:
    """Measured properties of a laid-out diagram."""

    width: float                    # natural size, SVG user units (px at 96 dpi)
    height: float
    bond_length: float
    n_residues: int
    n_interactions: int
    overlaps: int                   # residue pairs that overlap; 0 for a valid layout
    unplaced: int                   # residues the engine could not seat
    line_crossings: int             # interaction lines crossing each other
    bond_crossings: int             # interaction lines crossing ligand bonds
    max_stretch: float              # longest link relative to its ideal length
    min_font_pt: float              # smallest text when printed at the figure size
    #: Figure width at which the smallest text reaches the minimum; infinite where no width does,
    #: because the minimum is above what the text is at full size.
    required_width_mm: float
    readable: bool
    ligand_width: float = 0.0       # width of the ligand drawing alone
    estimated: bool = False         # True when no layout was run: sizes are lower bounds
    link_hits: int = 0              # interaction lines passing through another residue
    ligand_height: float = 0.0      # height of the ligand drawing alone
    #: Boxes on a plane: are they, as drawn, still a faithful picture of where the residues lie
    #: (:attr:`bindview.plane.Fidelity.ok`)? Always true of every other figure.
    faithful: bool = True

    @property
    def valid(self) -> bool:
        """Does nothing overlap, and was every residue placed?"""
        return self.overlaps == 0 and self.unplaced == 0 and self.link_hits == 0 and self.faithful


@dataclass
class RenderResult:
    status: str
    svg: str | None
    figure: str                     # diagram | empty | summary | none
    warnings: list[str] = field(default_factory=list)
    quality: Quality | None = None
    path: Path | None = None
    style: str = ""
    theme: str = ""
    scene: Scene | None = field(default=None, repr=False)
    #: Page coordinates of every ligand atom, keyed by structure-file atom name
    #: (only for atom-level figures). This is the map the library exists for:
    #: overlays and hit regions on a web page can be positioned from it.
    atoms: dict[str, tuple[float, float]] = field(default_factory=dict, repr=False)
    #: Figure-level confidence that was drawn (ipTM, pocket and ligand pLDDT, the
    #: caller's flag), or ``None`` when the model carries none or it was switched off.
    confidence: dict[str, Any] | None = None
    #: What the figure says beside its drawing, as plain data. Of a diagram: how a ligand with a chain of
    #: residues is drawn (``ligand``), what an interface says of itself (``interface``), the predicted
    #: waters (``solvent_sites``), the residues another protein lacks (``absent``) and what the atoms drawn
    #: of a residue stand at (``plddt_basis``), each where it applies. Of a comparison view: the alignment
    #: decision, the shared / only-A / only-B classification or the consensus rows.
    data: dict[str, Any] = field(default_factory=dict, repr=False)
    #: The figure as nodes, contacts and rigid groups (see :mod:`bindview.graph`): what was laid out,
    #: before it became ink. ``None`` when no node-level figure was drawn (empty, summary, none).
    graph: Graph | None = field(default=None, repr=False)
    #: The ligand's solvent-accessible surface free and in the complex (square angstrom) and the
    #: buried fraction, with whether it fell below the calibrated cut. ``None`` without a structure.
    burial: dict[str, Any] | None = None
    #: The words the figure has around its drawing, in their order: the line of confidence numbers under
    #: the title, a warning above the drawing, the notes under the key. Whether they are drawn or were left
    #: off (``bare=True``, ``caption=False``): a figure without them takes them into its caption.
    notes: list[str] = field(default_factory=list)
    #: The style sheet that the figure goes with, where it was asked for apart (``stylesheet="external"``): for
    #: the page, once for all its figures (:func:`bindview.stylesheet`).
    css: str | None = field(default=None, repr=False)

    @property
    def ok(self) -> bool:
        """Is it the diagram that was asked for, with nothing left out or assumed (status ``ok``)? Not so where the
        figure is drawn and says that something is missing, nor where another figure or none was drawn."""
        return self.status == "ok"

    @property
    def is_empty(self) -> bool:
        """Is it the figure of a complex in which no contact was found?"""
        return self.status == "empty"

    def __str__(self) -> str:
        return f"<RenderResult {self.status}: {self.figure}, {len(self.warnings)} warning(s)>"


@dataclass
class _Composed:
    scene: Scene
    layout: LayoutResult
    quality: Quality
    notes: list[str]
    atoms: dict[str, tuple[float, float]] = field(default_factory=dict)
    graph: Graph | None = None
    #: Where atoms of residues are drawn at a level of detail that was asked for and the model has
    #: confidence: what each of them is drawn at, ``"atom"`` (its own value) or ``"residue"``.
    basis: "str | None" = None
    #: Does a contact stand on a glow for its predicted aligned error?
    glows: bool = False
    #: What a figure that answers the pointer has beside (:mod:`bindview.styles.closeup`): made when it is
    #: first asked for, from what the figure was drawn from.
    parts: "tuple[Any, ...] | None" = field(default=None, repr=False)
    over: "list[Group] | None" = field(default=None, repr=False)

    def beside(self) -> list[Group]:
        if self.over is None:
            self.over = []
            if self.parts is not None:
                style_obj, ctx, bodies, poses, fld, params, content, box = self.parts
                made = closeups(ctx, bodies, poses, fld, params, style_obj.settings, box) + atom_hits(ctx)
                for g in made:
                    g.translate = content.translate
                self.over = made
        return self.over


class _Same:
    """An object for what it is and not for what it says: how the key of a kept figure names what a diagram holds as
    one thing and has no value to compare by (its ligand, its structure, the confidence of its model)."""

    __slots__ = ("it",)

    def __init__(self, it: Any) -> None:
        self.it = it

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Same) and other.it is self.it

    def __hash__(self) -> int:
        return id(self.it)


#: What ``interactions`` can be in the place of a table: make one from the positions of the heavy atoms.
DISTANCES = "distances"

#: How the sentence on the drawn model of an ensemble ends (:meth:`bindview.ensemble.Ensemble.diagram`).
_WHOSE_DISTANCES = " Distances are that model's."


class Diagram:
    """A ligand, its named atoms, and the interactions resolved onto them."""

    def __init__(self, ligand: Ligand, interactions: Sequence[ResolvedInteraction], *,
                 structure: Structure | None = None, ligand_key: ResidueKey | None = None,
                 name: str | None = None, source: str = "supplied",
                 warnings: Sequence[str] = (), confidence: Confidence | None = None,
                 ensemble: str | None = None, ligand_as: str = "auto",
                 detail: "str | Mapping[str, str | None] | None" = None) -> None:
        self.ligand = ligand
        #: How the ligand is drawn -- as atoms or as a chain of residues -- and why
        #: (:mod:`bindview.backends`). ``ligand_as`` is ``"auto"``, ``"atoms"`` or ``"chain"``.
        self.backend: "SmallMolecule | Polymer" = choose_backend(ligand, ligand_as)
        #: How much of a residue is drawn where it is drawn with atoms, on either side: of the
        #: ``"ligand"`` (where it is drawn as a chain) and of the ``"protein"`` around it. ``None`` asks
        #: for no level: a residue of the ligand is its box, and one of the protein is drawn as the style
        #: draws it. Otherwise one of :data:`bindview.fragments.DETAILS`. It is not ``ligand_as``, which
        #: says how the ligand is read.
        self.detail: dict[str, "str | None"] = _details(detail)
        #: Set by an interface: the level of each chain around its column (a participant's own).
        self.detail_by_chain: dict[str, "str | None"] = {}
        if isinstance(self.backend, Polymer) and self.detail["ligand"] is not None:
            self.backend.peptide = replace(self.backend.peptide, detail=self.detail["ligand"])
        self.interactions = list(interactions)
        self.structure = structure
        self.ligand_key = ligand_key
        self.name = name
        self.source = source
        self.warnings = list(warnings)
        #: Per-atom / per-residue pLDDT and model-level evidence, when it may be drawn.
        self.confidence = confidence
        #: What the records' ``occupancy`` is a share of ("1,000 frames of a 100 ns trajectory",
        #: "the 10 ligands of the set"). Printed in the legend; the numbers do not say.
        self.ensemble = ensemble
        #: Sentences the figure prints under itself, after its own: what the one who built the diagram
        #: has to say about it and the figure cannot know. Empty for every diagram but one that stands
        #: for several models (:meth:`bindview.ensemble.Ensemble.diagram`), which says here which model it
        #: is and which contacts of the others it does not make. Not a note that ``notes=False`` drops:
        #: without it such a figure reads as the whole ensemble.
        self.remarks: list[str] = []
        #: Contacts that this structure does not make and other models of its ensemble do, where the
        #: ensemble was asked to have them drawn (``Ensemble.diagram(absent="faint")``): each is drawn
        #: faint, with its share where a distance would stand. They are not among ``interactions``:
        #: not counted, not in a fingerprint, and a diagram without a contact of its own stays empty.
        self.elsewhere: list[ResolvedInteraction] = []
        #: What a figure says of them: where it draws them, and where it cannot (a summary table).
        self.elsewhere_said: tuple[str, str] = ("", "")
        #: A diagram of all the models of an ensemble (:meth:`bindview.ensemble.Ensemble.consensus`): for every
        #: residue the models that contact it, which is what its ring shows, and how each is drawn unless
        #: ``residues=`` says otherwise. Empty for every other diagram.
        self.shares: "Mapping[tuple[str, int, str], ResidueShare]" = {}
        self.forms: "Mapping[str, str]" = {}
        self._exposure: np.ndarray | None = None
        self._burial: Burial | None = None
        self._sites: list[SolventSite] | None = None
        #: The matrix of predicted aligned errors the contacts' values are from, where one was given.
        self.pae: "Pae | PaeDraws | None" = None
        #: What was drawn, each figure under its options and what the diagram was when it was drawn
        #: (:meth:`_drawn_from`), and under keys of their own the layouts it was drawn from (:meth:`_laid_out`).
        self._composed: dict[tuple, Any] = {}

    # ------------------------------------------------------------------ build
    @classmethod
    def from_structure(cls, structure: "str | Path | Structure", ligand: "str | tuple[str, int] | ResidueKey",
                       smiles: str | None = None,
                       interactions: "Iterable[Interaction | Mapping[str, Any]] | str | Path | None" = None, *,
                       name: str | None = None, ionize_acids: bool = True,
                       plddt: "str | bool | Confidence | None" = "auto",
                       model_confidence: "str | Path | Mapping[str, Any] | ModelConfidence | None" = "auto",
                       trusted: bool | None = None, flag: str | None = None,
                       ensemble: str | None = None, ligand_as: str = "auto",
                       detail: "str | Mapping[str, str | None] | None" = None, covalent: bool = False,
                       pae: Any = None, ions: bool = False) -> "Diagram":
        """Build a diagram from a structure file.

        Parameters
        ----------
        structure:
            Path to an mmCIF or PDB file, or an already parsed :class:`Structure`.
        ligand:
            Which residue is the ligand: a residue name (``"LIG_B"``),
            ``"chain:number"``, or a ``(chain, number)`` tuple. A peptide
            ligand that is a polymer chain of its own, as in a deposited
            structure, is named by its chain: ``"P"`` or ``"chain:P"``
            (:meth:`bindview.structure.Structure.resolve_ligand`). Its residues,
            their names and their numbers are then those of the file.
        smiles:
            The ligand's SMILES. Optional but strongly recommended: bond orders
            cannot be recovered reliably from coordinates alone. When omitted,
            ``_chem_comp.pdbx_smiles`` from the file is used if present.
        interactions:
            The contacts: a list of :class:`Interaction` (or dict-like rows), or the path of a table
            (:func:`bindview.load_interactions`). ``"distances"`` makes the table from the positions of the
            heavy atoms (:mod:`bindview.distance_table`), which needs nothing installed and is cruder than a
            detector. ``None`` means *detect them*, which needs the optional PLIP extra. An empty list means
            "analysed, nothing found" and yields an ``empty`` result.
        plddt:
            Where per-atom confidence comes from. ``"auto"`` (default) reads the
            B-factor column **only if the file declares it to be a pLDDT**
            (``_ma_qa_metric``, as AlphaFold-style models do) and otherwise draws
            no confidence at all: in a crystal structure that column is a
            temperature factor, and the value range is never used to guess.
            ``"bfactor"`` is the caller's assertion that the column is pLDDT on
            the 0-100 scale (AlphaFold 2 / ColabFold PDB files, which cannot
            declare it), ``"bfactor01"`` that it is pLDDT on the 0-1 scale. The
            scale is never guessed: a column in which nothing exceeds 1 is
            refused as 0-100, one in which something does as 0-1. A
            :class:`~bindview.confidence.Confidence` supplies values explicitly;
            ``None`` switches confidence off.
        model_confidence:
            Model-level numbers to display (ipTM, pTM, ranking score): a
            :class:`~bindview.confidence.ModelConfidence`, a mapping, or the path
            of an AlphaFold 3 ``summary_confidences.json``. ``"auto"`` picks up
            that file when it sits next to the model under its standard name.
        trusted, flag:
            The caller's verdict. ``trusted=False`` marks the figure as coming
            from a model that should not be read as a result; ``flag`` is the
            verdict in the caller's words (``"grade C"``). bindview never decides
            this itself.
        ensemble:
            What the ``occupancy`` of the interaction records is a share of, in
            words for the legend: ``"1,000 frames of a 100 ns trajectory"``.
            Occupancy itself comes with the records; nothing here reads a
            trajectory.
        ligand_as:
            How the ligand is drawn. ``"auto"`` (default) decides by what the
            molecule is: a ligand with a chain of at least eight amino-acid
            residues is drawn as that chain, every other ligand as atoms,
            whatever its size (:func:`bindview.peptide.decide`). ``"atoms"`` and
            ``"chain"`` overrule it.
        detail:
            How much of a residue is drawn where it is drawn with atoms: ``"box"``
            (no atoms: its name), ``"backbone"``, ``"sidechain"`` or ``"atoms"``.
            One level for both sides, or a mapping with one for the ``"ligand"``
            (where it is drawn as a chain) and one for the ``"protein"``. Only a
            residue that makes a directional contact is drawn with atoms: the
            part that makes the contact, and what the level adds to it. ``None``
            (default) asks for no level: a residue of a ligand drawn as a chain
            is its box, and a residue of the protein is drawn as the style draws
            it. Not ``ligand_as``, which says how the ligand is read.
        covalent:
            Look for covalent bonds between the ligand and a residue, as the file declares them and by
            distance, and draw them (:mod:`bindview.covalent`). Off unless asked for; a row of the kind
            ``"covalent"`` in ``interactions`` is drawn either way.
        pae:
            The model's matrix of predicted aligned errors (:mod:`bindview.pae`), as an
            :class:`~bindview.interface.Interface` takes it: AlphaFold 3's ``confidences.json``, Boltz's
            ``pae_*.npz``, or several. Each contact then has the PAE between its residue and the atoms of the
            ligand that make it (of a ligand that is a chain, the residue of it), drawn as a glow behind its
            line (``render(pae_colors=...)``).
        ions:
            Draw an ion that the model has between the ligand and the protein as the bridge it is, with the
            atoms of either that hold it (:mod:`bindview.ions`). Off unless asked for; where the model has
            none, none is predicted.
        """
        st = structure if isinstance(structure, Structure) else read_structure(structure)
        # A polymer chain can be the ligand: it then becomes one residue of the structure, like the
        # single component a co-folding model writes, and ``given`` keeps what the file said of its residues.
        whole = st
        st, key, given, notes = st.resolve_ligand(ligand)
        source = None
        if isinstance(interactions, (str, Path)):
            if str(interactions) == DISTANCES:
                from .distance_table import contact_table, ligand_contacts

                interactions = contact_table(whole, key.chain) if given else ligand_contacts(whole, ligand, smiles)
                source = DISTANCES
            else:
                interactions, source = load_interactions(interactions), "supplied"
        if interactions is None and given:
            # A chain of its own as ligand: PLIP takes it in its peptide mode, from the file as it was.
            from .detect import detect_interface

            others = sorted({a.chain for a in whole.atoms} - {key.chain})
            interactions = [c.as_interaction() for c in detect_interface(whole, [key.chain], others)]
            source = "detected"
        return cls._assemble(st, key, given, notes, smiles, interactions, name=name, ionize_acids=ionize_acids,
                             plddt=plddt, model_confidence=model_confidence, trusted=trusted, flag=flag,
                             ensemble=ensemble, ligand_as=ligand_as, source=source, detail=detail, covalent=covalent,
                             pae=read_pae(pae, whole), ions=ions)

    @classmethod
    def _assemble(cls, st: Structure, key: ResidueKey, given: tuple, notes: "Sequence[str]", smiles: "str | None",
                  interactions: "Iterable[Interaction | Mapping[str, Any]] | None", *, name: "str | None" = None,
                  ionize_acids: bool = True, plddt: Any = "auto", model_confidence: Any = "auto",
                  trusted: "bool | None" = None, flag: "str | None" = None, ensemble: "str | None" = None,
                  ligand_as: str = "auto", source: "str | None" = None,
                  detail: "str | Mapping[str, str | None] | None" = None, covalent: bool = False,
                  pae: "Pae | PaeDraws | None" = None, ions: bool = False) -> "Diagram":
        """The second half of :meth:`from_structure`: from a structure in which the ligand is one
        residue (``key``), whether the file had it so or a chain, or the residues of one side of an
        interface, were merged into one (``given`` then holds what the file said of them)."""
        warnings: list[str] = list(notes)
        confidence = _confidence_for(st, key, plddt, model_confidence, trusted, flag)
        if given and not smiles:
            from .chains import chain_smiles

            smiles, unknown = chain_smiles(st.residue_atoms(key), given)
            warnings.extend(unknown)
        elif not smiles:
            smiles = st.chem_comp_smiles.get(st.residue_name(key))
            if smiles is None:
                warnings.append("no SMILES was given and the structure file carries none")
        found_ions = bridges(st, key) if ions else []
        holding = holders(st, key) if ions else []
        whole = st
        if found_ions:
            st, smiles = with_ions(st, key, smiles, found_ions)
            warnings.extend(f"ion bridge: {b}" for b in found_ions)
        elif ions and not holding:
            warnings.append("ions were asked for, and the model has none between the ligand and the protein")
        lig = build_ligand(st.residue_atoms(key), smiles, ionize_acids=ionize_acids)
        if found_ions:
            seat(lig, found_ions)
        if given:
            lig.given, lig.chain = given, key.chain
        warnings.extend(lig.diagnostics)
        if interactions is None:
            from .detect import detect_interactions

            records = detect_interactions(whole, key)
            source = "detected"
        else:
            records = coerce_interactions(interactions)
            source = source or "supplied"
        check_confidence_scale(r.confidence for r in records)
        if found_ions or holding:
            records, dropped = without_the_bridge(list(records), found_ions, holding)
            records += [r for b in found_ions for r in b.records()]
            if dropped:
                warnings.append(f"{dropped} hydrogen bond{'s' if dropped != 1 else ''} of the table with an atom that holds "
                                f"an ion {'are' if dropped != 1 else 'is'} left out")
        if covalent:
            brought = {(r.residue_id, r.protein_atom) for r in records if r.type is InteractionType.COVALENT}
            found = [b for b in covalent_bonds(st, key)
                     if ((b.residue.chain, b.residue.number, b.residue.icode), b.atom) not in brought]
            records = list(records) + [b.as_interaction() for b in found]
            warnings.extend(f"covalent bond: {b}" for b in found)
            if not found and not brought:
                warnings.append("covalent bonds were asked for, and the model has none between the ligand and a residue")
        resolved = resolve_interactions(records, lig, st)
        if any(r.type is InteractionType.COVALENT for r in records):
            for r in resolved:
                if r.type is InteractionType.COVALENT and r.ligand is not None and r.ligand.kind == "atom":
                    said = as_bound(lig, r.ligand.atom)
                    if said:
                        warnings.append(said)
            resolved, bonded = drop_bonded(resolved, lig, st)
            if bonded:
                warnings.append(f"{len(bonded)} contact{'s' if len(bonded) != 1 else ''} between atoms that the covalent bond "
                                f"joins {'are' if len(bonded) != 1 else 'is'} left out: "
                                + ", ".join(f"{r.record.residue_label} {r.type.label.lower()}" for r in bonded))
        if pae is not None:
            _give_pae(resolved, pae, lig, st, key)
        out = cls(lig, resolved, structure=st, ligand_key=key, name=name, source=source, warnings=warnings,
                  confidence=confidence, ensemble=ensemble, ligand_as=ligand_as, detail=detail)
        out.pae = pae
        return out

    # ------------------------------------------------------------------ queries
    @property
    def n_interactions(self) -> int:
        """How many contacts the diagram has."""
        return len(self.interactions)

    def fingerprint(self, name: str | None = None) -> "Fingerprint":
        """The per-(residue, interaction type) fingerprint of this complex (see :mod:`bindview.fingerprint`)."""
        from .fingerprint import Fingerprint

        return Fingerprint.from_diagram(self, name)

    @property
    def unresolved(self) -> list[ResolvedInteraction]:
        """The contacts that could not be attached to the ligand: they are not drawn, and the result says so."""
        return [r for r in self.interactions if r.ligand is None]

    def ligand_exposure(self) -> np.ndarray | None:
        """Relative solvent exposure (0..1) of every ligand atom, or ``None`` without a structure."""
        if self.structure is None or self.ligand_key is None:
            return None
        if self._exposure is None:
            from .exposure import relative_exposure

            st = self.structure
            env = [a for a in st.atoms if a.residue != self.ligand_key and a.element not in ("H", "D")]
            env_xyz = np.array([a.xyz for a in env], dtype=float).reshape(-1, 3)
            self._exposure = relative_exposure(self.ligand.coords3d,
                                               [self.ligand.element(i) for i in range(self.ligand.n_atoms)],
                                               env_xyz, [a.element for a in env])
        return self._exposure

    def burial(self) -> Burial | None:
        """How much of the ligand's surface the rest of the model covers.

        The ligand's solvent-accessible surface alone and in the complex, and
        from the two the buried fraction (see :mod:`bindview.exposure`). Waters
        are not counted as cover; everything else in the model is. ``None``
        without a structure.
        """
        if self.structure is None or self.ligand_key is None:
            return None
        if self._burial is None:
            self._burial = structure_burial(self.structure, self.ligand_key)
        return self._burial

    def solvent_sites(self) -> list[SolventSite]:
        """Predicted solvent sites that could bridge ligand and protein, best first.

        Empty without a structure, and empty for a structure that has waters
        of its own: observed waters take precedence over guessed ones, always.
        See :mod:`bindview.solvent` for what the prediction is and is not.
        """
        if self.structure is None or self.ligand_key is None or has_waters(self.structure):
            return []
        if self._sites is None:
            self._sites = predict_solvent_sites(self.structure, self.ligand_key)
        return self._sites

    def _predicted_bridges(self) -> tuple[list[ResolvedInteraction], str]:
        """Predicted sites as water-bridge records marked ``predicted``, and a note saying what was done."""
        if self.structure is None or self.ligand_key is None:
            return [], "solvent sites were asked for, but there is no structure to predict them in"
        if has_waters(self.structure):
            n = len({a.residue for a in self.structure.atoms if a.resname.upper() in WATER_NAMES})
            return [], (f"no solvent site was predicted: the structure has {n} water molecule"
                        f"{'s' if n != 1 else ''} of its own, and observed waters take precedence over guessed ones")
        sites = self.solvent_sites()
        if not sites:
            return [], "no solvent site was predicted between ligand and protein"
        # A water between two atoms that already touch is not a bridge: a site is drawn only
        # towards a residue that the ligand atom does not already contact directionally.
        direct = {(r.record.residue_id, self.ligand.names[i]) for r in self.interactions
                  if r.ligand is not None and r.type is not InteractionType.HYDROPHOBIC for i in r.ligand.atoms}
        records = []
        for k, site in enumerate(sites):
            atom_name = site.ligand[0][0]                                     # the nearest ligand partner
            for chain, number, icode, resname, atom, _ in site.protein:       # nearest protein partner first
                if ((chain, number, icode), atom_name) not in direct:
                    records.append(Interaction(type=InteractionType.WATER_BRIDGE, residue_number=number,
                                               residue_name=resname, chain=chain, insertion_code=icode,
                                               protein_atom=atom, ligand_atom=atom_name, water=f"predicted:{k}",
                                               predicted=True))
                    break
        resolved = [r for r in resolve_interactions(records, self.ligand, self.structure) if r.ligand is not None]
        for k, r in enumerate(resolved):
            r.index = len(self.interactions) + len(self.elsewhere) + k
        if not resolved:
            return [], (f"{len(sites)} solvent site{'s were' if len(sites) != 1 else ' was'} predicted, none of them "
                        "between partners that are not already in direct contact: nothing was drawn")
        skipped = len(sites) - len(resolved)
        return resolved, (f"{len(resolved)} predicted water site{'s are' if len(resolved) != 1 else ' is'} drawn"
                          + (f" ({skipped} more lie between partners already in direct contact)" if skipped else "")
                          + ": geometric guesses for a model without solvent, not observations")

    @property
    def _interface(self) -> Any:
        """What an interface figure says about itself; ``None`` for every other figure."""
        return self.backend.interface if isinstance(self.backend, Polymer) else None

    @property
    def _burial_cut_applies(self) -> bool:
        """Is the calibrated cut of :data:`~bindview.exposure.BURIAL_WARNING_BELOW` this ligand's?

        It was calibrated on small molecules, none of them a chain of residues. A ligand that is
        one -- a peptide, a strand of a nucleic acid, by what it is and however it is drawn --
        lies along a surface and leaves more of itself open than a molecule in a pocket: its buried
        fraction is measured and printed, and no warning is drawn from it.
        """
        return self._interface is None and not self.backend.decision.is_chain and not self.shares

    def _burial_warning(self) -> str | None:
        return burial_warning(self.burial()) if self._burial_cut_applies else None

    def _burial_report(self) -> dict[str, Any] | None:
        # A buried fraction is a ligand's number, with a cut calibrated on ligands. The column of an
        # interface is a piece of a chain: nothing is reported, warned of or printed for it. Nor for a
        # figure of several models, each of which buries its ligand otherwise.
        burial = self.burial() if self._interface is None and not self.shares else None
        if burial is None:
            return None
        cut = self._burial_cut_applies
        return {**burial.as_dict(), "warn_below": BURIAL_WARNING_BELOW if cut else None,
                "warning": cut and burial_warning(burial) is not None}

    def proximal_residues(self, cutoff: float = 4.5) -> list[str]:
        """Labels of residues with a heavy atom within ``cutoff`` angstrom of the ligand."""
        keys = self.pocket_residues(cutoff)
        if not keys or self.structure is None:
            return []
        chains = {k.chain for k in keys}
        return [format_residue_label(self.structure.residue_name(k), k.number, k.icode,
                                     k.chain if len(chains) > 1 else None) for k in keys]

    def pocket_residues(self, cutoff: float = 4.5) -> list[ResidueKey]:
        """Residues with a heavy atom within ``cutoff`` angstrom of the ligand, in chain/number order."""
        if self.structure is None or self.ligand_key is None:
            return []
        return list(pocket_atoms(self.structure, self.ligand_key, cutoff))

    def _shown_confidence(self, opts: RenderOptions) -> Confidence | None:
        return self.confidence if opts.confidence else None

    # ------------------------------------------------------------------ render
    def render(self, style: str = "detailed", theme: "str | Palette" = "light", *,
               metrics: Metrics | None = None, graph: Graph | None = None, **options: Any) -> RenderResult:
        """Lay out and draw the diagram.

        Parameters
        ----------
        style:
            ``"detailed"`` (the atoms that make the contacts) or ``"schematic"`` (residues as bubbles).
        theme:
            ``"light"``, ``"dark"``, ``"auto"`` (one file that follows the reader's setting), or a
            :class:`~bindview.Palette` of your own (``LIGHT.with_(hbond="#123456")``).
        graph:
            A layout to draw from (below).
        metrics:
            The dimensions of a figure (fonts, widths, margins), as a :class:`~bindview.Metrics`.
        options:
            Everything else: ``bindview.options()`` lists them, with a line on each, and
            ``help(bindview.styles.base.RenderOptions)`` does the same. Often used: ``residues``, ``title``,
            ``bare``, ``plddt_colors``, ``solvent_sites``, ``interactive``, ``figure_width_mm``, ``on_unreadable``.

        Returns a :class:`RenderResult`: ``svg``, ``status`` (``"ok"``, ``"empty"``, ``"degraded"``,
        ``"unreadable"``), ``warnings`` in words, ``quality``, ``graph``, and what the figure says as data.

        With ``graph`` nothing is laid out: the figure is drawn from the node
        positions the graph holds -- one returned by an earlier render
        (``result.graph``), read back from JSON, or edited by moving nodes.
        An untouched graph gives the identical figure. An edited one is held to
        the same audit as an engine layout, so a node dragged onto another
        makes the result ``unreadable`` unless ``on_unreadable="render"``.
        """
        opts = RenderOptions(**options)
        _check_sizes(opts)
        asked_for_water = opts.solvent_sites
        if self.shares:
            # All the models of an ensemble: what belongs to one of them is not drawn unless it is asked for.
            opts = replace(opts, burial=False, solvent_sites=False,
                           residues=dict(self.forms) if opts.residues is None else opts.residues,
                           lining="none" if opts.lining is None else opts.lining,
                           exposure=False if opts.exposure is None else opts.exposure)
        if graph is not None and graph.view != style:
            raise GraphError(f"the graph was laid out for the {graph.view!r} style, not {style!r}")
        if graph is not None:
            graph.validate()                             # (it may have been changed since it was read)
            if graph.bond_length > _BOND_LENGTH_MAX:
                raise GraphError(f"the bond_length of the graph is {graph.bond_length:g}: a figure is drawn with one "
                                 f"of at most {_BOND_LENGTH_MAX:g}")
        css_vars = isinstance(theme, str) and theme.lower() == "auto"
        light, dark = scheme(opts.colors)
        palette = light if css_vars else get_palette(theme, opts.colors)
        # (The theme, whatever the colour scheme: the name of a palette is the theme only where the caller passed one.)
        theme_name = "auto" if css_vars else palette.name if isinstance(theme, Palette) else str(theme).lower()
        base = metrics or Metrics()
        warnings = list(self.warnings)
        conf = self._shown_confidence(opts)
        if conf is not None and conf.marked:
            verdict = f" ({conf.model.flag})" if conf.model is not None and conf.model.flag else ""
            warnings.append(f"the model is flagged as not trustworthy{verdict}; the figure is marked accordingly")
        if asked_for_water and not opts.solvent_sites:
            warnings.append("solvent_sites was asked for, and this is a figure of several models: predicted waters "
                            "are those of one")
        if opts.plddt_colors and conf is None:
            # An option that was asked for and changes nothing is said, so that a figure without colour
            # is not read as a model without doubt.
            warnings.append("plddt_colors was asked for, and this figure has no pLDDT to colour by: "
                            + ("confidence is switched off (confidence=False)" if not opts.confidence else
                               "none is read from this model (it declares none, or plddt is off)"))
        if graph is not None and opts.bond_length is not None and abs(opts.bond_length - graph.bond_length) > 1e-9:
            # An option that was asked for and changes nothing is said.
            warnings.append(f"bond_length={opts.bond_length:g} was asked for with a graph, which has its own scale: the "
                            f"figure is drawn at {graph.bond_length:g}")
        if graph is not None and opts.coords != "auto":
            warnings.append(f"coords={opts.coords!r} was asked for with a graph, which has the atoms of the ligand where "
                            "they are drawn: it changes nothing")
        exposed = self._burial_warning()
        if exposed is not None:
            warnings.append(exposed)
        for r in self.interactions:
            where = f"{r.record.type.value} {r.record.residue_label}"
            if r.ligand is None:
                warnings.append(f"{where} was not drawn: " + ("; ".join(r.notes) or "it could not be attached"))
            elif r.method in ("approximate", "nearest"):
                warnings.append(f"{where} attached by {r.method} match: " + "; ".join(r.notes))
            else:
                warnings.extend(f"{where}: {n}" for n in r.notes)
        drawable = [r for r in self.interactions if r.ligand is not None]
        if self.interactions:                            # never on a figure that says "nothing was found"
            drawable = drawable + [r for r in self.elsewhere if r.ligand is not None]
        extra_data: dict[str, Any] = {}
        if self.backend.decision.longest_run >= 2:
            # There is a chain of amino-acid residues in the ligand: say how it is drawn, and why.
            extra_data["ligand"] = describe_backend(self.backend)
        if isinstance(self.backend, Polymer) and self.backend.interface is not None:
            extra_data["interface"] = self.backend.interface.meta_for(None)
        if isinstance(self.backend, Polymer):
            unknown = [f"{r.code3}{r.position}" for r in self.backend.peptide.residues if not r.known]
            if unknown:
                warnings.append(f"{len(unknown)} residue{'s' if len(unknown) != 1 else ''} of the ligand's chain "
                                f"{'were' if len(unknown) != 1 else 'was'} not recognised and "
                                f"{'are' if len(unknown) != 1 else 'is'} shown as Xaa (position "
                                f"{', '.join(str(r.position) for r in self.backend.peptide.residues if not r.known)})")
        if opts.solvent_sites:
            bridges, said = self._predicted_bridges()
            warnings.append(said)
            extra_data["solvent_sites"] = [s.as_dict() for s in self.solvent_sites()]
            if self.interactions:                        # never on a figure that says "nothing was found"
                drawable = drawable + bridges

        def finish(scene: Scene | None, status: str, figure: str, quality: Quality | None,
                   extra: Sequence[str] = (), atoms: dict[str, tuple[float, float]] | None = None,
                   laid_out: Graph | None = None, composed: "_Composed | None" = None) -> RenderResult:
            svg = css = None
            if scene is not None:
                over = composed.beside() if opts.interactive and composed is not None else ()
                svg, css = svg_and_css(scene, palette, background=opts.background, css_vars=css_vars,
                                       dark_palette=dark if css_vars else None, interactive=opts.interactive, over=over,
                                       stylesheet=opts.stylesheet)
                if opts.interactive and ROOT_CLASS not in svg.split(">", 1)[0]:
                    # An option that was asked for and changes nothing is said.
                    extra = list(extra) + [NOTHING_TO_POINT_AT]
            return RenderResult(status=status, svg=svg, figure=figure, warnings=_dedupe(warnings + list(extra)),
                                quality=quality, style=style, theme=theme_name,
                                # (A copy, like the graph below: the scene of a figure is kept for the next one.)
                                scene=None if scene is None else scene.copy(),
                                atoms=dict(atoms or {}),
                                confidence=self._confidence_report(conf, extra_data.get("plddt_basis"))
                                if conf is not None else None,
                                burial=self._burial_report(), data=dict(extra_data),
                                notes=list(scene.notes) if scene is not None else [], css=css,
                                # A copy: layouts are cached, and a graph is made to be edited.
                                graph=None if laid_out is None else copy.deepcopy(laid_out))

        if not self.interactions:
            scene = self._empty_scene(palette, base, opts)
            return finish(scene, "empty", "empty", None)

        style_obj = style_for(style, opts)
        strangers = unknown_residues(opts, group_residues(drawable, opts.show_chain))
        if strangers:
            # An option that was asked for and changes nothing is said.
            warnings.append(f"residues names {', '.join(strangers)}, which no residue of this figure is called")
        if opts.absent is not None:
            everyone = group_residues(drawable, opts.show_chain)
            mark_groups(everyone, opts)
            extra_data["absent"] = Absence(everyone, opts.absent_label).as_dict()
        best: _Composed | None = None
        queue: list[float] = [opts.bond_length] if opts.bond_length else list(_BOND_LENGTHS)
        if graph is not None:
            queue = [graph.bond_length]                  # the layout is given, and with it its scale
        chain = isinstance(self.backend, Polymer) and graph is None
        plane = isinstance(self.backend, Polymer) and self.backend.arrangement is not None
        pitches = list(_CHAIN_PITCHES) if chain else [base.chain_pitch]
        pitch = pitches[0]
        attempts = 0
        given_up: _Composed | None = None
        hasty = (opts.effort == "quick" and chain and self._interface is None and opts.on_unreadable != "render")
        if graph is None and opts.on_unreadable != "render" and not opts.bond_length:
            hopeless = self._hopeless(base, opts, drawable)
            if hopeless is not None:
                q, reason = hopeless
                message = (f"{self._what()} is not readable: {self._size()}, "
                           f"{q.n_residues} residues, {q.n_interactions} interactions; {reason}")
                if opts.on_unreadable == "none":
                    return finish(None, "unreadable", "none", q, [message])
                scene = self._summary_scene(base, opts, drawable, q, reason)
                return finish(scene, "unreadable", "summary", q,
                              [message + "; a per-residue summary was drawn instead"])
        while queue:
            bl = queue.pop(0)
            scaled = _scaled(base, bl).with_(chain_pitch=pitch) if chain else _scaled(base, bl)
            if hasty:
                given_up = self._looks_hopeless(style_obj, palette, scaled, opts, drawable, pitches)
                hasty = False
                if given_up is not None:
                    best = given_up
                    break
            comp = self._compose(style_obj, palette, scaled, opts, drawable, graph)
            if best is None or _better(comp.quality, best.quality):
                best = comp
            if comp.quality.valid and comp.quality.readable:
                best = comp
                break
            attempts += 1
            if chain and attempts >= _CHAIN_ATTEMPTS:
                break                                    # a crowded chain costs several seconds each time
            if chain and not comp.quality.valid:
                # Too crowded along the chain: the same scale again, with more room from residue to
                # residue -- as long as the chain then still fits the page at a legible size.
                wider = [p for p in pitches if p > pitch and self._chain_fits(comp.quality, scaled, opts, p)]
                if wider:
                    assert isinstance(self.backend, Polymer)
                    want = _ROOM_PER_RESIDUE * comp.quality.n_residues / max(1, len(self.backend.peptide.residues))
                    if plane:
                        # Boxes on a plane are spread in both directions at once: every step makes room.
                        pitch = wider[0]
                    elif attempts == 1 and want > 1.55:
                        pitch = next((p for p in wider if p >= want), wider[-1])
                    elif attempts >= 3 and len(wider) > 1:
                        # Three layouts were too crowded: small steps are not what is missing. Every
                        # second step from here, so that the wide ones are reached before the tries run out.
                        pitch = wider[1]
                    else:
                        pitch = wider[0]
                    queue.insert(0, bl)
                    continue
            if queue and comp.quality.valid:
                # A full layout is expensive: skip bond lengths that cannot fit either.
                # The ligand shrinks in proportion, what surrounds it only partly.
                q = comp.quality
                limit = q.width * q.min_font_pt / opts.min_font_pt
                lig_w = q.ligand_width

                def estimate(b: float) -> float:
                    return lig_w * b / bl + (q.width - lig_w) * (0.55 + 0.45 * b / bl)
                fitting = [b for b in queue if estimate(b) <= limit * 1.04]
                queue = fitting if fitting else queue[-1:]
        assert best is not None
        q = best.quality
        warnings.extend(best.notes)
        if best.graph is not None and "interface" in best.graph.meta:
            extra_data["interface"] = dict(best.graph.meta["interface"])     # as it was drawn
        if best.basis is not None:
            extra_data["plddt_basis"] = best.basis
        if opts.absent_style == "marked" and extra_data.get("absent", {}).get("absent") and (
                (opts.plddt_colors and conf is not None) or best.glows):
            warnings.append("the marker under what is absent is yellow, and so is one of the colours that the "
                            "confidence is drawn in here; absent_style='faint' tells the two apart")
        if q.valid and q.readable:
            degraded = bool(self.unresolved) or self.ligand.bond_source == "connectivity"
            return finish(best.scene, "degraded" if degraded else "ok", "diagram", q, atoms=best.atoms,
                          laid_out=best.graph, composed=best)

        if not q.valid:
            problems = []
            if q.unplaced or q.overlaps or q.link_hits or q.faithful:
                problems.append(f"{q.unplaced} residue(s) could not be placed, {q.overlaps} pair(s) overlap and "
                                f"{q.link_hits} interaction line(s) cross another residue")
            if not q.faithful:
                found: dict[str, Any] = (extra_data.get("interface") or {}).get("plane") or {}
                covered = found.get("boxes_covering_each_other", 0)
                problems.append(
                    "the boxes as they stand are not a faithful picture of where the residues lie "
                    f"({100.0 * found.get('neighbours_kept', 0.0):.0f} % of neighbour relations kept, distances off "
                    f"by {100.0 * found.get('distance_error', 0.0):.0f} %; at least {100.0 * KEPT_MIN:.0f} % and at "
                    f"most {100.0 * ERROR_MAX:.0f} % are required"
                    + (f"; {covered} pair{'s' if covered != 1 else ''} of boxes lie on top of each other"
                       if covered else "") + ")")
            reason = "; ".join(problems)
            if given_up is not None:
                reason += " (effort='quick': with the most room that the page allows, and not tried any further)"
        else:
            reason = beyond_any_page(_smallest_text_pt(base), opts) or too_small_on_page(
                q.min_font_pt, q.required_width_mm, q.width, q.height, opts)
        message = (f"{self._what()} is not readable: {self._size()}, "
                   f"{q.n_residues} residues, {q.n_interactions} interactions; {reason}")
        if opts.on_unreadable == "render":
            return finish(best.scene, "unreadable", "diagram", q, [message + "; rendered anyway on request"],
                          atoms=best.atoms, laid_out=best.graph, composed=best)
        if opts.on_unreadable == "none":
            return finish(None, "unreadable", "none", q, [message])
        scene = self._summary_scene(base, opts, drawable, q, reason)
        return finish(scene, "unreadable", "summary", q, [message + "; a per-residue summary was drawn instead"])

    def _looks_hopeless(self, style_obj: Any, palette: Palette, m: Metrics, opts: RenderOptions,
                        drawable: Sequence[ResolvedInteraction], pitches: Sequence[float]) -> "_Composed | None":
        """The roomiest figure of a chain that the page allows, if that is too crowded: then none is tried further.

        Looked at where the first layout, the closest one, has no place for several residues; both are left as they
        are as soon as that is known of them. What is drawn is never drawn in haste: where this gives no reason to
        give the figure up, it is laid out as ever, from the first.
        """
        first = self._compose(style_obj, palette, m, opts, drawable, haste=1)
        n = len(first.layout.poses) + len(first.layout.unplaced)
        if first.layout.left < _crowded(n):
            return None
        wider = [p for p in pitches if p > m.chain_pitch and self._chain_fits(first.quality, m, opts, p)]
        roomy = first if not wider else self._compose(style_obj, palette, m.with_(chain_pitch=wider[-1]), opts, drawable,
                                                      haste=2)
        return roomy if roomy.layout.left >= _hopeless(n) else None

    def _what(self) -> str:
        """What kind of diagram this is, for a message that it could not be made readable."""
        if isinstance(self.backend, Polymer):
            return "an interface diagram" if self.backend.interface is not None else "a chain diagram"
        return "an atom-level diagram"

    def _size(self) -> str:
        if isinstance(self.backend, Polymer):
            # An interface has no ligand: the residues in question are those of its column.
            whose = ("ligand residues" if self.backend.interface is None else
                     "residues on the plane" if self.backend.arrangement is not None else "residues in the column")
            return f"{len(self.backend.peptide)} {whose} ({self.ligand.n_atoms} atoms)"
        return f"{self.ligand.n_atoms} ligand atoms"

    def _hopeless(self, base: Metrics, opts: RenderOptions, drawable: Sequence[ResolvedInteraction]
                  ) -> tuple[Quality, str] | None:
        """A lower bound on the figure size, without laying anything out.

        If the ligand drawn at the smallest bond length, with nothing but a
        margin around it, is already too large to print legibly at the figure
        size, no arrangement of residues can rescue it and the (expensive)
        layout is skipped.
        """
        m = _scaled(base, _BOND_LENGTHS[-1])
        # (A chain with the atoms that these contacts put beside its boxes, not with those of the last figure.)
        self._set_beside(drawable)
        dep = self._depicted(m, replace(opts, atom_labels="names"))          # (as it is laid out, whatever is written)
        box = dep.bounds()
        ring = 2.0 * (m.margin + 1.6 * m.bond_length) if drawable else 2.0 * m.margin
        width, height = box[2] - box[0] + ring, box[3] - box[1] + ring
        natural = _smallest_text_pt(m)
        fit = min(opts.figure_width_mm * _PX_PER_MM / width, opts.figure_height_mm * _PX_PER_MM / height, 1.0)
        font_pt = natural * fit
        if font_pt >= opts.min_font_pt - 1e-9:
            return None
        # (A minimum above the text's own size is beyond every page, not beyond this one.)
        never = beyond_any_page(natural, opts)
        required = math.inf if never else width * opts.min_font_pt / natural / _PX_PER_MM
        groups = group_residues(drawable, opts.show_chain)
        q = Quality(width=width, height=height, bond_length=m.bond_length, n_residues=len(groups),
                    n_interactions=len(drawable), overlaps=0, unplaced=0, line_crossings=0, bond_crossings=0,
                    max_stretch=0.0, min_font_pt=font_pt, required_width_mm=required, readable=False,
                    ligand_width=box[2] - box[0], estimated=True, ligand_height=box[3] - box[1])
        if never:
            return q, never
        mm, word, asked = page_that_would_do(required, width, height, opts)
        return q, (f"the ligand alone, at the smallest bond length, needs a figure at least {mm} mm {word} "
                   f"for {opts.min_font_pt:g} pt text (requested: {asked:g} mm)")

    def _chain_fits(self, q: Quality, m: Metrics, opts: RenderOptions, pitch: float) -> bool:
        """Would this chain figure, laid out again with ``pitch`` bond lengths from residue to
        residue, still be legible on the page? An estimate from the figure as it is: a wider pitch
        makes it taller by that much for every residue, and nothing else changes."""
        assert isinstance(self.backend, Polymer)
        n = len(self.backend.peptide.residues)
        page_w, page_h = opts.figure_width_mm * _PX_PER_MM, opts.figure_height_mm * _PX_PER_MM
        unscaled = q.min_font_pt / min(page_w / q.width, page_h / q.height, 1.0)
        now = max(POLYMER_MIN_PITCH, m.chain_pitch * m.bond_length)
        if self.backend.arrangement is not None:
            # Boxes on a plane: the drawing of the boxes grows in both directions with the pitch, and
            # what is around it stays as it is.
            grow = max(POLYMER_MIN_PITCH, pitch * m.bond_length) / now
            width = q.width + q.ligand_width * (grow - 1.0)
            height = q.height + q.ligand_height * (grow - 1.0)
            return unscaled * min(page_w / width, page_h / height, 1.0) >= opts.min_font_pt * 1.003
        height = q.height + (n - 1) * (max(POLYMER_MIN_PITCH, pitch * m.bond_length) - now)
        return unscaled * min(page_w / q.width, page_h / height, 1.0) >= opts.min_font_pt * 1.003

    def render_svg(self, path: "str | Path | None" = None, style: str = "detailed",
                   theme: "str | Palette" = "light", **options: Any) -> RenderResult:
        """Render and, when ``path`` is given and there is a figure, write it."""
        result = self.render(style=style, theme=theme, **options)
        if path is not None and result.svg is not None:
            p = Path(path)
            try:
                p.write_text(result.svg, encoding="utf-8")
            except OSError as exc:
                raise RenderError(f"cannot write {p}: {exc}") from exc
            result.path = p
        return result

    # ------------------------------------------------------------------ pieces
    def arrange(self, path: "str | Path | None" = None, style: str = "detailed", *, title: "str | None" = None,
                port: int = 0, browser: bool = True, **options: Any) -> "Graph | Path | None":
        """Arrange the figure by hand, in a browser (:mod:`bindview.arrange`).

        Residues and the key are dragged, turned and flipped; bindview draws the figure again whenever one is let
        go. Returns the layout when the page is done, for ``render(graph=...)`` (None if it is stopped with Ctrl-C).
        With ``path`` the page is written there as one file instead, with a preview and without bindview behind it,
        and the path is returned. ``options`` are those of :meth:`render`.
        """
        from . import arrange

        options = {"style": style, **options}
        if path is not None:
            return arrange.page(self, path, title, **options)
        return arrange.serve(self, title, port=port, browser=browser, **options)

    def _compose(self, style_obj: Any, palette: Palette, m: Metrics, opts: RenderOptions,
                 drawable: Sequence[ResolvedInteraction], given: Graph | None = None, haste: int = 0) -> _Composed:
        if given is not None:
            return self._compose_uncached(style_obj, palette, m, opts, drawable, given)
        # A scene holds colour *tokens*, so it is the same for every palette that
        # defines the same elements: themes share one layout. Whether the file answers the pointer is
        # decided where the scene is written, so a figure and its interactive copy share one too.
        # Nor does a mask of absent residues move anything: with and without it a figure has one layout.
        key = (style_obj.name, m, replace(opts, interactive=False, stylesheet="inline", effort="full"),
               frozenset(palette.elements), haste, self._drawn_from())
        hit = self._kept(key)
        if hit is None:
            hit = self._keep(key, self._compose_uncached(style_obj, palette, m, opts, drawable, haste=haste))
        return hit

    def _kept(self, key: tuple) -> Any:
        """What is kept under ``key``, or ``None``. It was used now, and is so the last of its kind to go."""
        hit = self._composed.pop(key, None)
        if hit is not None:
            self._composed[key] = hit
        return hit

    def _keep(self, key: tuple, value: Any) -> Any:
        """Keep ``value`` under ``key``; where there are then more of its kind than are kept
        (:data:`_KEPT_AT_MOST`), those that were used longest ago go."""
        self._composed[key] = value
        kind = _kind(key)
        same = [k for k in self._composed if _kind(k) == kind]
        for old in same[:max(0, len(same) - _KEPT_AT_MOST[kind])]:
            del self._composed[old]
        return value

    def _drawn_from(self) -> tuple:
        """What a figure is drawn from beside its options, for the key of one that is kept.

        The contacts by what they say, the name, the remarks and the levels of detail by their value; the ligand,
        the structure, the confidence and what else the diagram holds as one thing by which object it is. A
        diagram that was given other contacts or another name after a figure of it was drawn, or a copy of it
        that was, is so drawn again, and one that was left as it is, is not.
        """
        def rows(contacts: Sequence[ResolvedInteraction]) -> tuple:
            return tuple((r.record, r.index, r.residue, r.ligand, r.protein, r.method, r.delta, r.computed_distance,
                          tuple(r.notes), r.absent) for r in contacts)

        backend = self.backend
        held = (self.ligand, self.structure, self.confidence, self.pae, self.shares, backend,
                getattr(backend, "peptide", None), getattr(backend, "interface", None),
                getattr(backend, "arrangement", None))
        return (rows(self.interactions), rows(self.elsewhere), self.name, self.ensemble, self.ligand_key,
                tuple(self.remarks), tuple(self.elsewhere_said), tuple(self.detail.items()),
                tuple(self.detail_by_chain.items()), *(_Same(it) for it in held))

    def _depicted(self, m: Metrics, opts: RenderOptions, given: Graph | None = None) -> Depiction:
        """The ligand as it is drawn, or where a graph has it."""
        if isinstance(self.backend, Polymer):
            self.backend.atom_labels = opts.atom_labels
        return self.backend.depict(m, opts.coords) if given is None else self.backend.depict_from(m, given)

    def _field(self, dep: Depiction, m: Metrics, params: LayoutParams, style_obj: Any, ctx: Context) -> LigandField:
        fld = LigandField(dep, reach=(params.reach + 3.0) * m.bond_length, step=max(2.0, m.bond_length / 10.0),
                          extra=list(style_obj.ligand_obstacles(ctx)) + self.backend.obstacles(dep))
        # (One that was made before is as good, and knows its distances.)
        key = ("field", fld.key)
        hit = self._kept(key)
        return hit if hit is not None else self._keep(key, fld)

    def _laid_out(self, fld: LigandField, bodies: Sequence[Any], params: LayoutParams) -> LayoutResult:
        """The placement of the residues, kept for the next figure that differs in what moves nothing."""
        key: tuple = ("layout", layout_key(fld, bodies, replace(params, give_up=0)))
        hit = self._kept(key)
        if hit is None or 0 < params.give_up <= hit.left:
            # (One that was left as it was is kept apart: it is not what a figure that takes its time is drawn from,
            # and what a hasty one says does not depend on which of the two was drawn first.)
            hasty = (*key, params.give_up)
            hit = self._kept(hasty)
            if hit is None:
                hit = self._keep(hasty if (hit := layout_bodies(fld, bodies, params)).abandoned else key, hit)
        return hit

    def _compose_uncached(self, style_obj: Any, palette: Palette, m: Metrics, opts: RenderOptions,
                          drawable: Sequence[ResolvedInteraction], given: Graph | None = None, haste: int = 0) -> _Composed:
        self._set_beside(drawable)
        dep = self._depicted(m, opts, given)
        groups = split_along_chain(group_residues(drawable, opts.show_chain), dep)
        mark_groups(groups, opts)
        twice = list(dict.fromkeys(g.label for g in groups if g.parts > 1))
        conf = self._shown_confidence(opts)
        ctx = Context(self.ligand, dep, groups, m, palette, opts, confidence=conf, ensemble=self.ensemble,
                      solvent_sites=tuple(self.solvent_sites()) if opts.solvent_sites else (),
                      detail={"*": self.detail["protein"], **self.detail_by_chain}, pae=self.pae is not None,
                      shares=self.shares)
        if style_obj.settings.exposure:
            ctx.exposure = self.ligand_exposure()
        bodies = style_obj.bodies(ctx)
        _set_fans(bodies)
        params = LayoutParams(unit=m.bond_length, give_up=(0, _crowded(len(bodies)), _hopeless(len(bodies)))[haste])
        # What is written on an atom moves nothing: the figure is laid out with the atoms' names whatever is written.
        laid, laid_dep = bodies, dep
        if given is None and opts.atom_labels != "names":
            plain = replace(opts, atom_labels="names")
            laid_dep = self._depicted(m, plain)
            laid = style_obj.bodies(replace(ctx, dep=laid_dep, options=plain, notes=[]))
            _set_fans(laid)
        fld = self._field(dep, m, params, style_obj, ctx)
        if given is None:
            layout = self._laid_out(fld if laid_dep is dep else self._field(laid_dep, m, params, style_obj, ctx), laid,
                                    params)
        else:
            layout = audit_layout(fld, bodies, *poses_from(given, bodies), params)   # placed elsewhere: measure only
        ctx.field = fld
        # The engine's result becomes the graph, and the renderer draws from the graph: it is the
        # one description of where things are, whether it was made just now or handed in.
        graph = diagram_graph(style_obj.name, ctx, bodies, layout, params, name=self.name or "",
                              exposure=ctx.exposure)
        poses, label_t = poses_from(graph, bodies)
        drawing = style_obj.draw(ctx, bodies, poses, label_t)
        notes = list(dep.notes) + list(drawing.notes)

        by_colour = ctx.by_colour
        ligand_group, ligand_levels = self.backend.draw(dep, palette, m, conf, by_colour=by_colour, by_bond=ctx.by_bond)
        levels = set(drawing.levels) | ligand_levels
        legend = list(drawing.legend)
        if dep.units:
            assert isinstance(self.backend, Polymer)
            legend.append(LegendEntry(self.backend.legend_text(), "unit"))
            if dep.plane is not None:
                # Boxes on a plane are drawn to scale, and a reader will measure: the scale they are drawn at.
                legend.append(LegendEntry(f"{SCALE_BAR:g} \u00c5 between the centres of two boxes", "scale",
                                          length=SCALE_BAR * dep.plane.scale))
            if self.backend.interface is not None and not ctx.glows:
                # The key of a PAE tag is a tag as this figure writes them: plain, bold, or one of each.
                legend.extend(LegendEntry(text, "tag", marks=pae_key(drawing.tags))
                              for text in self.backend.interface.legend_extra)
        if not any(e.label == "Confidence:" for e in legend):
            legend.extend(confidence_legend(levels))          # only the ligand is below full strength
        under = None
        if by_colour:
            assert conf is not None
            from .styles.ligand_draw import plddt_underlay

            legend.extend(plddt_legend())
            # The ligand's atoms, on colour. (Where the bonds have it, that is for the atoms outside a chain's boxes.)
            under = plddt_underlay(dep, self.backend.bands(dep, conf)) if not ctx.by_bond or dep.units else None
        content = Group("diagram")
        for name in ("backdrop", "halos", "interactions"):
            if drawing.layers.get(name):
                content.add(Group(name, list(drawing.layers[name])))
            if name == "backdrop" and under is not None:
                content.add(under)
        content.add(ligand_group)
        for name in ("contacts", "residues"):
            if drawing.layers.get(name):
                content.add(Group(name, list(drawing.layers[name])))
        offset = given.legend_offset() if given is not None else (0.0, 0.0)
        # Atoms of residues drawn at a level that was asked for carry confidence of their own: the
        # figure says whether that is each atom's value or its residue's, which depends on the model.
        basis = None
        if conf is not None and (drawing.atoms_at_level or drawing.atoms_by_own or any(u.beside for u in dep.units)):
            basis = "atom" if conf.per_atom else "residue"
        lost = absence(groups, opts)
        foot = self._footnotes(opts, drawing.has_distances, levels, drawing.occupancy, drawing.predicted,
                               polymer=bool(dep.units), twice=twice, dep=dep, basis=basis, by_colour=by_colour,
                               glows=drawing.glows, lost=lost)
        scene = self._frame(content, m, opts, legend=legend,
                            footnotes=foot if opts.caption else (), unsaid=() if opts.caption else foot,
                            legend_offset=offset,
                            strip_data={"plddt-basis": basis} if basis is not None else None)
        if dep.plane is not None and self._interface is not None:
            # What was measured of the boxes belongs in what is read out of the figure, too.
            scene.description = " ".join(t for t in (scene.description, self._interface.described(dep)) if t)
        lig_box = laid_dep.bounds()
        quality = self._quality(scene, m, opts, layout, len({g.key for g in groups}), len(drawable),
                                lig_box[2] - lig_box[0], lig_box[3] - lig_box[1],
                                faithful=dep.plane is None or dep.plane.ok)
        dx, dy = content.translate or (0.0, 0.0)
        atoms = {a.name: (a.x + dx, a.y + dy) for a in dep.atoms if a.name}
        graph.origin = (dx, dy)
        graph.size = (scene.width, scene.height)
        graph.meta = {"ligand": self.name or "", "n_interactions": len([r for r in drawable if not r.absent])}
        if any(v is not None for v in self.detail.values()) and not (
                isinstance(self.backend, Polymer) and self.backend.interface is not None):
            # Said only where a level was asked for (an interface says it by participant, in its own part).
            graph.meta["detail"] = dict(self.detail)
        if basis is not None:
            graph.meta["plddt_basis"] = basis
        if isinstance(self.backend, Polymer):
            graph.meta["chain"] = describe_backend(self.backend)
            if self.backend.interface is not None:
                graph.meta["interface"] = self.backend.interface.meta_for(dep)
        report = self._burial_report()
        if report is not None:
            graph.meta["burial"] = report
        for item in scene.items:
            if isinstance(item, Group) and item.cls == "legend":
                # The legend is a node too: it has a place and can be moved, by an offset from
                # where the page puts it.
                box = items_bounds(item.items) or (0.0, 0.0, 0.0, 0.0)
                tx, ty = item.translate or (0.0, 0.0)
                graph.nodes.append(Node(id="legend", kind="legend", label="Legend", x=tx - dx, y=ty - dy,
                                        attrs={"width": round(box[2] - box[0], 3), "height": round(box[3] - box[1], 3),
                                               "offset": [offset[0], offset[1]]}))
        box = items_bounds(content.items) or (0.0, 0.0, 1.0, 1.0)
        if opts.pae_colors and not drawing.glows:
            # An option that was asked for and changes nothing is said.
            notes.append("pae_colors was asked for, and no contact of this figure has a predicted aligned error")
        return _Composed(scene, layout, quality, notes, atoms, graph, basis, drawing.glows,
                         parts=(style_obj, ctx, bodies, poses, fld, params, content, box))

    def _set_beside(self, drawable: Sequence[ResolvedInteraction]) -> None:
        """Tell a chain which atoms of which residue stand beside its box, and on which side.

        Decided by the contacts that are drawn and by the chain's level of detail: a residue with a
        directional contact is drawn with the part that makes it and what the level adds
        (:func:`bindview.polymer.shown_atoms`); two such residues that share a partner stand on one
        side (:func:`bindview.polymer.piece_sides`). A column only: boxes on a plane stay boxes.
        """
        if not isinstance(self.backend, Polymer):
            return
        pep = self.backend.peptide
        shown: dict[int, tuple[int, ...]] = {}
        sides: dict[int, int] = {}
        if pep.detail not in (None, "box") and self.backend.arrangement is None:
            involved: dict[int, set[int]] = {}
            directional = [r for r in drawable if r.ligand is not None and r.type is not InteractionType.HYDROPHOBIC]
            for r in directional:
                assert r.ligand is not None
                for i in r.ligand.atoms:
                    res = pep.residue_of(i)
                    if res is not None:
                        involved.setdefault(res.position, set()).add(i)
            shown = shown_atoms(pep, involved)
            partners: dict[int, set[tuple]] = {}
            for r in directional:
                assert r.ligand is not None
                for i in r.ligand.atoms:
                    res = pep.residue_of(i)
                    if res is not None and i in shown.get(res.position, ()):
                        partners.setdefault(res.position, set()).add(
                            (r.record.chain, r.record.residue_number, r.record.insertion_code))
            sides = piece_sides([res.position for res in pep.residues if res.position in shown], partners)
        self.backend.shown, self.backend.sides = shown, sides

    def _confidence_report(self, conf: Confidence, basis: "str | None" = None) -> dict[str, Any]:
        """Figure-level confidence as plain data, plus how the contacts fall into bands, each by the
        number it is drawn at (the one its record carries, else the lower of its two ends); and, where
        atoms of residues were drawn at a level that was asked for, what each was drawn at (``basis``)."""
        report = conf.summary()
        if basis is not None:
            report["drawn_atoms_at"] = ("each atom's own pLDDT" if basis == "atom" else
                                        "their residue's pLDDT: the model gives one value for a residue")
        bands: dict[str, int] = {}
        values = []
        for r in self.interactions:
            v: "float | None"
            if r.record.confidence is not None:            # as the drawing has it (styles.base.contact_plddt)
                v = r.record.confidence
            elif r.ligand is None:
                continue
            else:
                v = conf.contact(r.residue if r.residue is not None else r.record.residue_id,
                                 [self.ligand.names[i] for i in r.ligand.atoms])
            if v is not None:
                values.append(v)
                band = look_of(v).band or ""
                bands[band] = bands.get(band, 0) + 1
        report["contacts_by_band"] = {k: bands[k] for k in ("very_high", "confident", "low", "very_low") if k in bands}
        report["contact_plddt_range"] = [round(min(values), 1), round(max(values), 1)] if values else None
        return report

    def _footnotes(self, opts: RenderOptions, has_distances: bool, levels: "set[int] | None" = None,
                   occupancy: bool = False, predicted: bool = False, polymer: bool = False,
                   twice: Sequence[str] = (), dep: "Depiction | None" = None,
                   basis: "str | None" = None, by_colour: bool = False, glows: bool = False,
                   lost: "Absence | None" = None) -> list[str]:
        out = []
        if lost is not None:
            # First: it is what the figure is for.
            out.append(lost.sentence)
        if has_distances:
            out.append("Distances in \u00c5.")
        if glows:
            ends = ("the two residues" if isinstance(self.backend, Polymer) else "the residue and the ligand")
            out.append(f"A contact stands on a glow in the colour of its PAE (predicted aligned error): how far off, in "
                       f"\u00e5ngstr\u00f6m, the model expects {ends} to be placed against each other. It is not pLDDT.")
        if by_colour:
            # What is coloured depends on what the figure is of: an interface has no ligand.
            what = ("a residue has the colour of its own value, on either side"
                    if isinstance(self.backend, Polymer) and self.backend.interface is not None else
                    "a residue has the colour of its own value, and so has every residue of the ligand"
                    if isinstance(self.backend, Polymer) else
                    "a residue has the colour of its own value, and so has every atom of the ligand")
            out.append(f"Colour follows the model's pLDDT, in AlphaFold's four bands: {what}. No line is drawn "
                       "weaker for it: a contact is as sure as the less sure of its two ends.")
        if levels and any(lv > 0 for lv in levels):
            # An interface has no ligand: there the two ends are the neighbour residue and, on the
            # column's side, the atoms that make the contact.
            ends = ("two ends' values" if isinstance(self.backend, Polymer) and self.backend.interface is not None
                    else "ligand-atom and residue values")
            out.append("Stroke weight and opacity follow the model's pLDDT: a contact is drawn at the lower of its "
                       f"{ends}, a residue at its own. {MARK} marks pLDDT below 50.")
        if basis == "atom":
            # Said whether or not anything is drawn weaker: what a drawn atom stands at is a claim either way.
            out.append("The atoms that are drawn of a residue are each at their own pLDDT: the model gives one "
                       "for every atom.")
        elif basis == "residue":
            out.append("The atoms that are drawn of a residue are all at their residue's pLDDT: this model gives "
                       "one value for a residue, not one for each of its atoms.")
        if occupancy and opts.occupancy_style == "steps":
            out.append("Stroke weight and opacity follow occupancy, the share of "
                       f"{self.ensemble or 'the ensemble'} in which a contact is made, in three steps (90 % and "
                       "more, 50 to 90 %, less than 50 %); the share is written on "
                       + ("the line, or under the name of a residue that has none." if opts.occupancy_labels == "all" else
                          "the line of a contact that is not made in all."))
        elif occupancy:
            out.append("Stroke weight and opacity follow occupancy: the share of "
                       f"{self.ensemble or 'the ensemble'} in which a contact is made.")
        # (A figure without a distance on it has nothing to say of whose they are.)
        out.extend(r if has_distances else r.replace(_WHOSE_DISTANCES, "") for r in self.remarks)
        if self.elsewhere and self.elsewhere_said[0]:
            out.append(self.elsewhere_said[0])
        if predicted:
            out.append("Predicted water sites are geometric guesses for a model that has no solvent, "
                       f"not observations. {VALIDATION}")
        if opts.notes:
            lig = self.ligand
            if lig.stereo_mismatches and isinstance(self.backend, Polymer):
                n = len(lig.stereo_mismatches)
                out.append(f"D and L are those of the reference SMILES; in the model {n} "
                           f"stereocentre{'s' if n != 1 else ''} differ{'' if n != 1 else 's'} from it.")
            elif lig.stereo_mismatches:
                n = len(lig.stereo_mismatches)
                out.append(f"Stereo is drawn as modelled; {n} stereocentre{'s' if n != 1 else ''} "
                           f"differ{'' if n != 1 else 's'} from the reference SMILES.")
            if lig.artefact_bonds and not self.shares:   # of one model, and nothing of its geometry is drawn
                n = len(lig.artefact_bonds)
                out.append(f"The model has {n} unbonded ligand atom pair{'s' if n != 1 else ''} at bonding distance.")
            if not isinstance(self.backend, Polymer):          # a chain's gaps are drawn, and said, as gaps
                out.extend(apart_note(lig, *part) for part in lig.apart())
            if lig.bond_source == "geometry":
                out.append("Bond orders inferred from geometry (no SMILES).")
            elif lig.bond_source == "connectivity":
                out.append("Bond orders unknown (no SMILES): only connectivity is drawn.")
            n_un = len(self.unresolved)
            if n_un:
                out.append(f"{n_un} reported interaction{'s' if n_un != 1 else ''} could not be attached "
                           f"and {'are' if n_un != 1 else 'is'} not shown.")
        if polymer:
            # (Not a note that can be switched off: it says what a number means. A figure that draws the PAE as a
            # glow has no number on a line.)
            tags = self._interface.tag_notes if glows and self._interface is not None else ()
            out.extend(t for t in self.backend.footnotes(dep) if t not in tags)
        if twice:
            # Nor is this one: without it two drawings of one residue read as two residues.
            out.append("Drawn more than once, each time beside the stretch of the "
                       + ("chain" if getattr(self.backend, "interface", None) is not None else "ligand")
                       + " it touches there: "
                       + ", ".join(twice) + ".")
        return out

    def _frame(self, content: Group, m: Metrics, opts: RenderOptions, *, legend: Sequence[LegendEntry] = (),
               footnotes: Sequence[str] = (),
               extra_blocks: Sequence[tuple[Group, float, float]] = (),
               legend_offset: tuple[float, float] = (0.0, 0.0),
               strip_data: "Mapping[str, str] | None" = None, unsaid: Sequence[str] = ()) -> Scene:
        """Put title, content, legend and footnotes on a page sized to fit them.

        The words around the drawing (the line under the title, a warning above it, the footnotes, and
        ``unsaid``: notes that are not drawn) are kept on the scene as ``notes``; ``bare`` draws none of
        them."""
        said: list[str] = []
        bare = opts.bare
        box = items_bounds([content]) or (0.0, 0.0, 1.0, 1.0)
        cw, ch = box[2] - box[0], box[3] - box[1]
        pad = m.margin
        inner = max(cw, 150.0)
        conf = self._shown_confidence(opts)

        top: list[Item] = []
        y = pad
        title_w = 0.0
        banner: list[str] = []
        if conf is not None and conf.marked:
            # The caller says this model is not to be trusted: say so in words, at the
            # top, and frame the whole figure. Faint drawing alone would not be a mark.
            verdict = f" ({conf.model.flag})" if conf.model is not None and conf.model.flag else ""
            banner = wrap_text(f"Model flagged as not trustworthy{verdict} \u2014 shown for inspection, "
                               "not as a result.", m.title_font, max(inner, 330.0), bold=True)
            title_w = max(text_width(line, m.title_font, bold=True) for line in banner)
            y += len(banner) * m.title_font * 1.45 + pad * 0.9
        if opts.title:
            top.append(Text.plain(pad, y + CAP_HEIGHT * m.title_font, opts.title, m.title_font, "text",
                                  anchor="start", bold=True))
            title_w = text_width(opts.title, m.title_font, bold=True)
            y += m.title_font * 1.5
        if opts.subtitle:
            top.append(Text.plain(pad, y + CAP_HEIGHT * m.caption_font, opts.subtitle, m.caption_font,
                                  "text_muted", anchor="start"))
            title_w = max(title_w, text_width(opts.subtitle, m.caption_font))
            y += m.caption_font * 1.6
        shown = self.burial() if opts.burial and self._interface is None else None
        # An interface has neither pocket nor ligand: the column, and the residues around it.
        spine = isinstance(self.backend, Polymer) and self.backend.interface is not None
        more = (tuple(self.backend.interface.strip)
                if isinstance(self.backend, Polymer) and self.backend.interface is not None else ())
        strip = (_confidence_strip(conf, m.caption_font + 1.0, shown,
                                   names=(("residues around", "boxes" if self._interface.shape == "plane" else "column")
                                          if spine else ("pocket", "ligand")), more=more,
                                   more_data=self._interface.strip_data if self._interface is not None else None)
                 if (conf is not None or shown is not None or more) else None)
        if strip is not None:
            said.append(" ".join("".join(s.text for s in strip[0]).split()))
        if strip is not None and not bare:
            top.append(Group("confidence", [Text(pad, y + CAP_HEIGHT * strip[1], strip[0], strip[1], "text_muted",
                                                 anchor="start")],
                             data={k: str(v) for k, v in {**strip[3], **(strip_data or {})}.items()}))
            title_w = max(title_w, strip[2])
            y += strip[1] * 1.7
        warn = shown is not None and self._burial_warning() is not None
        if warn:
            said.append(_BURIAL_LINE)
        if warn and shown is not None and not bare:
            # Below the calibrated cut the figure says what the number may mean, in words.
            lines = wrap_text(_BURIAL_LINE, m.caption_font, max(inner, title_w, 330.0))
            items, height = text_block(lines, pad, y - 2.0, m.caption_font, "warning")
            top.append(Group("burial-warning", items, data={"buried-fraction": f"{shown.buried_fraction:.3f}"}))
            title_w = max([title_w] + [text_width(line, m.caption_font) for line in lines])
            y += height
        for warned in (self._interface.warnings if self._interface is not None else ()):
            said.append(warned)
            if bare:
                continue
            # The same place and the same colour for what an interface has to be told about as a whole:
            # that none of its contacts is placed by the model. The tags on the lines say it one by one.
            lines = wrap_text(warned, m.caption_font, max(inner, title_w, 330.0))
            items, height = text_block(lines, pad, y - 2.0, m.caption_font, "warning")
            top.append(Group("pae-warning", items))
            title_w = max([title_w] + [text_width(line, m.caption_font) for line in lines])
            y += height
        if top:
            y += 4.0
        inner = max(inner, title_w)
        content_top = y
        y += ch

        blocks: list[tuple[Group, float, float]] = list(extra_blocks)
        if opts.legend and legend:
            lg, lw, lh = legend_group(legend, m, max(inner, 300.0))
            if lh > 0:
                blocks.append((lg, lw, lh))
        said.extend(footnotes)
        said.extend(unsaid)
        foot_lines: list[str] = []
        for note in (() if bare else footnotes):
            foot_lines.extend(wrap_text(note, m.caption_font, max(inner, 300.0)))
        inner = max([inner] + [bw for _, bw, _ in blocks] +
                    [text_width(line, m.caption_font) for line in foot_lines])

        bottom: list[Item] = []
        for grp, _bw, bh in blocks:
            y += 10.0
            if grp.cls == "legend":
                grp.translate = (pad + legend_offset[0], y + legend_offset[1])
            else:
                grp.translate = (pad, y)
            bottom.append(grp)
            y += bh
        if foot_lines:
            y += 6.0
            items, fh = text_block(foot_lines, pad, y, m.caption_font, "text_muted")
            bottom.append(Group("footnotes", items))
            y += fh
        width = inner + 2.0 * pad
        height = y + pad
        content.translate = (pad + (inner - cw) / 2.0 - box[0], content_top - box[1])
        if opts.break_lines:
            keep_lines_off_words(content)
        scene = Scene(width, height, title=opts.title or self.name or "Ligand interaction diagram", notes=said)
        scene.items = top + [content] + bottom
        if banner:
            bh = len(banner) * m.title_font * 1.45 + pad * 0.5
            items, _ = text_block(banner, pad, pad * 0.35, m.title_font, "background", bold=True)
            flag = Group("model-flag", [Rect(0.0, 0.0, width, bh, Paint(fill="warning"))] + items,
                         data={"trusted": "false"})
            frame = Rect(1.5, 1.5, width - 3.0, height - 3.0, Paint(stroke="warning", width=3.0, join="miter"))
            scene.items = [flag] + scene.items + [Group("model-flag-frame", [frame])]
        interface = self.backend.interface if isinstance(self.backend, Polymer) else None
        if conf is not None or (interface is not None and interface.strip):
            scene.description = _confidence_sentence(conf, interface)
        return scene

    def _quality(self, scene: Scene, m: Metrics, opts: RenderOptions, layout: LayoutResult,
                 n_residues: int, n_interactions: int, ligand_width: float = 0.0, ligand_height: float = 0.0,
                 faithful: bool = True) -> Quality:
        natural = _smallest_text_pt(m)
        fit = min(opts.figure_width_mm * _PX_PER_MM / scene.width,
                  opts.figure_height_mm * _PX_PER_MM / scene.height, 1.0)
        font_pt = natural * fit
        required = (math.inf if beyond_any_page(natural, opts) else
                    scene.width * (opts.min_font_pt / natural) / _PX_PER_MM)
        return Quality(
            width=scene.width, height=scene.height, bond_length=m.bond_length, n_residues=n_residues,
            n_interactions=n_interactions, overlaps=layout.overlaps, unplaced=len(layout.unplaced),
            line_crossings=layout.line_crossings, bond_crossings=layout.bond_crossings,
            max_stretch=layout.max_stretch, min_font_pt=font_pt, required_width_mm=required,
            readable=font_pt >= opts.min_font_pt - 1e-9, ligand_width=ligand_width, link_hits=layout.link_hits,
            ligand_height=ligand_height, faithful=faithful)

    def _ligand_only(self, palette: Palette, m: Metrics, opts: RenderOptions) -> Group:
        dep = self.backend.depict(m, opts.coords)
        return Group("diagram", [self.backend.draw(dep, palette, m)[0]])

    def _empty_scene(self, palette: Palette, m: Metrics, opts: RenderOptions) -> Scene:
        content = self._ligand_only(palette, m, opts)
        headline = "No interactions detected" if self.source == "detected" else "No interactions in the supplied set"
        near = self.proximal_residues()
        if self.source == "detected":
            body = "Detection ran on this complex and no contact met the criteria for a typed interaction."
        else:
            body = "The interaction list supplied for this ligand is empty: it was analysed and nothing was found."
        if near:
            body += (f" {len(near)} residue{'s' if len(near) != 1 else ''} "
                     f"{'lie' if len(near) != 1 else 'lies'} within 4.5 Å of the ligand "
                     f"({', '.join(near)}).")
        elif self.structure is not None:
            body += " No residue lies within 4.5 Å of the ligand."
        if self.remarks:
            body += " " + " ".join(self.remarks + [s for s in self.elsewhere_said[1:] if s])
        wrap = 330.0
        head_items, hh = text_block([headline], 0.0, 0.0, m.title_font, "text", bold=True)
        body_items, bh = text_block(wrap_text(body, m.caption_font, wrap), 0.0, hh + 3.0, m.caption_font, "text_muted")
        block = Group("empty-state", head_items + body_items)
        block.translate = None
        spacer = Group("spacer")
        return self._frame(content, m, opts, extra_blocks=[(spacer, 0.0, 2.0), (block, wrap, hh + 3.0 + bh)])

    def _summary_scene(self, m: Metrics, opts: RenderOptions, drawable: Sequence[ResolvedInteraction], q: Quality,
                       reason: str) -> Scene:
        # A table of this structure's contacts: what only other models make is not one of them.
        groups = group_residues([r for r in drawable if not r.absent], opts.show_chain)
        mark_groups(groups, opts)
        max_w = opts.figure_width_mm * _PX_PER_MM - 2.0 * m.margin
        conf = self._shown_confidence(opts)
        plddt = {}
        if conf is not None:
            for g in groups:
                first = g.interactions[0]
                value = conf.residue(first.residue if first.residue is not None else g.key)
                if value is not None:
                    plddt[g.key] = value
        table, tw, _ = summary_table(groups, m, max_w, plddt or None)
        headline = "Residue-level summary"
        why = (f"{self._what().capitalize()} of this complex is not readable at {opts.figure_width_mm:g} mm "
               f"({self._size()}, {q.n_residues} residues, {q.n_interactions} interactions): "
               f"{reason}. Contacts are listed per residue instead.")
        head_items, hh = text_block([headline], 0.0, 0.0, m.title_font, "text", bold=True)
        why_items, wh = text_block(wrap_text(why, m.caption_font, max(tw, 360.0)), 0.0, hh + 3.0,
                                   m.caption_font, "warning")
        table.translate = (0.0, hh + wh + 14.0)
        content = Group("diagram", head_items + why_items + [table])
        foot = ["Dots mark interaction types; ×n gives the count. 'closest' is the shortest reported distance."]
        if plddt:
            foot.append(f"pLDDT is the residue's own (mean over its atoms); {MARK} marks values below 50.")
        if any(g.beyond for g in groups):
            foot.append("† outside the supplied contact set")
        lost = absence(groups, opts)
        if lost is not None:
            how = "In italics" if opts.absent_style == "faint" else "Marked"
            foot.append((f"{how}: absent from {from_whom(opts.absent_label)}. " if lost.absent else "") + lost.sentence)
        foot.extend(self.remarks + [s for s in self.elsewhere_said[1:] if s])
        if isinstance(self.backend, Polymer) and self.backend.interface is not None:
            # No line is left to carry a predicted aligned error: what it is, and the warning, stay.
            foot.extend(self.backend.interface.summary_notes)
        return self._frame(content, m, RenderOptions(**{**opts.__dict__, "legend": False}), footnotes=foot)


_SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
_SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def _give_pae(resolved: "Sequence[ResolvedInteraction]", pae: "Pae | PaeDraws", lig: Ligand, st: Structure,
              key: ResidueKey) -> None:
    """Give every contact that has none of its own the predicted aligned error between its two ends: the residue,
    and the atoms of the ligand that make it (or, of a ligand that is a chain, the residue of it they are of)."""
    heavy = [a.name for a in st.residue_atoms(key) if a.element not in ("H", "D")]
    unit = {}                                                    # an atom of a chain ligand: the residue it is of
    for r in lig.given:
        digits = r.number.rstrip("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
        for name in r.atoms:
            unit[name] = ResidueKey(lig.chain or key.chain, int(digits), r.number[len(digits):])
    found = False
    for r in resolved:
        if r.ligand is None or r.residue is None or r.record.pae is not None:
            found = found or r.record.pae is not None
            continue
        names = [lig.names[i] for i in r.ligand.atoms]
        values: list[Any] = []
        if unit:
            for end in dict.fromkeys(unit[n] for n in names if n in unit):
                values.append((pae.between(r.residue, end), pae.each(r.residue, end) if isinstance(pae, PaeDraws) else ()))
        else:
            atoms = ([heavy.index(n) for n in names if n in heavy], len(heavy))
            values.append((pae.between(r.residue, key, atoms),
                           pae.each(r.residue, key, atoms) if isinstance(pae, PaeDraws) else ()))
        known = [v for v in values if v[0] is not None]
        if known:
            value, draws = min(known, key=lambda v: v[0])
            found = True
            r.record = r.record.with_(pae=value, pae_draws=tuple(v for v in draws if v is not None)
                                      if all(v is not None for v in draws) else ())
    if resolved and not found:
        raise RenderError(f"{pae.source or 'the PAE matrix'} has no value for any contact of this ligand: its rows are of "
                          f"chains {', '.join(repr(n) for n in sorted(set(pae.chains)))}, residue numbers "
                          f"{min(pae.numbers)} to {max(pae.numbers)}")


def apart_note(lig: Ligand, part: "tuple[int, ...]", metal: "tuple[int, float] | None",
               nearest: "tuple[int, float]") -> str:
    """The note on a part of the ligand that the model binds to nothing else of it (:meth:`Ligand.apart`).

    It is drawn beside the rest, as it is in the model; without a word a reader takes it for a part the
    drawing lost, or one that sits where chemistry says (a hydroxide on its cobalt)."""
    if len(part) == 1:
        a = lig.mol.GetAtomWithIdx(part[0])
        h, q = a.GetTotalNumHs(), a.GetFormalCharge()
        formula = a.GetSymbol() + ("H" + (str(h) if h > 1 else "") if h else "").translate(_SUB)
        if q:
            formula += (str(abs(q)) if abs(q) > 1 else "").translate(_SUP) + ("⁺" if q > 0 else "⁻")
        what = f"{lig.names[part[0]]} ({formula})" if formula != a.GetSymbol() else lig.names[part[0]]
    else:
        names = [lig.names[i] for i in part]
        what = ("the part " + ", ".join(names[:-1]) + " and " + names[-1]) if len(names) <= 4 \
            else f"a part of {len(names)} atoms ({', '.join(names[:3])}, ...)"
    where = [f"{nearest[1]:.1f} Å from {lig.names[nearest[0]]}, the nearest atom"]
    if metal is not None and metal[0] != nearest[0]:
        where.insert(0, f"{metal[1]:.1f} Å from the metal {lig.names[metal[0]]}")
    elif metal is not None:
        where = [f"{nearest[1]:.1f} Å from the metal {lig.names[metal[0]]}, the nearest atom"]
    return (f"In the model {what} is bound to no other atom of the ligand, and is drawn apart: "
            + " and ".join(where) + ".")


def _details(detail: "str | Mapping[str, str | None] | None") -> dict[str, "str | None"]:
    """``detail`` as it was given, for the two sides of a diagram: one level for both, or one for each
    under ``"ligand"`` and ``"protein"``."""
    if detail is None or isinstance(detail, str):
        level = check_detail(detail)
        return {"ligand": level, "protein": level}
    unknown = [k for k in detail if k not in ("ligand", "protein")]
    if unknown:
        raise ValueError(f"detail is a level, or one for each of 'ligand' and 'protein'; not for {unknown[0]!r}")
    return {side: check_detail(detail.get(side), f"detail of the {side}") for side in ("ligand", "protein")}


def _confidence_for(st: Structure, key: ResidueKey, plddt: Any, model_confidence: Any,
                    trusted: bool | None, flag: str | None) -> Confidence | None:
    """Decide whether, and from where, confidence may be read. The guard lives here.

    The B-factor column is read as pLDDT only if the file declares it
    (``plddt="auto"``) or the caller asserts it (``"bfactor"``); never because
    the numbers happen to look like one.
    """
    conf: Confidence | None
    if isinstance(plddt, Confidence):
        conf = plddt
    elif plddt is None or plddt is False or (isinstance(plddt, str) and plddt.lower() in ("off", "none")):
        conf = None
    elif plddt is True or (isinstance(plddt, str) and plddt.lower() == "auto"):
        conf = Confidence.from_structure(st, key, scale=st.plddt_scale or 1.0, origin="declared") \
            if st.declares_plddt else None
    elif isinstance(plddt, str) and plddt.lower() in ("bfactor", "bfactor01"):
        # The caller's assertion, and with it the scale: 0-100, or 0-1 ("bfactor01"). A column that
        # cannot be on the scale asserted is refused there (Confidence.from_structure), not rescaled.
        conf = Confidence.from_structure(st, key, scale=100.0 if plddt.lower() == "bfactor01" else 1.0,
                                         origin="asserted")
        if conf is None:
            raise ValueError(f"plddt={plddt!r} was asked for, but {st.source or 'the structure'} has no B-factor "
                             f"column")
    else:
        raise ValueError(f"plddt must be 'auto', 'bfactor', 'bfactor01', None or a Confidence, not {plddt!r}")

    model: ModelConfidence | None
    if isinstance(model_confidence, str) and model_confidence.lower() == "auto":
        # A sidecar is only looked for next to a file that is a predicted model.
        beside = ModelConfidence.beside(st.source) if (st.source and conf is not None) else None
        model = ModelConfidence.from_json(beside) if beside is not None else (conf.model if conf else None)
    else:
        model = ModelConfidence.coerce(model_confidence)
    if trusted is not None or flag is not None:
        base = model or ModelConfidence()
        model = ModelConfidence(base.iptm, base.ptm, base.ranking_score,
                                trusted=base.trusted if trusted is None else bool(trusted),
                                flag=flag if flag is not None else base.flag)
    if model is None:
        return conf
    if conf is None:
        # Model-level evidence without per-atom values: still worth displaying.
        return Confidence({}, {}, origin="supplied", model=model)
    return conf.with_model(model)


#: What the figure says, when asked to show burial, about a ligand below the calibrated cut.
_BURIAL_LINE = ("Most of the ligand's surface is open to solvent in this model: the site may be incomplete here "
                "(one chain of an oligomer, a missing cofactor), or the ligand may not sit in a pocket.")


def _confidence_strip(conf: "Confidence | None", size: float, burial: "Burial | None" = None,
                      names: tuple[str, str] = ("pocket", "ligand"), more: "Sequence[tuple[str, str]]" = (),
                      more_data: "Mapping[str, str] | None" = None
                      ) -> "tuple[tuple[Span, ...], float, float, dict[str, Any]] | None":
    """The figure-level line of provenance numbers: ipTM, mean pocket pLDDT, mean ligand pLDDT,
    and -- when ``burial`` is given -- the ligand's buried fraction beside them.

    Returns ``(spans, font size, width, data attributes)``, or ``None`` when
    there is nothing to say.
    """
    parts: list[tuple[str, str]] = []
    data: dict[str, Any] = {}
    m = conf.model if conf is not None else None
    if m is not None and m.iptm is not None:
        parts.append(("ipTM", f"{m.iptm:.2f}"))
        data["iptm"] = f"{m.iptm:.2f}"
    if conf is not None and conf.pocket_plddt is not None:
        parts.append((f"{names[0]} pLDDT", f"{conf.pocket_plddt:.0f}"))
        data["pocket-plddt"] = f"{conf.pocket_plddt:.1f}"
    if conf is not None and conf.ligand_plddt is not None:
        parts.append((f"{names[1]} pLDDT", f"{conf.ligand_plddt:.0f}"))
        data["ligand-plddt"] = f"{conf.ligand_plddt:.1f}"
    for label, value in more:
        # Not pLDDT: a number of its own, named (the predicted aligned error of an interface's contacts).
        parts.append((label, value))
    data.update(more_data or {})
    if m is not None and m.flag and m.trusted:
        parts.append(("", m.flag))
    head = "Model confidence  " if parts else ""
    if burial is not None and burial.sasa_free > 0.0:
        # Not a confidence, but the same kind of number: what stands behind the picture.
        parts.append(("ligand buried" if parts else "Ligand buried", f"{100.0 * burial.buried_fraction:.0f} %"))
        data["buried-fraction"] = f"{burial.buried_fraction:.3f}"
    if not parts:
        return None
    spans: list[Span] = [Span(head)] if head else []
    width = text_width(head, size) if head else 0.0
    for k, (label, value) in enumerate(parts):
        lead = ("  \u00b7  " if k else "") + (label + " " if label else "")
        spans.append(Span(lead))
        spans.append(Span(value, fill="text", bold=True))
        width += text_width(lead, size) + text_width(value, size, bold=True)
    return tuple(spans), size, width, data


def _confidence_sentence(conf: "Confidence | None", interface: Any = None) -> str:
    """The same evidence as one plain sentence, for the SVG description.

    ``interface`` is what an interface figure says about itself: it has
    neither pocket nor ligand, and its predicted aligned error and what that
    means for the figure belong in the sentence as much as under the title.
    """
    bits = []
    m = conf.model if conf is not None else None
    whole = "the boxes" if getattr(interface, "shape", "column") == "plane" else "the column"
    if m is not None and m.iptm is not None:
        bits.append(f"ipTM {m.iptm:.2f}")
    if conf is not None and conf.pocket_plddt is not None:
        bits.append(f"mean pocket pLDDT {conf.pocket_plddt:.0f}" if interface is None
                    else f"mean pLDDT {conf.pocket_plddt:.0f} of the residues around {whole}")
    if conf is not None and conf.ligand_plddt is not None:
        bits.append(f"mean ligand pLDDT {conf.ligand_plddt:.0f}" if interface is None
                    else f"mean pLDDT {conf.ligand_plddt:.0f} of {whole}")
    if interface is not None:
        bits.extend(f"{label} {value}" for label, value in interface.strip)
    text = "Predicted model: " + ", ".join(bits) + "." if bits else ""
    if conf is not None and conf.marked:
        verdict = f" ({m.flag})" if m is not None and m.flag else ""
        text = (text + " " if text else "") + f"Flagged as not trustworthy{verdict}."
    if interface is not None:
        for said in interface.warnings:
            text = (text + " " if text else "") + said[:1].upper() + said[1:]
    return text


def _check_sizes(opts: RenderOptions) -> None:
    """Refuse a page, a size of text or a scale that is none, before anything is drawn for it: a
    :class:`~bindview.errors.RenderError`, which the command line and the page for arranging report in words."""
    def number(value: Any) -> bool:
        return isinstance(value, numbers.Real) and not isinstance(value, bool) and math.isfinite(value)

    for name, side in (("figure_width_mm", "width"), ("figure_height_mm", "height")):
        value = getattr(opts, name)
        if not number(value) or value <= 0:
            raise RenderError(f"{name} is the {side} of the page in millimetres, a positive number, not {value!r}")
    if not number(opts.min_font_pt) or opts.min_font_pt < 0:
        raise RenderError("min_font_pt is the smallest text that counts as legible, in points: a number that is not "
                          f"negative, not {opts.min_font_pt!r}")
    scale = opts.bond_length
    if scale is not None and (not number(scale) or not 0 < scale <= _BOND_LENGTH_MAX):
        raise RenderError(f"bond_length is the length of a bond in page units, a positive number of at most "
                          f"{_BOND_LENGTH_MAX:g} (or None: the largest that fits the page), not {scale!r}")


def _kind(key: tuple) -> str:
    """Which kind of thing a diagram keeps under ``key`` (:data:`_KEPT_AT_MOST`)."""
    return key[0] if key[0] in ("field", "layout") else "figure"


def _crowded(n: int) -> int:
    """With this many of ``n`` residues without a place after the first round of exchanges the first layout of a chain is so
    crowded that ``effort="quick"`` looks at the roomiest (docs/reference.md has the numbers) ..."""
    return max(4, math.ceil(n / 10.0))


def _hopeless(n: int) -> int:
    """... and with this many that one is."""
    return max(3, math.ceil(n / 15.0))


def _set_fans(bodies: Sequence[Any]) -> None:
    """Tell links that share a ligand site how far apart they should like to be.

    ``n`` links into one atom, ring or group ideally split the circle evenly;
    asking for 80 % of that (capped at a right angle) leaves room for the site's
    own bonds while still making the partners surround it.
    """
    count: dict[tuple[float, float], int] = {}
    for body in bodies:
        for port in body.ports:
            if port.drawn:
                key = (round(port.target[0], 2), round(port.target[1], 2))
                count[key] = count.get(key, 0) + 1
    for body in bodies:
        changed = False
        for port in body.ports:
            n = count.get((round(port.target[0], 2), round(port.target[1], 2)), 0)
            if port.drawn and n >= 2:
                port.fan = min(math.pi / 2.0, 0.8 * 2.0 * math.pi / n)
                changed = True
        if changed:
            body.__post_init__()


def _smallest_text_pt(m: Metrics) -> float:
    """The smallest text of a diagram at full size, in points: what a page can at most leave of it."""
    return _PT_PER_PX * min(m.distance_font, m.protein_atom_font, m.sub_font, m.charge_font, m.legend_font,
                            m.caption_font)


def _scaled(base: Metrics, bond_length: float) -> Metrics:
    """Metrics for a different bond length: geometry scales, text does not."""
    k = bond_length / base.bond_length
    if abs(k - 1.0) < 1e-9:
        return base
    return base.with_(bond_length=bond_length, bond_width=max(1.2, base.bond_width * (0.55 + 0.45 * k)),
                      double_offset=max(3.0, base.double_offset * k), wedge_width=max(4.4, base.wedge_width * k),
                      hash_spacing=max(2.2, base.hash_spacing * k))


def _better(a: Quality, b: Quality) -> bool:
    """Is layout ``a`` preferable to ``b`` when neither is fully acceptable?"""
    if a.valid != b.valid:
        return a.valid
    return a.min_font_pt > b.min_font_pt + 1e-9


def _dedupe(items: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out
