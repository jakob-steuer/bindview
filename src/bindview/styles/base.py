"""Pieces every style needs: grouping interactions by residue, finding where a
link lands on the ligand, drawing link lines, eyelashes, spokes and the legend.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np

from ..confidence import (LOOKS, MARK, OCCUPANCY_SAMPLES, OCCUPANCY_STEPS, PLAIN, SHARE_WRITTEN_BELOW, Confidence,
                          Look, look_of, occupancy_factors)
from ..depict import Depiction
from ..pae import PAE_BANDS, pae_band, pae_tag
from ..geometry import (Point, add, angle_diff, dist, lerp, mul, perp, polygon_area, resample_closed, sub, unit)
from ..layout.body import SPRINGS, Port
from ..layout.field import LigandField
from ..interactions import InteractionType
from ..interactive import NUMBER
from ..ligand import Ligand
from ..residues import ATOM_LABELS, can_donate, charged_groups, format_residue_label, is_backbone_atom, is_nucleotide, residue_class
from ..resolve import Anchor, ResolvedInteraction
from ..scene import Circle, Group, Item, Line, Paint, Polygon, Polyline, Rect, Text
from ..textmetrics import CAP_HEIGHT, text_width
from ..theme import Metrics, Palette
from . import absence as other_protein
from .rings import ring_sample

if TYPE_CHECKING:
    from ..consensus import ResidueShare

__all__ = [
    "link_legend",
    "ResidueGroup", "group_residues", "Target", "ligand_target", "LinkStyle", "LINK_STYLES",
    "link_items", "eyelash_items", "eyelash_mark", "spoke_items", "legend_group", "Context", "RenderOptions",
    "hbond_direction", "halo_items", "LABELLED_BY_DEFAULT", "interaction_title", "spoke_reach", "halo_radius",
    "group_skeleton", "PortInfo", "port_infos", "make_ports", "label_size", "LegendEntry", "StyleDrawing",
    "VISIBLE_LENGTH", "interaction_data", "contact_plddt", "residue_plddt", "confidence_legend", "charge_legend",
    "marked_label", "residue_occupancy", "occupancy_legend", "absent_legend", "ABSENT_OPACITY", "share_label",
    "PREDICTED_LABEL", "predicted_style",
    "predicted_items", "predicted_legend", "CHAIN_CONTACT", "CHAIN_CONTACT_LENGTH", "set_lanes", "tag_slack",
    "split_along_chain", "chain_index", "SPLIT_GAP", "beyond_any_page",
    "page_that_would_do", "too_small_on_page",
    "RESIDUE_FORMS", "LININGS", "LINES", "LineSet", "Settings", "STYLE_SETTINGS", "residue_names", "Drawn",
    "finish_drawing", "LiningClaim", "lining_claims", "lining_items", "contact_face", "EXPOSED_FROM",
]

#: What a predicted solvent site is called on the figure, on the line where a water would be.
PREDICTED_LABEL = "pred. H\u2082O"

T = InteractionType

#: Interaction types whose distance is printed on the line unless asked otherwise.
#: Atom-to-atom distances are; centre-to-centre ones (rings, charge centres) are
#: carried in the SVG as data and tooltips but would only clutter the drawing.
LABELLED_BY_DEFAULT = frozenset({T.HBOND, T.HALOGEN, T.METAL, T.WATER_BRIDGE})


#: How a residue beside the ligand can be drawn (``RenderOptions.residues``).
RESIDUE_FORMS = ("atoms", "bubbles", "names")
#: What the lining of the pocket can follow (``RenderOptions.lining``).
LININGS = ("contacts", "nearest", "none")


def _residues_option(value: Any) -> Any:
    """``residues`` as it is kept: None, one form, or the pairs of a mapping in a fixed order."""
    if value is None or isinstance(value, str):
        if value is not None and value not in RESIDUE_FORMS:
            raise ValueError(f"residues is one of {', '.join(map(repr, RESIDUE_FORMS))}, or a mapping from a "
                             f"residue to one of them, not {value!r}")
        return value
    pairs = tuple(sorted((str(k), str(v)) for k, v in dict(value).items()))
    for name, form in pairs:
        if form not in RESIDUE_FORMS:
            raise ValueError(f"residues: {name} is to be drawn as one of {', '.join(map(repr, RESIDUE_FORMS))}, "
                             f"not {form!r}")
    return pairs


#: Where ``plddt_colors`` puts the colour, and how a predicted water stands on its line.
PLDDT_PLACES = ("behind", "bonds")
SOLVENT_MARKS = ("glow", "filled", "outline")
#: The words of the options that are one word of a few and have no list elsewhere.
_WORDS = {"distance_labels": ("auto", "all", "none"), "show_chain": ("auto", "always", "never"),
          "on_unreadable": ("summary", "render", "none"), "coords": ("auto", "coordgen", "rdkit")}


@dataclass(frozen=True)
class RenderOptions:
    title: str | None = None
    subtitle: str | None = None
    legend: bool = True
    caption: bool = True
    notes: bool = True
    distance_labels: str = "auto"            # auto | all | none
    show_chain: str = "auto"                 # auto | always | never
    figure_width_mm: float = 180.0
    figure_height_mm: float = 240.0
    min_font_pt: float = 5.0
    bond_length: float | None = None         # None = pick the largest readable one
    on_unreadable: str = "summary"           # summary | render | none
    #: How long to try with a ligand that is drawn as a chain of residues, whose figure is tried with more and
    #: more room along the chain while it is too crowded: ``"full"``, until it is legible or six layouts were
    #: not; or ``"quick"``, which, when the first layout is very crowded, looks at the roomiest one that the
    #: page allows and tries no further when that has no place for several residues either. A figure that is
    #: drawn is the same file either way; one that is given up says so. With ``on_unreadable="render"``, and
    #: for anything but a ligand drawn as a chain, it changes nothing.
    effort: str = "full"
    background: bool = True
    coords: str = "auto"                     # auto | coordgen | rdkit: how the ligand is laid out
    #: Draw the model's confidence when the diagram has it (see :mod:`bindview.confidence`).
    confidence: bool = True
    #: Print the ligand's buried fraction with the other provenance numbers under the title
    #: (and, below the calibrated cut, a line saying what that may mean). It is always reported
    #: in the result; this only decides whether it is also in the figure: ``True`` prints it there,
    #: ``False`` leaves it out. An interface has no buried fraction, on or off.
    burial: bool = True
    #: Predict solvent sites that could bridge ligand and protein, and draw them -- marked as
    #: predicted. Off unless asked for, and never done for a structure that has waters of its own.
    solvent_sites: bool = False
    #: The colour scheme: ``"standard"``, ``"contrast"``, whose residue classes are further apart, or one of the
    #: caller's own registered in :data:`bindview.theme.SCHEMES`. The theme picks its light or its dark palette.
    colors: str = "standard"
    #: Show the model's pLDDT by AlphaFold's four colours and not by stroke weight: residues and the
    #: atoms of the ligand are coloured by their band, and every line is drawn at full strength.
    #: ``True`` or ``"behind"``: the atoms stand on the colour. ``"bonds"``: the bonds are drawn in it, half
    #: by half, those of the ligand and those that are drawn of a residue.
    #: Off unless asked for; without a pLDDT to draw it changes nothing, and the result says so.
    plddt_colors: "bool | str" = False
    #: Show the predicted aligned error of a contact (:mod:`bindview.pae`) as a glow behind its line, in the four
    #: colours of pLDDT (blue: the two ends are placed against each other; orange: they are not). ``None``, the
    #: default, does so in the diagram of a ligand that was given a ``pae``, and writes the value on the line in
    #: the figure of an interface; ``True`` draws the glow there too, ``False`` in neither.
    pae_colors: "bool | None" = None
    #: Draw each atom that the detailed style draws of a residue (the fragment of one that makes a
    #: directional contact) at its own pLDDT, where the model gives the atoms of a residue values of their
    #: own (AlphaFold 3 does), and say so in a footnote; a model with one value for a residue is the same
    #: figure either way. ``False`` draws a fragment at its residue's value. At a level of ``detail`` the
    #: atoms are at their own values either way.
    plddt_by_atom: bool = True
    #: How a contact's occupancy (the share of an ensemble that makes it) is drawn: ``"steps"``, a line in
    #: three steps of weight and opacity (90 % and more, 50 to 90 %, below 50 %) with the share written on
    #: every line made in fewer than all, in the place of its distance; or ``"continuous"``, weight and
    #: opacity in proportion to the share and nothing written. A figure without occupancy is the same file
    #: either way.
    occupancy_style: str = "steps"

    def __post_init__(self) -> None:
        for name, words in _WORDS.items():
            if not isinstance(getattr(self, name), str) or getattr(self, name) not in words:
                raise ValueError(f"{name} is one of {', '.join(map(repr, words))}, not {getattr(self, name)!r}")
        if self.occupancy_style not in ("steps", "continuous"):
            raise ValueError(f"occupancy_style is 'steps' or 'continuous', not {self.occupancy_style!r}")
        if self.occupancy_labels not in ("all", "partial"):
            raise ValueError(f"occupancy_labels is 'all' or 'partial', not {self.occupancy_labels!r}")
        if self.plddt_colors not in (True, False, *PLDDT_PLACES):
            raise ValueError(f"plddt_colors is True, False or one of {', '.join(map(repr, PLDDT_PLACES))}, "
                             f"not {self.plddt_colors!r}")
        if self.lines is not None and self.lines not in LINES:
            raise ValueError(f"lines is one of {', '.join(map(repr, LINES))} (or None, the style's), "
                             f"not {self.lines!r}")
        if self.lining not in (None, True, False, *LININGS):
            raise ValueError(f"lining is one of {', '.join(map(repr, LININGS))} (or None, the style's), "
                             f"not {self.lining!r}")
        if self.stylesheet not in ("inline", "scoped", "external"):
            raise ValueError(f"stylesheet is 'inline', 'scoped' or 'external', not {self.stylesheet!r}")
        if self.absent_style not in other_protein.ABSENT_STYLES:
            raise ValueError(f"absent_style is one of {', '.join(map(repr, other_protein.ABSENT_STYLES))}, "
                             f"not {self.absent_style!r}")
        if self.effort not in ("full", "quick"):
            raise ValueError(f"effort is 'full' or 'quick', not {self.effort!r}")
        if self.atom_labels not in ATOM_LABELS:
            raise ValueError(f"atom_labels is one of {', '.join(map(repr, ATOM_LABELS))}, not {self.atom_labels!r}")
        if self.solvent_mark not in SOLVENT_MARKS:
            raise ValueError(f"solvent_mark is one of {', '.join(map(repr, SOLVENT_MARKS))}, not {self.solvent_mark!r}")
        # (Held as a tuple: options are the key of a layout that is kept.)
        object.__setattr__(self, "residues", _residues_option(self.residues))
        object.__setattr__(self, "absent", other_protein.absent_option(self.absent))
    #: The figure answers the pointer (:mod:`bindview.interactive`): pointing at a residue, or at one of
    #: its contacts, keeps it with its contacts and what they reach, draws those contacts heavier and steps
    #: the other residues and contacts far back; a click keeps that until a click elsewhere. A style sheet in the file does it and
    #: no script, and nothing that is drawn changes: where the file is not shown by a browser as a
    #: document or as part of a page (an image, a PDF, a drawing program) it is the figure it would be
    #: without this. Off unless asked for; for a figure with no residue to point at it changes nothing,
    #: and the result says so.
    interactive: bool = False
    #: Break a contact's line where it runs over a word that is not its own (:mod:`bindview.clearance`): the
    #: letter of a ligand atom that lies on its way to a ring centre, the name of a residue, the distance
    #: written on another line. The line is broken around the word as it is around its own distance.
    #: ``False`` draws every line whole; a figure in which no line runs over a word is the same file either
    #: way.
    break_lines: bool = True
    #: Shade the charged group that a salt bridge or a pi-cation contact is made with, in the colour of the
    #: contact: such a contact is with the group and not with one of its atoms. ``True`` draws the shading and
    #: ``False`` none: the line points at the group, and ends where the shading would begin, either way;
    #: nothing else in the figure moves.
    charge_halos: bool = False
    #: Name that shading, where it is drawn, in the key: "Charged group", a soft disc in the colour of the
    #: contact (one for each kind of contact that shades a group, where both do). On unless ``False`` is
    #: passed, which leaves the entry out.
    charge_key: bool = True
    #: The drawing and its key, and no words around them: no line of confidence numbers under the title, no
    #: warning above the drawing and no notes under the key. For a figure that goes into a paper or a slide
    #: with a caption of its own. The words that are left off are what the figure needs to be read (what a
    #: ``?`` marks, the unit of a distance, what a PAE is): they are in ``RenderResult.notes``, for that
    #: caption. A title or subtitle that is asked for is drawn, and so is the mark of a model the caller
    #: flagged as not to be trusted. Off unless asked for.
    bare: bool = False
    #: How the residues are drawn, in either style: ``"atoms"`` (the part that makes a directional contact,
    #: and an eyelash for a residue that makes none), ``"bubbles"``, or ``"names"``; or a mapping, residue
    #: by residue (``{"Gln712": "atoms", "*": "bubbles"}``; a residue is named as the figure names it, with
    #: or without its chain). ``None``: the style's own (:data:`STYLE_SETTINGS`), as for the next three.
    residues: Any = None
    #: The colours and dashes of the contact lines: ``"detailed"`` or ``"schematic"``, in either style.
    lines: str | None = None
    #: Grey halos behind the ligand's atoms that are open to solvent, in either style.
    exposure: bool | None = None
    #: The wall of the pocket around the ligand, in either style: ``"contacts"``, a stretch for each contact
    #: in the colour of what makes it (hydrophobic, polar, charged), which is what the schematic style draws
    #: unless told otherwise; ``"nearest"``, a band in the class of the nearest residue, open where none is
    #: near; ``"none"`` (docs/reference.md). ``True`` is ``"contacts"`` and ``False`` is ``"none"``.
    lining: "str | bool | None" = None
    #: What is written on an atom that is drawn of a residue: the letter of its element (``"elements"``: N, O), as in a
    #: chemical formula, or its name in the structure file (``"names"``: NZ, OE1, O4'), by which it is found in the file
    #: and in a table of contacts; that is in the tooltip of the contact either way. Nothing moves with it. A name
    #: is centred on its atom, so a bond that comes from the right ends at its last letter, and NZ can read as a Z.
    atom_labels: str = "elements"
    #: How a predicted water stands on its line: on a glow (``"glow"``), in a filled box with a heavy broken
    #: border (``"filled"``), or in a thin broken outline (``"outline"``).
    solvent_mark: str = "glow"
    #: On which contacts the share of an ensemble is written (``occupancy_style="steps"``): on every line, and
    #: beside the name of a residue that has hydrophobic contacts only and no line (``"all"``); or on the line of
    #: a contact that is made in fewer than all (``"partial"``), where one made in all has its distance. With
    #: ``distance_labels="all"`` a contact made in all keeps its distance either way.
    occupancy_labels: str = "all"
    #: Residues that another protein does not have (:mod:`bindview.styles.absence`): numbers (``702``, ``"60A"``) or
    #: names (``"Leu702"``, ``"A:702"``), any of which the figure may not have. Those that touch the ligand are
    #: marked with their contacts, and the figure says how many of the contact residues they are. ``None``: there
    #: is no other protein; an empty list: there is one, and none is absent from it, which the figure says too.
    absent: Any = None
    #: The other protein's name, for the key and the sentence under it.
    absent_label: str | None = None
    #: What becomes of them: ``"marked"``, a stroke of a marker pen under them; ``"faint"``, set back.
    absent_style: str = "marked"
    #: Where the figure's style sheet is, for a page that has several figures in it: ``"inline"``, in the figure,
    #: where it declares the colours for every figure of the page; ``"scoped"``, in the figure and for figures in
    #: the same colours alone; ``"external"``, in ``result.css`` and not in the figure, for the page to have once
    #: (:func:`bindview.stylesheet`). Where a figure has no style sheet it changes nothing.
    stylesheet: str = "inline"

    def labelled(self, kind: InteractionType) -> bool:
        if self.distance_labels == "none":
            return False
        if self.distance_labels == "all":
            return kind is not T.HYDROPHOBIC
        return kind in LABELLED_BY_DEFAULT


def beyond_any_page(natural_pt: float, options: RenderOptions, fixed: bool = False) -> "str | None":
    """Why no page can make a figure legible, where that is so; ``None`` where a page of some size can.

    A figure is fitted to its page by making it smaller, never larger. So its smallest text is at most
    what it is at full size (``natural_pt``, in points), and a ``min_font_pt`` above that is met by no
    width or height. A message that named a width to try would send the caller after a page that does
    not exist: this is the reason to give in its place, and it names what does help. ``fixed``: the
    text of this kind of figure has one size, which no ``Metrics`` changes.
    """
    if options.min_font_pt <= natural_pt + 1e-9:
        return None
    # (Both sizes are cut off, not rounded: rounded up, 6.375 pt would be said as "is 6.4 pt" to a caller
    # who asked for 6.4 pt and is told that no page gives it.)
    is_ = math.floor(natural_pt * 100.0 + 1e-9) / 100.0
    most = math.floor(natural_pt * 10.0 + 1e-9) / 10.0
    return (f"its smallest text is {is_:g} pt at full size, and a figure is fitted to a page by making "
            f"it smaller, never larger: no page gives it the {options.min_font_pt:g} pt that min_font_pt asks "
            f"for. Ask for {most:g} pt or less"
            + (": the text of this view has one size" if fixed else ", or hand in larger fonts (metrics=)"))


def page_that_would_do(required_width_mm: float, width: float, height: float,
                       options: RenderOptions) -> "tuple[str, str, float]":
    """The page a figure needs for its text, as the words name it: millimetres, ``"wide"`` or ``"high"``,
    and what was asked for in that direction.

    ``required_width_mm`` is the width at which the smallest text reaches ``min_font_pt``; ``width`` and
    ``height`` are the figure's own. The width is named, in whole millimetres, wherever that is more than
    the width asked for. Two cases read wrong that way and are said differently:

    * rounded, the number is the width that was asked for (90.3 mm of a page of 90 mm): it is then
      rounded up to a tenth, so the page named is one that would do;
    * the width asked for would do, and the page's height is what makes the text small: the height is
      named, and a caller is not sent after a width they already have.
    """
    needed, asked, word = required_width_mm, options.figure_width_mm, "wide"
    if required_width_mm <= options.figure_width_mm + 1e-9:
        needed, asked, word = required_width_mm * height / width, options.figure_height_mm, "high"
    said = f"{needed:.0f}"
    if float(said) <= asked:
        said = f"{math.ceil(needed * 10.0 - 1e-9) / 10.0:g}"
    return said, word, asked


def too_small_on_page(printed_pt: float, required_width_mm: float, width: float, height: float,
                      options: RenderOptions) -> str:
    """What to say of a figure whose text the page asked for makes too small (``printed_pt``), with the
    page that would do. The size is cut off where rounding would write the minimum itself: 4.98 pt is
    not "5.0 pt (minimum 5 pt)"."""
    size = f"{printed_pt:.1f}"
    if float(size) >= options.min_font_pt:
        size = f"{math.floor(printed_pt * 100.0) / 100.0:.2f}"
    mm, word, asked = page_that_would_do(required_width_mm, width, height, options)
    where = f"at {asked:g} mm" if word == "wide" else f"on a page {asked:g} mm high"
    return (f"{where} the smallest text would print at {size} pt "
            f"(minimum {options.min_font_pt:g} pt); it needs a figure about {mm} mm {word}")


@dataclass
class ResidueGroup:
    """All drawable interactions of one residue."""

    key: tuple[str, int, str]
    resname: str
    label: str
    interactions: list[ResolvedInteraction]
    #: Beside a chain ligand a residue whose contacts lie far apart along the chain is drawn once for
    #: each stretch (:func:`split_along_chain`): which of its drawings this is, and how many there are.
    part: int = 0
    parts: int = 1
    #: Does the protein that the figure is compared with lack this residue (``RenderOptions.absent``), and is it
    #: set back for that?
    lost: bool = False
    faint: bool = False

    @property
    def uid(self) -> tuple:
        """What tells this drawing from every other: the residue, and which of its drawings it is."""
        return self.key if self.parts == 1 else (*self.key, self.part)

    @property
    def node_ident(self) -> str:
        """The identifier of this drawing: the residue's, numbered where the residue is drawn more than once."""
        return self.ident if self.parts == 1 else f"{self.ident}~{self.part + 1}"

    @property
    def beyond(self) -> bool:
        return any(r.record.beyond_contact_set for r in self.interactions)

    @property
    def absent(self) -> bool:
        """Does the structure that is shown make none of this residue's contacts (other models do)?"""
        return bool(self.interactions) and all(r.absent for r in self.interactions)

    @property
    def types(self) -> list[InteractionType]:
        seen: list[InteractionType] = []
        for r in self.interactions:
            if r.type not in seen:
                seen.append(r.type)
        return seen

    @property
    def only_hydrophobic(self) -> bool:
        return all(r.type is T.HYDROPHOBIC for r in self.interactions)

    @property
    def residue_class(self) -> str:
        return residue_class(self.resname)

    @property
    def ident(self) -> str:
        chain, number, icode = self.key
        return f"{chain}:{self.resname}{number}{icode}" if chain else f"{self.resname}{number}{icode}"


    @property
    def numbered(self) -> str:
        """The residue's number, with its insertion code."""
        return f"{self.key[1]}{self.key[2]}"


def group_residues(resolved: Sequence[ResolvedInteraction], show_chain: str = "auto") -> list[ResidueGroup]:
    """Group resolved, drawable interactions by residue, in chain/number order."""
    table: dict[tuple[str, int, str], list[ResolvedInteraction]] = {}
    for r in resolved:
        if r.ligand is None:
            continue
        table.setdefault(r.record.residue_id, []).append(r)
    chains = {k[0] for k in table}
    with_chain = show_chain == "always" or (show_chain == "auto" and len(chains) > 1)
    out = []
    for key in sorted(table):
        rs = table[key]
        rec = rs[0].record
        label = format_residue_label(rec.residue_name, rec.residue_number, rec.insertion_code,
                                     key[0] if with_chain and key[0] else None)
        out.append(ResidueGroup(key, rec.residue_name, label, rs))
    return out


#: How far apart along a chain ligand, in residues of the ligand, the contacts of one residue of the
#: protein may lie for it still to be drawn once.
SPLIT_GAP = 8


def chain_index(dep: Depiction, atoms: Sequence[int]) -> int | None:
    """Where along a chain ligand these atoms are: the index of their residue, and for atoms outside
    the chain that of the residue which carries them."""
    for a in atoms:
        res = dep.home_of(a)
        if res is not None:
            return res.index
    seen, queue = set(atoms), list(atoms)
    while queue:
        for j in dep.neighbours(queue.pop(0), False):
            res = dep.unit_of(j)
            if res is not None:
                return res.index
            if j not in seen:
                seen.add(j)
                queue.append(j)
    return None


def split_along_chain(groups: Sequence[ResidueGroup], dep: Depiction) -> list[ResidueGroup]:
    """Beside a chain ligand: one drawing of a residue for each stretch of the chain it touches.

    A chain is drawn as a straight column whatever its path in space. A residue
    of the protein that touches both ends of a folded peptide would have to
    stand beside two places at once; drawn in one of them, its other contact is
    a line across the whole figure, or cannot be drawn at all. Where the
    contacts of a residue lie more than :data:`SPLIT_GAP` residues apart, it is
    therefore drawn once for each stretch, with the contacts it makes there --
    and the figure says so in a footnote.
    """
    if not dep.units or dep.arranged == "plane":
        # On a plane the boxes are where the residues are: what one residue touches lies together.
        return list(groups)
    out: list[ResidueGroup] = []
    for grp in groups:
        places = []
        for r in grp.interactions:
            assert r.ligand is not None
            where = chain_index(dep, r.ligand.atoms)
            places.append(0 if where is None else where)
        order = sorted(range(len(places)), key=lambda i: (places[i], i))
        stretch = [0] * len(places)
        for before, i in zip(order, order[1:]):
            stretch[i] = stretch[before] + (1 if places[i] - places[before] > SPLIT_GAP else 0)
        n = max(stretch) + 1
        if n == 1:
            out.append(grp)
            continue
        for k in range(n):
            out.append(ResidueGroup(grp.key, grp.resname, grp.label,
                                    [r for r, s in zip(grp.interactions, stretch) if s == k], part=k, parts=n))
    return out


@dataclass
class Context:
    ligand: Ligand
    dep: Depiction
    groups: list[ResidueGroup]
    metrics: Metrics
    palette: Palette
    options: RenderOptions
    notes: list[str] = field(default_factory=list)
    exposure: "np.ndarray | None" = None          # relative solvent exposure per ligand atom
    field: "LigandField | None" = None            # the ligand distance field, once built
    confidence: "Confidence | None" = None        # per-atom / per-residue pLDDT, when it may be drawn
    ensemble: str | None = None                   # what the records' occupancy is a share of (legend text)
    solvent_sites: tuple = ()                     # the predicted sites behind any predicted water bridge
    #: How much of a residue beside the ligand is drawn where it is drawn with atoms: by chain, and
    #: under ``"*"`` for every chain that is not named (:data:`bindview.fragments.DETAILS`). None: as always.
    detail: "dict[str, str | None] | None" = None
    #: Was the diagram given a matrix of predicted aligned errors of its own (``from_structure(pae=...)``)?
    pae: bool = False
    #: A figure of all the models of an ensemble: the models that contact each residue, which its ring shows.
    shares: "Mapping[tuple[str, int, str], ResidueShare] | None" = None

    def share_of(self, grp: "ResidueGroup") -> "ResidueShare | None":
        """The models that contact a residue, where the figure is of several and the residue stands in a ring."""
        return self.shares.get(grp.key) if self.shares else None

    @property
    def glows(self) -> bool:
        """Is the predicted aligned error of a contact a glow behind its line (``pae_colors``)?"""
        return self.pae if self.options.pae_colors is None else self.options.pae_colors

    def detail_of(self, grp: "ResidueGroup") -> "str | None":
        """The level of detail of a residue beside the ligand: its chain's, or the figure's."""
        if not self.detail:
            return None
        return self.detail.get(grp.key[0], self.detail.get("*"))

    @property
    def by_colour(self) -> bool:
        """Does this figure show pLDDT by AlphaFold's colours (``plddt_colors``) and not by weight?"""
        return bool(self.options.plddt_colors) and self.confidence is not None

    @property
    def by_bond(self) -> bool:
        """... and is it the bonds that have the colour?"""
        return self.by_colour and self.options.plddt_colors == "bonds"

    @property
    def shares_written(self) -> bool:
        """Is the share of an ensemble written on every contact that has one (``occupancy_labels``)?"""
        return self.options.occupancy_style == "steps" and self.options.occupancy_labels == "all"

    def look(self, value: "float | None") -> Look:
        """The look of a pLDDT value in this figure (:func:`bindview.confidence.look_of`)."""
        return look_of(value, by_colour=self.by_colour)

    def fill_of(self, look: Look, otherwise: str) -> str:
        """The fill of something that has a pLDDT band: the band's colour where the figure colours by
        pLDDT and the thing has a value, and ``otherwise`` (its own token) in every other case."""
        return f"plddt_{look.band}" if self.by_colour and look.band is not None else otherwise

    def text_on(self, look: Look, otherwise: str) -> str:
        """The colour of a name that stands on :meth:`fill_of`."""
        return f"plddt_text_{look.band}" if self.by_colour and look.band is not None else otherwise


# ---------------------------------------------------------------------------
# Confidence: what a contact and a residue are drawn with
# ---------------------------------------------------------------------------

def contact_plddt(ctx: Context, r: ResolvedInteraction) -> float | None:
    """Confidence of one contact: the number the record carries, and otherwise the lower of the
    residue's pLDDT and the ligand site's."""
    if not ctx.options.confidence:
        return None
    if r.record.confidence is not None:
        return r.record.confidence
    if ctx.confidence is None or r.ligand is None:
        return None
    names = [ctx.ligand.names[i] for i in r.ligand.atoms]
    return ctx.confidence.contact(r.residue if r.residue is not None else r.record.residue_id, names)


def residue_plddt(ctx: Context, grp: "ResidueGroup") -> float | None:
    """pLDDT of the residue itself (mean over its atoms)."""
    if ctx.confidence is None:
        return None
    first = grp.interactions[0]
    return ctx.confidence.residue(first.residue if first.residue is not None else grp.key)


def residue_occupancy(grp: "ResidueGroup") -> float | None:
    """Occupancy a residue node is drawn at: that of its most persistent contact.

    This is a drawing level, not a reported number: the share of the ensemble
    in which the residue makes *any* contact is at least this, and cannot be
    known from per-contact occupancies alone.
    """
    values = [r.record.occupancy for r in grp.interactions if r.record.occupancy is not None]
    return max(values) if values else None


def marked_label(text: str, look: Look) -> str:
    """``text`` with the lowest band's mark appended."""
    return f"{text}{MARK}" if look.mark else text


# ---------------------------------------------------------------------------
# Where a link lands on the ligand
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Target:
    point: Point
    radius: float
    caps: tuple[np.ndarray, np.ndarray, np.ndarray] | None
    kind: str                       # atom | ring | group
    atoms: tuple[int, ...]
    extent: float = 3.0             # how far the atom / ring / group reaches from ``point``
    unit: int | None = None         # in a chain ligand: the residue these atoms belong to (index of the unit)

    @property
    def key(self) -> tuple:
        """What tells two targets apart: the residue of a chain ligand, or else the kind and the atoms."""
        return ("unit", self.unit) if self.unit is not None else (self.kind, self.atoms)


def _box_capsule(box: tuple[float, float, float, float], pad: float) -> tuple[Point, Point, float]:
    w, h = box[2] - box[0], box[3] - box[1]
    cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
    r = min(w, h) / 2.0
    if w >= h:
        return (box[0] + r, cy), (box[2] - r, cy), r + pad
    return (cx, box[1] + r), (cx, box[3] - r), r + pad


def halo_radius(metrics: Metrics) -> float:
    return 0.33 * metrics.bond_length


def spoke_reach(dep: Depiction, atom: int, metrics: Metrics) -> float:
    """Outer radius of the contact spokes drawn on a ligand atom."""
    r0 = metrics.spoke_inner
    label = dep.atoms[atom].label
    if label is not None:
        r0 = max(r0, math.hypot(label.box[2] - label.box[0], label.box[3] - label.box[1]) / 2.0 + 1.5)
    return r0 + (metrics.spoke_outer - metrics.spoke_inner)


def group_skeleton(dep: Depiction, atoms: Sequence[int]) -> list[tuple[Point, Point]]:
    """Bond segments among ``atoms`` (the backbone a group halo is drawn along)."""
    members = set(atoms)
    segs = [(dep.pos(b.a), dep.pos(b.b)) for b in dep.bonds
            if b.kind == "covalent" and b.a in members and b.b in members]
    if not segs:
        segs = [(dep.pos(i), dep.pos(i)) for i in atoms]
    return segs


def ligand_target(dep: Depiction, anchor: Anchor, metrics: Metrics, whole: bool = False) -> Target:
    """The point a link aims at, and the ink that stops it short of that point.

    ``whole``: a contact without a direction, which is one with the residue of a chain as a whole. It
    lands on the residue's box even where atoms of the residue are drawn beside it; a directional
    contact lands on its atom there.
    """
    res = dep.unit_for(anchor.atoms, whole) if dep.units else None
    if res is None and dep.units:
        own = dep.landing(anchor.atoms)
        if own != tuple(anchor.atoms):
            # Some of its atoms are drawn and some are in the box: it lands on those that are drawn.
            anchor = Anchor("atom" if len(own) == 1 else anchor.kind, own, anchor.group_kind)
    if res is not None:
        # A residue of a chain ligand: the link stops at its box, whichever of its atoms makes the contact.
        c0, c1, cr = _box_capsule(res.box, metrics.label_gap + 0.5)
        caps = (np.array([c0], dtype=float), np.array([c1], dtype=float), np.array([cr], dtype=float))
        return Target(res.pos, 8.0, caps, "atom", anchor.atoms, extent=8.0, unit=res.index)
    if anchor.kind in ("ring", "group"):
        centre = dep.centroid(anchor.atoms)
        spread = max(dist(centre, dep.pos(i)) for i in anchor.atoms)
        if anchor.kind == "ring":
            return Target(centre, 3.6, None, "ring", anchor.atoms, extent=spread)
        hr = halo_radius(metrics)
        segs = group_skeleton(dep, anchor.atoms)
        p0 = np.array([s[0] for s in segs], dtype=float)
        p1 = np.array([s[1] for s in segs], dtype=float)
        caps = (p0, p1, np.full(len(segs), hr + 1.0))
        return Target(centre, hr, caps, "group", anchor.atoms, extent=spread + hr)
    atom = dep.atoms[anchor.atom]
    if atom.label is None:
        return Target(atom.pos, 3.0, None, "atom", anchor.atoms, extent=3.0)
    caps_l = [_box_capsule(box, metrics.label_gap + 0.5) for box in atom.label.boxes]
    caps = (np.array([c[0] for c in caps_l], dtype=float), np.array([c[1] for c in caps_l], dtype=float),
            np.array([c[2] for c in caps_l], dtype=float))
    return Target(atom.pos, 8.0, caps, "atom", anchor.atoms, extent=8.0)


# ---------------------------------------------------------------------------
# Link lines
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LinkStyle:
    token: str
    dash: tuple[float, ...]
    width: float = 1.0
    cap: str = "butt"
    end_dots: bool = False          # dot at a ring-centroid end


LINK_STYLES: dict[InteractionType, LinkStyle] = {
    T.HBOND: LinkStyle("hbond", (4.2, 2.8)),
    T.SALTBRIDGE: LinkStyle("saltbridge", (0.01, 3.7), width=1.75, cap="round"),
    T.PI_STACKING: LinkStyle("pi_stacking", (6.5, 2.4, 1.3, 2.4), end_dots=True),
    T.PI_CATION: LinkStyle("pi_cation", (8.0, 3.0), end_dots=True),
    T.HALOGEN: LinkStyle("halogen", (6.5, 2.2, 1.3, 2.2, 1.3, 2.2)),
    T.METAL: LinkStyle("metal", ()),
    T.WATER_BRIDGE: LinkStyle("water_bridge", (2.2, 2.2)),
    T.HYDROPHOBIC: LinkStyle("hydrophobic", (1.0, 3.0)),
    # A bond like those of the ligand, on a highlight (link_items).
    T.COVALENT: LinkStyle("ligand_bond", (), width=1.35, cap="round"),
}

_ARROW_LEN = 6.4
_ARROW_HALF = 2.5
#: The highlight a covalent bond stands on: its colour, and its width in widths of a contact's line.
HIGHLIGHT = "warning"
COVALENT_GLOW = 6.5
#: The glow of a contact's predicted aligned error: its width in widths of a contact's line, and its opacity.
GLOW = 6.0
GLOW_OPACITY = 0.5
#: ... of the darkest of its colours, so that a distance is legible on it.
_GLOW_OPACITIES = {"plddt_very_high": 0.34}

#: ... and of the line a hydrophobic contact gets on a chain ligand, in the detailed style: in the colour of
#: the arc it leaves, and finer than any other.
CHAIN_CONTACT = LinkStyle("hydrophobic", (), width=0.6)

#: The lines of the schematic style.
SCHEMATIC_LINKS: dict[InteractionType, LinkStyle] = {
    T.HBOND: LinkStyle("sch_hbond", ()),
    T.SALTBRIDGE: LinkStyle("sch_saltbridge", (0.01, 3.7), width=1.75, cap="round"),
    T.PI_STACKING: LinkStyle("sch_pi_stacking", (), end_dots=True),
    T.PI_CATION: LinkStyle("sch_pi_cation", (), end_dots=True),
    T.HALOGEN: LinkStyle("sch_halogen", (6.5, 2.2, 1.3, 2.2)),
    T.METAL: LinkStyle("sch_metal", ()),
    T.WATER_BRIDGE: LinkStyle("sch_water_bridge", (2.2, 2.2)),
    # Only on a chain ligand, where a contact without a line cannot be told from its neighbour's.
    T.HYDROPHOBIC: LinkStyle("bubble_line", (), width=0.6),
    T.COVALENT: LinkStyle("ligand_bond", (), width=1.35, cap="round"),
}

SCHEMATIC_SIDECHAIN = LinkStyle("sch_hbond", (4.2, 2.6))


@dataclass(frozen=True)
class LineSet:
    """How the contacts of a figure are drawn: a line style for each kind of contact, and one for a hydrogen
    bond of a side chain or a base where it is told from one of the backbone (``sidechain``)."""

    name: str
    styles: Mapping[InteractionType, LinkStyle]
    sidechain: LinkStyle | None = None

    def hbond(self, info: "PortInfo", grp: "ResidueGroup") -> tuple[LinkStyle, str]:
        """The line of a hydrogen bond, and what it is called in the key: ``backbone``, ``sidechain``,
        ``base``, or nothing in particular."""
        if self.sidechain is None:
            return self.styles[T.HBOND], ""
        if info.site_kind == "atom" and not is_backbone_atom(grp.resname, info.site_atoms[0]):
            return self.sidechain, "base" if is_nucleotide(grp.resname) else "sidechain"
        return self.styles[T.HBOND], "backbone"


#: The line sets there are, by name (``RenderOptions.lines``).
LINES: dict[str, LineSet] = {
    "detailed": LineSet("detailed", {**LINK_STYLES, T.HYDROPHOBIC: CHAIN_CONTACT}),
    "schematic": LineSet("schematic", SCHEMATIC_LINKS, SCHEMATIC_SIDECHAIN),
}


@dataclass(frozen=True)
class Settings:
    """The settings of a style: how the residues are drawn, the lines, solvent exposure, the lining of the
    pocket, and whether a distance is written without being asked for (``distances``)."""

    residues: Any
    lines: LineSet
    exposure: bool
    lining: str
    distances: bool

    def with_options(self, options: "RenderOptions") -> "Settings":
        """These settings with what the options set by hand in their place."""
        asked = options.lining
        lining = (self.lining if asked is None else asked if isinstance(asked, str) else
                  "contacts" if asked else "none")
        return Settings(self.residues if options.residues is None else options.residues,
                        self.lines if options.lines is None else LINES[options.lines],
                        self.exposure if options.exposure is None else bool(options.exposure),
                        lining, self.distances)

    def labelled(self, options: "RenderOptions"):
        """``labelled(kind)`` for :func:`port_infos`: is the distance of this kind of contact written?"""
        if self.distances:
            return options.labelled
        # Maestro prints no distances; they are available on request.
        return lambda kind: options.distance_labels == "all" and kind is not T.HYDROPHOBIC

    def form_of(self, grp: "ResidueGroup") -> str:
        """How this residue is drawn: one of :data:`RESIDUE_FORMS`."""
        if isinstance(self.residues, str):
            return self.residues
        table = dict(self.residues)
        for name in residue_names(grp):
            if name in table:
                return table[name]
        lowered = {k.lower(): v for k, v in table.items()}
        for name in residue_names(grp):
            if name.lower() in lowered:
                return lowered[name.lower()]
        return table.get("*", "")


def residue_names(grp: "ResidueGroup") -> list[str]:
    """What a residue can be called in ``residues=``: as the figure names it, with its chain, by its number."""
    chain, number, icode = grp.key
    plain = f"{grp.resname}{number}{icode}"
    out = [grp.label, grp.ident, plain, f"{number}{icode}"]
    if chain:
        out += [f"{chain}:{plain}", f"{chain}:{number}{icode}"]
    return out


#: What each style is: its settings. Every one of them can be set by hand (:class:`RenderOptions`).
STYLE_SETTINGS: dict[str, Settings] = {
    "detailed": Settings("atoms", LINES["detailed"], False, "none", True),
    "schematic": Settings("bubbles", LINES["schematic"], True, "contacts", False),
}


def _cut_by_box(a: Point, b: Point, box: tuple[float, float, float, float]) -> list[tuple[Point, Point]]:
    """Segment ``a->b`` with the part inside ``box`` removed."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, a[0] - box[0]), (dx, box[2] - a[0]), (-dy, a[1] - box[1]), (dy, box[3] - a[1])):
        if abs(p) < 1e-12:
            if q < 0:
                return [(a, b)]
            continue
        r = q / p
        if p < 0:
            t0 = max(t0, r)
        else:
            t1 = min(t1, r)
    if t0 >= t1:
        return [(a, b)]
    out = []
    if t0 > 1e-6:
        out.append((a, lerp(a, b, t0)))
    if t1 < 1.0 - 1e-6:
        out.append((lerp(a, b, t1), b))
    return out


def link_items(a: Point, b: Point, kind: InteractionType, metrics: Metrics, *, label: str | None = None,
               arrow: str | None = None, opacity: float = 1.0, label_at: float = 0.5,
               style: "LinkStyle | None" = None, weight: float = 1.0,
               tag: "tuple[str, bool] | None" = None, glow: "str | None" = None) -> list[Item]:
    """A link from ``a`` (protein end) to ``b`` (ligand end).

    ``arrow`` is ``"a"`` or ``"b"``: put an arrowhead at that end (the acceptor
    of an H-bond). ``label`` is printed at the midpoint with the line broken
    around it, so no background patch is needed and the figure works on any
    backdrop. ``weight`` scales the stroke width (confidence; see
    :class:`~bindview.confidence.Look`).
    """
    style = style or LINK_STYLES[kind]
    paint = Paint(stroke=style.token, width=metrics.interaction_width * style.width * weight, dash=style.dash,
                  cap=style.cap, opacity=opacity)
    out: list[Item] = []
    length = dist(a, b)
    if length < 1e-6:
        return out
    u = unit(sub(b, a))
    start, end = a, b
    if glow is not None:
        out.append(Line(a[0], a[1], b[0], b[1],
                        Paint(stroke=glow, width=GLOW * metrics.interaction_width, cap="round",
                              opacity=_GLOW_OPACITIES.get(glow, GLOW_OPACITY))))
    if kind is T.COVALENT:
        # A bond to the target is what a reader looks for first: it stands on a highlight.
        out.append(Line(a[0], a[1], b[0], b[1],
                        Paint(stroke=HIGHLIGHT, width=COVALENT_GLOW * metrics.interaction_width, cap="round",
                              opacity=0.4 * opacity)))
    if arrow in ("a", "b") and length > 2.5 * _ARROW_LEN:
        tip = a if arrow == "a" else b
        back = add(tip, mul(u, _ARROW_LEN if arrow == "a" else -_ARROW_LEN))
        n = perp(u)
        half = _ARROW_HALF * (0.55 + 0.45 * weight)
        out.append(Polygon((tip, add(back, mul(n, half)), add(back, mul(n, -half))),
                           Paint(fill=style.token, opacity=opacity)))
        if arrow == "a":
            start = lerp(tip, back, 0.85)
        else:
            end = lerp(tip, back, 0.85)
    pieces = [(start, end)]
    if label and not tag:
        mid = lerp(a, b, label_at)
        w = text_width(label, metrics.distance_font) + 3.0
        h = CAP_HEIGHT * metrics.distance_font + 3.4
        pieces = _cut_by_box(start, end, (mid[0] - w / 2.0, mid[1] - h / 2.0, mid[0] + w / 2.0, mid[1] + h / 2.0))
        out.append(Text.plain(mid[0], mid[1] + CAP_HEIGHT * metrics.distance_font / 2.0, label,
                              metrics.distance_font, style.token, opacity=opacity))
    elif tag:
        # The predicted aligned error of the contact, written out: in the colour of the figure's text and at
        # full strength whatever the line's, because it is not a property of the line. What pLDDT does to the
        # stroke it does not do to this.
        mid = lerp(a, b, label_at)
        words, bold = tag
        lead = (label + " ") if label else ""
        w = text_width(lead, metrics.distance_font) + text_width(words, metrics.distance_font, bold=bold) + 3.0
        h = CAP_HEIGHT * metrics.distance_font + 3.4
        pieces = _cut_by_box(start, end, (mid[0] - w / 2.0, mid[1] - h / 2.0, mid[0] + w / 2.0, mid[1] + h / 2.0))
        left = mid[0] - (w - 3.0) / 2.0
        baseline = mid[1] + CAP_HEIGHT * metrics.distance_font / 2.0
        if label:
            out.append(Text.plain(left, baseline, label, metrics.distance_font, style.token, anchor="start",
                                  opacity=opacity))
        out.append(Text.plain(left + text_width(lead, metrics.distance_font), baseline, words, metrics.distance_font,
                              "text", anchor="start", bold=bold))
    for p, q in pieces:
        if dist(p, q) > 0.4:
            out.append(Line(p[0], p[1], q[0], q[1], paint))
    return out


def hbond_direction(r: ResolvedInteraction, dep: Depiction) -> tuple[str | None, str | None]:
    """``(arrow_end, note)`` for an H-bond-like interaction.

    The arrowhead goes at the acceptor: ``"a"`` (protein end) when the ligand
    donates, ``"b"`` (ligand end) when the protein does. No arrowhead is drawn
    when the direction is unknown, or when the reported donor cannot donate in
    the protonation state that is actually depicted -- the line is kept, the
    impossible claim is not made.
    """
    donor = r.record.ligand_is_donor
    if donor is None or r.ligand is None:
        return None, None
    if donor:
        if r.ligand.kind == "atom" and dep.atoms[r.ligand.atom].hcount == 0:
            name = dep.atoms[r.ligand.atom].name
            return None, (f"{r.record.residue_label}–{name}: reported with the ligand as donor, but {name} "
                          f"carries no hydrogen in the depicted protonation state; drawn without direction")
        return "a", None
    if r.protein.kind == "atom":
        ok = can_donate(r.record.residue_name, r.protein.atoms[0])
        if ok is False:
            return None, (f"{r.record.residue_label} {r.protein.atoms[0]}: reported as donor, but that atom "
                          f"carries no hydrogen in the standard protonation state; drawn without direction")
    return "b", None


def interaction_title(r: ResolvedInteraction, lig: Ligand, plddt: float | None = None,
                      dep: Depiction | None = None) -> str:
    """Tooltip text: what interacts with what, at what distance (and how confidently).

    With the depiction of a chain ligand, the ligand's end names the residue
    that makes the contact, its atoms, and whether they are backbone or side
    chain: what the line itself does not show.
    """
    rec = r.record
    prot = rec.residue_label
    if r.protein.kind == "atom":
        prot += f" {r.protein.atoms[0]}"
    elif r.protein.kind == "ring":
        prot += " ring"
    elif r.protein.kind == "group":
        prot += " " + ("(+)" if r.protein.charge > 0 else "(−)" if r.protein.charge < 0 else "group")
    res = dep.unit_for(r.ligand.atoms, True) if (dep is not None and dep.units and r.ligand is not None) else None
    if r.ligand is None:
        lg = "?"
    elif res is not None:
        part = res.part(r.ligand.atoms)
        lg = (f"{res.text} {res.number or res.position} of {dep.called if dep is not None else 'the ligand'} "
              f"({part}: "
              f"{', '.join(lig.names[i] for i in r.ligand.atoms)})")
    elif r.ligand.kind == "atom":
        lg = lig.names[r.ligand.atom]
    elif r.ligand.kind == "ring":
        lg = "ring " + "-".join(lig.names[i] for i in r.ligand.atoms)
    else:
        lg = f"{r.ligand.group_kind or 'group'} " + "-".join(lig.names[i] for i in r.ligand.atoms)
    text = f"{rec.type.label}: {prot} – {lg}"
    if rec.predicted:
        text = f"Predicted water site, not observed: {prot} – {lg}"
    if r.absent:
        text = f"{rec.type.label}, made by other models and not by the one shown: {prot} – {lg}"
    if rec.distance is not None:
        text += f", {rec.distance:.2f} Å"
    if plddt is not None:
        text += f", contact pLDDT {plddt:.0f}"
    if rec.pae is not None:
        text += f", PAE {rec.pae:.1f} \u00c5"
        if len(rec.pae_draws) > 1:
            text += (f" (median of {len(rec.pae_draws)} draws, {min(rec.pae_draws):.1f} to "
                     f"{max(rec.pae_draws):.1f})")
    if rec.occupancy is not None:
        text += f", occupancy {100.0 * rec.occupancy:.0f} %"
    return text


# ---------------------------------------------------------------------------
# Hydrophobic contacts: eyelashes and spokes
# ---------------------------------------------------------------------------

def _arc_points(centre: Point, radius: float, a0: float, a1: float) -> tuple[Point, ...]:
    n = max(6, int(math.ceil(abs(a1 - a0) * radius / 3.0)))
    return tuple((centre[0] + radius * math.cos(a0 + (a1 - a0) * k / n),
                  centre[1] + radius * math.sin(a0 + (a1 - a0) * k / n)) for k in range(n + 1))


def _direction_clusters(centre: Point, targets: Sequence[Point]) -> list[tuple[float, float]]:
    """``[(mid_angle, half_width)]`` arcs facing ``targets`` as seen from ``centre``."""
    angles = sorted(math.atan2(t[1] - centre[1], t[0] - centre[0]) % (2 * math.pi) for t in targets)
    if not angles:
        return []
    if len(angles) == 1:
        return [(angles[0], math.radians(56.0))]
    gaps = [((angles[(i + 1) % len(angles)] - angles[i]) % (2 * math.pi), i) for i in range(len(angles))]
    big, at = max(gaps)
    spread = 2 * math.pi - big
    start = angles[(at + 1) % len(angles)]
    if spread <= math.radians(110.0):
        return [(start + spread / 2.0, math.radians(56.0))]
    # Two faces: split the run at its own widest internal gap.
    run = [(a - start) % (2 * math.pi) for a in angles]
    run.sort()
    inner = [(run[i + 1] - run[i], i) for i in range(len(run) - 1)]
    _, cut = max(inner)
    out = []
    for part in (run[:cut + 1], run[cut + 1:]):
        mid = start + (part[0] + part[-1]) / 2.0
        out.append((mid, math.radians(40.0)))
    return out


def eyelash_items(centre: Point, radius: float, targets: Sequence[Point], metrics: Metrics,
                  opacity: float = 1.0, weight: float = 1.0) -> list[Item]:
    """Arc(s) with lashes, bulging toward the ligand atoms in contact."""
    paint = Paint(stroke="hydrophobic", width=1.25 * weight, opacity=opacity)
    out: list[Item] = []
    for mid, half in _direction_clusters(centre, targets):
        out.append(Polyline(_arc_points(centre, radius, mid - half, mid + half), paint))
        n = 7 if half > math.radians(50.0) else 5
        for k in range(n):
            ang = mid - half + 2 * half * (k + 0.5) / n
            c, s = math.cos(ang), math.sin(ang)
            out.append(Line(centre[0] + radius * c, centre[1] + radius * s,
                            centre[0] + (radius + metrics.lash_length) * c,
                            centre[1] + (radius + metrics.lash_length) * s, paint))
    return out


def eyelash_mark(centre: Point, radius: float, targets: Sequence[Point], metrics: Metrics) -> list[Item]:
    """The marker pen under the arcs of :func:`eyelash_items`, lashes and all."""
    reach = metrics.lash_length
    return [other_protein.stroke(_arc_points(centre, radius + reach / 2.0, mid - half, mid + half), reach + 4.0, 1.0)
            for mid, half in _direction_clusters(centre, targets)]


def spoke_items(dep: Depiction, atom: int, toward: Point, metrics: Metrics, opacity: float = 1.0,
                weight: float = 1.0) -> list[Item]:
    """Short ticks on a ligand atom, fanned toward the residue it contacts.

    Ticks are only drawn in the free sectors between the atom's bonds, so they
    never lie on top of the skeleton.
    """
    p = dep.pos(atom)
    want = math.atan2(toward[1] - p[1], toward[0] - p[0])
    bonds = [math.atan2(q[1] - p[1], q[0] - p[0]) for q in (dep.pos(j) for j in dep.neighbours(atom, False))]
    keep_off = math.radians(21.0)

    def free(ang: float) -> bool:
        return all(abs(angle_diff(ang, b)) >= keep_off for b in bonds)

    centre = want
    if not free(centre):
        # Slide to the nearest free direction.
        best = None
        for step in range(1, 61):
            for sgn in (1.0, -1.0):
                cand = want + sgn * math.radians(3.0 * step)
                if free(cand):
                    best = cand
                    break
            if best is not None:
                break
        if best is None:
            return []
        centre = best
    r1 = spoke_reach(dep, atom, metrics)
    r0 = r1 - (metrics.spoke_outer - metrics.spoke_inner)
    paint = Paint(stroke="hydrophobic", width=1.1 * weight, opacity=opacity)
    out: list[Item] = []
    for off in (-0.42, 0.0, 0.42):
        ang = centre + off
        if not free(ang):
            continue
        c, s = math.cos(ang), math.sin(ang)
        out.append(Line(p[0] + r0 * c, p[1] + r0 * s, p[0] + r1 * c, p[1] + r1 * s, paint))
    return out


def halo_items(segments: Sequence[tuple[Point, Point]], radius: float, token: str) -> list[Item]:
    """A soft highlight along a charged group: the thing a salt bridge attaches to."""
    paint = Paint(stroke=token, width=2.0 * radius, opacity=0.17, cap="round")
    return [Line(p[0], p[1], q[0] + (0.01 if p == q else 0.0), q[1], paint) for p, q in segments]


# ---------------------------------------------------------------------------
# Ports shared by the styles
# ---------------------------------------------------------------------------

# Length of the *visible* part of a link, in bond lengths: (ideal, minimum). What
# the atoms, rings and groups at either end take up is added on top.
VISIBLE_LENGTH = {
    T.HBOND: (1.45, 1.1), T.HALOGEN: (1.45, 1.1), T.METAL: (1.35, 1.05), T.WATER_BRIDGE: (1.5, 1.15),
    T.SALTBRIDGE: (1.3, 0.75), T.PI_STACKING: (1.3, 0.75), T.PI_CATION: (1.3, 0.75), T.HYDROPHOBIC: (1.9, 0.0),
    T.COVALENT: (0.85, 0.6),
}
#: ... and of the line a hydrophobic contact gets on a chain ligand.
CHAIN_CONTACT_LENGTH = (0.9, 0.3)
_PRIMARY_ORDER = {T.COVALENT: -1, T.METAL: 0, T.HBOND: 1, T.HALOGEN: 2, T.WATER_BRIDGE: 3, T.SALTBRIDGE: 4,
                  T.PI_STACKING: 5, T.PI_CATION: 6, T.HYDROPHOBIC: 9}


@dataclass
class PortInfo:
    """One link of a residue: which interactions it stands for and where it lands."""

    interactions: list[ResolvedInteraction]
    kind: InteractionType
    site_kind: str                         # atom | ring | group | residue
    site_atoms: tuple[str, ...]
    target: Target
    drawn: bool
    label: str | None
    plddt: float | None = None             # contact confidence: the lower of its two ends
    look: Look = PLAIN
    occupancy: float | None = None         # share of the ensemble in which the contact is made
    #: Predicted aligned error between the two residues of the contact, where the caller gave one.
    #: It is never drawn by the stroke (that is pLDDT's): it is written on the line (:attr:`tag`),
    #: in the figure of an interface, which is the one that explains the tag (``written``).
    pae: float | None = None
    #: ... and, where that is the median over several draws of the model, the value of each draw.
    pae_draws: tuple[float, ...] = ()
    written: bool = False
    #: A residue of a chain ligand is one place. When a residue of the protein has several links to
    #: it (an H-bond and a salt bridge), each gets a lane: a shift across the line, so that they run
    #: side by side instead of on top of each other. Zero everywhere else.
    lane: float = 0.0
    #: Is occupancy drawn in steps (``RenderOptions.occupancy_style``)?
    steps: bool = False
    #: Is the PAE drawn as a glow (``RenderOptions.pae_colors``)?
    glowing: bool = False
    #: Is the residue set back because another protein lacks it (``absent_style="faint"``)?
    faint: bool = False

    @property
    def glow(self) -> str | None:
        """The colour a contact's line stands on for its PAE, where that is drawn."""
        if not self.glowing or not self.drawn or self.pae is None:
            return None
        return f"plddt_{pae_band(self.pae)}"

    @property
    def beyond(self) -> bool:
        return all(r.record.beyond_contact_set for r in self.interactions)

    @property
    def absent(self) -> bool:
        """Is this a contact that the structure shown does not make, and other models of its ensemble do?"""
        return all(r.absent for r in self.interactions)

    @property
    def tag(self) -> "tuple[str, bool] | None":
        """``("PAE 11", bold)`` for a drawn contact whose PAE is worth saying (:func:`bindview.pae.pae_tag`)."""
        return pae_tag(self.pae) if (self.drawn and self.written) else None

    @property
    def shown(self) -> str | None:
        """Everything that is printed on the line: the label and the tag, for the room they need."""
        parts = [t for t in (self.label, self.tag[0] if self.tag else None) if t]
        return " ".join(parts) if parts else None

    @property
    def predicted(self) -> bool:
        """Is this a contact through a predicted solvent site (not observed, not detected)?"""
        return all(r.record.predicted for r in self.interactions)

    @property
    def weight(self) -> float:
        """Stroke-width factor: confidence and occupancy, each scaling the line."""
        return self.look.width * occupancy_factors(self.occupancy, self.steps)[0]

    @property
    def opacity(self) -> float:
        """Stroke opacity: confidence, occupancy, and the fading of contacts outside the contact set."""
        return min(self.look.opacity * occupancy_factors(self.occupancy, self.steps)[1], 0.5 if self.beyond else 1.0,
                   ABSENT_OPACITY if self.absent or self.faint else 1.0)

    def absorb(self, other: "PortInfo | ResolvedInteraction") -> None:
        """Let this link also stand for further records. Its occupancy is then that of the most
        persistent of them: the link is there whenever any of them is, so this is a lower bound."""
        records = other.interactions if isinstance(other, PortInfo) else [other]
        self.interactions.extend(records)
        values = [v for v in [self.occupancy] + [r.record.occupancy for r in records] if v is not None]
        self.occupancy = max(values) if values else None
        for r in records:
            # The smallest value stands for the link, and brings its own draws with it.
            if r.record.pae is not None and (self.pae is None or r.record.pae < self.pae):
                self.pae, self.pae_draws = r.record.pae, r.record.pae_draws


def _ring_systems(rings: "Sequence[frozenset]") -> "dict[frozenset, frozenset]":
    """Each ring's fused system: all the atoms of the rings joined to it by a shared bond, and so on."""
    parent = list(range(len(rings)))

    def root(i: int) -> int:
        while parent[i] != i:
            i = parent[i]
        return i

    for i in range(len(rings)):
        for j in range(i + 1, len(rings)):
            if len(rings[i] & rings[j]) >= 2 and root(i) != root(j):
                parent[root(i)] = root(j)
    members: dict[int, set] = {}
    for i, ring in enumerate(rings):
        members.setdefault(root(i), set()).update(ring)
    return {ring: frozenset(members[root(i)]) for i, ring in enumerate(rings)}


def _stacked_pairs(ctx: Context, records: "Sequence[ResolvedInteraction]"
                   ) -> "tuple[list[ResolvedInteraction], dict[int, tuple]]":
    """One stacked pair, one line.

    A π-stack is found ring by ring: an indole over a pteridine is up to four pairs of rings, and would be
    four lines from one residue into one place. Where one fused ring system of a residue stacks on one
    fused ring system of the ligand (or on one residue of a chain ligand) with more than one pair of rings,
    those records are one link, drawn at the shortest of them; the others stand in its title. Returns the
    records, the stacking ones among them in order of distance where they are merged, and the key of the
    link each merged record belongs to. A stack made by one pair of rings is left as it was."""
    stacks = [k for k, r in enumerate(records) if r.type is T.PI_STACKING and r.ligand is not None
              and r.protein.kind == "ring" and r.ligand.kind == "ring"]
    if len(stacks) < 2:
        return list(records), {}
    protein = _ring_systems(list(dict.fromkeys(frozenset(records[k].protein.atoms) for k in stacks)))
    ligand = _ring_systems(list(dict.fromkeys([*(frozenset(r) for r in ctx.dep.rings),
                                               *(frozenset(records[k].ligand.atoms)  # type: ignore[union-attr]
                                                 for k in stacks)])))
    by_pair: dict[tuple, list[int]] = {}
    for k in stacks:
        r = records[k]
        assert r.ligand is not None
        res = ctx.dep.unit_for(r.ligand.atoms, False) if ctx.dep.units else None
        place = ("unit", res.index) if res is not None else ("ring", ligand[frozenset(r.ligand.atoms)])
        by_pair.setdefault((T.PI_STACKING, "ring", protein[frozenset(r.protein.atoms)], *place), []).append(k)
    out, keys = list(records), {}
    for key, ks in by_pair.items():
        if len({(records[k].protein.atoms, records[k].ligand.atoms) for k in ks}) < 2:  # type: ignore[union-attr]
            continue
        nearest = sorted((records[k] for k in ks),
                         key=lambda r: r.record.distance if r.record.distance is not None else 99.0)
        for k, r in zip(ks, nearest):
            out[k] = r
            keys[id(r)] = key
    return out, keys


def port_infos(ctx: Context, grp: ResidueGroup, labelled) -> list[PortInfo]:
    """One port per distinct (protein site, ligand anchor, type); primary first.

    ``labelled(kind)`` says whether that type's distance is printed.
    """
    table: dict[tuple, PortInfo] = {}
    records = grp.interactions
    if ctx.dep.pairwise:
        # One line per pair of residues and kind: the closest pair of atoms speaks for all of them.
        records = sorted(records, key=lambda r: r.record.distance if r.record.distance is not None else 99.0)
    records, stacked = _stacked_pairs(ctx, records)
    for r in records:
        assert r.ligand is not None
        site = r.protein
        # A contact without a direction is one with the residue, whatever is drawn of the residue.
        whole = r.type is T.HYDROPHOBIC
        res = ctx.dep.unit_for(r.ligand.atoms, whole) if ctx.dep.units else None
        # In a chain ligand the residue is what a contact lands on: two atoms of one residue are one place.
        where = ("unit", res.index) if res is not None else (r.ligand.kind, r.ligand.atoms)
        key = (r.type, site.kind, site.atoms, *where, r.record.predicted, r.absent)
        if id(r) in stacked:
            key = (*stacked[id(r)], r.record.predicted, r.absent)
        if res is not None and (r.type is T.HYDROPHOBIC or ctx.dep.pairwise):
            # ... and a hydrophobic contact is one between two residues, whichever of their atoms are close.
            # In the figure of an interface every contact is: a column of residues on one side leaves
            # no room, and no use, for three lines from three atoms of one residue into one box.
            key = (r.type, *where, r.record.predicted, r.absent)
        info = table.get(key)
        if info is not None:
            info.absorb(r)
            continue
        # A hydrophobic contact has no line: it is read off by proximity, spokes facing spokes. Beside a
        # column of residue boxes that does not work -- there is not room for everything beside its own
        # box, and a contact drawn two boxes away is read as a contact with the wrong residue. On a
        # chain ligand it therefore gets a line as well, a fine one without a label.
        drawn = r.type is not T.HYDROPHOBIC or res is not None
        label = None
        if r.type is T.WATER_BRIDGE:
            # The bridging water sits on the link, where a distance would -- and says so if it is a guess.
            label = PREDICTED_LABEL if r.record.predicted else "H\u2082O"
        elif r.type is T.HYDROPHOBIC:
            pass
        elif drawn and labelled(r.type) and r.record.distance is not None:
            label = f"{r.record.distance:.2f}"
        if r.absent and drawn:
            # No distance: the structure shown has none for it. The share of the models that make it
            # stands where a distance would.
            label = share_label(r.record.occupancy, label)
        plddt = contact_plddt(ctx, r)
        look = ctx.look(plddt)
        if drawn and look.mark and not r.record.predicted and r.type is not T.HYDROPHOBIC:
            # The lowest band carries a visible mark; it sits where a distance would,
            # so the layout reserves room for it like for any other label.
            label = f"{label} {MARK}" if label else MARK
        table[key] = PortInfo([r], r.type, site.kind, site.atoms,
                              ligand_target(ctx.dep, r.ligand, ctx.metrics, whole),
                              drawn, label, plddt, look, r.record.occupancy, pae=r.record.pae,
                              pae_draws=r.record.pae_draws, written=ctx.dep.pairwise and not ctx.glows,
                              glowing=ctx.glows, faint=grp.faint,
                              steps=ctx.options.occupancy_style == "steps")
    if ctx.options.occupancy_style == "steps":
        # The share of a contact made in fewer than all the models stands on its line, where a distance would.
        # Written once the links are merged: a link is drawn at the share of its most persistent record.
        for info in table.values():
            if info.occupancy is None or not info.drawn or info.absent or info.kind is T.HYDROPHOBIC:
                continue
            in_all = info.occupancy >= SHARE_WRITTEN_BELOW
            if not in_all or (ctx.shares_written and not (info.label and ctx.options.distance_labels == "all")):
                water = (PREDICTED_LABEL if info.predicted else "H\u2082O") if info.kind is T.WATER_BRIDGE else None
                text = share_label(info.occupancy, water)
                info.label = f"{text} {MARK}" if text and info.look.mark and not info.predicted else text

    def order(info: PortInfo) -> tuple:
        d = info.interactions[0].record.distance
        return (0 if info.drawn else 1, _PRIMARY_ORDER[info.kind], d if d is not None else 99.0,
                info.site_atoms, info.target.atoms)
    set_lanes(sorted((i for i in table.values() if i.drawn and i.target.unit is not None), key=order))
    return sorted(table.values(), key=order)


#: Distance between two links of one residue to one residue of a chain ligand, and the room there is for them.
_LANE = 8.0
_LANES_SPAN = 12.0


def set_lanes(infos: "Sequence[PortInfo]") -> None:
    """Give every drawn link its lane where several of them end on the same residue of a chain ligand."""
    by_unit: dict[int, list[PortInfo]] = {}
    for info in infos:
        assert info.target.unit is not None
        by_unit.setdefault(info.target.unit, []).append(info)
    for same in by_unit.values():
        if len(same) < 2:
            for info in same:                # alone after all (its companions were merged into it): back to the middle
                x, y = info.target.point
                info.target = replace(info.target, point=(x, y - info.lane))
                info.lane = 0.0
            continue
        step = min(_LANE, _LANES_SPAN / (len(same) - 1))
        for k, info in enumerate(same):
            lane = (k - (len(same) - 1) / 2.0) * step
            x, y = info.target.point
            info.target = replace(info.target, point=(x, y + lane - info.lane))
            info.lane = lane


def tag_slack(dep: Depiction, info: "PortInfo") -> float:
    """Extra length a line with a tag on it needs where it ends on a residue box of a chain.

    A line into a box is reckoned to end a fixed distance from the box's centre, which is right from
    above or below and too little from the side, where the box is widest. A distance label has always
    had room to spare; a tag is long enough to cover what is left of a short line. With this much more
    a piece of the line is seen on either side of it from any direction.
    """
    if info.tag is None or info.target.unit is None:
        return 0.0
    box = dep.units[info.target.unit].box
    return max(0.0, (box[2] - box[0]) / 2.0 - info.target.extent)


def label_size(text: str, font: float) -> tuple[float, float]:
    return (text_width(text, font) + 2.0, CAP_HEIGHT * font + 2.4)


def make_ports(ctx: Context, infos: Sequence[PortInfo], locate) -> list[Port]:
    """Engine ports for a residue. ``locate(info)`` gives, for the protein end of
    a link, ``(local point, ink radius, upright index, extent, clip)``."""
    m = ctx.metrics
    bl = m.bond_length
    ports = []
    n_sight = sum(1 for i in infos if not i.drawn) or 1
    for info in infos:
        local, port_radius, upright, extent, clip = locate(info)
        ideal, minimum = CHAIN_CONTACT_LENGTH if (info.kind is T.HYDROPHOBIC and info.drawn) \
            else VISIBLE_LENGTH[info.kind]
        label = label_size(info.shown, m.distance_font) if info.shown else None
        if label is not None and info.predicted:
            label = (label[0] + 8.0, label[1] + 4.0)      # room for the outline drawn around a predicted site
        ideal_px, min_px = ideal * bl, minimum * bl
        if label is not None:
            # A printed distance needs room whatever the bond length.
            ideal_px = max(ideal_px, label[0] + 24.0)
            min_px = max(min_px, label[0] + 13.0)
        ends = info.target.extent + extent + tag_slack(ctx.dep, info)
        ports.append(Port(
            local=local, target=info.target.point, ideal=ideal_px + ends,
            min_length=min_px + ends if info.drawn else 0.0,
            weight=SPRINGS["link"] if info.drawn else SPRINGS["sightline"] / n_sight, drawn=info.drawn,
            port_radius=port_radius, target_radius=info.target.radius, target_reach=info.target.extent,
            label_size=label,
            target_caps=info.target.caps, port_upright=upright,
            port_caps=clip[0] if clip else None, port_uprights=clip[1] if clip else ()))
    return ports


def interaction_data(ctx: Context, grp: ResidueGroup, info: PortInfo) -> dict[str, str]:
    """``data-*`` attributes that make an interaction addressable from a web page."""
    rec = info.interactions[0].record
    data = {"residue": grp.ident, NUMBER: grp.numbered, "type": info.kind.value,
            "ligand-atoms": ",".join(ctx.ligand.names[i] for i in info.target.atoms)}
    if info.target.unit is not None:
        # A residue of a chain ligand makes it: name the residue, and every atom of it that takes part.
        res = ctx.dep.units[info.target.unit]
        atoms = list(dict.fromkeys(a for r in info.interactions if r.ligand is not None for a in r.ligand.atoms))
        data["ligand-atoms"] = ",".join(ctx.ligand.names[i] for i in atoms)
        data["ligand-residue"] = f"{res.text}{res.number or res.position}"
        data["ligand-part"] = res.part(atoms).replace(" ", "-")
    if info.pae is not None:
        data["pae"] = f"{info.pae:.1f}"
        if len(info.pae_draws) > 1:
            data["pae-draws"] = " ".join(f"{v:.1f}" for v in info.pae_draws)
    if info.site_atoms:
        data["protein-atoms"] = ",".join(info.site_atoms)
    if rec.distance is not None:
        data["distance"] = f"{rec.distance:.2f}"
    if info.beyond:
        data["beyond"] = "true"
    if info.plddt is not None and info.look.band is not None:
        data["plddt"] = f"{info.plddt:.1f}"
        data["confidence"] = info.look.band
    if info.occupancy is not None:
        data["occupancy"] = f"{info.occupancy:.3f}"
    if info.predicted:
        data["predicted"] = "true"
    if info.absent:
        data["absent"] = "true"
    if grp.lost:
        data[other_protein.FROM] = other_protein.said_of(ctx.options)
    return data


# ---------------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LegendEntry:
    """One legend entry: ``sample`` says what is drawn in front of the text.

    ``link``: a line in ``style`` (with an arrowhead if ``arrow``); ``eyelash``;
    ``swatch``: a filled disc in ``token``; ``halo``: a soft disc in ``token``;
    ``scale``: a bar ``length`` page units long, for a figure in which a length means something;
    ``tag``: words that are written on lines of the figure (``marks``), as they are written there;
    ``lining``: a piece of the pocket's lining in ``token``; ``site``: a predicted water on its line;
    ``glow``: a piece of the glow of a contact's predicted aligned error in ``token``;
    ``ring``: the ring of a residue (:mod:`bindview.styles.rings`);
    ``marker``: a line on a stroke of the marker pen (:mod:`bindview.styles.absence`);
    ``text``: no sample at all.
    """

    label: str
    sample: str = "text"
    kind: InteractionType | None = None
    style: LinkStyle | None = None
    token: str | None = None
    arrow: bool = False
    stroke: str | None = None
    weight: float = 1.0             # ``weight`` sample: stroke width factor ...
    opacity: float = 1.0            # ... and opacity of a confidence band
    mark: str = ""                  # ... and the mark printed on it
    dash: tuple[float, ...] = ()    # ``node`` sample: outline dash (``weight`` is its width)
    length: float = 0.0             # ``scale`` sample: the length of the bar
    #: ``tag`` sample: words that are written on lines of the figure, each with whether it is bold there.
    marks: "tuple[tuple[str, bool], ...]" = ()


@dataclass
class StyleDrawing:
    """What a style hands back: items by layer (bottom first) and bookkeeping."""

    layers: dict[str, list[Item]]
    legend: list[LegendEntry]
    notes: list[str] = field(default_factory=list)
    has_distances: bool = False
    levels: set[int] = field(default_factory=set)      # confidence levels that occur (0 = unqualified)
    occupancy: bool = False                            # some contact carries an occupancy
    predicted: bool = False                            # some contact runs through a predicted solvent site
    #: For every PAE tag that was written on a line: is it bold? What the legend's key of a tag follows.
    tags: list[bool] = field(default_factory=list)
    #: Was a residue beside the ligand drawn with atoms at a level of detail that was asked for? Then
    #: its atoms carry confidence of their own, and the figure has to say on what basis.
    atoms_at_level: bool = False
    #: Was the fragment of a residue drawn with every atom at its own pLDDT without a level of detail
    #: (``plddt_by_atom``)? The figure then says so, as it does for a level.
    atoms_by_own: bool = False
    #: Does a contact stand on a glow for its predicted aligned error?
    glows: bool = False


@dataclass
class Drawn:
    """What drawing the residues of a figure gathers: the items, and what the key has to name."""

    layers: dict[str, list[Item]]
    kinds: list[InteractionType] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)       # residue classes that colour a bubble
    halo_done: set[tuple] = field(default_factory=set)
    dots_done: set[tuple] = field(default_factory=set)
    hbonds: set[str] = field(default_factory=set)          # backbone | sidechain | base, where they are told apart
    lashes: bool = False                                   # a hydrophobic contact is an eyelash or a spoke
    whole: bool = False                                    # ... or a line into a residue of a chain ligand
    has_distances: bool = False
    beyond: bool = False
    occupancy: bool = False
    predicted: bool = False
    absent: bool = False
    lost: bool = False                                     # a residue that the protein compared with does not have
    levels: set[int] = field(default_factory=set)
    tags: list[bool] = field(default_factory=list)
    glows: bool = False                                    # a contact stands on a glow for its PAE
    atoms_at_level: bool = False
    atoms_by_own: bool = False
    claims: "list[LiningClaim]" = field(default_factory=list)
    rings: bool = False                                    # a residue stands in a ring (``Context.shares``)
    #: Where bubbles stand, with their radius and the colour they are filled with (``lining="nearest"``).
    placed: list[tuple[Point, float, str]] = field(default_factory=list)


_CLASS_LABEL = {
    "hydrophobic": "Hydrophobic", "polar": "Polar", "positive": "Charged (+)", "negative": "Charged (−)",
    "glycine": "Glycine", "nucleotide": "Nucleotide", "metal": "Metal", "water": "Water", "other": "Other",
}
_CLASS_ORDER = ["hydrophobic", "polar", "positive", "negative", "glycine", "nucleotide", "metal", "water", "other"]


def finish_drawing(ctx: Context, settings: "Settings", d: Drawn) -> StyleDrawing:
    """What every style does once its residues are drawn: the backdrop (exposure, lining) and the key."""
    back: list[Item] = []
    if settings.exposure:
        back.extend(exposure_items(ctx))
    faces: set[str] = set()
    if settings.lining == "contacts":
        items, faces = lining_items(ctx, d.claims)
        back.extend(items)
    elif settings.lining == "nearest":
        back.extend(ribbon_items(ctx, d.placed))
    if back:
        d.layers["backdrop"] = back + d.layers.get("backdrop", [])
    styles = settings.lines.styles
    legend = [LegendEntry(_CLASS_LABEL[c], "swatch", token=f"class_{c}", stroke="bubble_line")
              for c in _CLASS_ORDER if c in d.classes]
    if d.rings:
        legend.append(LegendEntry(f"Ring: share of {ctx.ensemble or 'the models'} that contact the residue", "ring"))
    kinds = list(d.kinds)
    if settings.lines.sidechain is not None:
        if "backbone" in d.hbonds:
            legend.append(LegendEntry("H-bond (backbone)", "link", kind=T.HBOND, style=styles[T.HBOND], arrow=True))
        if d.hbonds & {"sidechain", "base"}:
            words = ("side chain" if "base" not in d.hbonds else "base" if "sidechain" not in d.hbonds
                     else "side chain or base")
            legend.append(LegendEntry(f"H-bond ({words})", "link", kind=T.HBOND, style=settings.lines.sidechain,
                                      arrow=True))
        kinds = [k for k in kinds if k is not T.HBOND]
    legend.extend(link_legend(kinds, styles))
    if d.whole:
        legend.append(LegendEntry(T.HYDROPHOBIC.label, "link", kind=T.HYDROPHOBIC, style=styles[T.HYDROPHOBIC]))
    if d.lashes:
        legend.append(LegendEntry(T.HYDROPHOBIC.label, "eyelash"))
    if ctx.options.charge_key:
        legend.extend(charge_legend({key[1] for key in d.halo_done}, styles))
    if settings.exposure and ctx.exposure is not None and any(e >= EXPOSED_FROM for e in ctx.exposure):
        legend.append(LegendEntry("Solvent exposure", "halo", token="exposure"))
    legend.extend(lining_legend(faces, "Lining:" if ctx.dep.pairwise else "Pocket lining:"))
    if d.beyond:
        legend.append(LegendEntry("† outside the supplied contact set"))
    legend.extend(predicted_legend(d.predicted, styles, ctx.options.solvent_mark))
    legend.extend(confidence_legend(d.levels))
    legend.extend(pae_legend(d.glows))
    legend.extend(occupancy_legend(ctx.ensemble, d.occupancy, ctx.options.occupancy_style == "steps"))
    legend.extend(absent_legend(d.absent))
    if d.lost:
        words = other_protein.legend_text(ctx.options)
        legend.append(LegendEntry(words, "weight", opacity=ABSENT_OPACITY)
                      if ctx.options.absent_style == "faint" else LegendEntry(words, "marker"))
    return StyleDrawing(d.layers, legend, d.notes, d.has_distances, d.levels, d.occupancy, d.predicted, d.tags,
                        d.atoms_at_level, atoms_by_own=d.atoms_by_own, glows=d.glows)


# ---------------------------------------------------------------------------
# Behind the ligand: solvent exposure and the lining of the pocket
# ---------------------------------------------------------------------------

#: An atom of the ligand that is open to solvent by this share or more has a halo.
EXPOSED_FROM = 0.25


def exposure_items(ctx: Context) -> list[Item]:
    """Grey halos behind ligand atoms that are open to solvent."""
    if ctx.exposure is None:
        return []
    out: list[Item] = []
    bl = ctx.metrics.bond_length
    for i, e in enumerate(ctx.exposure[: ctx.dep.n_heavy]):
        if e < EXPOSED_FROM or (ctx.dep.units and ctx.dep.home_of(i) is not None):
            continue
        x, y = ctx.dep.pos(i)
        out.append(Circle(x, y, (0.30 + 0.30 * min(1.0, float(e))) * bl, Paint(fill="exposure", opacity=0.55)))
    for res in ctx.dep.units:
        # A residue of a chain ligand: one halo behind its box, as wide as its atoms are exposed on average.
        values = [float(ctx.exposure[a]) for a in res.atoms if a < len(ctx.exposure)]
        e = sum(values) / len(values) if values else 0.0
        if e < EXPOSED_FROM:
            continue
        pad = (0.08 + 0.12 * min(1.0, e)) * bl
        x0, y0, x1, y1 = res.box
        out.append(Rect(x0 - pad, y0 - pad, x1 - x0 + 2.0 * pad, y1 - y0 + 2.0 * pad,
                        Paint(fill="exposure", opacity=0.55), rx=5.0 + pad))
    return out


#: How far from the ligand's ink the lining runs, and how far along it a contact's stretch reaches to either
#: side (in bond lengths); how near the middles of two stretches of different colours may be; and its line.
LINING_OFF, LINING_HALF, LINING_APART = 0.62, 0.7, 0.5
LINING_WIDTH, LINING_OPACITY = 4.4, 0.62
_LINING_STEP = 3.0


@dataclass(frozen=True)
class LiningClaim:
    """One contact's stretch of the lining: from where (the ligand's end) to where (the residue's) it lies,
    the colour of what makes it, and who makes it."""

    point: Point
    toward: Point
    token: str
    residue: str
    label: str


def contact_face(r: ResolvedInteraction) -> str:
    """What part of the residue makes a contact, as a class of surface: ``hydrophobic`` (a carbon, the face of
    a ring), ``polar`` (a nitrogen, oxygen or sulphur), ``positive`` or ``negative`` (a charged group),
    ``metal`` or ``water``; the class of the residue where the record does not say which part."""
    name, site = r.record.residue_name, r.protein
    cls = residue_class(name)
    if cls in ("metal", "water"):
        return cls
    if r.type is T.HYDROPHOBIC or site.kind == "ring":
        return "hydrophobic"
    signs = {sign for sign, _ in charged_groups(name)}
    if site.kind == "group" or r.type is T.SALTBRIDGE:
        sign = site.charge or (signs.pop() if len(signs) == 1 else 0)
        return "positive" if sign > 0 else "negative" if sign < 0 else cls
    if site.kind == "atom" and site.atoms:
        atom = site.atoms[0]
        element = atom.lstrip("0123456789")[:1].upper()
        if element == "C":
            return "hydrophobic"
        if cls in ("positive", "negative") and not is_backbone_atom(name, atom):
            return cls
        if is_nucleotide(name) and atom.upper() in ("OP1", "OP2", "O1P", "O2P"):
            return "negative"
        return "polar"
    return cls


def lining_claims(ctx: Context, grp: "ResidueGroup", look: Look, infos: "Sequence[PortInfo]",
                  ends: Sequence[Point]) -> list[LiningClaim]:
    """The stretches of the lining that a residue's contacts ask for (``ends``: where each leaves it)."""
    out: list[LiningClaim] = []
    for info, end in zip(infos, ends):
        if info.predicted or info.absent or info.faint or info.kind is T.COVALENT:
            continue          # a guess, a contact that is not there or is set back, and a bond, which is no wall
        for face in dict.fromkeys(contact_face(r) for r in info.interactions):
            token = ctx.fill_of(look, f"class_{face}")
            out.append(LiningClaim(info.target.point, end, token, grp.ident, grp.label))
    return out


def _outline(ctx: Context) -> list[np.ndarray]:
    """The outline of the ligand that the lining runs along: closed curves, smooth, in even steps."""
    assert ctx.field is not None
    loops = ctx.field.contours(LINING_OFF * ctx.metrics.bond_length)
    if not loops:
        return []
    sign = polygon_area(loops[0]) >= 0
    out = []
    for loop in loops:
        if (polygon_area(loop) >= 0) != sign or len(loop) < 12:
            continue                      # holes (the inside of a macrocycle) carry none
        pts = resample_closed(loop, _LINING_STEP)
        for _ in range(3):                # take the staircase out of the grid contour
            pts = (np.roll(pts, 1, axis=0) + pts + np.roll(pts, -1, axis=0)) / 3.0
        out.append(pts)
    return out


def _leaves_at(loops: Sequence[np.ndarray], a: Point, b: Point) -> tuple[int, float]:
    """Where the line from ``a`` (in the ligand) to ``b`` leaves the outline: which curve, and where on it.
    The first crossing from ``a``; where it crosses none, the place of the outline nearest to the line."""
    A, d = np.array(a, dtype=float), np.array(b, dtype=float) - np.array(a, dtype=float)
    best: tuple[float, int, float] | None = None
    for k, pts in enumerate(loops):
        e = np.roll(pts, -1, axis=0) - pts
        den = d[0] * e[:, 1] - d[1] * e[:, 0]
        ok = np.abs(den) > 1e-9
        w = pts - A
        safe = np.where(ok, den, 1.0)
        t = (w[:, 0] * e[:, 1] - w[:, 1] * e[:, 0]) / safe
        u = (w[:, 0] * d[1] - w[:, 1] * d[0]) / safe
        hit = np.nonzero(ok & (t >= 0.0) & (t <= 1.0) & (u >= 0.0) & (u < 1.0))[0]
        if len(hit):
            i = int(hit[np.argmin(t[hit])])
            found = (float(t[i]), k, i + float(u[i]))
            if best is None or found < best:
                best = found
    if best is not None:
        return best[1], best[2]
    near: tuple[float, float, int, int] | None = None
    for k, pts in enumerate(loops):
        w = pts - A
        along = np.clip((w @ d) / max(float(d @ d), 1e-9), 0.0, 1.0)
        off = np.linalg.norm(w - along[:, None] * d, axis=1)
        i = int(np.argmin(np.round(off, 6) + 1e-7 * np.linalg.norm(w, axis=1)))
        cand = (round(float(off[i]), 6), float(np.linalg.norm(w[i])), k, i)
        if near is None or cand < near:
            near = cand
    assert near is not None
    return near[2], float(near[3])


def lining_items(ctx: Context, claims: Sequence[LiningClaim]) -> tuple[list[Item], set[str]]:
    """The lining of the pocket from the contacts (docs/reference.md): a stretch of the ligand's outline for each,
    where the line to the residue crosses it, in the colour of what makes it. Returns the items, and the
    colours that are drawn."""
    if ctx.field is None or not claims or ctx.dep.arranged == "plane":
        return [], set()                  # boxes where residues lie have no outline: there is no pocket round them
    loops = _outline(ctx)
    if not loops:
        return [], set()
    bl = ctx.metrics.bond_length
    half, apart = LINING_HALF * bl, LINING_APART * bl
    at: dict[int, list[tuple[float, LiningClaim]]] = {}
    for claim in claims:
        k, where = _leaves_at(loops, claim.point, claim.toward)
        at.setdefault(k, []).append((where, claim))
    out: list[Item] = []
    drawn: set[str] = set()
    for loop in sorted(at):
        pts = loops[loop]
        n = len(pts)
        step = float(np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1).sum()) / n
        places = sorted(at[loop], key=lambda c: (c[0], c[1].residue, c[1].token))
        mids = np.array([c[0] for c in places])
        for _ in range(4):                # two colours in one place: moved apart, so that each is seen
            for i in range(len(places) if len(places) > 2 else len(places) - 1):
                j = (i + 1) % len(places)
                if places[i][1].token == places[j][1].token:
                    continue
                gap = float(mids[j] - mids[i]) + (n if j < i else 0.0)
                short = apart / step - gap
                if short > 0.0:
                    mids[i] -= short / 2.0
                    mids[j] += short / 2.0
        steps = np.arange(n)[:, None]
        off = np.abs(steps - mids[None, :]) % n
        off = np.minimum(off, n - off) * step
        nearest = np.argmin(off, axis=1)
        held = off[np.arange(n), nearest] <= half
        label = [places[j][1].token if h else "" for j, h in zip(nearest, held)]
        start = next((i for i in range(n) if label[i] != label[i - 1]), 0)
        i = 0
        while i < n:
            token = label[(start + i) % n]
            run = [(start + i) % n]
            i += 1
            while i < n and label[(start + i) % n] == token:
                run.append((start + i) % n)
                i += 1
            if not token or len(run) < 2:
                continue
            whose = list(dict.fromkeys(places[nearest[q]][1] for q in run))
            if len(run) == n:
                run.append(run[0])        # one colour all the way round: close the curve
            drawn.add(token)
            face = token.removeprefix("class_")
            names = list(dict.fromkeys(c.label for c in whose))
            g = Group("lining", data={"surface": face,
                                      "residues": " ".join(dict.fromkeys(c.residue for c in whose))},
                      title=("Lining" + (f", {_LINING_LABEL.get(face, face)}" if token.startswith("class_") else "")
                             + ": " + ", ".join(names)))
            g.add(Polyline(tuple((float(pts[q, 0]), float(pts[q, 1])) for q in run),
                           Paint(stroke=token, width=LINING_WIDTH, opacity=LINING_OPACITY, cap="round")))
            out.append(g)
    return out, drawn


_LINING_LABEL = {"hydrophobic": "hydrophobic", "polar": "polar", "positive": "charged (+)",
                 "negative": "charged (−)", "metal": "metal", "water": "water", "nucleotide": "nucleotide",
                 "glycine": "glycine", "other": "other"}


def lining_legend(faces: "set[str]", heading: str = "Pocket lining:") -> list[LegendEntry]:
    """The key of the lining: what each colour that is drawn says of the part of the pocket that is there.
    (Where the figure colours by pLDDT the lining does too, and that key is the figure's.)"""
    named = [c for c in _CLASS_ORDER if f"class_{c}" in faces]
    if not named:
        return []
    return [LegendEntry(heading)] + [LegendEntry(_LINING_LABEL[c], "lining", token=f"class_{c}")
                                              for c in named]


def ribbon_items(ctx: Context, placed: Sequence[tuple[Point, float, str]]) -> list[Item]:
    """The lining that follows the nearest residue (``lining="nearest"``): each stretch in the colour of the
    residue nearest to it (``placed`` holds that colour as its token) and left open where none is near."""
    if ctx.field is None or not placed:
        return []
    if ctx.dep.arranged == "plane":
        return []                         # boxes where residues lie have no outline: there is no pocket round them
    m = ctx.metrics
    centres = np.array([p[0] for p in placed], dtype=float)
    radii = np.array([p[1] for p in placed], dtype=float)
    out: list[Item] = []
    for pts in _outline(ctx):
        d = np.linalg.norm(pts[:, None, :] - centres[None, :, :], axis=2) - radii[None, :]
        nearest = np.argmin(d, axis=1)
        near = d[np.arange(len(pts)), nearest] <= 1.25 * m.bond_length
        label = np.where(near, nearest, -1)
        cls = [placed[j][2] if j >= 0 else "" for j in label]
        cls = _despeckle(cls, 5)
        n = len(pts)
        # Walk the closed loop in runs of equal class, starting at a class boundary.
        start = next((k for k in range(n) if cls[k] != cls[k - 1]), 0)
        k = 0
        while k < n:
            i0 = (start + k) % n
            c = cls[i0]
            run = [i0]
            k += 1
            while k < n and cls[(start + k) % n] == c:
                run.append((start + k) % n)
                k += 1
            if not c or len(run) < 2:
                continue
            if len(run) == n:
                run.append(run[0])        # one class all the way round: close the loop
            out.append(Polyline(tuple((float(pts[i, 0]), float(pts[i, 1])) for i in run),
                                Paint(stroke=c, width=LINING_WIDTH, opacity=LINING_OPACITY, cap="round")))
    return out


def _despeckle(values: list[str], min_run: int) -> list[str]:
    """Absorb runs shorter than ``min_run`` (on a closed loop) into the preceding run."""
    n = len(values)
    if n == 0:
        return values
    out = list(values)
    for _ in range(2):
        start = next((k for k in range(n) if out[k] != out[k - 1]), None)
        if start is None:
            return out
        k = 0
        while k < n:
            i0 = (start + k) % n
            j = k
            while j < n and out[(start + j) % n] == out[i0]:
                j += 1
            if j - k < min_run:
                prev = out[(i0 - 1) % n]
                for t in range(k, j):
                    out[(start + t) % n] = prev
            k = j
    return out


_LEGEND_ORDER = [T.COVALENT, T.HBOND, T.SALTBRIDGE, T.PI_STACKING, T.PI_CATION, T.HALOGEN, T.METAL, T.WATER_BRIDGE]


def link_legend(kinds: Sequence[InteractionType], styles: "Mapping[InteractionType, LinkStyle]"
                ) -> list[LegendEntry]:
    return [LegendEntry(k.label, "link", kind=k, style=styles[k], arrow=k is T.HBOND)
            for k in _LEGEND_ORDER if k in kinds]


def charge_legend(shaded: "set[str]", styles: "Mapping[InteractionType, LinkStyle]") -> list[LegendEntry]:
    """The key of the shading of a charged group (``charge_halos``): one entry for each kind of contact
    whose colour shades a group in the figure (``shaded``: the tokens drawn), named by the kind where
    there are two."""
    kinds = [k for k in (T.SALTBRIDGE, T.PI_CATION) if k in styles and styles[k].token in shaded]
    return [LegendEntry("Charged group" if len(kinds) == 1 else f"Charged group ({k.label.lower()})", "halo",
                        token=styles[k].token) for k in kinds]


def confidence_legend(levels: "set[int]") -> list[LegendEntry]:
    """Legend samples for the confidence bands, when anything is drawn below full strength."""
    if not any(lv > 0 for lv in levels):
        return []
    out = [LegendEntry("Confidence:")]
    for band, text in (("confident", "pLDDT \u2265 70"), ("low", "50\u201370"), ("very_low", "< 50")):
        look = LOOKS[band]
        out.append(LegendEntry(text, "weight", weight=look.width, opacity=look.opacity,
                               mark=MARK if look.mark else ""))
    return out


def pae_legend(present: bool) -> list[LegendEntry]:
    """The key of the glow of a contact's predicted aligned error: its four bands, all of them."""
    if not present:
        return []
    out = [LegendEntry("PAE of the contact:")]
    bounds = [least for _, least in PAE_BANDS]
    for k, (band, least) in enumerate(PAE_BANDS):
        text = (f"< {bounds[1]:g} \u00c5" if k == 0 else f"\u2265 {least:g} \u00c5" if k == len(PAE_BANDS) - 1 else
                f"{least:g}\u2013{bounds[k + 1]:g} \u00c5")
        out.append(LegendEntry(text, "glow", token=f"plddt_{band}"))
    return out


def plddt_legend() -> list[LegendEntry]:
    """The key of a figure that colours by pLDDT: AlphaFold's four bands, all of them, so that the
    key is the same on every figure."""
    out = [LegendEntry("pLDDT:")]
    for band, text in (("very_high", "\u2265 90"), ("confident", "70\u201390"), ("low", "50\u201370"),
                       ("very_low", "< 50")):
        out.append(LegendEntry(text, "swatch", token=f"plddt_{band}", stroke="bubble_line"))
    return out


def predicted_style(base: LinkStyle) -> LinkStyle:
    """How a line through a predicted solvent site is drawn: fine round dots, in the water colour
    of the style. A different kind of line from an observed water bridge, not a paler one."""
    return LinkStyle(base.token, (0.01, 3.4), width=1.25, cap="round")


def _predicted_box(a: Point, b: Point, label: str, label_at: float, metrics: Metrics
                   ) -> tuple[float, float, float, float]:
    mid = lerp(a, b, label_at)
    w = text_width(label, metrics.distance_font) + 9.0
    h = CAP_HEIGHT * metrics.distance_font + 7.0
    return mid[0] - w / 2.0, mid[1] - h / 2.0, w, h


def predicted_items(a: Point, b: Point, info: "PortInfo", label_at: float, metrics: Metrics, style: LinkStyle,
                    opacity: float, mark: str) -> list[Item]:
    """The line through a predicted site, with its label set off as a node of its own, the way LigPlot sets a
    water between ligand and residue, and unlike anything an observed water gets: on a glow (``mark="glow"``), in a
    filled box with a heavy broken border (``"filled"``), or in a thin broken outline (``"outline"``)."""
    label = info.label or ""
    line = link_items(a, b, info.kind, metrics, label=info.label, opacity=opacity, label_at=label_at,
                      weight=info.weight, style=style)
    x, y, w, h = _predicted_box(a, b, label, label_at, metrics)
    if mark == "outline":
        return line + [Rect(x, y, w, h, Paint(stroke=style.token, width=0.85, dash=(2.2, 1.7), opacity=opacity),
                            rx=h / 2.0)]
    if mark == "glow":
        return water_glow(x, y, w, h, style.token, opacity) + line
    # The fill lies under the words, the border over the line.
    return [Rect(x, y, w, h, Paint(fill=style.token, opacity=0.24 * opacity), rx=h / 2.0)] + line + [
        Rect(x, y, w, h, Paint(stroke=style.token, width=1.9, dash=(4.0, 1.8), opacity=opacity), rx=h / 2.0)]


#: The glow of a predicted water, from the outside in: how far each layer reaches beyond its words, and how strong
#: it is. (Soft edges without a filter, which not every reader of an SVG has.)
_WATER_GLOW = ((6.4, 0.07), (4.3, 0.09), (2.2, 0.12), (0.0, 0.16))


def water_glow(x: float, y: float, w: float, h: float, token: str, opacity: float = 1.0) -> list[Item]:
    return [Rect(x - g, y - g, w + 2.0 * g, h + 2.0 * g, Paint(fill=token, opacity=o * opacity), rx=h / 2.0 + g)
            for g, o in _WATER_GLOW]


def predicted_legend(present: bool, styles: "Mapping[InteractionType, LinkStyle]", mark: str
                     ) -> list[LegendEntry]:
    if not present:
        return []
    return [LegendEntry("Predicted water site (not observed)", {"outline": "link", "glow": "site-glow"}.get(mark, "site"),
                        kind=T.WATER_BRIDGE, style=predicted_style(styles[T.WATER_BRIDGE]))]


#: The most a contact is drawn at that the structure shown does not make.
ABSENT_OPACITY = 0.45


def is_distance(label: "str | None") -> bool:
    """Is what stands on a line a distance (and not a share, the word of a water, a mark)?"""
    return label is not None and label[:1].isdigit() and "%" not in label


def share_label(occupancy: "float | None", label: "str | None" = None) -> "str | None":
    """What stands on the line of a contact that only other models make: the share of the models that
    make it, after whatever stood there already (the water of a water bridge)."""
    if occupancy is None:
        return label
    share = f"{100.0 * occupancy:.0f} %"
    return f"{label} {share}" if label else share


def absent_legend(present: bool) -> list[LegendEntry]:
    """The key of a contact that other models of the ensemble make and the one shown does not."""
    if not present:
        return []
    return [LegendEntry("Faint, with a share: made by other models, not by the one shown", "weight",
                        weight=occupancy_factors(0.5)[0], opacity=ABSENT_OPACITY)]


def occupancy_legend(ensemble: str | None, present: bool, steps: bool) -> list[LegendEntry]:
    """Legend samples for occupancy, naming the ensemble it is a share of: its three steps, where it is drawn
    in steps (and then the share is written on the line, as the heading says), else three shares."""
    if not present:
        return []
    of = ensemble or "the ensemble"
    if steps:
        out = [LegendEntry(f"Occupancy (share of {of}, written on the line):")]
        bounds = [least for least, _, _ in OCCUPANCY_STEPS]
        names = [f"{100 * bounds[0]:.0f}\u2013100 %"] + [f"{100 * lo:.0f}\u2013{100 * hi:.0f} %" for lo, hi in
                                                         zip(bounds[1:-1], bounds[:-2])] + [f"< {100 * bounds[-2]:.0f} %"]
        for name, (_, width, opacity) in zip(names, OCCUPANCY_STEPS):
            out.append(LegendEntry(name, "weight", weight=width, opacity=opacity))
        return out
    out = [LegendEntry(f"Occupancy (share of {of}):")]
    for value in OCCUPANCY_SAMPLES:
        width, opacity = occupancy_factors(value)
        out.append(LegendEntry(f"{100.0 * value:.0f} %", "weight", weight=width, opacity=opacity))
    return out


#: Between two samples of one key (a plain tag and a bold one).
_KEY_GAP = 6.0


def legend_group(entries: Sequence[LegendEntry], metrics: Metrics, max_width: float) -> tuple[Group, float, float]:
    """A wrapped row of legend entries. Returns ``(group, width, height)`` with the
    group's origin at its top-left corner."""
    g = Group("legend")
    font = metrics.legend_font
    spacing = 18.0
    line_h = font * 1.75
    x, y = 0.0, 0.0
    widest = 0.0

    def size(e: LegendEntry) -> tuple[float, float, float]:
        """The width of an entry's sample, of the gap after it, and of the whole entry."""
        sample = {"link": 26.0, "eyelash": 26.0, "swatch": 11.0, "halo": 13.0, "weight": 26.0,
                  "node": 13.0, "unit": 22.0, "lining": 18.0, "glow": 20.0, "site": 38.0, "site-glow": 38.0,
                  "ring": 15.0, "marker": 26.0}.get(e.sample, 0.0)
        if e.sample == "tag":
            # Words that are written in the figure, shown as they are written there: plain, bold, or one of each.
            sample = sum(text_width(words, metrics.distance_font, bold=bold) for words, bold in e.marks) \
                + _KEY_GAP * max(len(e.marks) - 1, 0)
        if e.sample == "scale":
            sample = e.length
        gap = 8.0 if e.sample in ("link", "eyelash", "weight", "scale", "site", "site-glow", "marker") else (5.0 if sample else 0.0)
        return sample, gap, sample + gap + text_width(e.label, font)

    sizes = [size(e) for e in entries]
    for k, e in enumerate(entries):
        sample, gap, w = sizes[k]
        # A heading ("Confidence:") goes with the first entry after it: it is not left at the end of a row.
        heading = e.sample == "text" and e.label.endswith(":") and k + 1 < len(entries)
        need = w + (spacing + sizes[k + 1][2] if heading else 0.0)
        if x > 0 and x + need > max_width:
            x, y = 0.0, y + line_h
        cy = y + line_h / 2.0
        if e.sample == "eyelash":
            centre = (x + sample / 2.0, cy + 5.0)
            g.extend(eyelash_items(centre, 9.0, [(centre[0], centre[1] - 50.0)], metrics.with_(lash_length=4.0)))
        elif e.sample == "link" and e.kind is not None:
            a, b = (x, cy), (x + sample, cy)
            st = e.style or LINK_STYLES[e.kind]
            g.extend(link_items(a, b, e.kind, metrics, arrow="b" if e.arrow else None, style=st))
            if st.end_dots:
                g.add(Circle(b[0], b[1], 2.4, Paint(fill=st.token)))
        elif e.sample == "lining" and e.token:
            g.add(Line(x + 2.2, cy, x + sample - 2.2, cy,
                       Paint(stroke=e.token, width=LINING_WIDTH, opacity=LINING_OPACITY, cap="round")))
        elif e.sample == "glow" and e.token:
            width = GLOW * metrics.interaction_width
            g.add(Line(x + width / 2.0, cy, x + sample - width / 2.0, cy,
                       Paint(stroke=e.token, width=width, opacity=_GLOW_OPACITIES.get(e.token, GLOW_OPACITY),
                             cap="round")))
        elif e.sample == "site-glow" and e.style is not None and e.kind is not None:
            # A predicted water: its glow on its line, as it stands in the figure.
            g.extend(water_glow(x + 13.0, cy - 2.6, sample - 26.0, 5.2, e.style.token))
            g.extend(link_items((x, cy), (x + sample, cy), e.kind, metrics, style=e.style))
        elif e.sample == "site" and e.style is not None and e.kind is not None:
            # A predicted water: its box on its line, as it stands in the figure.
            box = (x + 11.0, cy - 5.0, sample - 22.0, 10.0)
            g.add(Rect(*box, Paint(fill=e.style.token, opacity=0.24), rx=5.0))
            g.extend(link_items((x, cy), (x + 11.0, cy), e.kind, metrics, style=e.style))
            g.extend(link_items((x + sample - 11.0, cy), (x + sample, cy), e.kind, metrics, style=e.style))
            g.add(Rect(*box, Paint(stroke=e.style.token, width=1.9, dash=(4.0, 1.8)), rx=5.0))
        elif e.sample == "weight":
            paint = Paint(stroke="text", width=metrics.interaction_width * 1.3 * e.weight, opacity=e.opacity,
                          cap="butt")
            if e.mark:
                mid = x + sample / 2.0
                mw = text_width(e.mark, metrics.distance_font) / 2.0 + 2.0
                g.add(Line(x, cy, mid - mw, cy, paint), Line(mid + mw, cy, x + sample, cy, paint))
                g.add(Text.plain(mid, cy + CAP_HEIGHT * metrics.distance_font / 2.0, e.mark, metrics.distance_font,
                                 "text", opacity=e.opacity))
            else:
                g.add(Line(x, cy, x + sample, cy, paint))
        elif e.sample == "marker":
            g.add(Group("", [other_protein.stroke([(x + 4.0, cy), (x + sample - 4.0, cy)], metrics.interaction_width)],
                        opacity=other_protein.OPACITY))
            g.add(Line(x, cy, x + sample, cy, Paint(stroke="text", width=metrics.interaction_width, dash=(4.2, 2.6),
                                                   cap="butt")))
        elif e.sample == "node":
            g.add(Circle(x + sample / 2.0, cy, sample / 2.0 - e.weight / 2.0,
                         Paint(fill=e.token, stroke=e.stroke or "bubble_line", width=e.weight, dash=e.dash,
                               opacity=e.opacity)))
        elif e.sample == "ring":
            g.extend(ring_sample(x + sample / 2.0, cy, sample / 2.0))
        elif e.sample == "unit":
            g.add(Rect(x + 0.6, cy - 5.5, sample - 1.2, 11.0, Paint(fill="background", stroke="ligand_bond", width=1.15),
                       rx=3.5))
        elif e.sample == "swatch" and e.token:
            g.add(Circle(x + sample / 2.0, cy, sample / 2.0,
                         Paint(fill=e.token, stroke=e.stroke, width=0.8) if e.stroke else Paint(fill=e.token)))
        elif e.sample == "halo" and e.token:
            g.add(Circle(x + sample / 2.0, cy, sample / 2.0, Paint(fill=e.token, opacity=0.6)))
        elif e.sample == "tag":
            at = x
            for words, bold in e.marks:
                g.add(Text.plain(at, cy + CAP_HEIGHT * metrics.distance_font / 2.0, words, metrics.distance_font,
                                 "text", anchor="start", bold=bold))
                at += text_width(words, metrics.distance_font, bold=bold) + _KEY_GAP
        elif e.sample == "scale":
            # A scale bar: a length of the figure, with a tick at either end.
            paint = Paint(stroke="text_muted", width=1.1, cap="butt")
            g.add(Line(x, cy, x + sample, cy, paint), Line(x, cy - 3.5, x, cy + 3.5, paint),
                  Line(x + sample, cy - 3.5, x + sample, cy + 3.5, paint))
        g.add(Text.plain(x + sample + gap, cy + CAP_HEIGHT * font / 2.0, e.label, font, "text_muted", anchor="start"))
        x += w + spacing
        widest = max(widest, x - spacing)
    return g, widest, (y + line_h if entries else 0.0)
