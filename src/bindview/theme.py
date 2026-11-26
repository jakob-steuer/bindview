"""Colours and dimensions.

Every colour the drawing code uses is a named *token* looked up on a
:class:`Palette`; no literal colour appears in what a style draws. (One is in
the rules of a figure that answers the pointer, :mod:`bindview.interactive`: the
ring around an atom that is pointed at, which is the same in every palette.) A
theme is a palette plus the dimensional :class:`Metrics`. Two palettes ship
(``light``, ``dark``); a caller can pass its own.

A *colour scheme* is a pair of palettes, one for each theme (:data:`SCHEMES`):
``standard``, which is those two, and ``contrast``, whose residue classes are
further apart for every kind of colour vision. The four colours of the pLDDT bands are AlphaFold's and belong
to no palette (:data:`PLDDT_COLORS`): they mean the same on every page.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from typing import Mapping

from .interactions import InteractionType

__all__ = ["Palette", "Metrics", "LIGHT", "DARK", "get_palette", "SCHEMES", "scheme",
           "CONTRAST", "CONTRAST_DARK", "EXTRAS", "PLDDT_COLORS", "PLDDT_TEXT", "fixed_tokens"]


@dataclass(frozen=True)
class Palette:
    """Named colours. Tokens are the attribute names; element and residue-class
    colours live in the two mappings."""

    name: str
    background: str
    text: str                    # residue names, captions
    text_muted: str              # secondary captions, footnotes
    ligand_bond: str
    ligand_carbon: str           # explicit carbon labels on the ligand
    protein_bond: str            # the "second colour": residue fragments
    protein_label: str           # residue names
    protein_atom: str            # atom-name labels on fragments that are not element-coloured
    hbond: str
    hydrophobic: str
    saltbridge: str
    pi_stacking: str
    pi_cation: str
    halogen: str
    metal: str
    water_bridge: str
    coordination: str            # intra-ligand metal coordination
    beyond: str                  # contacts flagged beyond the caller's contact set
    exposure: str                # solvent-exposure halo (schematic style)
    warning: str
    panel: str                   # legend / card background
    panel_border: str
    # Link colours of the schematic style, which follows the Maestro conventions.
    sch_hbond: str = "#c026d3"
    sch_saltbridge: str = "#2563eb"
    sch_pi_stacking: str = "#16a34a"
    sch_pi_cation: str = "#dc2626"
    sch_halogen: str = "#a16207"
    sch_metal: str = "#4b5563"
    sch_water_bridge: str = "#0891b2"
    bubble_text: str = "#1f2933"
    bubble_line: str = "#5b6472"
    elements: Mapping[str, str] = field(default_factory=dict)
    residue_classes: Mapping[str, str] = field(default_factory=dict)
    #: Colours for what few figures hold, by token (``class_nucleotide``). They are not among
    #: :meth:`tokens`: a figure that follows the reader's theme declares every token of its palette, whether
    #: it draws in that colour or not. These are declared by the figures that use them, and by no other.
    extra: Mapping[str, str] = field(default_factory=dict)

    def element(self, symbol: str, on_protein: bool = False) -> str:
        """The colour of an element's letter: of a carbon, the colour of the bonds it is drawn with."""
        if symbol == "C":
            return self.protein_bond if on_protein else self.ligand_carbon
        return self.elements.get(symbol, self.elements.get("*", self.text))

    def interaction(self, kind: str) -> str:
        """The colour of a kind of contact, by its name in a table (``"hbond"``, ``"pi_stacking"``)."""
        try:
            return str(getattr(self, InteractionType(kind).value))
        except (ValueError, AttributeError):
            raise ValueError(f"{kind!r} is no kind of contact that a palette has a colour for") from None

    def element_token(self, symbol: str, on_protein: bool = False) -> str:
        """The colour *token* for an element label (see :meth:`tokens`)."""
        if symbol == "C":
            return "protein_atom" if on_protein else "ligand_carbon"
        return f"element_{symbol}" if symbol in self.elements and symbol != "*" else "element_other"

    def residue_class(self, cls: str) -> str:
        """The colour of a class of residues."""
        if cls in self.residue_classes:
            return self.residue_classes[cls]
        return self.extra.get(f"class_{cls}", self.residue_classes.get("other", self.panel))

    def with_(self, **changes: object) -> "Palette":
        """The same palette with some of its colours changed."""
        return replace(self, **changes)  # type: ignore[arg-type]

    def tokens(self) -> dict[str, str]:
        """Flat ``token -> colour`` map (used for CSS-variable output)."""
        out = {f.name: getattr(self, f.name) for f in fields(self)
               if isinstance(getattr(self, f.name), str) and f.name != "name"}
        out.update({f"element_{k}": v for k, v in self.elements.items() if k != "*"})
        out["element_other"] = self.elements.get("*", self.text)
        out.update({f"class_{k}": v for k, v in self.residue_classes.items()})
        return out


LIGHT = Palette(
    name="light",
    background="#ffffff",
    text="#1f2933",
    text_muted="#667085",
    ligand_bond="#1f2933",
    ligand_carbon="#1f2933",
    protein_bond="#6b7a90",
    protein_label="#1f2933",
    protein_atom="#4b5768",
    hbond="#1a9641",
    hydrophobic="#c0392b",
    saltbridge="#7b3fa0",
    pi_stacking="#00838f",
    pi_cation="#d97706",
    halogen="#c2185b",
    metal="#37474f",
    water_bridge="#1e88e5",
    coordination="#8b95a5",
    beyond="#9aa4b2",
    exposure="#c9ced6",
    warning="#b45309",
    panel="#f6f7f9",
    panel_border="#d5dae1",
    elements={
        "N": "#2f5fd0", "O": "#d6332a", "S": "#a67c00", "P": "#d9730d",
        "F": "#2e9e44", "Cl": "#2e9e44", "Br": "#a0522d", "I": "#7b3fbf",
        "B": "#c77c5e", "Se": "#b8860b", "H": "#667085",
        "Fe": "#b5541a", "Co": "#c2417a", "Zn": "#5b6f8a", "Mg": "#3f8f3f", "Mn": "#8a5fb0",
        "Ca": "#4f8f4f", "Cu": "#b06a2c", "Ni": "#3f8a7a", "Na": "#7b5fc0", "K": "#7b5fc0",
        "*": "#5b6472",
    },
    residue_classes={
        "hydrophobic": "#b8dc7a", "polar": "#9fdcee", "positive": "#a9a6f0", "negative": "#f4a98a",
        "glycine": "#ecebd8", "metal": "#b9bec7", "water": "#dfe7f2", "other": "#d9dde3",
    },
    extra={"class_nucleotide": "#f3d477", "absent": "#facc15"},
)

DARK = Palette(
    sch_hbond="#e879f9", sch_saltbridge="#60a5fa", sch_pi_stacking="#4ade80", sch_pi_cation="#f87171",
    sch_halogen="#facc15", sch_metal="#cbd5e1", sch_water_bridge="#22d3ee",
    bubble_text="#f1f5f9", bubble_line="#94a3b8",
    name="dark",
    background="#0f172a",
    text="#e2e8f0",
    text_muted="#94a3b8",
    ligand_bond="#e2e8f0",
    ligand_carbon="#e2e8f0",
    protein_bond="#8fa0b8",
    protein_label="#e2e8f0",
    protein_atom="#a9b6c8",
    hbond="#4ade80",
    hydrophobic="#f87171",
    saltbridge="#c084fc",
    pi_stacking="#22d3ee",
    pi_cation="#fb923c",
    halogen="#f472b6",
    metal="#cbd5e1",
    water_bridge="#60a5fa",
    coordination="#7c8aa0",
    beyond="#64748b",
    exposure="#3b475c",
    warning="#fbbf24",
    panel="#172036",
    panel_border="#2b3956",
    elements={
        "N": "#7aa2f7", "O": "#f7768e", "S": "#e0af68", "P": "#ff9e64",
        "F": "#73daca", "Cl": "#73daca", "Br": "#d19a66", "I": "#bb9af7",
        "B": "#e0a58b", "Se": "#e0af68", "H": "#94a3b8",
        "Fe": "#f0a070", "Co": "#f28cb1", "Zn": "#9fb4d0", "Mg": "#86d086", "Mn": "#c4a3e8",
        "Ca": "#8fd08f", "Cu": "#e0a070", "Ni": "#7fd0c0", "Na": "#b9a3f0", "K": "#b9a3f0",
        "*": "#b6c0cf",
    },
    residue_classes={
        "hydrophobic": "#3f6212", "polar": "#155e75", "positive": "#3730a3", "negative": "#9a3412",
        "glycine": "#57534e", "metal": "#475569", "water": "#334155", "other": "#3b4658",
    },
    extra={"class_nucleotide": "#854d0e", "absent": "#a16207"},
)

# Navy text on white (#0f172a, #475569), with cyan and fuchsia (#0891b2, #d946ef) and the accents beside them
# (emerald, sky, indigo). What is told apart by colour alone here is the class of a residue, so the five
# fills were chosen by measurement and not by eye: of the tints of that family, those in which the worst of
# all pairs is further apart than in the standard palette, for full colour vision and for red-green and
# blue-yellow colour blindness (docs/reference.md has the numbers). Lines keep the colours of the standard
# palette: what a line means does not change with the scheme. What is drawn of a residue is no nearer to
# the colour of any line than the standard's grey is: on white that is a step of the blue, and on the dark
# ground, where the light blues are the colours of lines, the muted grey.
CONTRAST = LIGHT.with_(
    name="contrast",
    background="#ffffff", text="#0f172a", text_muted="#475569",
    ligand_bond="#0f172a", ligand_carbon="#0f172a",
    protein_bond="#0369a1", protein_label="#0f172a", protein_atom="#075985",
    coordination="#94a3b8", beyond="#94a3b8", exposure="#cbd5e1",
    panel="#f8fafc", panel_border="#cbd5e1",
    bubble_text="#0f172a", bubble_line="#64748b",
    residue_classes={
        "hydrophobic": "#10b981", "polar": "#67e8f9", "positive": "#a5b4fc", "negative": "#d946ef",
        "glycine": "#f1f5f9", "metal": "#cbd5e1", "water": "#e0f2fe", "other": "#e2e8f0",
    },
    extra={"class_nucleotide": "#fbbf24", "absent": "#facc15"},
)

CONTRAST_DARK = DARK.with_(
    name="contrast-dark",
    background="#0f172a", text="#f8fafc", text_muted="#94a3b8",
    ligand_bond="#f8fafc", ligand_carbon="#f8fafc",
    protein_bond="#94a3b8", protein_label="#f8fafc", protein_atom="#cbd5e1",
    coordination="#64748b", beyond="#64748b", exposure="#334155",
    panel="#1e293b", panel_border="#334155",
    bubble_text="#f8fafc", bubble_line="#94a3b8",
    residue_classes={
        "hydrophobic": "#064e3b", "polar": "#0e7490", "positive": "#3730a3", "negative": "#c026d3",
        "glycine": "#1e293b", "metal": "#475569", "water": "#0c4a6e", "other": "#334155",
    },
    extra={"class_nucleotide": "#b45309", "absent": "#a16207"},
)

#: The colour schemes: for each, the palette of the light theme and that of the dark one.
SCHEMES: dict[str, tuple[Palette, Palette]] = {
    "standard": (LIGHT, DARK),
    "contrast": (CONTRAST, CONTRAST_DARK),
}

#: What stands for a colour of :attr:`Palette.extra` in a palette of the caller's own that has none by that name.
EXTRAS: dict[str, str] = {"absent": "#facc15"}

#: AlphaFold's colours for the four bands of pLDDT, best first, as its database and viewers draw them.
#: They are the same in every scheme and theme: a reader knows them from there, and reads them as that.
PLDDT_COLORS: dict[str, str] = {"very_high": "#0053d6", "confident": "#65cbf3", "low": "#ffdb13",
                                "very_low": "#ff7d45"}
#: The colour of a name that stands on each of them.
PLDDT_TEXT: dict[str, str] = {"very_high": "#ffffff", "confident": "#1f2933", "low": "#1f2933",
                              "very_low": "#1f2933"}


def fixed_tokens() -> dict[str, str]:
    """The tokens that no palette holds, because they are the same in all of them: ``plddt_<band>`` and
    ``plddt_text_<band>``. A figure names them only where it was asked to colour by pLDDT."""
    out = {f"plddt_{band}": colour for band, colour in PLDDT_COLORS.items()}
    out.update({f"plddt_text_{band}": colour for band, colour in PLDDT_TEXT.items()})
    return out


def scheme(colors: str = "standard") -> tuple[Palette, Palette]:
    """The two palettes of a colour scheme, light and dark."""
    try:
        return SCHEMES[str(colors).lower()]
    except KeyError:
        known = ", ".join(repr(k) for k in SCHEMES)
        raise ValueError(f"unknown colours {colors!r}; the schemes are {known}") from None


def get_palette(theme: "str | Palette", colors: str = "standard") -> Palette:
    """The palette of a theme (``"light"`` or ``"dark"``) in a colour scheme. A :class:`Palette` of the
    caller's own is its colours already, and is returned as it is."""
    if isinstance(theme, Palette):
        if str(colors).lower() != "standard":
            raise ValueError(f"a Palette of your own is its colours: leave colors at 'standard', not {colors!r}")
        return theme
    light, dark = scheme(colors)
    name = str(theme).lower()
    if name == "light":
        return light
    if name == "dark":
        return dark
    if name == "auto":
        raise ValueError("the theme 'auto' is a figure with both palettes and no palette itself: ask for 'light' or "
                         "'dark' here, or for the two with scheme()")
    raise ValueError(f"unknown theme {theme!r}; use 'light', 'dark', 'auto' or a Palette")


@dataclass(frozen=True)
class Metrics:
    """Dimensions, in SVG user units (px at natural size). Everything scales
    from ``bond_length``; fonts are absolute because legibility is."""

    bond_length: float = 30.0
    bond_width: float = 1.7
    double_offset: float = 4.3          # distance between the lines of a double bond
    wedge_width: float = 6.2            # full width of a wedge at its wide end
    hash_spacing: float = 2.7
    label_gap: float = 2.2              # clearance between a bond end and an atom label
    atom_font: float = 13.5
    sub_font: float = 9.5               # hydrogen counts
    charge_font: float = 9.5
    protein_bond_width: float = 1.35
    protein_bond_scale: float = 0.82    # fragment bond length relative to the ligand's
    protein_atom_font: float = 9.5
    residue_font: float = 12.0
    interaction_width: float = 1.35
    distance_font: float = 9.5
    legend_font: float = 10.5
    title_font: float = 13.0
    caption_font: float = 10.5
    lash_length: float = 6.0
    spoke_inner: float = 6.5            # ligand-atom spokes start/end radius
    spoke_outer: float = 11.5
    margin: float = 18.0
    #: A ligand drawn as a chain: centre-to-centre distance of two of its residues, in bond lengths.
    #: A crowded chain is drawn with more (the residues around it need the room), never with less.
    chain_pitch: float = 0.9

    def with_(self, **changes: float) -> "Metrics":
        """The same dimensions with some of them changed."""
        return replace(self, **changes)
