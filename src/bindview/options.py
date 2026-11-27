"""Every keyword of ``render``, by what it is for: ``bindview.options()``.

One line for each, with its default. ``options("colour")`` gives those that have a word in their name or their line. The same
table is the help of :class:`~bindview.styles.base.RenderOptions` and the list of ``docs/api.md``; the comments on the fields of
that class say more of each.
"""

from __future__ import annotations

import dataclasses

from .styles.base import RenderOptions

__all__ = ["options", "GROUPS"]

#: ``(what they are for, ((keyword, what it does), ...))``: every field of ``RenderOptions`` once.
GROUPS: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
    ("How the residues and contacts are drawn", (
        ("residues", '"atoms", "bubbles" or "names", or residue by residue: {"Gln712": "atoms", "*": "bubbles"}; None: the style\'s'),
        ("lines", 'the colours and dashes of the contact lines: "detailed" or "schematic"; None: the style\'s'),
        ("lining", 'the wall of the pocket: "contacts", "nearest" or "none"; None: the style\'s'),
        ("exposure", "grey halos on the ligand's atoms that are open to solvent; None: the style's"),
        ("charge_halos", "shade the charged group that a salt bridge or a pi-cation contact is made with"),
        ("charge_key", "name that shading in the key"),
        ("break_lines", "break a line where it runs over a word that is not its own"),
        ("atom_labels", 'what is written on an atom of a residue: "elements" (N, O), or "names", as in the file (NZ, OE1)'),
        ("distance_labels", 'the lengths written on lines: "auto" (hydrogen and halogen bonds, metals), "all" or "none"'),
        ("show_chain", 'name residues with their chain: "auto" (where there are several), "always" or "never"'),
        ("interactive", "the figure answers the pointer in a browser, and data-active on its root (a style sheet, no script)"),
        ("stylesheet", 'in a page with several figures: "inline", "scoped" (colours for this one alone) or "external" (result.css)'),
    )),
    ("Compared with another protein", (
        ("absent", "the residues the other one does not have (numbers or names); those in contact are marked and counted"),
        ("absent_label", "its name, for the key and the sentence under it"),
        ("absent_style", '"marked" (a stroke of a marker pen) or "faint"'),
    )),
    ("Colours and words", (
        ("colors", 'the colour scheme: "standard", "contrast" or one registered in theme.SCHEMES; the theme picks '
                   'its light or dark palette'),
        ("background", "False: no background, transparent"),
        ("title", "a title above the figure"),
        ("subtitle", "a line under the title"),
        ("legend", "the key"),
        ("caption", "the sentences under the key"),
        ("notes", "of those, the ones on the ligand's own quirks"),
        ("bare", "the drawing and its key and no words; they are in result.notes"),
    )),
    ("The confidence of a predicted model", (
        ("confidence", "draw the model's confidence, where the diagram has it"),
        ("plddt_colors", 'pLDDT in AlphaFold\'s colours and not by weight: True (behind the atoms) or "bonds"'),
        ("plddt_by_atom", "each atom drawn of a residue at its own pLDDT, where the model gives one for each"),
        ("pae_colors", "the predicted aligned error of a contact as a glow behind its line; None: for a ligand, not for an interface"),
    )),
    ("The pocket and an ensemble", (
        ("burial", "the ligand's buried fraction under the title"),
        ("solvent_sites", "predict waters that could bridge ligand and protein, and draw them, marked as predicted"),
        ("solvent_mark", 'how a predicted water stands on its line: "glow", "filled" or "outline"'),
        ("occupancy_style", 'the share of an ensemble: "steps" (three weights, with the number) or "continuous" (weight alone)'),
        ("occupancy_labels", 'where the number stands: "all" (every contact) or "partial" (lines that fewer than all models make)'),
    )),
    ("Size and legibility", (
        ("figure_width_mm", "the width of the page the figure is for"),
        ("figure_height_mm", "its height"),
        ("min_font_pt", "the smallest text that counts as legible on it"),
        ("on_unreadable", 'a figure that is not legible: "summary" (a table), "render" (drawn all the same) or "none"'),
        ("effort", 'for a peptide drawn as a chain: "full", or "quick" (one that is far too crowded is given up early)'),
        ("bond_length", "the scale, in page units for a bond; None: the largest that fits the page"),
        ("coords", 'how the ligand is laid out in two dimensions: "auto" (CoordGen unless its atoms clash), "coordgen" or "rdkit"'),
    )),
)


class _Text(str):
    """Text that shows itself as it is when it is asked for at the prompt."""

    def __repr__(self) -> str:
        return str(self)


def options(word: str = "") -> str:
    """The keywords of ``render``, one line each, by what they are for; with ``word`` those that mention it."""
    defaults = {f.name: f.default for f in dataclasses.fields(RenderOptions)}
    wanted = word.strip().lower()
    out = []
    for title, rows in GROUPS:
        hits = [(name, line) for name, line in rows if wanted in name.lower() or wanted in line.lower()]
        if hits:
            out.append(f"{title}\n" + "\n".join(f"  {name + '=' + repr(defaults[name]):<28}{line}" for name, line in hits))
    return _Text("\n\n".join(out) if out else f"no keyword of render mentions {word!r}")


RenderOptions.__doc__ = ("Every keyword of ``render`` beside ``style``, ``theme``, ``metrics`` and ``graph``.\n\n"
                         + "\n".join("    " + line if line else "" for line in options().split("\n")) + "\n")
