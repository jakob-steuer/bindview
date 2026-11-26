"""A figure whose residues are not all drawn alike (``residues={...}``).

Each residue is drawn by the style that draws its form: with atoms, or as its
name, by the detailed one, and as a bubble by the schematic one. What the two
share -- the lines, the lining, the halos, the key -- is the figure's and is
drawn once, from the one set of settings.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from ..layout import Body
from .base import (STYLE_SETTINGS, Context, Drawn, RenderOptions, ResidueGroup, Settings, StyleDrawing,
                   finish_drawing, residue_names)
from .detailed import DetailedStyle
from .schematic import SchematicStyle, _Bubble

__all__ = ["MixedStyle", "style_for", "unknown_residues"]


class MixedStyle:
    """``name``: the style whose settings are the figure's where a residue is not named."""

    def __init__(self, name: str, settings: Settings) -> None:
        self.name, self.settings = name, settings
        self._atoms, self._bubbles = DetailedStyle(settings), SchematicStyle(settings)

    def _of(self, grp: ResidueGroup) -> "DetailedStyle | SchematicStyle":
        return self._bubbles if self.settings.form_of(grp) == "bubbles" else self._atoms

    def ligand_obstacles(self, ctx: Context) -> list[tuple[float, float, float]]:
        return self._atoms.ligand_obstacles(ctx, [g for g in ctx.groups if self._of(g) is self._atoms])

    def bodies(self, ctx: Context) -> list[Body]:
        return [self._of(grp).body(ctx, grp) for grp in ctx.groups]

    def draw(self, ctx: Context, bodies: Sequence[Body], poses: dict, label_ts: dict | None = None) -> StyleDrawing:
        d = Drawn({"backdrop": [], "halos": [], "interactions": [], "contacts": [], "residues": []})
        bubbles = [b for b in bodies if isinstance(b.payload, _Bubble)]
        others = [b for b in bodies if not isinstance(b.payload, _Bubble)]
        self._atoms.draw_bodies(ctx, others, poses, label_ts or {}, d)
        self._bubbles.draw_bodies(ctx, bubbles, poses, label_ts or {}, d)
        return finish_drawing(ctx, self.settings, d)


def style_for(name: str, options: RenderOptions) -> "DetailedStyle | SchematicStyle | MixedStyle":
    """The style called ``name`` with what the options set by hand."""
    try:
        own = STYLE_SETTINGS[name]
    except KeyError:
        raise ValueError(f"unknown style {name!r}; choose from {', '.join(sorted(STYLE_SETTINGS))}") from None
    settings = own.with_options(options)
    if not isinstance(settings.residues, str):
        if "*" not in dict(settings.residues):
            # A residue that is not named is drawn as the style draws it.
            settings = replace(settings, residues=settings.residues + (("*", own.residues),))
        return MixedStyle(name, settings)
    drawn = SchematicStyle(settings) if settings.residues == "bubbles" else DetailedStyle(settings)
    drawn.name = name
    return drawn


def unknown_residues(options: RenderOptions, groups: Sequence[ResidueGroup]) -> list[str]:
    """The names in ``residues={...}`` that no residue of the figure answers to."""
    if options.residues is None or isinstance(options.residues, str):
        return []
    known = {n.lower() for g in groups for n in residue_names(g)}
    return [name for name, _ in options.residues if name != "*" and name.lower() not in known]
