"""Placing residues around the ligand.

The engine is style-agnostic. A style describes each residue as a :class:`Body`
(collision shapes, a text label, and *ports* that tie it to points on the
ligand); the engine returns a pose for every body such that nothing overlaps,
and reports honestly when it could not manage that.
"""

from .body import Body, Port, Pose
from .body import SPRINGS
from .engine import LayoutParams, LayoutResult, audit_layout, layout_bodies, layout_key
from .field import LigandField

__all__ = ["Body", "Port", "Pose", "LayoutParams", "LayoutResult", "layout_bodies", "layout_key", "audit_layout", "LigandField",
           "SPRINGS"]
