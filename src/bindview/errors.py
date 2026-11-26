"""Exception types.

Errors are reserved for *failures*: a file that cannot be parsed, a ligand that
is not in the structure, a molecule that cannot be built. A complex that was
analysed and has no interactions is **not** an error -- it produces a normal
result whose ``status`` is ``"empty"``. Keeping the two apart is a hard
requirement of the library: callers must be able to tell "nothing found" from
"could not look".
"""

from __future__ import annotations


class BindviewError(Exception):
    """Base class for every error raised by this package."""


class StructureError(BindviewError):
    """The structure file could not be read or contains no usable atoms."""


class LigandNotFoundError(BindviewError):
    """The ligand selection matched nothing (or matched ambiguously)."""


class LigandBuildError(BindviewError):
    """An RDKit molecule could not be built for the ligand."""


class InteractionError(BindviewError):
    """An interaction record is malformed."""


class DetectionUnavailableError(BindviewError):
    """Interaction detection was requested but the optional detector is not installed."""


class EnsembleError(BindviewError):
    """Models given as one ensemble are not models of one complex: another ligand, or another pocket."""


class RenderError(BindviewError):
    """The diagram could not be rendered."""


class GraphError(BindviewError, ValueError):
    """A diagram graph is malformed, of an unsupported version, or does not fit the diagram."""
