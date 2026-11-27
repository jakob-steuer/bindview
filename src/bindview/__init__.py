"""bindview -- publication-quality figures of molecular interactions.

A figure::

    from bindview import Diagram

    dia = Diagram.from_structure("model.cif", ligand="LIG_B", smiles="CC(=O)N1CCN(CC1)...", interactions="distances")
    result = dia.render_svg("figure.svg")           # result.status, .warnings, .svg, .graph, ...

``interactions`` is a table of contacts (a path, or a list of records), ``"distances"`` for a plain one made from the heavy
atoms, or ``None`` to detect them with PLIP (optional, GPL-2.0).

Where to look::

    bindview.options()                              every keyword of render, with its default
    bindview.options("colour")                      those that mention a word
    help(bindview.Diagram.from_structure)           what goes in
    help(bindview.Diagram.render)                   what comes out

Other figures::

    dia.arrange()                                   by hand in a browser: drag, turn, flip
    Comparison([dia_a, dia_b, ...])                 several ligands of one pocket
    Ensemble.from_structures(models, "LIG", smiles, "distances")       several models of one complex
        .consensus()  .diagram()                    all of them in one figure; one of them, with shares
    Interface(structure, ["A", "B"], contacts)      the contacts between two chains
    BasePairs(structure, ["A", "B"], pairs)         two strands of RNA or DNA

A predicted model brings its confidence into the figure, read only where the file declares it. A peptide is drawn as a chain of
its residues. The core imports RDKit and NumPy and nothing else.
"""

from __future__ import annotations

from typing import Any

from .backends import LigandBackend, Polymer, SmallMolecule
from .basepairs import (BasePair, BasePairs, BaseStack, pairs_from_contacts, pairs_from_dot_bracket,
                        pairs_within_from_table,
                        read_fr3d_pairs, read_fr3d_stacks, read_mmcif_pairs, read_rnapolis_pairs,
                        read_rnapolis_stacks)
from .interface import Interface, InterfaceContact, Participant, load_interface_contacts
from .compare import VIEWS, Comparison, Difference
from .confidence import Confidence, ModelConfidence
from .diagram import STYLES, Diagram, Quality, RenderResult
from .consensus import ResidueShare
from .ensemble import Ensemble, EnsembleContact
from .errors import (DetectionUnavailableError, EnsembleError, GraphError, InteractionError, LigandBuildError,
                     LigandNotFoundError, BindviewError, RenderError, StructureError)
from .fingerprint import Fingerprint, FingerprintTable
from .graph import Graph
from .interactions import Interaction, InteractionType, load_interactions, parse_interactions
from .options import options
from .peptide import Peptide, perceive_peptide
from .structure import Structure, read_models, read_structure
from .svg import stylesheet
from .theme import DARK, LIGHT, Metrics, Palette

__version__ = "0.1.1"

__all__ = [
    "Diagram", "RenderResult", "Quality", "STYLES", "options", "stylesheet",
    "Interaction", "InteractionType", "load_interactions", "parse_interactions", "detect_interactions",
    "Structure", "read_structure", "read_models",
    "Confidence", "ModelConfidence",
    "Comparison", "Difference", "VIEWS", "Fingerprint", "FingerprintTable", "Graph",
    "Ensemble", "EnsembleContact", "ResidueShare",
    "LigandBackend", "SmallMolecule", "Polymer", "Peptide", "perceive_peptide",
    "Interface", "InterfaceContact", "Participant", "load_interface_contacts",
    "BasePairs", "BasePair", "BaseStack", "read_mmcif_pairs", "read_fr3d_pairs", "read_fr3d_stacks",
    "read_rnapolis_pairs", "read_rnapolis_stacks", "pairs_from_dot_bracket", "pairs_from_contacts",
    "pairs_within_from_table",
    "Palette", "Metrics", "LIGHT", "DARK",
    "BindviewError", "StructureError", "LigandNotFoundError", "LigandBuildError", "InteractionError",
    "DetectionUnavailableError", "RenderError", "GraphError", "EnsembleError",
    "__version__",
]


def detect_interactions(structure: Any, ligand: Any) -> "list[Interaction]":
    """Detect interactions with PLIP (optional, GPL-2.0). See :mod:`bindview.detect`.

    Defined here as a thin wrapper so that importing :mod:`bindview` never imports
    the detector.
    """
    from .detect import detect_interactions as _detect

    return _detect(structure, ligand)
