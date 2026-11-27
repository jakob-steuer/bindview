"""Command line interface.

    bindview render model.cif --ligand LIG_B --smiles-from meta.json -o out.svg
    bindview compare case_a/ case_b/ -o difference.svg          # two ligands: difference view
    bindview compare series/*/ -o consensus.svg --fingerprints fp.csv
    bindview fingerprint series/*/ --form wide > fp.csv
    bindview report model.cif -o report.html                    # looks what the input is, and draws that
    bindview basepairs model.pdb --chains A B --pairs pairs.json # two strands and their base pairs

Exit status: 0 when a figure was produced for a valid or empty complex; 3 when
it is not readable at the size asked for (``render`` and ``compare`` still write
a summary figure unless ``--on-unreadable none``, ``basepairs`` the figure
itself); 2 on an error, which is one line on standard error.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

from . import __version__
from .align import MIN_FRACTION
from .compare import VIEWS, Comparison
from .diagram import DISTANCES, STYLES, Diagram
from .theme import SCHEMES
from .errors import BindviewError
from .graph import Graph
from .interactions import Interaction, load_interactions
from .residues import ATOM_LABELS
from .styles.absence import ABSENT_STYLES
from .styles.base import LINES, LININGS, RESIDUE_FORMS, SOLVENT_MARKS

__all__ = ["main", "build_parser"]

_SIBLING_TABLES = ("interactions.tsv", "interactions.csv")


class _Help(argparse.RawDescriptionHelpFormatter):
    """Descriptions and examples as written (they are wrapped by hand); options in a wider column."""

    def __init__(self, prog: str) -> None:
        super().__init__(prog, max_help_position=32)


def _command(sub: Any, name: str, summary: str, usage: str, description: str, examples: str) -> argparse.ArgumentParser:
    return sub.add_parser(name, help=summary, usage=usage, description=description,
                          epilog="examples:\n" + examples, formatter_class=_Help)


_EXIT = "\n\nexit status: 0 a figure was written; 3 it is not readable at the size asked for; 2 an error."


def _from(least: float, above: bool = False, whole: bool = False) -> "Callable[[str], Any]":
    """The type of an argument that is a number from ``least`` up (``above``: and not ``least`` itself): what is
    no number, not finite or too small is refused by the parser, which names the argument."""
    kind = ("a whole number " if whole else "a number ") + (f"above {least:g}" if above else f"from {least:g} up")

    def parse(text: str) -> Any:
        try:
            value = int(text) if whole else float(text)
        except ValueError:
            value = math.nan
        if not math.isfinite(value) or value < least or (above and value == least):
            raise argparse.ArgumentTypeError(f"{text!r} is not {kind}")
        return value

    return parse


def _fraction(text: str) -> float:
    """The type of an argument that is a fraction: a number from 0 to 1."""
    try:
        value = float(text)
    except ValueError:
        value = math.nan
    if not 0.0 <= value <= 1.0:                          # (nan is not either)
        raise argparse.ArgumentTypeError(f"{text!r} is not a fraction from 0 to 1")
    return value


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bindview", formatter_class=_Help,
        description="Figures of molecular interactions, as SVG.",
        epilog="'bindview COMMAND --help' lists the options of a command; docs/manual.md shows how they are used.")
    p.add_argument("--version", action="version", version=f"bindview {__version__}")
    sub = p.add_subparsers(dest="command", required=True, title="commands", metavar="COMMAND")

    # -- render ---------------------------------------------------------------------------------------
    r = _command(
        sub, "render", "draw one ligand in its pocket", "bindview render STRUCTURE --ligand LIGAND [options]",
        "Draw the contacts of one ligand in a structure file as an SVG figure.\n\n"
        "The contacts come from a table you give (--interactions), from interactions.tsv beside the\n"
        "structure, or from PLIP, which is optional and GPL-2.0 (pip install \"bindview[plip]\", or\n"
        "pip install \".[plip]\" in a clone of the repository)." + _EXIT,
        "  bindview render model.cif --ligand LIG_B --smiles \"CC(=O)Nc1ccc(O)cc1\" -o figure.svg\n"
        "  bindview render model.cif --ligand B:1 --interactions contacts.tsv --style schematic\n"
        "  bindview render frame.pdb --ligand LIG --interactions md.tsv --ensemble \"1,000 frames\" \\\n"
        "      --bare --notes-file caption.txt")
    a = r.add_argument_group("input")
    a.add_argument("structure", metavar="STRUCTURE", help="mmCIF or PDB file")
    a.add_argument("--ligand", required=True,
                   help="a residue name (LIG_B), chain:number (B:1), chain:name:number, or a peptide chain by its "
                        "name (P)")
    one = a.add_mutually_exclusive_group()
    one.add_argument("--smiles", help="the ligand's SMILES: it gives the bond orders")
    one.add_argument("--smiles-from", metavar="JSON",
                     help="a JSON file with 'smiles' or 'ligand.smiles' (also looked for beside the structure)")
    a.add_argument("--ligand-as", choices=("auto", "atoms", "chain"), default="auto",
                   help="the ligand as atoms or as a chain of its residues (auto: a chain from 8 amino acids up)")
    a.add_argument("--no-ionize", action="store_true", help="draw acids as the SMILES writes them, not deprotonated")
    k = r.add_argument_group("contacts")
    k.add_argument("--interactions", metavar="TABLE",
                   help="a table of contacts (.tsv/.csv); default: interactions.tsv beside the structure, else PLIP")
    k.add_argument("--detect", action="store_true", help="find the contacts with PLIP even where a table lies beside")
    k.add_argument("--distances-only", action="store_true",
                   help="make the table from the distances of the heavy atoms: nothing else is needed, and it is cruder "
                        "than PLIP")
    k.add_argument("--ensemble", metavar="TEXT",
                   help="what the table's 'occupancy' column is a share of, for the key (e.g. '1,000 frames')")
    k.add_argument("--covalent", action="store_true",
                   help="look for covalent bonds between the ligand and a residue, as the file declares them and by "
                        "distance, and draw them")
    k.add_argument("--ions", action="store_true",
                   help="draw an ion that the model has between the ligand and the protein as the bridge it is")
    k.add_argument("--pae", metavar="FILE", action="append",
                   help="predicted aligned error: AlphaFold 3's confidences.json(.gz) or Boltz's pae_*.npz; each contact "
                        "stands on a glow in the colour of its own; repeat it for several draws of one model")
    k.add_argument("--solvent-sites", action="store_true",
                   help="predict water sites that could bridge ligand and protein, drawn as predicted")
    k.add_argument("--solvent-mark", choices=SOLVENT_MARKS, default="glow",
                   help="a predicted water on a glow, in a filled box with a heavy border, or in a thin broken "
                        "outline")
    _confidence_arguments(r)
    d = r.add_argument_group("what is drawn")
    d.add_argument("--style", choices=STYLES, default="detailed",
                   help="residues with the atoms that make the contacts (default), or as bubbles")
    d.add_argument("--detail", choices=("box", "backbone", "sidechain", "atoms"), default=None,
                   help="how much is drawn of a residue that makes a directional contact")
    d.add_argument("--distances", choices=("auto", "all", "none"), default="auto",
                   help="which contact lines carry their distance")
    d.add_argument("--atom-labels", choices=ATOM_LABELS, default="elements",
                   help="on an atom of a residue the letter of its element (N, O), or its name in the file (NZ, OE1)")
    d.add_argument("--show-chain", choices=("auto", "always", "never"), default="auto",
                   help="name residues with their chain")
    d.add_argument("--absent", metavar="RESIDUES",
                   help="residues that another protein (a splice isoform) does not have: 702,705-712,A:744; those in "
                        "contact with the ligand are marked, and the figure counts them")
    d.add_argument("--absent-label", metavar="NAME", help="that protein's name (AR-V7)")
    d.add_argument("--absent-style", choices=ABSENT_STYLES, default="marked",
                   help="mark them with a stroke of a marker pen, or set them back")
    d.add_argument("--burial", action=argparse.BooleanOptionalAction, default=True,
                   help="the ligand's buried fraction under the title (on by default)")
    _look_arguments(r)
    t = r.add_argument_group("page and text")
    t.add_argument("--title")
    t.add_argument("--subtitle", metavar="TEXT")
    _bare_argument(t)
    t.add_argument("--no-legend", action="store_true", help="leave the key off")
    t.add_argument("--no-notes", action="store_true", help="leave off the notes on the ligand's own quirks")
    t.add_argument("--scoped-css", action="store_true",
                   help="with --theme auto: the colours are declared for figures in these colours alone, for a page that "
                        "has figures in others beside it")
    t.add_argument("--no-background", action="store_true", help="a transparent background")
    t.add_argument("--width-mm", type=_from(0, above=True), default=180.0, metavar="MM",
                   help="the width the figure has to be readable at (180)")
    t.add_argument("--min-font-pt", type=_from(0), default=5.0, metavar="PT",
                   help="the smallest text allowed at that width (5)")
    t.add_argument("--bond-length", type=_from(0, above=True), metavar="UNITS",
                   help="a fixed bond length instead of the largest that fits")
    t.add_argument("--on-unreadable", choices=("summary", "render", "none"), default="summary",
                   help="what is drawn where the figure is not readable at that width (a per-residue summary)")
    t.add_argument("--effort", choices=("full", "quick"), default="full",
                   help="quick: a peptide drawn as a chain that is far too crowded is given up early")
    o = r.add_argument_group("output")
    o.add_argument("-o", "--output", metavar="SVG", help="the figure (default: <structure>.<style>.svg)")
    _output_arguments(o)
    o.add_argument("--layout", metavar="JSON",
                   help="draw from a graph written by --graph, as it is or with nodes moved")
    o.add_argument("--arrange", action="store_true",
                   help="arrange the figure by hand in a browser first (drag, turn, flip), then draw it")
    o.add_argument("--arrange-page", metavar="HTML", help="write the page to arrange it as one file, and no figure")
    o.add_argument("-q", "--quiet", action="store_true", help="print nothing but --json")
    plainer = _plainer_arguments(r)
    plainer.add_argument("--occupancy-style", choices=("steps", "continuous"), default="steps",
                         help="continuous: occupancy by line weight alone, with no share written")
    plainer.add_argument("--occupancy-labels", choices=("all", "partial"), default="all",
                         help="partial: no share on a contact that is made in all, and none under a name")

    # -- interface ------------------------------------------------------------------------------------
    i = _command(
        sub, "interface", "draw the contacts between two chains",
        "bindview interface STRUCTURE --between A B [options]",
        "Draw the contacts between two chains, or between parts of a complex: the residues of one as\n"
        "a column of boxes (or each where it lies, --layout plane), those of the other around them.\n"
        "Three participants in a row (MHC, peptide, receptor) are drawn around the middle one.\n\n"
        "Without --contacts, PLIP (optional, GPL-2.0) finds the contacts." + _EXIT,
        "  bindview interface model.cif --between A B --pae confidences.json -o interface.svg\n"
        "  bindview interface model.cif --between H+L C --layout auto --detail sidechain\n"
        "  bindview interface model_0.cif --between A B --pae pae_model_0.npz --ipsae")
    a = i.add_argument_group("input")
    a.add_argument("structure", metavar="STRUCTURE", help="mmCIF or PDB file")
    a.add_argument("--between", nargs="+", required=True, metavar="CHAINS",
                   help="the participants, each one or more chains: A B, or A C D+E, or receptor=D+E")
    a.add_argument("--spine", help="the participant drawn as the column (default: the one that touches all others; "
                                   "of two, the smaller side)")
    a.add_argument("--site", metavar="CHAIN:NUMBER", help="only the site around this residue")
    a.add_argument("--radius", type=float, default=8.0, metavar="A", help="... within this many angstrom of it (8)")
    a.add_argument("--max-residues", type=_from(1, whole=True), default=30, metavar="N",
                   help="the most residues of the column; a larger interface is narrowed to a site (30)")
    k = i.add_argument_group("contacts and confidence")
    k.add_argument("--contacts", metavar="TSV",
                   help="a table of contacts with residues and atoms of both sides (docs/reference.md); "
                        "default: PLIP")
    k.add_argument("--pae", metavar="FILE", action="append",
                   help="predicted aligned error: AlphaFold 3's confidences.json(.gz), Boltz's pae_*.npz, or the "
                        "AlphaFold database's predicted_aligned_error file of an entry; repeat it for several "
                        "draws of one model")
    k.add_argument("--ipsae", nargs="?", const=True, default=False, type=float, metavar="CUTOFF",
                   help="also work out ipSAE from --pae, with this PAE cut-off (10 angstrom)")
    k.add_argument("--model-confidence", metavar="JSON",
                   help="ipTM: AlphaFold 3's summary_confidences.json or Boltz's confidence_*.json")
    k.add_argument("--pae-colors", action="store_true",
                   help="the predicted aligned error of a contact as a glow behind its line, not as a number on it")
    d = i.add_argument_group("layout")
    d.add_argument("--layout", choices=("column", "auto", "plane"), default="column",
                   help="a column (default); a plane, each box where its residue lies; or let a rule choose")
    d.add_argument("--order", choices=("auto", "sequence", "interface"), default="auto",
                   help="the order of the column: of the sequence, along the interface, or (auto) the one in which "
                        "the contacts of each partner residue are spread over fewer boxes")
    d.add_argument("--detail", choices=("box", "backbone", "sidechain", "atoms"), default=None,
                   help="how much is drawn of a residue that makes a directional contact, on both sides")
    d.add_argument("--style", choices=("detailed", "schematic"), default="detailed")
    _look_arguments(i)
    t = i.add_argument_group("page and text")
    t.add_argument("--title")
    _bare_argument(t)
    o = i.add_argument_group("output")
    o.add_argument("-o", "--output", metavar="SVG", help="the figure (default: <structure>.interface.<style>.svg)")
    _output_arguments(o)
    o.add_argument("-q", "--quiet", action="store_true", help="print nothing but --json")
    _plainer_arguments(i)

    # -- basepairs ------------------------------------------------------------------------------------
    bp = _command(
        sub, "basepairs", "draw two strands of RNA or DNA and their base pairs",
        "bindview basepairs STRUCTURE --chains A B [options]",
        "Draw two nucleic-acid chains as their secondary structures facing each other: which\n"
        "nucleotides pair across, by which edges, where on each strand's own fold, what stacks\n"
        "across, and a ligand between them with its contacts.\n\n"
        "The pairs are read, not detected: RNApolis's annotator JSON, RNA 3D Hub's FR3D tables, or the\n"
        "PDB's own annotation in a deposited mmCIF. Without --pairs, the structure's own annotation is\n"
        "used where it has one, else a stand-in from bindview's plain distance table, without a family\n"
        "(docs/manual.md)." + _EXIT,
        "  annotator -j pairs.json model.pdb          # RNApolis, in an environment of its own\n"
        "  bindview basepairs model.pdb --chains A B --pairs pairs.json --ligand BGD -o pairs.svg\n"
        "  bindview basepairs 1xpf.cif --chains A B --theme auto --interactive\n"
        "  bindview basepairs model.pdb --chains A B --dot-bracket A='(((((((....)))))))' --family cWW")
    a = bp.add_argument_group("input")
    a.add_argument("structure", metavar="STRUCTURE", help="mmCIF or PDB file")
    a.add_argument("--chains", nargs=2, required=True, metavar="CHAIN", help="the two strands")
    k = bp.add_argument_group("pairs and stacks")
    k.add_argument("--pairs", action="append", default=[], metavar="FILE",
                   help="RNApolis's JSON (.json), an FR3D table (.csv) or a deposited mmCIF with the PDB's "
                        "annotation (repeatable; their pairs are added together)")
    k.add_argument("--stacks", action="append", default=[], metavar="FILE",
                   help="RNApolis's JSON or FR3D's stacking table (default: those of an RNApolis --pairs, "
                        "else the plain table's)")
    k.add_argument("--dot-bracket", action="append", default=[], metavar="CHAIN=STRUCTURE",
                   help="a strand's own pairs as a dot-bracket, one character a nucleotide (repeatable)")
    k.add_argument("--family", metavar="FAMILY", help="the family of the pairs of --dot-bracket (cWW); none by default")
    k.add_argument("--no-contacts", action="store_true",
                   help="no plain distance table: no stand-in pairs, no stacks from it, no other hydrogen bonds")
    g = bp.add_argument_group("a ligand beside the pairs")
    g.add_argument("--ligand", metavar="RESIDUE", help="a residue that is no part of the two chains: BGD, or C:1")
    g.add_argument("--ligand-contacts", metavar="TSV", help="its contacts (PLIP's table, say); default: the plain table")
    g.add_argument("--ligand-smiles", metavar="SMILES", help="its bond orders, for the plain table")
    g.add_argument("--ligand-name", metavar="TEXT", help="the word in its box (default: the residue name)")
    lk = bp.add_argument_group("look")
    lk.add_argument("--theme", choices=("light", "dark", "auto"), default="light",
                    help="auto: one file with both palettes, following the reader's setting")
    lk.add_argument("--colors", choices=tuple(SCHEMES), default="standard", help="the colour scheme")
    lk.add_argument("--interactive", action="store_true",
                    help="in a browser, a nucleotide and its pairs stand out under the pointer (no script)")
    t = bp.add_argument_group("page and text")
    t.add_argument("--title")
    t.add_argument("--no-legend", action="store_true", help="no key under the drawing")
    t.add_argument("--no-caption", action="store_true",
                   help="no line saying where the pairs came from: it stays in the file's description, in --json "
                        "and in --notes-file")
    t.add_argument("--step", type=_from(0, above=True), default=24.0, metavar="PX",
                   help="the distance of two neighbours (24)")
    t.add_argument("--width-mm", type=_from(0, above=True), default=180.0, metavar="MM",
                   help="the width the figure has to be readable at (180)")
    t.add_argument("--min-font-pt", type=_from(0), default=5.0, metavar="PT",
                   help="the smallest text allowed at that width (5)")
    o = bp.add_argument_group("output")
    o.add_argument("-o", "--output", metavar="SVG", help="the figure (default: <structure>.basepairs.svg)")
    _output_arguments(o)
    o.add_argument("--from-graph", metavar="JSON", help="draw from a graph written by --graph, its nodes moved")
    o.add_argument("-q", "--quiet", action="store_true", help="print nothing but --json")

    # -- compare --------------------------------------------------------------------------------------
    members = ("a structure file, or a folder holding model.cif or model.cif.gz (with interactions.tsv and "
               "meta.json, if there)")
    c = _command(
        sub, "compare", "compare ligands of one pocket", "bindview compare MEMBER MEMBER [MEMBER ...] [options]",
        "Draw several ligands of one pocket against one frame of residues: two ligands give a\n"
        "difference view, more a consensus view." + _EXIT,
        "  bindview compare case_a/ case_b/ -o difference.svg\n"
        "  bindview compare series/*/ -o consensus.svg --fingerprints fingerprints.csv")
    a = c.add_argument_group("input")
    a.add_argument("members", nargs="+", metavar="MEMBER", help=members)
    a.add_argument("--ligand", help="the ligand residue in every structure (default: meta.json's "
                                    "ligand.residue_name_in_cif)")
    a.add_argument("--plddt", choices=("auto", "bfactor", "bfactor01", "off"), default="auto",
                   help="where pLDDT comes from (see 'bindview render --help')")
    a.add_argument("--untrusted", action="append", default=[], metavar="NAME",
                   help="mark this ligand's model as not to be trusted (repeatable)")
    d = c.add_argument_group("what is drawn")
    d.add_argument("--view", choices=("auto",) + VIEWS, default="auto")
    d.add_argument("--min-ligands", type=int, default=1, metavar="N",
                   help="consensus view: leave out residues contacted by fewer ligands")
    d.add_argument("--no-thumbnails", action="store_true", help="consensus view: no drawing of each single ligand")
    d.add_argument("--min-fraction", type=_fraction, default=MIN_FRACTION, metavar="F",
                   help=f"align on the common substructure where it covers at least this fraction of the smallest "
                        f"ligand's heavy atoms ({MIN_FRACTION:g})")
    d.add_argument("--burial", action=argparse.BooleanOptionalAction, default=True,
                   help="each ligand's buried fraction in the figure (on by default)")
    d.add_argument("--theme", choices=("light", "dark", "auto"), default="light")
    d.add_argument("--colors", choices=tuple(SCHEMES), default="standard", help="the colour scheme")
    t = c.add_argument_group("page and text")
    t.add_argument("--title")
    t.add_argument("--subtitle", metavar="TEXT")
    t.add_argument("--width-mm", type=_from(0, above=True), default=180.0, metavar="MM",
                   help="the width the figure has to be readable at (180)")
    t.add_argument("--min-font-pt", type=_from(0), default=5.0, metavar="PT",
                   help="the smallest text allowed at that width (5)")
    t.add_argument("--on-unreadable", choices=("summary", "render", "none"), default="summary",
                   help="what is drawn where the figure is not readable at that width")
    o = c.add_argument_group("output")
    o.add_argument("-o", "--output", metavar="SVG", default="comparison.svg", help="the figure (comparison.svg)")
    o.add_argument("--fingerprints", metavar="CSV", help="also write the fingerprints (long form)")
    o.add_argument("--json", action="store_true", help="print status, alignment and classification as JSON")
    o.add_argument("--graph", metavar="JSON", help="also write the figure as a graph of nodes and contacts")
    o.add_argument("-q", "--quiet", action="store_true", help="print nothing but --json")

    # -- fingerprint ----------------------------------------------------------------------------------
    f = _command(
        sub, "fingerprint", "write interaction fingerprints as CSV", "bindview fingerprint MEMBER [MEMBER ...] [options]",
        "One row per contact (long form) or per ligand (wide form): residue and kind of contact, in a\n"
        "fixed order of columns.",
        "  bindview fingerprint series/*/ --form wide -o fingerprints.csv")
    f.add_argument("members", nargs="+", metavar="MEMBER", help=members)
    f.add_argument("--ligand", help="the ligand residue in every structure")
    f.add_argument("--form", choices=("long", "wide"), default="long",
                   help="long: one row per contact, lossless; wide: one row per ligand, one column per contact")
    f.add_argument("--counts", action="store_true", help="wide form: counts of records instead of 1/0")
    f.add_argument("-o", "--output", metavar="CSV", help="the file (default: standard output)")
    f.add_argument("--plddt", choices=("auto", "bfactor", "bfactor01", "off"), default="auto",
                   help="where pLDDT comes from (see 'bindview render --help')")
    f.add_argument("-q", "--quiet", action="store_true")

    # -- report ---------------------------------------------------------------------------------------
    rp = _command(
        sub, "report", "look at an input, draw what it holds, write one page", "bindview report INPUT [options]",
        "Look at a model, or a folder that holds one, and say what it holds: a small molecule in a\n"
        "pocket, a peptide, an interface, an ensemble. Draw the figure for each, under the call that\n"
        "drew it, and then its views: the same figure in the other style, with pLDDT in AlphaFold's\n"
        "colours where the model has it, and as a figure that answers the pointer. List what was\n"
        "found and not drawn.\n\n"
        "Contacts of a small molecule come from a table beside the model, else from PLIP where it is\n"
        "installed (optional, GPL-2.0), else from a table of heavy atoms, which is cruder than PLIP.",
        "  bindview report model.cif -o report.html\n"
        "  bindview report series/ -o series.html\n"
        "  bindview report screen/ --each -o screen.html\n"
        "  bindview report model.cif --domain N=A:28-124 --domain ZP=A:126-269")
    rp.add_argument("input", metavar="INPUT", help="a model (mmCIF or PDB, gzipped or not), or a folder that holds one")
    rp.add_argument("-o", "--output", metavar="HTML", help="the page (default: <input>.report.html beside the input)")
    rp.add_argument("--smiles", action="append", default=[], metavar="NAME=SMILES",
                   help="the SMILES of a residue by its name, for its bond orders (repeatable)")
    rp.add_argument("--domain", action="append", default=[], metavar="NAME=CHAIN:FROM-TO",
                   help="a part of a chain by its first and last residue; two or more: the interface between them")
    rp.add_argument("--look", action="store_true", help="say what the input holds and draw nothing")
    rp.add_argument("--each", action="store_true",
                    help="a folder of different inputs: a page for each, and an index of them as a page and a CSV")
    rp.add_argument("--one-view", action="store_true",
                    help="each figure alone: not also in the other style, in pLDDT colours and interactive")
    rp.add_argument("-q", "--quiet", action="store_true")
    return p


def _look_arguments(r: argparse.ArgumentParser) -> None:
    """How a figure looks: its colours, and whether it answers the pointer."""
    g = r.add_argument_group("look")
    g.add_argument("--theme", choices=("light", "dark", "auto"), default="light",
                   help="auto: one file with both palettes, following the reader's setting")
    g.add_argument("--colors", choices=tuple(SCHEMES), default="standard",
                   help="the colour scheme: standard, or contrast (residue classes further apart)")
    c = g.add_mutually_exclusive_group()
    c.add_argument("--plddt-colors", action="store_true",
                   help="pLDDT in AlphaFold's four colours behind the atoms, instead of by line weight")
    c.add_argument("--plddt-bonds", action="store_true", help="the same four colours, as the colours of the bonds")
    g.add_argument("--interactive", action="store_true",
                   help="in a browser, a residue and its contacts stand out under the pointer (no script)")
    g.add_argument("--charge-halos", action=argparse.BooleanOptionalAction, default=False,
                   help="shade the charged group that a salt bridge or pi-cation contact is made with (not shaded "
                        "by default)")
    s = r.add_argument_group("the settings of a style, one by one",
                             "Each is the style's own unless it is given here (docs/reference.md).")
    s.add_argument("--residues", nargs="+", metavar="HOW", default=None,
                   help="atoms, bubbles or names; or residue by residue: Met743=atoms Gln712=atoms '*=bubbles'")
    s.add_argument("--lines", choices=tuple(LINES), default=None, help="the colours and dashes of the contact lines")
    s.add_argument("--exposure", action=argparse.BooleanOptionalAction, default=None,
                   help="grey halos behind the ligand's atoms that are open to solvent")
    s.add_argument("--lining", choices=LININGS, default=None,
                   help="the wall of the pocket: a stretch for each contact in the colour of what makes it "
                        "(contacts: the schematic style's own), the class of the nearest residue (nearest), or "
                        "none")


def _plainer_arguments(r: argparse.ArgumentParser) -> Any:
    """The options that each draw one thing more plainly than a figure does by itself; last in the help."""
    b = r.add_argument_group("one thing drawn more plainly",
                             "Each leaves off one refinement of the drawing; the figure is otherwise the same.")
    b.add_argument("--no-break-lines", action="store_true",
                   help="draw a contact's line whole where it runs over a word that is not its own")
    b.add_argument("--no-charge-key", action="store_true", help="leave the charged-group shading out of the key")
    b.add_argument("--no-plddt-by-atom", action="store_true", help="draw a residue's atoms at the residue's pLDDT")
    return b


def _residues_given(given: "list[str] | None") -> Any:
    """``--residues``: one word for all, or residue=how for some."""
    if given is None:
        return None
    if len(given) == 1 and given[0] in RESIDUE_FORMS:
        return given[0]
    table = {}
    for item in given:
        name, sep, how = item.partition("=")
        if not sep or not name:
            raise BindviewError(f"--residues {item}: one word ({', '.join(RESIDUE_FORMS)}) for all, or "
                                "RESIDUE=HOW for each that is named")
        table[name] = how
    for how in table.values():
        if how not in RESIDUE_FORMS:
            raise BindviewError(f"--residues: {how!r} is not one of {', '.join(RESIDUE_FORMS)}")
    return table


def _style_settings(args: argparse.Namespace) -> dict[str, Any]:
    return dict(residues=_residues_given(args.residues), lines=args.lines, exposure=args.exposure,
                lining=args.lining)


def _absent_given(text: "str | None") -> "list[str] | None":
    """``702,705-712,A:744`` as the names of the residues (a range is of numbers, with or without its chain).
    Nothing at all (``""``) is the other protein that lacks none; a list that only looks empty is an error."""
    if text is None:
        return None
    if not text.strip():
        return []
    out: list[str] = []
    for part in (p.strip() for p in text.split(",")):
        if not part:
            raise BindviewError(f"--absent {text}: an empty name between its commas; write the residues as "
                                "702,705-712,A:744")
        span = re.fullmatch(r"(?:([^:]+):)?(\d+)-(\d+)", part)
        if span is None:
            out.append(part)
            continue
        chain = f"{span[1]}:" if span[1] else ""
        if int(span[2]) > int(span[3]):
            raise BindviewError(f"--absent {text}: the range {part} runs backwards and holds no residue; "
                                f"write it as {chain}{span[3]}-{span[2]}")
        out += [f"{chain}{n}" for n in range(int(span[2]), int(span[3]) + 1)]
    return out


def _bare_argument(t: Any) -> None:
    t.add_argument("--bare", action="store_true",
                   help="the drawing and its key only: no confidence line, warning or notes (they are in --json "
                        "and --notes-file)")


def _output_arguments(o: Any) -> None:
    o.add_argument("--json", action="store_true", help="print status, warnings, notes and measures as JSON")
    o.add_argument("--notes-file", metavar="TXT",
                   help="write the figure's notes, one a line: the words a caption needs (see --bare)")
    o.add_argument("--graph", metavar="JSON",
                   help="also write the figure as a graph of nodes and contacts (docs/graph-schema.md)")


def _confidence_arguments(r: argparse.ArgumentParser) -> None:
    c = r.add_argument_group(
        "model confidence",
        "For predicted models. pLDDT is read from the B-factor column only where the file declares it\n"
        "(_ma_qa_metric) or you say it is, with --plddt; it is never guessed from the values.")
    c.add_argument("--plddt", choices=("auto", "bfactor", "bfactor01", "off"), default="auto",
                   help="auto: where the file declares it; bfactor, bfactor01: you say it is pLDDT on 0-100, "
                        "on 0-1; off: none")
    c.add_argument("--model-confidence", metavar="JSON",
                   help="ipTM and pTM: AlphaFold 3's summary_confidences.json (found beside <job>_model.cif)")
    c.add_argument("--untrusted", action="store_true", help="mark the figure as from a model not to be read as a result")
    c.add_argument("--flag", metavar="TEXT", help="your verdict on the model, printed on the figure (e.g. 'grade C')")


def _confidence_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return dict(plddt=None if args.plddt == "off" else args.plddt,
                model_confidence=args.model_confidence or "auto",
                trusted=False if args.untrusted else None, flag=args.flag)


@contextmanager
def _reading(what: str, path: "str | Path") -> "Iterator[None]":
    """Whatever a file that is read fails with is said with the file's name."""
    try:
        yield
    except BindviewError as exc:                             # (the reader's own words, which name no file)
        if str(path) in str(exc):
            raise
        raise BindviewError(f"{what} {path}: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise BindviewError(f"{what} {path} is not text in UTF-8: {exc}") from exc
    except OSError as exc:
        raise BindviewError(f"cannot read {what} {path}: {exc.strerror or exc}") from exc
    except ValueError as exc:                                # (a JSON file that is none)
        raise BindviewError(f"cannot read {what} {path}: {exc}") from exc


def _stem(path: Path) -> str:
    """The name of a model file without its suffixes: ``model`` of ``model.cif.gz``."""
    return path.name.removesuffix(".gz").rsplit(".", 1)[0]


def _folder_there(option: str, path: "str | Path | None") -> None:
    """A file that this module writes when the figure is made has a folder to go into: asked before it is made."""
    if path and not Path(path).parent.is_dir():
        raise BindviewError(f"{option} {path}: there is no folder {Path(path).parent} to write it into")


def _write(option: str, path: "str | Path", text: str) -> None:
    try:
        Path(path).write_text(text, encoding="utf-8")
    except OSError as exc:
        raise BindviewError(f"{option}: cannot write {path}: {exc.strerror or exc}") from exc


def _write_notes(args: argparse.Namespace, result: Any, say) -> None:
    """``--notes-file``: the figure's notes, one a line, whether they are drawn or left off."""
    if args.notes_file:
        _write("--notes-file", args.notes_file, "".join(note + "\n" for note in result.notes))
        say(f"notes -> {args.notes_file}")


def _smiles_from(path_text: str, structure: Path) -> str:
    path = Path(path_text)
    if not path.exists() and (structure.parent / path_text).exists():
        path = structure.parent / path_text
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise BindviewError(f"cannot read {path}: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise BindviewError(f"{path} is not text in UTF-8: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise BindviewError(f"{path} is not valid JSON: {exc}") from exc
    for getter in (lambda d: d["ligand"]["smiles"], lambda d: d["smiles"]):
        try:
            value = getter(data)
        except (KeyError, TypeError):
            continue
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise BindviewError(f"{path} has no 'smiles' or 'ligand.smiles' entry")


def _interactions(args: argparse.Namespace, structure: Path, say) -> "list[Interaction] | str | None":
    if getattr(args, "distances_only", False):
        say("interactions: from the distances of the heavy atoms")
        return DISTANCES
    if args.interactions:
        say(f"interactions: {args.interactions}")
        with _reading("the table of contacts", args.interactions):
            return load_interactions(args.interactions)
    if not args.detect:
        for name in _SIBLING_TABLES:
            cand = structure.parent / name
            if cand.exists():
                say(f"interactions: {cand} (found next to the structure)")
                with _reading("the table of contacts", cand):
                    return load_interactions(cand)
    say("interactions: detecting with PLIP")
    return None


def _render(args: argparse.Namespace) -> int:
    def say(msg: str) -> None:
        if not args.quiet:
            print(msg, file=sys.stderr)

    absent = _absent_given(args.absent)                      # (a list that is wrong is said before anything is read)
    _folder_there("--notes-file", args.notes_file)
    structure = Path(args.structure)
    smiles = args.smiles
    if args.smiles_from:
        smiles = _smiles_from(args.smiles_from, structure)
    records = _interactions(args, structure, say)
    dia = Diagram.from_structure(structure, args.ligand, smiles=smiles, interactions=records,
                                 ionize_acids=not args.no_ionize, ensemble=args.ensemble,
                                 ligand_as=args.ligand_as, detail=args.detail, covalent=args.covalent, ions=args.ions,
                                 pae=None if not args.pae else args.pae[0] if len(args.pae) == 1 else args.pae,
                                 **_confidence_kwargs(args))
    out = Path(args.output) if args.output else structure.with_name(f"{_stem(structure)}.{args.style}.svg")
    options: dict[str, Any] = dict(
        title=args.title, subtitle=args.subtitle, legend=not args.no_legend, notes=not args.no_notes,
        distance_labels=args.distances, show_chain=args.show_chain, figure_width_mm=args.width_mm,
        min_font_pt=args.min_font_pt, bond_length=args.bond_length, on_unreadable=args.on_unreadable, effort=args.effort,
        background=not args.no_background, burial=args.burial, solvent_sites=args.solvent_sites,
        colors=args.colors, plddt_colors="bonds" if args.plddt_bonds else args.plddt_colors, interactive=args.interactive,
        break_lines=not args.no_break_lines, charge_halos=args.charge_halos,
        charge_key=not args.no_charge_key, plddt_by_atom=not args.no_plddt_by_atom,
        occupancy_style=args.occupancy_style, occupancy_labels=args.occupancy_labels, bare=args.bare,
        solvent_mark=args.solvent_mark, absent=absent, absent_label=args.absent_label,
        absent_style=args.absent_style, atom_labels=args.atom_labels, stylesheet="scoped" if args.scoped_css else "inline",
        **_style_settings(args))
    given = Graph.from_json(Path(args.layout)) if args.layout else None
    if args.arrange_page:
        made = dia.arrange(args.arrange_page, style=args.style, theme=args.theme, graph=given, **options)
        say(f"wrote {made}")
        return 0
    if args.arrange:
        given = dia.arrange(style=args.style, theme=args.theme, graph=given, **options)      # type: ignore[assignment]
        if given is None:
            raise BindviewError("the page was not done: nothing is drawn")
        options["on_unreadable"] = "render"                  # a layout made by hand is drawn as it was left
    result = dia.render_svg(out, style=args.style, theme=args.theme, graph=given, **options)
    if args.graph:
        if result.graph is None:
            say(f"no graph written: the result is a {result.figure} figure, which has no nodes")
        else:
            result.graph.to_json(args.graph)
            say(f"graph -> {args.graph}")

    q = result.quality
    summary: dict[str, Any] = {
        "status": result.status, "figure": result.figure, "output": str(result.path) if result.path else None,
        "style": result.style, "theme": result.theme, "n_interactions": dia.n_interactions,
        "n_unresolved": len(dia.unresolved), "warnings": result.warnings, "notes": result.notes,
        "confidence": result.confidence,
        "burial": result.burial, **result.data,
        "quality": None if q is None else {
            "width": round(q.width, 2), "height": round(q.height, 2), "bond_length": q.bond_length,
            "n_residues": q.n_residues, "overlaps": q.overlaps, "unplaced": q.unplaced,
            "line_crossings": q.line_crossings, "bond_crossings": q.bond_crossings,
            "max_stretch": round(q.max_stretch, 3), "min_font_pt": round(q.min_font_pt, 2),
            # (No width at all where the text asked for is larger than the figure's own: null, not a number.)
            "required_width_mm": None if math.isinf(q.required_width_mm) else round(q.required_width_mm, 1),
            "readable": q.readable},
    }
    _write_notes(args, result, say)
    if args.json:
        json.dump(summary, sys.stdout, indent=2)
        sys.stdout.write("\n")
    where = f" -> {result.path}" if result.path else " (no figure written)"
    say(f"{result.status}: {result.figure}{where}")
    for w in result.warnings:
        say(f"  note: {w}")
    return 3 if result.status == "unreadable" else 0


#: The model in a folder that is a member of a comparison, unless its meta.json names another file.
_MODEL_NAMES = ("model.cif", "model.cif.gz", "model.pdb", "model.pdb.gz")


def _member(text: str, args: argparse.Namespace, say) -> tuple[str, Diagram]:
    """Build one ligand of a set from a structure file or a case directory."""
    path = Path(text)
    meta: dict[str, Any] = {}
    meta_file = (path if path.is_dir() else path.parent) / "meta.json"
    if meta_file.exists():
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:                  # (not there, not text in UTF-8, not JSON)
            raise BindviewError(f"cannot read {meta_file}: {exc}") from exc
    structure = path
    if path.is_dir():
        named = meta.get("source", {}).get("file") if isinstance(meta.get("source"), dict) else None
        names = ([named] if named else []) + list(_MODEL_NAMES)
        structure = next((path / n for n in names if (path / n).exists()), path / names[0])
    if not structure.exists():
        raise BindviewError(f"no structure at {structure}")
    lig: dict[str, Any] = meta["ligand"] if isinstance(meta.get("ligand"), dict) else {}
    ligand = args.ligand or lig.get("residue_name_in_cif")
    if not ligand:
        raise BindviewError(f"{structure}: say which residue is the ligand with --ligand")
    table = next((structure.parent / n for n in _SIBLING_TABLES if (structure.parent / n).exists()), None)
    name = lig.get("name") or (path.name if path.is_dir() else _stem(structure))
    untrusted = name in getattr(args, "untrusted", [])
    records = None
    if table is not None:
        with _reading("the table of contacts", table):
            records = load_interactions(table)
    dia = Diagram.from_structure(structure, ligand, smiles=lig.get("smiles") or meta.get("smiles"),
                                 interactions=records, name=name,
                                 plddt=None if args.plddt == "off" else args.plddt,
                                 trusted=False if untrusted else None)
    say(f"{name}: {dia.n_interactions} interactions" + ("" if table is not None else " (detected)"))
    return name, dia


def _compare(args: argparse.Namespace) -> int:
    def say(msg: str) -> None:
        if not args.quiet:
            print(msg, file=sys.stderr)

    if len(args.members) < 2:
        raise BindviewError("a comparison needs at least two ligands")
    built = [_member(m, args, say) for m in args.members]
    unknown = [n for n in args.untrusted if n not in [b[0] for b in built]]
    if unknown:
        raise BindviewError(f"--untrusted names no ligand of the set: {', '.join(unknown)}")
    cmp_ = Comparison([d for _, d in built], [n for n, _ in built], min_fraction=args.min_fraction)
    result = cmp_.render_svg(args.output, view=args.view, theme=args.theme, title=args.title,
                             subtitle=args.subtitle, min_ligands=args.min_ligands,
                             thumbnails=not args.no_thumbnails, figure_width_mm=args.width_mm,
                             min_font_pt=args.min_font_pt, on_unreadable=args.on_unreadable, burial=args.burial,
                             colors=args.colors)
    if args.fingerprints:
        cmp_.table.to_csv(args.fingerprints)
        say(f"fingerprints -> {args.fingerprints}")
    if args.graph and result.graph is not None:
        result.graph.to_json(args.graph)
        say(f"graph -> {args.graph}")
    if args.json:
        json.dump({"status": result.status, "figure": result.figure, "view": result.style,
                   "output": str(result.path) if result.path else None, "warnings": result.warnings,
                   "confidence": result.confidence, "burial": result.burial, **result.data}, sys.stdout, indent=2)
        sys.stdout.write("\n")
    say(cmp_.core.describe())
    where = f" -> {result.path}" if result.path else " (no figure written)"
    say(f"{result.status}: {result.figure}{where}")
    for w in result.warnings:
        say(f"  note: {w}")
    return 3 if result.status == "unreadable" else 0


def _fingerprint(args: argparse.Namespace) -> int:
    def say(msg: str) -> None:
        if not args.quiet:
            print(msg, file=sys.stderr)

    from .fingerprint import FingerprintTable

    built = [_member(m, args, say) for m in args.members]
    table = FingerprintTable([d.fingerprint(n) for n, d in built])
    text = table.to_csv(args.output, form=args.form, counts=args.counts)
    if not args.output:
        sys.stdout.write(text)
    else:
        say(f"{len(built)} fingerprint(s), {len(table.columns)} columns -> {args.output}")
    return 0


def _interface(args: argparse.Namespace) -> int:
    from .interface import Interface, load_interface_contacts

    def say(msg: str) -> None:
        if not args.quiet:
            print(msg, file=sys.stderr)

    _folder_there("--notes-file", args.notes_file)
    structure = Path(args.structure)
    contacts = None
    if args.contacts:
        with _reading("the table of contacts", args.contacts):
            contacts = load_interface_contacts(args.contacts)
    say(f"contacts: {args.contacts}" if args.contacts else "contacts: detecting with PLIP (peptide mode)")
    face = Interface(structure, args.between, contacts, spine=args.spine, site=args.site, radius=args.radius,
                     max_residues=args.max_residues, order=args.order,
                     pae=None if not args.pae else args.pae[0] if len(args.pae) == 1 else args.pae,
                     layout=args.layout, detail=args.detail, ipsae=args.ipsae,
                     model_confidence=args.model_confidence or "auto")
    out = Path(args.output) if args.output else structure.with_name(f"{_stem(structure)}.interface.{args.style}.svg")
    result = face.render_svg(out, style=args.style, theme=args.theme, title=args.title, colors=args.colors,
                             plddt_colors="bonds" if args.plddt_bonds else args.plddt_colors, interactive=args.interactive,
                             break_lines=not args.no_break_lines, charge_halos=args.charge_halos,
                             charge_key=not args.no_charge_key, plddt_by_atom=not args.no_plddt_by_atom,
                             bare=args.bare, pae_colors=True if args.pae_colors else None, **_style_settings(args))
    _write_notes(args, result, say)
    if args.graph and result.graph is not None:
        result.graph.to_json(args.graph)
        say(f"graph -> {args.graph}")
    site = face.site
    say(f"spine: {face.spine.name}; {len(site.residues)} of {site.of_residues} residues at the interface, "
        f"{site.contacts} of {site.of_contacts} contacts ({site.rule})")
    drawn = result.data.get("interface", {})
    if drawn.get("layout_rule"):
        say(f"layout: {drawn['layout_rule']}")
    for s in face.ipsae:
        say(f"ipSAE (PAE below {s.cutoff:g} angstrom): {s.forward:.2f} aligned on {s.first} and scored over "
            f"{s.second}, {s.backward:.2f} the reverse")
    if args.json:
        json.dump({"status": result.status, "figure": result.figure, "output": str(result.path) if result.path
                   else None, "warnings": result.warnings, "notes": result.notes, **result.data}, sys.stdout,
                  indent=2)
        sys.stdout.write("\n")
    say(f"{result.status}: {result.figure}" + (f" -> {result.path}" if result.path else " (no figure written)"))
    for w in result.warnings:
        say(f"  note: {w}")
    return 3 if result.status == "unreadable" else 0


def _read_pairs(path: Path) -> list:
    from .basepairs import read_fr3d_pairs, read_mmcif_pairs, read_rnapolis_pairs

    name = path.name.lower()
    with _reading("the pairs of", path):
        if name.endswith(".json"):
            return read_rnapolis_pairs(path)
        if name.endswith(".csv"):
            return read_fr3d_pairs(path)
        return read_mmcif_pairs(path)


def _read_stacks(path: Path) -> list:
    from .basepairs import read_fr3d_stacks, read_rnapolis_stacks

    with _reading("the stacks of", path):
        return read_rnapolis_stacks(path) if path.name.lower().endswith(".json") else read_fr3d_stacks(path)


def _basepairs(args: argparse.Namespace) -> int:
    from .basepairs import (BasePairs, pairs_from_contacts, pairs_from_dot_bracket, pairs_within_from_table,
                            read_mmcif_pairs)
    from .distance_table import interface_contacts
    from .structure import read_structure

    def say(msg: str) -> None:
        if not args.quiet:
            print(msg, file=sys.stderr)

    _folder_there("--notes-file", args.notes_file)
    structure = Path(args.structure)
    model = read_structure(structure)
    a, b = args.chains
    contacts = [] if args.no_contacts else interface_contacts(model, [a], [b], stacking=True)
    pairs = [p for f in args.pairs for p in _read_pairs(Path(f))]
    said = [f"pairs: {', '.join(args.pairs)}"] if args.pairs else []
    bracketed = set()
    for spec in args.dot_bracket:
        chain, sep, text = spec.partition("=")
        if not sep:
            raise BindviewError(f"--dot-bracket {spec!r}: write it as CHAIN=STRUCTURE")
        pairs += pairs_from_dot_bracket(model, chain.strip(), text.strip(), family=args.family)
        bracketed.add(chain.strip())
        said.append(f"pairs within {chain.strip()}: the dot-bracket")
    if not args.pairs:
        own = read_mmcif_pairs(structure) if structure.name.lower().endswith((".cif", ".cif.gz", ".mmcif")) else []
        if own:
            pairs += own
            said.append("pairs: the PDB's annotation in the file")
        elif not args.no_contacts:
            pairs += pairs_from_contacts(contacts)[0]
            for chain in (a, b):
                if chain not in bracketed:
                    pairs += pairs_within_from_table(model, chain)
            said.append("pairs" + (" across" if bracketed else "") + ": the plain distance table's stand-in, "
                        "without a family")
    from_json = [f for f in args.pairs if f.lower().endswith(".json")]
    stack_files = args.stacks or from_json                      # left out: an RNApolis --pairs has them
    stacks = [st for f in stack_files for st in _read_stacks(Path(f))] if stack_files else None
    figure = BasePairs(model, [a, b], pairs, stacks=stacks, contacts=contacts, ligand=args.ligand,
                       ligand_contacts=args.ligand_contacts, ligand_name=args.ligand_name,
                       ligand_smiles=args.ligand_smiles)
    graph = Graph.from_json(Path(args.from_graph)) if args.from_graph else None
    out = Path(args.output) if args.output else structure.with_name(f"{_stem(structure)}.basepairs.svg")
    result = figure.render_svg(out, theme=args.theme, colors=args.colors, title=args.title,
                               legend=not args.no_legend, caption=not args.no_caption, step=args.step,
                               figure_width_mm=args.width_mm, min_font_pt=args.min_font_pt,
                               interactive=args.interactive, graph=graph)
    for line in said:
        say(line)
    _write_notes(args, result, say)
    if args.graph and result.graph is not None:
        result.graph.to_json(args.graph)
        say(f"graph -> {args.graph}")
    say(f"{result.data['pairs_across']} pairs across, {result.data['pairs_within']} within, "
        f"{result.data['stacks']} stacks ({result.data['layout']})")
    unreadable = result.status == "unreadable"
    status = result.status
    if args.json:
        json.dump({"status": status, "figure": "basepairs", "output": str(result.path) if result.path
                   else None, "warnings": result.warnings, "notes": result.notes, **result.data}, sys.stdout,
                  indent=2)
        sys.stdout.write("\n")
    say(f"{status}: base pairs" + (f" -> {result.path}" if result.path else " (no figure written)"))
    for w in result.warnings:
        say(f"  note: {w}")
    return 3 if unreadable else 0


def _report(args: argparse.Namespace) -> int:
    from .report import survey, write_each

    def say(msg: str) -> None:
        if not args.quiet:
            print(msg, file=sys.stderr)

    given: dict[str, str] = {}
    for item in args.smiles:
        name, sep, smiles = item.partition("=")
        if not (sep and name and smiles):
            raise BindviewError(f"--smiles takes NAME=SMILES, not {item!r}")
        given[name] = smiles
    parts: list[tuple[str, str, int, int]] = []
    for item in args.domain:
        name, sep, where = item.partition("=")
        chain, colon, span = where.partition(":")
        ends = re.fullmatch(r"(-?[0-9]+)-(-?[0-9]+)", span)             # (a residue number may be negative: -5-10)
        if not (sep and colon and name and chain and ends):
            raise BindviewError(f"--domain takes NAME=CHAIN:FROM-TO, not {item!r}")
        parts.append((name, chain, int(ends[1]), int(ends[2])))
    if len(parts) == 1:
        raise BindviewError("one --domain names one part: an interface is between two")
    if args.each:
        if args.look or parts:
            raise BindviewError("--each draws every input of a folder: not with --look or --domain")
        say(f"{args.input}: a page for each input")
        rows, index = write_each(args.input, args.output, given, views=not args.one_view, say=say)
        drawn = sum(1 for r in rows if r.first is not None)
        say(f"{drawn} of {len(rows)} inputs have a figure; index -> {index} and {index.with_suffix('.csv')}")
        return 0
    source = Path(args.input)
    out = Path(args.output) if args.output else source.parent / (
        (_stem(source) if source.is_file() else source.name) + ".report.html")
    if not args.look:
        _folder_there("-o", args.output)                     # (without -o the page goes beside the input)
    found = survey(args.input, given, parts or None)
    if found.kind == "series":
        say(f"{args.input}: a series of {len(found.members)} ligands on one protein: "
            + ", ".join(m.name for m in found.members))
    say(f"{args.input}: " + ", ".join(
        f"chain {c.name}: {c.what if c.kind else 'no polymer'}" + (f" of {c.residues}" if c.residues else "")
        for c in found.found.chains))
    say("  other residues: " + (", ".join(f"{c.name} x{c.copies} ({c.kind}, {c.heavy} heavy atom{'s' if c.heavy != 1 else ''})"
                                          for c in found.found.candidates) or "none")
        + f"; waters: {found.found.waters}; models in the file: {found.models_in_file}")
    if not args.look:
        found.draw(views=not args.one_view)
    for track in found.tracks:
        say(f"  {track.kind}: {track.title}: " + (track.status or "to draw")
            + (", in the schematic style: the detailed one is not readable" if track.schematic else ""))
    for what, why in found.left:
        say(f"  no figure: {what}: {why}")
    if args.look:
        return 0
    _write("-o", out, found.html())
    say(f"report -> {out}")
    return 0


def _said(exc: Exception) -> str:
    """An exception that is no BindviewError, as a message."""
    if isinstance(exc, UnicodeDecodeError):
        return f"a file that was read is not text in UTF-8: {exc}"
    if isinstance(exc, OSError) and exc.filename is not None:
        return f"{exc.strerror or type(exc).__name__}: {exc.filename}"
    return str(exc) or type(exc).__name__


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "render":
            return _render(args)
        if args.command == "interface":
            return _interface(args)
        if args.command == "basepairs":
            return _basepairs(args)
        if args.command == "compare":
            return _compare(args)
        if args.command == "fingerprint":
            return _fingerprint(args)
        return _report(args)                                  # (the parser takes no other command)
    except BindviewError as exc:
        print(f"bindview: error: {exc}", file=sys.stderr)
        return 2
    except (OSError, ValueError) as exc:
        # What a wrong input raises where nothing made it a BindviewError: a file that cannot be read or written, one
        # that is not text (UnicodeDecodeError is a ValueError), a value that the library refuses. A mistake of the
        # program is none of these as a rule (TypeError, KeyError, IndexError), and keeps its traceback.
        print(f"bindview: error: {_said(exc)}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
