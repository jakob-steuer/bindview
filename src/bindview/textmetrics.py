"""Text extents without a font engine.

Layout has to know how wide a label is before anything is drawn, and it has to
give the same answer on every machine (hard requirement: deterministic output).
So widths come from a fixed table -- the advance widths of Helvetica, with which
Arial and Liberation Sans are metric-compatible -- rather than from whatever
font happens to be installed. The SVG asks for exactly that font family.
"""

from __future__ import annotations

__all__ = ["text_width", "CAP_HEIGHT", "DESCENT", "FONT_FAMILY"]

FONT_FAMILY = "Helvetica, Arial, 'Liberation Sans', sans-serif"

#: Vertical metrics as a fraction of the font size.
CAP_HEIGHT = 0.718
DESCENT = 0.207

_REGULAR = {
    " ": 278, "!": 278, '"': 355, "#": 556, "$": 556, "%": 889, "&": 667, "'": 191, "(": 333, ")": 333,
    "*": 389, "+": 584, ",": 278, "-": 333, ".": 278, "/": 278, ":": 278, ";": 278, "<": 584, "=": 584,
    ">": 584, "?": 556, "@": 1015, "[": 278, "\\": 278, "]": 278, "^": 469, "_": 556, "`": 333,
    "{": 334, "|": 260, "}": 334, "~": 584,
    "A": 667, "B": 667, "C": 722, "D": 722, "E": 667, "F": 611, "G": 778, "H": 722, "I": 278, "J": 500,
    "K": 667, "L": 556, "M": 833, "N": 722, "O": 778, "P": 667, "Q": 778, "R": 722, "S": 667, "T": 611,
    "U": 722, "V": 667, "W": 944, "X": 667, "Y": 667, "Z": 611,
    "a": 556, "b": 556, "c": 500, "d": 556, "e": 556, "f": 278, "g": 556, "h": 556, "i": 222, "j": 222,
    "k": 500, "l": 222, "m": 833, "n": 556, "o": 556, "p": 556, "q": 556, "r": 333, "s": 500, "t": 278,
    "u": 556, "v": 500, "w": 722, "x": 500, "y": 500, "z": 500,
    "Å": 667, "°": 400, "·": 278, "–": 556, "—": 1000, "−": 584,
    "π": 549, "×": 584, "′": 247, "→": 1000, "†": 556, "≥": 549, "≤": 549,
    "…": 1000, "±": 584,
}

_BOLD = {
    " ": 278, "!": 333, '"': 474, "#": 556, "$": 556, "%": 889, "&": 722, "'": 238, "(": 333, ")": 333,
    "*": 389, "+": 584, ",": 278, "-": 333, ".": 278, "/": 278, ":": 333, ";": 333, "<": 584, "=": 584,
    ">": 584, "?": 611, "@": 975, "[": 333, "\\": 278, "]": 333, "^": 584, "_": 556, "`": 333,
    "{": 389, "|": 280, "}": 389, "~": 584,
    "A": 722, "B": 722, "C": 722, "D": 722, "E": 667, "F": 611, "G": 778, "H": 722, "I": 278, "J": 556,
    "K": 722, "L": 611, "M": 833, "N": 722, "O": 778, "P": 667, "Q": 778, "R": 722, "S": 667, "T": 611,
    "U": 722, "V": 667, "W": 944, "X": 667, "Y": 667, "Z": 611,
    "a": 556, "b": 611, "c": 556, "d": 611, "e": 556, "f": 333, "g": 611, "h": 611, "i": 278, "j": 278,
    "k": 556, "l": 278, "m": 889, "n": 611, "o": 611, "p": 611, "q": 611, "r": 389, "s": 556, "t": 333,
    "u": 611, "v": 556, "w": 778, "x": 556, "y": 556, "z": 500,
    "Å": 722, "°": 400, "·": 278, "–": 556, "—": 1000, "−": 584,
    "π": 549, "×": 584, "′": 247, "→": 1000, "†": 556, "≥": 549, "≤": 549,
    "…": 1000, "±": 584,
}
for _d in "0123456789":
    _REGULAR[_d] = 556
    _BOLD[_d] = 556
for _sub in "\u2080\u2081\u2082\u2083\u2084":        # subscript digits
    _REGULAR[_sub] = 370
    _BOLD[_sub] = 370

_DEFAULT = 600


def text_width(text: str, size: float, bold: bool = False) -> float:
    """Advance width of ``text`` at ``size`` (same units as ``size``)."""
    table = _BOLD if bold else _REGULAR
    return sum(table.get(ch, _DEFAULT) for ch in text) * size / 1000.0
