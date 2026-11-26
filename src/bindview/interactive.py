"""A figure that answers the pointer, with a style sheet and no script.

``render(interactive=True)`` asks for it. Pointing at a residue, or at one of its contacts, keeps the
residue, its contacts and what they reach, steps the rest of the residues and contacts far back, and draws
the lines of the contacts heavier, so that what is pointed at is what the figure shows. A click keeps that
until a click elsewhere, and the tab key goes from one residue to the next.

Nothing is added to what a figure draws. A figure already says of every residue which one it is, of every
contact which residue makes it, and of every residue of a chain ligand which atoms are its own
(``data-residue``, ``data-ligand-atoms``, ``data-atoms``). From that this module gives each of those groups
a few classes and lets it take the focus, and writes one rule per residue. A browser that shows the file
as a document or as part of a page follows the rules; everything else (an ``<img>``, a PDF, a drawing
program) shows the figure as it is without them.

A figure of a ligand that is drawn as atoms has two things beside, in a group of their own at its end
(:mod:`bindview.styles.closeup`). The atoms of every residue that is drawn without them, which take its place
while it is pointed at and are not shown otherwise (``display="none"``, which every reader of the file
honours). And a place to point at on every atom of the ligand, which is no ink: pointing at an atom that makes
contacts keeps those and the residues they are with.

A page around the figure can ask for the same, with one attribute on its root: ``<svg data-active="712">`` is
what pointing at residue 712 is, until the attribute is taken off or says another. The number is the one that
``data-residue-number`` says of the residue and its contacts (with its insertion code, ``60A``). A number that
the figure does not have changes nothing; one that two chains have lights both, and there ``A:712`` lights one.
Whoever points into the figure, or has clicked in it, is followed first.

The classes, where ``X`` is a residue of the protein (``r712``: its number; where two have the same, each is
``r712_A`` with its chain and ``r712`` as well), of a chain ligand (``u0``, ...), or an atom of the ligand that makes a
contact (``a0``, ...). What is no letter, digit or hyphen in a number or in the name of a chain is in a class by
its code (``'`` is ``__27_``); ``data-active`` says it as it is:

``lf-part``  a residue, a residue of the chain, a contact, or such an atom: what can step back
``lf-t-X``   pointing at this is pointing at ``X`` (``X`` itself, and the contacts of a protein residue)
``lf-o-X``   this is ``X`` or one of its contacts
``lf-p-X``   this is in contact with ``X``
``lf-m-X``   the atoms of ``X``, shown while it is pointed at
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from .scene import Group, Line, Scene

__all__ = ["Hooks", "hooks", "hooks_of_links", "ROOT_CLASS", "DIMMED", "HEAVY", "NOTHING_TO_POINT_AT", "OVER",
           "ACTIVE", "NUMBER", "UNDER"]

#: On the root of a figure that answers the pointer.
ROOT_CLASS = "lf-live"
#: How much is left of what steps back.
DIMMED = 0.12
#: The group at the end of a figure that holds what is not drawn until it is pointed at.
OVER = "lf-over"
#: The attribute of the root by which a page says which residue it means.
ACTIVE = "data-active"
#: What a group says of its residue's number (``data-residue-number``). Written with the hooks or not at all.
NUMBER = "residue-number"
#: The ring around an atom of the ligand that is pointed at.
_RING = "#b45309"
#: How heavy the line of a contact is drawn while a residue is pointed at (what stepped back is faint all the same).
HEAVY = 2.6
#: The class of a line in the group of a contact that is wider than ``HEAVY``: it lies under the contact's line (the
#: glow of a PAE, the highlight of a covalent bond, whole or in the pieces left around a word) and keeps its width.
UNDER = "lf-under"
#: The lines of a contact that are drawn heavier.
_LINES = f":is(line,polyline,path):not(.{UNDER})"
#: What a result says where the option was asked for and changes nothing.
NOTHING_TO_POINT_AT = ("interactive was asked for, and this figure has no residue to point at: it is the figure it "
                       "would be without")


@dataclass
class Hooks:
    """What is added to a figure: classes for some of its groups (by the group's ``id``), the groups
    that take the focus (0: in the order of the tab key; -1: by a click only), and the rules."""

    classes: dict[int, str] = field(default_factory=dict)
    focus: dict[int, int] = field(default_factory=dict)
    css: str = ""
    #: The same, rule by rule, where each says the same in every figure that has it.
    shared: list[str] = field(default_factory=list)
    #: The groups that are written hidden.
    hidden: set[int] = field(default_factory=set)
    #: The lines that are written with the class ``UNDER`` (by their ``id``).
    under: set[int] = field(default_factory=set)


def _kind(group: Group) -> str:
    return group.cls.split(" ", 1)[0]


def _names(text: str) -> set[str]:
    return {n for n in text.split(",") if n}


def hooks(scene: Scene, over: "Sequence[Group]" = ()) -> Hooks | None:
    """The hooks of a figure, or ``None`` for one with nothing to point at (a table, an empty figure).
    ``over``: what is written after the figure (:mod:`bindview.styles.closeup`)."""
    residues: dict[str, list[Group]] = {}          # which residue: its groups (a large one is drawn in parts)
    units: list[Group] = []
    contacts: list[Group] = []
    marks: list[Group] = []                        # what is marked of a residue (bindview.styles.absence)

    def collect(items: list) -> None:
        for it in items:
            if not isinstance(it, Group):
                continue
            kind = _kind(it)
            if kind == "residue" and it.data.get("residue") and it.items:
                residues.setdefault(it.data["residue"], []).append(it)
            elif kind == "unit" and it.items:
                units.append(it)
            elif kind == "absence" and it.data.get("residue") and it.items:
                marks.append(it)
            elif kind in ("interaction", "contact") and it.data.get("residue") and it.items:
                contacts.append(it)
            else:
                collect(it.items)

    collect(scene.items)
    if not residues and not units:
        return None
    token, numbered, also = _tokens(residues)
    atoms = [(_names(u.data.get("atoms", "")), f"u{n}") for n, u in enumerate(units)]

    # The atoms of the ligand that make a contact, by name: a contact that is not drawn as a line counts too.
    hits = [g for g in over if _kind(g) == "hit" and any(r in token for r in _names(g.data.get("residues", "")))]
    spot = {g.data["atom"]: f"a{n}" for n, g in enumerate(hits)}

    own: dict[int, list[str]] = {}                 # a group: whose it is
    target: dict[int, str] = {}                    # a group: pointing at it is pointing at whom
    near: dict[str, set[str]] = {}                 # who: in contact with whom
    for c in contacts:
        mine = [token[c.data["residue"]]] if c.data["residue"] in token else []
        reached = _names(c.data.get("ligand-atoms", ""))
        theirs = [t for names, t in atoms if names & reached]
        if not mine and not theirs:
            continue
        own[id(c)] = mine + theirs
        target[id(c)] = (mine or theirs)[0]
        for a in mine:
            for b in theirs:
                near.setdefault(a, set()).add(b)
                near.setdefault(b, set()).add(a)

    out = Hooks()
    at_atoms: dict[int, list[str]] = {}            # a contact: the atoms of the ligand that make it
    for c in contacts:
        if id(c) not in own:
            continue
        there = [spot[n] for n in sorted(_names(c.data.get("ligand-atoms", ""))) if n in spot]
        at_atoms[id(c)] = there
    for g in hits:
        for ident in _names(g.data["residues"]):
            if ident in token:
                near.setdefault(token[ident], set()).add(spot[g.data["atom"]])

    rank = {t: n for n, t in enumerate([*(t for _, t in atoms), *spot.values(), *token.values()])}

    def named(prefix: str, whos: "Sequence[str]") -> list[str]:
        # (A residue whose number another has is called by the number alone as well: _tokens.)
        return [f"lf-{prefix}-{t}" for t in dict.fromkeys(t for w in whos for t in (w, *also.get(w, ())))]

    def classes(group: Group, who: str, owners: list[str], partners: bool) -> None:
        parts = ["lf-part"] + named("t", [who]) + named("o", owners)
        if partners:
            parts += named("p", sorted(near.get(who, ()), key=rank.__getitem__))
        out.classes[id(group)] = " ".join(parts)

    for ident, groups in residues.items():
        for g in groups:
            classes(g, token[ident], [token[ident]], True)
            out.focus[id(g)] = 0
    for g in marks:
        if g.data["residue"] in token:                # it stays with its residue, and steps back with it
            classes(g, token[g.data["residue"]], [token[g.data["residue"]]], True)
    for n, u in enumerate(units):
        classes(u, f"u{n}", [f"u{n}"], True)
        out.focus[id(u)] = 0
    for c in contacts:
        if id(c) in own:
            classes(c, target[id(c)], own[id(c)] + at_atoms[id(c)], False)
            out.focus[id(c)] = -1
            out.under.update(_wide(c))
    shown: list[str] = []                          # the residues that have atoms to show
    for g in over:
        if _kind(g) == "more" and g.data.get("residue") in token:
            who = token[g.data["residue"]]
            shown.append(who)
            out.classes[id(g)] = f"lf-m-{who}"
            out.hidden.add(id(g))
        elif _kind(g) == "hit" and g.data["atom"] in spot:
            classes(g, spot[g.data["atom"]], [spot[g.data["atom"]]], False)
            out.focus[id(g)] = -1

    root = f"svg.bindview.{ROOT_CLASS}"
    rules = [f"{root} .lf-part{{transition:opacity .12s;cursor:pointer}}",
             f"{root} .lf-part:focus:not(:focus-visible){{outline:none}}",
             f"{root}:has(.lf-part:is(:hover,:focus)) .lf-part:is(.interaction,.contact) {_LINES}"
             f"{{stroke-width:{HEAVY:g}px}}"]
    if over:
        rules.append(f"{root} .more{{pointer-events:none}}{root} .hit{{cursor:pointer}}"
                     f"{root} .hit circle{{pointer-events:all}}{root} .hit:is(:hover,:focus) circle"
                     f"{{stroke:{_RING};stroke-width:1.6px}}")
    for who in list(dict.fromkeys(token.values())) + [t for _, t in atoms] + list(spot.values()):
        # A residue that has the focus is kept; pointing counts where nothing has the focus.
        at = f":is(:has(.lf-t-{who}:focus),:not(:has(.lf-part:focus)):has(.lf-t-{who}:hover))"
        rules.append(f"{root}{at} .lf-part:not(.lf-o-{who},.lf-p-{who}){{opacity:{DIMMED:g}}}")
        if who in shown:
            # Its atoms take its place: it steps back with the rest, and its lines go with it. (Said of a figure
            # that has them: a page with several figures in it reads the rules of each for all.)
            rules.append(f"{root}:has(.lf-m-{who}){at} .lf-part.lf-o-{who}{{opacity:{DIMMED:g}}}"
                         f"{root}{at} .lf-m-{who}{{display:inline}}")
    out.css = "".join(rules) + _asked_for(numbered, set(shown))
    out.shared = _RULE.findall("".join(rules) + _asked_for(numbered, set(shown), each=True))
    return out


_RULE = re.compile(r"[^{}]+\{[^{}]*\}")


def _wide(group: Group) -> "list[int]":
    """The lines of a contact's group that drawing it heavier would make thinner."""
    out: list[int] = []
    for it in group.items:
        if isinstance(it, Group):
            out += _wide(it)
        elif isinstance(it, Line) and it.paint.stroke is not None and it.paint.width > HEAVY:
            out.append(id(it))
    return out


def _name(text: str) -> str:
    """``text`` as part of a class name: a letter, a digit and a hyphen as they are, anything else by its code
    (``'`` is ``__27_``). The name is then one word of a ``class`` attribute and needs no escape in a selector."""
    return "".join(c if (c.isascii() and c.isalnum()) or c == "-" else f"__{ord(c):x}_" for c in text)


def _quoted(text: str) -> str:
    """``text`` as a string in a selector. The style sheet is written into the file as it is, so what would end
    the string (a quote, a backslash) and what is markup in XML is said by its code, as CSS has it."""
    return '"' + "".join(c if c.isprintable() and c not in '"\\<>&' else f"\\{ord(c):x} " for c in text) + '"'


def _tokens(residues: "dict[str, list[Group]]"
            ) -> "tuple[dict[str, str], dict[str, tuple[str, list[str]]], dict[str, tuple[str, ...]]]":
    """What each residue is called in the classes; for everything that ``data-active`` can say (a number, and
    ``chain:number`` where more than one residue has the number), the name that it keeps and the residues that
    are meant; and what a residue whose number another has is called as well.

    A name follows from the residue alone, never from what else the figure holds: a page reads the rules of each
    of its figures for all of them. So a residue whose number another has is named with its chain (``r25_A``), and
    by the number alone too (``r25``), which is what every residue of that number answers to in any figure.
    """
    numbers: dict[str, list[str]] = {}
    for n, (ident, groups) in enumerate(residues.items()):
        numbers.setdefault(str(groups[0].data.get(NUMBER) or f"x{n}"), []).append(ident)
    token: dict[str, str] = {}
    numbered: dict[str, tuple[str, list[str]]] = {}
    also: dict[str, tuple[str, ...]] = {}
    for number, idents in numbers.items():
        name = f"r{_name(number)}"
        for ident in idents:
            if len(idents) == 1:
                token[ident] = name
                continue
            chain, colon, _ = ident.partition(":")
            chain = chain if colon else ""
            token[ident] = f"{name}_{_name(chain)}"
            also[token[ident]] = (name,)
            numbered[f"{chain}:{number}"] = (token[ident], [token[ident]])
        numbered[number] = (name, list(dict.fromkeys(token[ident] for ident in idents)))
    return token, numbered, also


def _asked_for(numbered: "dict[str, tuple[str, list[str]]]", shown: "set[str]", each: bool = False) -> str:
    """The rules by which the page around a figure asks for what pointing at a residue is (``data-active``).

    Each is said of a figure that has the residue: a page with several figures in it reads the rules of each for
    all, and a figure without the residue is left as it is. ``each``: no rule is of more than one residue.
    """
    if not numbered:
        return ""
    root = f"svg.{ROOT_CLASS}"
    # Said once, of a figure in which nothing is pointed at and nothing has the focus; what is asked for follows it.
    # (The rules of pointing are the more specific ones, so where they speak of a group they win.)
    rules = [f"{root}:not(:has(.lf-part:is(:hover,:focus))){{--lf-faint:{DIMMED:g};--lf-shown:inline}}"]
    any_of = []
    for said, (kept, whos) in numbered.items():
        asked = f"{root}[{ACTIVE}={_quoted(said)}]"
        here = f":has(.lf-t-{kept})"
        any_of.append(f"[{ACTIVE}={_quoted(said)}]{here}")
        rules.append(f"{asked}{here} .lf-part:not(.lf-o-{kept},.lf-p-{kept}){{opacity:var(--lf-faint,1)}}")
        for w in whos:
            if w in shown:
                rules.append(f"{asked}:has(.lf-m-{w}) .lf-part.lf-o-{w}{{opacity:var(--lf-faint,1)}}"
                             f"{asked} .lf-m-{w}{{display:var(--lf-shown,none)}}")
    heavy = f" .lf-part:is(.interaction,.contact) {_LINES}{{stroke-width:{HEAVY:g}px}}"
    rules += [f"{root}{a}{heavy}" for a in any_of] if each else [f"{root}:is({','.join(any_of)}){heavy}"]
    return "".join(rules)


def hooks_of_links(parts: "dict[str, list[Group]]", links: "list[tuple[Group, tuple[str, ...]]]") -> Hooks | None:
    """The hooks of a figure that says itself what its parts and the links between them are, as
    :class:`~bindview.BasePairs` does: ``parts`` maps a token (a nucleotide, the ligand) to its groups,
    ``links`` gives each line its group and the tokens it joins. Pointing at a part keeps it, its links and
    what they reach; pointing at a link (a click, as for a contact) keeps the link and its ends."""
    if not parts:
        return None
    near: dict[str, set[str]] = {}
    for _, ends in links:
        for a in ends:
            for b in ends:
                if a != b:
                    near.setdefault(a, set()).add(b)
    out = Hooks()
    for who, groups in parts.items():
        for g in groups:
            out.classes[id(g)] = " ".join(["lf-part", f"lf-t-{who}", f"lf-o-{who}"]
                                          + [f"lf-p-{t}" for t in sorted(near.get(who, ()))])
            out.focus[id(g)] = 0
    for g, ends in links:
        out.classes[id(g)] = " ".join(["lf-part", f"lf-t-{ends[0]}"] + [f"lf-o-{t}" for t in ends])
        out.focus[id(g)] = -1
    root = f"svg.bindview.{ROOT_CLASS}"
    rules = [f"{root} .lf-part{{transition:opacity .12s;cursor:pointer}}",
             f"{root} .lf-part:focus:not(:focus-visible){{outline:none}}"]
    for who in parts:
        rules.append(f"{root}:is(:has(.lf-t-{who}:focus),:not(:has(.lf-part:focus)):has(.lf-t-{who}:hover)) "
                     f".lf-part:not(.lf-o-{who},.lf-p-{who}){{opacity:{DIMMED:g}}}")
    out.css = "".join(rules)
    return out
