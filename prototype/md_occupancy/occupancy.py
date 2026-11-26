"""A trajectory's plan as figures: for each state the library's own diagram of the drawn frame, with every
contact weighted by its occupancy, and a strip of the contacts against time.

A SKETCH, not part of bindview: nothing here is tested or promised. A plan is what count.py writes from the
contacts that contacts.py found, and the frames it names are written by frame.py. The diagram is the
library's (``Diagram.from_structure`` with records that carry an occupancy, and ``ensemble=`` for what
that is a share of); the strip is drawn here.

    import occupancy

    plan = occupancy.load("testdata/md/riboswitch_bisguanidine/simulation_4.json")
    state = plan.state("A+B")                 # "all", or a state the plan has a figure for
    figure = occupancy.diagram(plan, state)   # the library's call, with the plan's words
    figure = occupancy.strip(plan)
    figure = occupancy.differences(plan, "B", "A+B")   # two states side by side, what one lacks highlighted
    figure = occupancy.encodings(plan)        # one figure, the share shown four ways: to choose from

    python occupancy.py [plan.json ...]       writes index.html beside this file
"""
from __future__ import annotations

import html
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from bindview import Diagram
from bindview.theme import LIGHT

FONT = "Helvetica, Arial, sans-serif"
INK, QUIET = "#1f2933", "#52606d"
SHADES = ("#7c8aa5", "#b08968", "#0f766e", "#7c3aed", "#be123c")
HIGHLIGHT = "#facc15"                # a marker pen: the colour of no kind of contact
OF_A_STATE = 0.10                    # a contact is of a state when this share of its frames makes it (count.py)
STEPS = ((0.9, 1.25, 1.0), (0.5, 0.7, 0.8), (0.0, 0.38, 0.5))       # from a share on: weight and opacity
DATA = Path(__file__).resolve().parents[2] / "testdata" / "md" / "riboswitch_bisguanidine"


@dataclass
class Sketch:
    """What a sketch hands back: the picture, and its name, facts, status and warnings, as a result of the library
    has them."""

    svg: str
    figure: str
    facts: dict = field(default_factory=dict)
    status: str = "sketch"
    warnings: list = field(default_factory=list)
    quality: None = None


@dataclass
class State:
    """One figure of a plan: the frames it counts, the frame that is drawn for them and its contacts."""

    name: str                   # "all", or the chains whose pockets the ligand occupies: "B", "A+B"
    words: str                  # the same in words
    frames: int                 # how many frames it counts
    share: float                # of the trajectory
    number: int                 # the drawn frame, counted from 0
    time_ns: float
    frame: str                  # the drawn frame, as a file
    records: list               # the drawn frame's contacts of 0.10 or more, each with its occupancy
    labels: list                # the name of each record in ``occupancy``, record by record
    occupancy: dict             # every contact of 0.10 or more in these frames, by its name
    not_drawn: list             # (contact, occupancy): made in 0.30 or more, and not by the drawn frame
    of: str                     # what the occupancies are a share of: for ``ensemble=``
    subtitle: str               # which frame the figure is drawn on

    @property
    def also(self) -> str:
        """The contacts that are made often and that the drawn frame does not make, as a sentence."""
        if not self.not_drawn:
            return ""
        return ("Made often and not by this frame: "
                + ", ".join(f"{name} ({share:.0%})" for name, share in self.not_drawn) + ".")


@dataclass
class Plan:
    """A trajectory as count.py counted it."""

    what: str                   # what the simulation is, in the words of whoever counted it
    ligand: str                 # the ligand's residue name
    smiles: str
    frames: int
    contacts_found: int         # different contacts, of any occupancy
    states: dict                # a state -> the share of the frames in it
    figures: list               # a State for the whole trajectory, and one for each state that is common
    strip: dict                 # what the strip is drawn from
    path: Path

    def state(self, name: str = "all") -> State:
        for figure in self.figures:
            if figure.name == name:
                return figure
        raise KeyError(f"{name!r}: {self.path.name} has a figure for "
                       + ", ".join(repr(f.name) for f in self.figures))


def words_of(state: str) -> str:
    """A state in words: which chains' pockets the ligand occupies."""
    if state in ("all", "none"):
        return "all frames" if state == "all" else "no pocket occupied"
    chains = state.split("+")
    if len(chains) == 1:
        return f"one pocket occupied (chain {chains[0]})"
    return "both pockets occupied" if len(chains) == 2 else f"the pockets of chains {', '.join(chains)} occupied"


def _order(states) -> list:
    """States with fewer pockets first."""
    return sorted(states, key=lambda s: (s.count("+"), s))


def load(path) -> Plan:
    """A plan of count.py, with the frames it names beside it."""
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    what = raw["what"]
    figures = []
    for fig in raw["figures"]:
        words = words_of(fig["name"])
        of = f"the {fig['frames']:,} frames of {what}" + ("" if fig["name"] == "all" else f" with {words}")
        figures.append(State(
            name=fig["name"], words=words, frames=fig["frames"], share=fig["share"], number=fig["frame"],
            time_ns=fig["time_ns"], frame=str(path.parent / fig["frame_file"]), records=fig["records"],
            labels=fig["labels"],
            occupancy=fig["occupancy"], not_drawn=[tuple(pair) for pair in fig["not_drawn"]], of=of,
            subtitle=f"{words}: the frame at {fig['time_ns'] / 1000:.3f} µs"))
    return Plan(what=what, ligand=raw["ligand"], smiles=raw["smiles"], frames=raw["frames"],
                contacts_found=raw["contacts_found"], states=raw["states"], figures=figures, strip=raw["strip"],
                path=path)


def diagram(plan: Plan, state: State | str = "all", style: str = "detailed", **options):
    """The library's diagram of a state's drawn frame: every contact with its occupancy.

    Nothing of this is the sketch's: it is ``Diagram.from_structure`` with the plan's records, which carry
    an occupancy, and ``render``. What comes back is the library's result.
    """
    state = plan.state(state) if isinstance(state, str) else state
    dia = Diagram.from_structure(state.frame, plan.ligand, smiles=plan.smiles, interactions=state.records,
                                 ensemble=state.of)
    return dia.render(style=style, subtitle=state.subtitle, **options)


def strip(plan: Plan) -> Sketch:
    """The contacts against time: a row a contact, time in bins, a cell as dark as the share of the bin's
    frames that make the contact. The states stand in rows of their own above the contacts."""
    data = plan.strip
    every = _order(plan.states)
    colour = {s: ("#cbd5e1" if s == "none" else SHADES[min(k, len(SHADES) - 1) if "+" not in s else 2])
              for k, s in enumerate(every)}
    rows = sorted(data["rows"], key=lambda r: (r["chain"] != "B", r["number"], r["kind"], r["label"]))
    left, cell, row_h, top = 318.0, 2.6, 13.0, 46.0
    n = data["bins"]
    width = left + n * cell + 52.0
    states = [s for s in every if s in data["state_share"]]
    height = top + (len(states) + len(rows)) * row_h + 34.0 + 8.0
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.0f} {height:.0f}" width="{width:.0f}" '
             f'font-family="{FONT}"><rect width="{width:.0f}" height="{height:.0f}" fill="#ffffff"/>',
             f'<text x="0" y="14" font-size="11.5" fill="{INK}" font-weight="600">{html.escape(plan.what)}: '
             'its contacts against time</text>',
             f'<text x="0" y="30" font-size="9.5" fill="{QUIET}">a cell is {data["bin_ns"]:.0f} ns; it is as dark as '
             'the share of its frames that make the contact. Contacts made in a tenth of the frames or more.</text>']
    y = top

    def band(values, fill, y):
        out = []
        for b, v in enumerate(values):
            if v > 0.004:
                out.append(f'<rect x="{left + b * cell:.1f}" y="{y:.1f}" width="{cell + 0.1:.1f}" '
                           f'height="{row_h - 2:.1f}" fill="{fill}" fill-opacity="{v:.2f}"/>')
        return "".join(out)

    for s in states:
        share = plan.states.get(s, 0.0)
        parts.append(f'<text x="{left - 8:.0f}" y="{y + 9.5:.1f}" font-size="9.5" fill="{INK}" text-anchor="end" '
                     f'font-weight="600">{words_of(s)}</text>' + band(data["state_share"][s], colour[s], y)
                     + f'<text x="{left + n * cell + 6:.0f}" y="{y + 9.5:.1f}" font-size="9" fill="{QUIET}">'
                     f'{share:.2f}</text>')
        y += row_h
    y += 6.0
    tokens = LIGHT.tokens()
    for r in rows:
        parts.append(f'<text x="{left - 8:.0f}" y="{y + 9.5:.1f}" font-size="9.5" fill="{INK}" text-anchor="end">'
                     f'{html.escape(r["label"])}</text>' + band(r["share"], tokens[r["kind"]], y)
                     + f'<text x="{left + n * cell + 6:.0f}" y="{y + 9.5:.1f}" font-size="9" fill="{QUIET}">'
                     f'{r["occupancy"]:.2f}</text>')
        y += row_h
    # the time axis, in microseconds of the simulation's own clock
    x0, x1 = left, left + n * cell
    parts.append(f'<line x1="{x0}" y1="{y + 3}" x2="{x1:.1f}" y2="{y + 3}" stroke="{INK}" stroke-width="0.8"/>')
    start, total = data["start_ns"], n * data["bin_ns"]
    tick = 500.0
    t = (int(start // tick) + (start % tick > 0)) * tick
    while t <= start + total + 1e-6:
        x = x0 + (t - start) / total * (x1 - x0)
        parts.append(f'<line x1="{x:.1f}" y1="{y + 3}" x2="{x:.1f}" y2="{y + 7}" stroke="{INK}" stroke-width="0.8"/>'
                     f'<text x="{x:.1f}" y="{y + 18}" font-size="9" fill="{QUIET}" text-anchor="middle">'
                     f'{t / 1000:g}</text>')
        t += tick
    parts.append(f'<text x="{x1 + 8:.1f}" y="{y + 18}" font-size="9" fill="{QUIET}">µs</text>'
                 f'<text x="{left + n * cell + 6:.0f}" y="{top - 6:.0f}" font-size="9" fill="{QUIET}">share</text></svg>')
    # Where the commonest state of a cell changes for the last time: what a reader takes from the strip.
    commonest = [max(states, key=lambda s: data["state_share"][s][b]) for b in range(n)]
    changes = [b for b in range(1, n) if commonest[b] != commonest[b - 1]]
    last = None
    if changes:
        b = changes[-1]
        last = {"at_us": round((start + b * data["bin_ns"]) / 1000.0, 2), "from": words_of(commonest[b - 1]),
                "to": words_of(commonest[b])}
    facts = {"rows": len(rows), "bins": n, "bin_ns": data["bin_ns"],
             "from_us": round(start / 1000.0, 2), "to_us": round((start + total) / 1000.0, 2),
             "states": {words_of(s): plan.states.get(s, 0.0) for s in states},
             "changes": len(changes), "last_change": last}
    return Sketch("".join(parts), "strip", facts)


# ---------------------------------------------------------------------------
# A figure of the library, drawn over: its contacts are groups that say what they are
# ---------------------------------------------------------------------------

_CONTACT = re.compile(r'<g class="interaction [^"]*"([^>]*)>(.*?)</g>', re.S)
_LINE = re.compile(r'<line x1="([-0-9.]+)" y1="([-0-9.]+)" x2="([-0-9.]+)" y2="([-0-9.]+)"')
_ROOT = re.compile(r'<svg[^>]*?width="([0-9.]+)" height="([0-9.]+)"[^>]*>')


def _said(head: str) -> dict:
    return dict(re.findall(r'data-([a-z-]+)="([^"]*)"', head))


def _answers(said: dict, record: dict) -> int:
    """How well a contact of a figure answers a row of the table: 0 is not at all."""
    chain, _, name = said.get("residue", "").partition(":")
    number = re.search(r"(-?[0-9]+)[^0-9]*$", name)
    if said.get("type") != record["interaction_type"] or chain != record["chain"] or number is None \
            or int(number.group(1)) != int(record["residue_number"]):
        return 0
    mine = set(record.get("ligand_atoms") or [record.get("ligand_atom")])
    return 1 + len(mine & set(said.get("ligand-atoms", "").split(",")))


def _reach(inner: str) -> "tuple | None":
    """The two ends of a contact's line: it may be drawn in pieces, around its distance."""
    ends = [(float(a), float(b)) for x1, y1, x2, y2 in _LINE.findall(inner) for a, b in ((x1, y1), (x2, y2))]
    if not ends:
        return None
    return max(((p, q) for p in ends for q in ends), key=lambda pq: (pq[0][0] - pq[1][0]) ** 2 + (pq[0][1] - pq[1][1]) ** 2)


def _highlighted(svg: str, records: list) -> "tuple[str, int]":
    """``svg`` with the contacts that answer ``records`` on a stroke of the marker pen, and with the name of
    a residue on one where every contact of the residue is. How many contacts were found."""
    contacts = [(m, _said(m.group(1))) for m in _CONTACT.finditer(svg)]
    marked = set()
    for record in records:
        best = max(contacts, key=lambda c: _answers(c[1], record), default=None)
        if best is not None and _answers(best[1], record):
            marked.add(best[0].start())
    strokes = []
    for m, said in contacts:
        if m.start() in marked and _reach(m.group(2)):
            (x1, y1), (x2, y2) = _reach(m.group(2))
            for width, opacity in ((13.0, 0.22), (8.0, 0.5)):
                strokes.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{HIGHLIGHT}" '
                               f'stroke-width="{width}" stroke-linecap="round" opacity="{opacity}"/>')
    for residue in dict.fromkeys(said["residue"] for _, said in contacts):      # in the figure's own order
        own = [m.start() in marked for m, said in contacts if said["residue"] == residue]
        at = svg.find(f'data-residue="{residue}">', svg.find('<g class="residues">'))
        if not all(own) or at < 0:
            continue
        end = svg.find('<g class="residue"', at)
        names = re.findall(r'<text x="([-0-9.]+)" y="([-0-9.]+)" font-size="([0-9.]+)"[^>]*font-weight="bold"[^>]*>'
                           r'([^<]*)</text>', svg[at:end if end > 0 else len(svg)])
        if names:
            x, y, size, name = float(names[-1][0]), float(names[-1][1]), float(names[-1][2]), names[-1][3]
            width = 0.62 * size * len(name) + 8.0
            strokes.append(f'<rect x="{x - width / 2:.1f}" y="{y - size + 0.5:.1f}" width="{width:.1f}" '
                           f'height="{size + 5:.1f}" rx="5" fill="{HIGHLIGHT}" opacity="0.5"/>')
    at = svg.index('<g class="interactions">')
    return svg[:at] + '<g class="differs">' + "".join(strokes) + "</g>" + svg[at:], len(marked)


_WORD = re.compile(r'<text x="([-0-9.]+)" y="([-0-9.]+)" font-size="([0-9.]+)"[^>]*>([^<]*)</text>')


def _restyled(svg: str, steps: bool, share: bool) -> str:
    """``svg`` with the share of a contact shown another way: in three steps of weight, written on the
    line where it is under 100 %, or both. The library's own way is a weight that follows it continuously."""
    def box(x, y, size, words):
        half = 0.3 * size * len(words) + 1.5
        return (x - half, y - size, x + half, y + 0.3 * size)

    inside = svg[svg.index('<g class="diagram"'):]
    taken = [box(float(x), float(y), float(size), words) for x, y, size, words in _WORD.findall(inside)]

    def free(x, y, words):
        mine = box(x, y, 9.5, words)
        return not any(mine[0] < b[2] and b[0] < mine[2] and mine[1] < b[3] and b[1] < mine[3] for b in taken)

    def again(m):
        said, inner = _said(m.group(1)), m.group(2)
        if "occupancy" not in said:
            return m.group(0)
        o = float(said["occupancy"])
        if steps:
            was = 0.4 + 0.6 * min(o, 1.0)                        # bindview.confidence.occupancy_factors
            weight, opacity = next((w, a) for least, w, a in STEPS if o >= least)
            inner = re.sub(r'stroke-width="([0-9.]+)"', lambda w: f'stroke-width="{float(w.group(1)) / was * weight:.2f}"', inner)
            inner = re.sub(r' opacity="[0-9.]+"', "", inner).replace("/>", f' opacity="{opacity}"/>')
            inner = re.sub(r'(<text[^>]*)>', rf'\1 opacity="{opacity}">', inner)
        if share and o < 0.995:
            words = f"{100.0 * o:.0f} %"
            if "<text" in inner:
                inner = re.sub(r'(<text[^>]*>)[^<]*(</text>)', rf'\g<1>{words}\g<2>', inner)
            elif _reach(inner):
                (x1, y1), (x2, y2) = _reach(inner)
                colour = re.search(r'stroke="(#[0-9a-f]+)"', inner).group(1)
                # In the middle of the line, or the nearest place along it where no other word stands.
                places = [(x1 + (x2 - x1) * f, y1 + (y2 - y1) * f + 3.3) for f in (0.5, 0.62, 0.38, 0.74, 0.26)]
                x, y = next((at for at in places if free(*at, words)), places[0])
                taken.append(box(x, y, 9.5, words))
                inner += (f'<text x="{x:.2f}" y="{y:.2f}" font-size="9.5" '
                          f'text-anchor="middle" fill="{colour}" stroke="#ffffff" stroke-width="3.4" '
                          f'stroke-linejoin="round" paint-order="stroke">{words}</text>')
        return m.group(0).replace(m.group(2), inner)

    return _CONTACT.sub(again, svg)


def _without_key(svg: str) -> str:
    """A figure cut off above its key: for a sheet that has one key for all its panels."""
    at = re.search(r'<g class="legend" transform="translate\(([-0-9.]+) ([-0-9.]+)\)">', svg)
    if at is None:
        return svg
    width, height = _ROOT.search(svg).groups()
    cut = float(at.group(2)) - 4.0
    svg = svg[:at.start()] + "</svg>"
    svg = svg.replace(f'height="{height}"', f'height="{cut:.1f}"').replace(f'viewBox="0 0 {width} {height}"',
                                                                         f'viewBox="0 0 {width} {cut:.1f}"')
    return svg


def _sheet(panels: list, columns: int, title: str, under: list, captions: "list | None" = None) -> str:
    """Figures set side by side on one sheet: a title, the panels in rows, and lines of words under them."""
    sizes = [tuple(float(v) for v in _ROOT.search(s).groups()) for s in panels]
    gap, top, line = 26.0, 34.0, 15.0
    widths = [max(w for k, (w, _) in enumerate(sizes) if k % columns == c) for c in range(min(columns, len(panels)))]
    rows = [panels[k:k + columns] for k in range(0, len(panels), columns)]
    heights = [max(h for _, h in sizes[r * columns:(r + 1) * columns]) for r in range(len(rows))]
    width = sum(widths) + gap * (len(widths) - 1) + 36.0
    out, y = [], top
    for r, row in enumerate(rows):
        x = 18.0
        for c, svg in enumerate(row):
            k = r * columns + c
            w, h = sizes[k]
            head = _ROOT.search(svg)
            family = re.search(r'font-family="([^"]*)"', head.group(0))
            out.append(f'<svg x="{x:.1f}" y="{y:.1f}" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
                       f'font-family="{family.group(1) if family else FONT}">' + svg[head.end():])
            if captions:
                out.append(f'<text x="{x:.1f}" y="{y + heights[r] + 13:.1f}" font-size="10.5" fill="{INK}">'
                           f'<tspan font-weight="700">{chr(97 + k)}</tspan>  {html.escape(captions[k])}</text>')
            x += widths[c] + gap
        y += heights[r] + (30.0 if captions else 6.0)
    y += 8.0
    for words in under:
        out.append(words.replace("{y}", f"{y:.1f}").replace("{mid}", f"{y - 3.6:.1f}"))
        y += line
    height = y + 4.0
    return (f'<svg xmlns="http://www.w3.org/2000/svg" class="bindview" width="{width:.0f}" height="{height:.0f}" '
            f'viewBox="0 0 {width:.0f} {height:.0f}" font-family="{FONT}" role="img">'
            f'<rect width="{width:.0f}" height="{height:.0f}" fill="#ffffff"/>'
            f'<text x="18" y="20" font-size="12.5" fill="{INK}" font-weight="600">{html.escape(title)}</text>'
            + "".join(out) + "</svg>")


def differences(plan: Plan, first: str, second: str) -> Sketch:
    """Two states of one ligand on one target side by side, each on the frame drawn for it, with what the
    other state does not have on a stroke of the marker pen.

    A contact is of a state when it is made in 0.10 of the state's frames or more: the rule that decides
    what a figure holds, and no other number. A contact of one state that is not of the other is marked,
    and so is the name of a residue whose every contact is. What both states have is told apart by its
    weight, as in every figure with an occupancy, and not marked.
    """
    a, b = plan.state(first), plan.state(second)
    panels, only = [], {}
    for state, other in ((a, b), (b, a)):
        lacking = [k for k, name in enumerate(state.labels) if name not in other.occupancy]
        svg, found = _highlighted(diagram(plan, state).svg, [state.records[k] for k in lacking])
        if found != len(lacking):
            raise ValueError(f"{state.words}: {len(lacking)} contacts differ, and {found} were found in the figure")
        only[state.words] = [state.labels[k] for k in lacking]
        panels.append(svg)
    shared = sorted(set(a.occupancy) & set(b.occupancy), key=lambda n: (-abs(a.occupancy[n] - b.occupancy[n]), n))
    key = [f'<line x1="18" y1="{{mid}}" x2="44" y2="{{mid}}" stroke="{HIGHLIGHT}" stroke-width="8" '
           f'stroke-linecap="round" opacity="0.6"/><text x="52" y="{{y}}" font-size="10.5" fill="{QUIET}">'
           f'Marked: a contact of this state that is not one of the other (made in under '
           f'{OF_A_STATE:.0%} of the other\'s frames), and a residue whose every contact is.</text>']
    facts = {"only": only, "shared": len(shared), "of_a_state": OF_A_STATE,
             "largest_shift": (shared[0], a.occupancy[shared[0]], b.occupancy[shared[0]]) if shared else None}
    return Sketch(_sheet(panels, 2, f"{plan.what}: two states of one ligand on one target", key), "states", facts)


def encodings(plan: Plan, state: "State | str" = "all") -> Sketch:
    """One figure four times, the share of the frames that make a contact shown four ways, to choose from:
    as the library draws it, in three steps of weight, written on the line, and both.

    Not a proposal that is settled: the steps (90 % and 50 %) are for this sheet. In the panels that write
    the share it stands where the distance stood, which is the distance in one frame.
    """
    state = plan.state(state) if isinstance(state, str) else state
    drawn = _without_key(diagram(plan, state).svg)
    panels = [drawn, _restyled(drawn, True, False), _restyled(drawn, False, True), _restyled(drawn, True, True)]
    captions = ["as the library draws it: weight and opacity follow the share continuously",
                "three steps: 90 % or more, 50 to 90 %, under 50 %, each about half the weight of the last",
                "the share written on the line where it is under 100 %; weight as in a",
                "both: three steps, and the share on the line"]
    shares = sorted(r["occupancy"] for r in state.records)
    facts = {"contacts": len(shares), "from": shares[0], "to": shares[-1],
             "steps": [sum(1 for s in shares if s >= least and (k == 0 or s < STEPS[k - 1][0]))
                       for k, (least, _, _) in enumerate(STEPS)]}
    under = [f'<text x="18" y="{{y}}" font-size="10.5" fill="{QUIET}">{html.escape(state.of[:1].upper() + state.of[1:])}: '
             f'{len(shares)} contacts, made in {shares[0]:.0%} to {shares[-1]:.0%} of them. Green dashes a hydrogen '
             'bond, purple dots a salt bridge, orange dashes a π–cation contact.</text>']
    return Sketch(_sheet(panels, 2, f"{plan.what}: the share of the frames that make a contact, four ways",
                         under, captions), "encodings", facts)


def main(paths) -> int:
    """A page with the strip and the diagrams of every plan: index.html beside this file."""
    plans = [load(p) for p in (paths or sorted(DATA.glob("simulation_*.json")))]
    out = ["<!doctype html><html lang='en'><head><meta charset='utf-8'>"
           "<title>Sketches: a ligand's contacts over a trajectory</title><style>"
           "body{font:15px/1.5 -apple-system,'Segoe UI',Helvetica,Arial,sans-serif;color:#1f2933;margin:0}"
           "main{max-width:1100px;margin:0 auto;padding:28px 24px 80px}h1{font-size:24px;margin:0 0 6px}"
           "h2{font-size:18px;margin:44px 0 4px}p{max-width:80ch}.quiet{color:#52606d}"
           ".sketch{color:#9a3412;font-weight:600}figure{margin:14px 0;padding:16px;border:1px solid #e4e7eb;"
           "border-radius:8px;overflow-x:auto}figure svg{display:block;max-width:none}"
           ".row{display:flex;gap:16px;flex-wrap:wrap}</style></head><body><main>"
           "<h1>A ligand's contacts over a trajectory</h1><p><span class='sketch'>Sketches.</span> Not part of "
           "bindview, not tested. Contacts of every frame by ProLIF; a diagram is the library's own, drawn on one "
           "frame with every contact weighted by the share of the frames that make it.</p>"]
    for plan in plans:
        shares = ", ".join(f"{html.escape(words_of(s))}: {share:.1%}"
                           for s, share in sorted(plan.states.items(), key=lambda kv: _order(plan.states).index(kv[0])))
        out.append(f"<h2>{html.escape(plan.what[:1].upper() + plan.what[1:])}</h2><p class='quiet'>{plan.frames:,} "
                   f"frames, {plan.contacts_found} different contacts. {shares}.</p>"
                   f"<figure>{strip(plan).svg}</figure><div class='row'>")
        states = [f for f in plan.figures if f.name != "all"] or plan.figures
        for state in sorted(states, key=lambda f: _order([g.name for g in states]).index(f.name)):
            result = diagram(plan, state)
            also = f"<figcaption class='quiet'>{html.escape(state.also)}</figcaption>" if state.also else ""
            out.append(f"<figure>{result.svg}{also}</figure>")
            print(f"{plan.path.stem} {state.name}: {result.status}, {len(state.records)} contacts drawn on frame "
                  f"{state.number}, warnings {len(result.warnings)}")
        out.append("</div>")
        if len(states) == 2:
            first, second = (s.name for s in sorted(states, key=lambda f: _order([g.name for g in states]).index(f.name)))
            out.append(f"<figure>{differences(plan, first, second).svg}</figure>")
        out.append(f"<figure>{encodings(plan).svg}<figcaption class='quiet'>A sheet to choose from, not a "
                   "proposal that is settled: the same figure with the share shown four ways.</figcaption></figure>")
    out.append("</main></body></html>")
    page = Path(__file__).with_name("index.html")
    page.write_text("".join(out), encoding="utf-8")
    print(f"wrote {page.name} ({page.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
