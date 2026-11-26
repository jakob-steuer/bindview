"""Serialising a :class:`~bindview.scene.Scene` to SVG text.

The writer is deliberately boring: fixed attribute order, numbers rounded and
formatted one way, no timestamps, no generated ids, no dependence on dict
ordering of anything that is not already ordered. The same scene always gives
the same bytes. Text is escaped, and a character that no XML file can hold (a
control character in a title, half a surrogate pair) is left out.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Iterable, Sequence
from xml.sax.saxutils import escape, quoteattr

from .interactive import NUMBER, OVER, ROOT_CLASS, UNDER, Hooks, hooks
from .scene import Circle, Group, Item, Line, Paint, Path, Polygon, Polyline, Rect, Scene, Span, Text
from .textmetrics import FONT_FAMILY
from .theme import EXTRAS, Palette, fixed_tokens

__all__ = ["to_svg", "svg_and_css", "stylesheet", "fmt"]


#: What XML 1.0 cannot hold, as text or in an attribute, escaped or not: the control characters other than tab
#: and the two that end a line, half a surrogate pair, and the last two code points of the first plane.
_NOT_XML = re.compile("[^\t\n\r\x20-\ud7ff\ue000-\ufffd\U00010000-\U0010ffff]")


def _plain(text: str) -> str:
    """``text`` without the characters that no XML file can hold."""
    return _NOT_XML.sub("", text)


def fmt(value: float) -> str:
    """Format a coordinate: two decimals, no trailing zeros, no negative zero."""
    s = f"{value:.2f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


#: What no colour has in it, and what would end the attribute or the declaration that a colour is written into.
_NOT_IN_A_COLOUR = re.compile(r"""["'<>&;{}\\]|/\*|[\x00-\x1f\x7f]""")
_TOKEN = re.compile(r"[A-Za-z0-9_-]+")


def _colour(token: str, value: str, palette: str = "") -> str:
    """``value`` as it is, where it can be written for a colour: a palette is the caller's, and its colours go into
    attributes and into a style sheet as they are given."""
    whose = f"the colour {token!r}" + (f" of the palette {palette!r}" if palette else "")
    if not _TOKEN.fullmatch(token):
        raise ValueError(f"{whose} has a name that cannot be written into a style sheet: letters, digits, _ and - can")
    if not isinstance(value, str) or not value.strip() or _NOT_IN_A_COLOUR.search(value):
        raise ValueError(f"{whose} is {value!r}: a colour is a name, #rrggbb or a function such as rgb(...), without "
                         "quotes, angle brackets, braces, semicolons, backslashes or line breaks")
    return value


class _Resolver:
    """Turns palette tokens into SVG paint values."""

    def __init__(self, palette: Palette, css_vars: bool) -> None:
        self.name = palette.name
        self.paints: dict[str, str] = {}
        # (The tokens that are the same in every palette come first: a palette that names one wins.)
        self.tokens = {**fixed_tokens(), **palette.tokens()}
        self.css_vars = css_vars
        # What a palette has beside its tokens (``Palette.extra``), and which of it this figure asked for:
        # only those are declared in its style sheet, so a figure that asks for none declares none.
        self.extra = {**EXTRAS, **palette.extra}
        self.used: set[str] = set()

    def colour(self, token: str | None) -> str:
        if token is None:
            return "none"
        paint = self.paints.get(token)
        if paint is None:
            paint = self.paints[token] = self._paint(token)
        return paint

    def _paint(self, token: str) -> str:
        if token.startswith("#") or token == "none":
            return _colour("a literal", token)
        try:
            value = self.tokens[token]
        except KeyError:
            if token not in self.extra:
                raise KeyError(f"unknown colour token {token!r}") from None
            value = self.extra[token]
            self.used.add(token)
        value = _colour(token, value, self.name)
        if self.css_vars:
            return f"var(--lf-{token.replace('_', '-')},{value})"
        return value


def _paint_attrs(p: Paint, res: _Resolver) -> str:
    if res.css_vars:
        style = [f"fill:{res.colour(p.fill)}"]
        if p.stroke is not None:
            style.append(f"stroke:{res.colour(p.stroke)}")
        out = [f'style="{";".join(style)}"']
    else:
        out = [f'fill="{res.colour(p.fill)}"']
        if p.stroke is not None:
            out.append(f'stroke="{res.colour(p.stroke)}"')
    if p.stroke is not None:
        out.append(f'stroke-width="{fmt(p.width)}"')
        if p.cap != "butt":
            out.append(f'stroke-linecap="{p.cap}"')
        if p.join != "miter":
            out.append(f'stroke-linejoin="{p.join}"')
        if p.dash:
            out.append(f'stroke-dasharray="{" ".join(fmt(d) for d in p.dash)}"')
    if p.opacity < 1.0:
        out.append(f'opacity="{fmt(p.opacity)}"')
    return " ".join(out)


def _text_fill(token: str, res: _Resolver) -> str:
    if res.css_vars:
        return f'style="fill:{res.colour(token)}"'
    return f'fill="{res.colour(token)}"'


def _span(s: Span, res: _Resolver) -> str:
    attrs = []
    if s.dy:
        attrs.append(f'dy="{fmt(s.dy)}"')
    if s.size is not None:
        attrs.append(f'font-size="{fmt(s.size)}"')
    if s.fill is not None:
        attrs.append(_text_fill(s.fill, res))
    if s.bold is not None:
        attrs.append(f'font-weight="{"bold" if s.bold else "normal"}"')
    if s.italic:
        attrs.append('font-style="italic"')
    body = escape(_plain(s.text))
    if not attrs:
        return body
    return f"<tspan {' '.join(attrs)}>{body}</tspan>"


def _item(it: Item, res: _Resolver, out: list[str], indent: str, live: "Hooks | None" = None) -> None:
    if isinstance(it, Group):
        cls = it.cls
        if live is not None and id(it) in live.classes:           # a figure that answers the pointer
            cls = f"{cls} {live.classes[id(it)]}".strip()
        attrs = [f'class="{cls}"'] if cls else []
        for k in sorted(it.data):
            if k != NUMBER or live is not None:                    # (the number is there for a page that has the hooks)
                attrs.append(f"data-{k}={quoteattr(_plain(str(it.data[k])))}")
        if it.translate is not None:
            attrs.append(f'transform="translate({fmt(it.translate[0])} {fmt(it.translate[1])})"')
        if it.opacity < 1.0:
            attrs.append(f'opacity="{fmt(it.opacity)}"')
        if live is not None and id(it) in live.focus:
            attrs.append(f'tabindex="{live.focus[id(it)]}"')
        if live is not None and id(it) in live.hidden:
            attrs.append('display="none"')
        out.append(f"{indent}<g{' ' if attrs else ''}{' '.join(attrs)}>")
        if it.title:
            out.append(f"{indent} <title>{escape(_plain(it.title))}</title>")
        for child in it.items:
            _item(child, res, out, indent + " ", live)
        out.append(f"{indent}</g>")
    elif isinstance(it, Line):
        under = f'class="{UNDER}" ' if live is not None and id(it) in live.under else ""
        out.append(f'{indent}<line {under}x1="{fmt(it.x1)}" y1="{fmt(it.y1)}" x2="{fmt(it.x2)}" y2="{fmt(it.y2)}" '
                   f"{_paint_attrs(it.paint, res)}/>")
    elif isinstance(it, Polyline):
        pts = " ".join(f"{fmt(x)},{fmt(y)}" for x, y in it.points)
        tag = "polygon" if it.closed else "polyline"
        out.append(f'{indent}<{tag} points="{pts}" {_paint_attrs(it.paint, res)}/>')
    elif isinstance(it, Polygon):
        pts = " ".join(f"{fmt(x)},{fmt(y)}" for x, y in it.points)
        out.append(f'{indent}<polygon points="{pts}" {_paint_attrs(it.paint, res)}/>')
    elif isinstance(it, Circle):
        out.append(f'{indent}<circle cx="{fmt(it.cx)}" cy="{fmt(it.cy)}" r="{fmt(it.r)}" '
                   f"{_paint_attrs(it.paint, res)}/>")
    elif isinstance(it, Rect):
        rx = f' rx="{fmt(it.rx)}"' if it.rx else ""
        out.append(f'{indent}<rect x="{fmt(it.x)}" y="{fmt(it.y)}" width="{fmt(it.w)}" height="{fmt(it.h)}"{rx} '
                   f"{_paint_attrs(it.paint, res)}/>")
    elif isinstance(it, Path):
        d = " ".join(c[0] + " ".join(fmt(v) if isinstance(v, float) else str(int(v)) for v in c[1:])
                     for c in it.commands)
        out.append(f'{indent}<path d="{d}" {_paint_attrs(it.paint, res)}/>')
    elif isinstance(it, Text):
        attrs = [f'x="{fmt(it.x)}"', f'y="{fmt(it.y)}"', f'font-size="{fmt(it.size)}"']
        if it.anchor != "start":
            attrs.append(f'text-anchor="{it.anchor}"')
        if it.bold:
            attrs.append('font-weight="bold"')
        if it.italic:
            attrs.append('font-style="italic"')
        attrs.append(_text_fill(it.fill, res))
        if it.opacity < 1.0:
            attrs.append(f'opacity="{fmt(it.opacity)}"')
        body = "".join(_span(s, res) for s in it.spans)
        out.append(f"{indent}<text {' '.join(attrs)}>{body}</text>")
    else:  # pragma: no cover - exhaustive over Item
        raise TypeError(f"cannot serialise {type(it).__name__}")


def _css_rules(light: Palette, dark: Palette | None, used: "Iterable[str]" = (), scope: str = "") -> list[str]:
    """The palette as custom properties. ``used`` names what the figure asked for of ``Palette.extra``; ``scope`` is a
    class that the figure has for itself."""
    def decl(p: Palette) -> str:
        tokens = dict(p.tokens())
        tokens.update({k: p.extra[k] for k in used if k in p.extra})
        return "".join(f"--lf-{k.replace('_', '-')}:{_colour(k, v, p.name)};" for k, v in sorted(tokens.items()))

    root = "svg.bindview" + (f".{scope}" if scope else "")
    rules = [f"{root}{{{decl(light)}}}"]
    if dark is not None:
        rules.append(f"@media (prefers-color-scheme:dark){{{root}:not([data-theme=light]){{{decl(dark)}}}}}")
        rules.append(f"{root}[data-theme=dark]{{{decl(dark)}}}")
    return rules


def _scope(rules: Sequence[str]) -> str:
    """The class of the figures that go with these rules: named by what the rules are, so that two figures in the same
    colours share it, and two in different ones do not."""
    return "lf-c" + hashlib.sha256("".join(rules).encode()).hexdigest()[:8]


def to_svg(scene: Scene, palette: Palette, *, background: bool = True, css_vars: bool = False,
           dark_palette: Palette | None = None, scale: float = 1.0, interactive: "bool | Hooks" = False,
           over: "Sequence[Group]" = (), stylesheet: str = "inline") -> str:
    """Serialise ``scene``: :func:`svg_and_css` without the sheet beside it."""
    return svg_and_css(scene, palette, background=background, css_vars=css_vars, dark_palette=dark_palette, scale=scale,
                       interactive=interactive, over=over, stylesheet=stylesheet)[0]


def svg_and_css(scene: Scene, palette: Palette, *, background: bool = True, css_vars: bool = False,
                dark_palette: Palette | None = None, scale: float = 1.0, interactive: "bool | Hooks" = False,
                over: "Sequence[Group]" = (), stylesheet: str = "inline") -> "tuple[str, str | None]":
    """``scene`` as SVG, and the style sheet that goes with it where that is kept apart.

    With ``css_vars`` every colour is emitted as a CSS custom property with the
    palette colour as fallback, and (given ``dark_palette``) the embedded style
    sheet switches palettes on ``prefers-color-scheme`` or a ``data-theme``
    attribute -- one cached file serves both themes.

    With ``interactive`` the figure answers the pointer (:mod:`bindview.interactive`):
    its residues and contacts get classes and can take the focus, and a second
    style sheet says what steps back when one of them is pointed at. A scene
    with nothing to point at is written as it is without it. A figure that
    knows its own parts better passes its :class:`~bindview.interactive.Hooks`.
    ``over`` is what such a figure has beside (:mod:`bindview.styles.closeup`); it is written with the
    hooks or not at all.

    ``stylesheet``: a ``<style>`` in an SVG that is part of a page speaks to the whole page. ``"inline"`` declares the
    palette for every figure there, so that of two in different colours the later one colours both; ``"scoped"``
    declares it for figures in these colours alone; ``"external"`` leaves both sheets out and hands them back, one
    rule on a line, each of which says the same in every figure that has it.
    """
    res = _Resolver(palette, css_vars)
    live = interactive if isinstance(interactive, Hooks) else hooks(scene, over) if interactive else None
    w, h = scene.width, scene.height
    apart = stylesheet == "external"
    # (The drawing is written first and set after the style sheet: the sheet declares what the drawing
    # asked for of the palette's extras, which is known once it is written.)
    body: list[str] = []
    if background:
        body.append(f' <rect class="background" width="{fmt(w)}" height="{fmt(h)}" '
                    f"{_paint_attrs(Paint(fill='background'), res)}/>")
    for it in scene.items:
        _item(it, res, body, " ", live)
    if live is not None and over:
        _item(Group(OVER, list(over)), res, body, " ", live)
    classes, rules = "bindview" + (f" {ROOT_CLASS}" if live is not None else ""), []
    if css_vars:
        # (A sheet that is shared declares all of a palette: which of it one figure uses does not matter there.)
        used = sorted(palette.extra) if apart else sorted(res.used)
        rules = _css_rules(palette, dark_palette, used)
        if stylesheet != "inline":
            scope = _scope(rules)
            classes, rules = f"{classes} {scope}", _css_rules(palette, dark_palette, used, scope)
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" class="{classes}" '
        f'width="{fmt(w * scale)}" height="{fmt(h * scale)}" '
        f'viewBox="0 0 {fmt(w)} {fmt(h)}" font-family="{FONT_FAMILY}" role="img">'
    ]
    if scene.title:
        out.append(f" <title>{escape(_plain(scene.title))}</title>")
    if scene.description:
        out.append(f" <desc>{escape(_plain(scene.description))}</desc>")
    if css_vars and not apart:
        out.append(f" <style>{''.join(rules)}</style>")
    if live is not None and not apart:
        out.append(f" <style>{live.css}</style>")
    out += body
    out.append("</svg>")
    css = None
    if apart and (rules or live is not None):
        css = "\n".join(rules + (live.shared or [live.css] if live is not None else [])) + "\n"
    return "\n".join(out) + "\n", css


def stylesheet(figures: "Iterable[Any]") -> str:
    """One style sheet for a page with several figures (``render(stylesheet="external")``): what each of them goes
    with, every rule once. ``figures``: their results, or their sheets (``result.css``)."""
    rules: dict[str, None] = {}
    for figure in figures:
        css = figure if isinstance(figure, str) else figure.css
        rules.update(dict.fromkeys((css or "").splitlines()))
    return "".join(f"{rule}\n" for rule in rules)
