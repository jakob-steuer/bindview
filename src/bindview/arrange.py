"""A figure arranged by hand: residues and the key are dragged, turned and flipped in a browser.

::

    graph = diagram.arrange()                       # opens a browser; hands the layout back when the page is done
    figure = diagram.render(graph=graph)

    diagram.arrange("arrange.html")                 # the same as one file, to open or to send

Drag a residue, or the key, to another place. With a residue picked, the wheel or ``r`` / ``shift``+``r`` turns it, ``f``
flips it, the arrow keys nudge it, a double click puts it back, and ``cmd``/``ctrl``+``z`` undoes.

**What follows.** While a residue is dragged the page shows a preview: the lines of its contacts, the arc of an eyelash,
the spokes on the atoms of the ligand and the tip of a bubble go with it. When it is let go, bindview draws the figure
from the moved layout (``render(graph=...)``), so that all that depends on a place is where it belongs: the lining of the
pocket, the lines broken around words, names and distances, the frame; and the page says where two things overlap. The
one file has no bindview behind it and stays with the preview.

**What comes out** is a layout (:class:`~bindview.Graph`), the same that ``graph.move``, ``graph.rotate`` and
``graph.flip`` make in Python. The page writes those calls, to copy, and saves the layout as the JSON that
``Graph.from_json`` and ``bindview render --layout`` read.

The page is served on 127.0.0.1 only, under an address that cannot be guessed, answers requests of its own page and no
other, and stops when it is done. Standard library, and nothing from the network.
"""

from __future__ import annotations

import hmac
import html
import json
import math
import secrets
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .errors import BindviewError, GraphError, RenderError
from .graph import Graph

if TYPE_CHECKING:
    from .diagram import Diagram, RenderResult

__all__ = ["page", "serve"]

#: The most a request of the page may hold: a layout is some tens of kilobytes.
_LARGEST = 16_000_000

_STYLE = """
body{font:15px/1.5 -apple-system,'Segoe UI',Helvetica,Arial,sans-serif;color:#1f2933;margin:0;background:#f5f7fa}
main{max-width:1100px;margin:0 auto;padding:20px 20px 60px}h1{font-size:22px;margin:0 0 4px}p{margin:4px 0 10px;max-width:90ch}
#figure{background:#fff;border:1px solid #d9e2ec;border-radius:8px;padding:10px;touch-action:none;overflow:auto}
#figure svg{display:block;max-width:100%;height:auto;margin:0 auto;user-select:none;-webkit-user-select:none;overflow:visible}
.picked{filter:drop-shadow(0 0 3px #f59e0b) drop-shadow(0 0 6px #f59e0b)}
.bar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:10px 0}button{font:inherit;padding:5px 12px;border:1px solid
#9aa5b1;border-radius:6px;background:#fff;cursor:pointer}button:hover{background:#eef2f6}#which{font-weight:600}#said{color:#9a3412}
textarea{width:100%;height:112px;font:13px/1.45 ui-monospace,Menlo,monospace;border:1px solid #d9e2ec;border-radius:6px;padding:8px;
box-sizing:border-box;background:#fff}kbd{font:12px ui-monospace,Menlo,monospace;border:1px solid #cbd2d9;border-radius:4px;
padding:0 4px;background:#fff}
"""


def _size(length: "str | None") -> "int | None":
    """The ``Content-Length`` of a request as a size that is read, or None: the digits 0 to 9 (``"\u00b2".isdigit()``
    is true, and ``int`` refuses it), no more of them than the largest size has, and not above that."""
    if not length or not (length.isascii() and length.isdigit()) or len(length) > len(str(_LARGEST)):
        return None
    return int(length) if int(length) <= _LARGEST else None


def _finite(number: str) -> float:
    """A number of a layout as it is read from JSON: ``Infinity``, ``NaN`` and ``1e999`` are none."""
    value = float(number)
    if not math.isfinite(value):
        raise ValueError(f"{number} is no number that a layout has")
    return value


def _bonds(diagram: "Diagram") -> dict[str, list[str]]:
    """For each atom of the ligand those it is bonded to: the spokes of a contact keep clear of the bonds."""
    names = diagram.ligand.names
    return {f"L:{names[a.GetIdx()]}": [f"L:{names[b.GetIdx()]}" for b in a.GetNeighbors() if b.GetAtomicNum() > 1]
            for a in diagram.ligand.mol.GetAtoms() if a.GetAtomicNum() > 1}


def _first(diagram: "Diagram", options: dict[str, Any]) -> "tuple[RenderResult, Graph]":
    """The figure that the page starts from, and its layout."""
    result = diagram.render(**options)
    if result.graph is None or result.svg is None:
        raise RenderError(f"there is nothing to arrange: the figure is {result.status}")
    return result, result.graph


def _html(diagram: "Diagram", title: str, behind: "str | None", options: dict[str, Any],
          first: "tuple[RenderResult, Graph] | None" = None) -> str:
    """The page. ``behind``: the address of the bindview that draws what is let go; None for the one file.
    ``first``: the figure it starts from, where that is drawn already."""
    result, graph = first or _first(diagram, options)
    # (Not one "<" in the block of data: "</script" would end it, and "<!--" with "<script" would keep it open.)
    given = json.dumps({"graph": graph.to_dict(), "bonds": _bonds(diagram),
                        "svg": result.svg if behind is None else ""}).replace("<", "\\u003c")
    script = resources.files(__package__).joinpath("arrange.js").read_text(encoding="utf-8")
    how = ("When it is let go, bindview draws the figure from the new layout." if behind else
           "This file shows a preview; bindview draws the figure from the layout that is saved.")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>{html.escape(title)}</title><style>{_STYLE}</style></head><body><main><h1>{html.escape(title)}</h1>
<p>Drag a residue or the key. With a residue picked, the wheel or <kbd>r</kbd> / <kbd>shift</kbd>+<kbd>r</kbd> turns it,
<kbd>f</kbd> flips it, the arrow keys nudge it; a double click puts it back, <kbd>cmd</kbd>/<kbd>ctrl</kbd>+<kbd>z</kbd> undoes.
{how}</p>
<div class="bar"><span>Picked: <span id="which">nothing</span></span><button id="back">Turn &#8634;</button>
<button id="turn">Turn &#8635;</button><button id="flip">Flip</button><button id="undo">Undo</button>
<button id="reset">Put all back</button><button id="save">Save layout.json</button>{'<button id="done">Done</button>' if behind else ''}
<span id="said"></span></div>
<div id="figure">{result.svg}</div>
<div class="bar"><span>The calls that make it:</span><button id="copy">Copy</button></div><textarea id="calls" readonly></textarea>
<p>Or from the saved file: <code>diagram.render(graph=Graph.from_json("layout.json"))</code>, or
<code>bindview render MODEL --ligand LIG --layout layout.json</code>.</p></main>
<script type="application/json" id="graph">{given}</script>
<script>window.EXACT = {json.dumps(behind)};</script><script>{script}</script></body></html>
"""


def page(diagram: "Diagram", path: "str | Path", title: "str | None" = None, **options: Any) -> Path:
    """Write the page as one file, with a preview and without bindview behind it. ``options``: those of ``render``."""
    out = Path(path)
    text = _html(diagram, title or _title(diagram), None, options)
    try:
        out.write_text(text, encoding="utf-8")
    except OSError as exc:
        raise RenderError(f"cannot write {out}: {exc}") from exc
    return out


def _title(diagram: "Diagram") -> str:
    return f"Arrange the figure{': ' + diagram.name if diagram.name else ''}"


class Server:
    """The page with bindview behind it, on 127.0.0.1: what :func:`serve` opens and waits for."""

    def __init__(self, diagram: "Diagram", title: "str | None" = None, port: int = 0, **options: Any) -> None:
        self.diagram, self.title = diagram, title or _title(diagram)
        self.options = {k: v for k, v in options.items() if k != "graph"}      # the first figure's; then the page's own
        self.key = secrets.token_urlsafe(18)
        #: The layout as the page had it when it was done.
        self.graph: "Graph | None" = None
        self.over = threading.Event()
        self._drawing = threading.Lock()
        first = _first(diagram, options)                                        # a figure that cannot be arranged says so here
        #: The page moves, turns and flips: a layout that comes back has the bond length of the first figure.
        self._bond_length = first[1].bond_length
        self._page = _html(diagram, self.title, f"/{self.key}/", options, first)
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:             # noqa: A002 (the name it has there)
                pass

            def answer(self, code: int, body: str, kind: str) -> None:
                data = body.encode("utf-8")
                self.send_response(code)
                for name, value in (("Content-Type", f"{kind}; charset=utf-8"), ("Content-Length", str(len(data))),
                                    ("Cache-Control", "no-store"), ("X-Content-Type-Options", "nosniff"),
                                    ("Referrer-Policy", "no-referrer")):
                    self.send_header(name, value)
                self.end_headers()
                self.wfile.write(data)

            def asked(self) -> "str | None":
                """What is asked of the page by the page: ``""`` for the page itself, else the word after the key
                (the path is ``/KEY/WORD`` and nothing more). None for another host, origin, key or path."""
                host = f"127.0.0.1:{server.port}"
                parts = self.path.split("/")
                if (self.headers.get("Host") != host or self.headers.get("Origin") not in (None, f"http://{host}")
                        or len(parts) != 3 or parts[0]
                        or not hmac.compare_digest(parts[1].encode("utf-8"), server.key.encode("utf-8"))):
                    return None
                return parts[2]

            def do_GET(self) -> None:                                            # noqa: N802
                if self.asked() == "":
                    self.answer(200, server._page, "text/html")
                else:
                    self.answer(404, "not here", "text/plain")

            def do_POST(self) -> None:                                           # noqa: N802
                size = _size(self.headers.get("Content-Length"))
                what = self.asked()
                if what not in ("render", "done") or size is None:
                    self.answer(404, "not here", "text/plain")
                    return
                body = server.answer(what, self.rfile.read(size))
                self.answer(400 if "error" in body else 200, json.dumps(body), "application/json")
                if what == "done" and "error" not in body:
                    server.over.set()

        self._http = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.port = int(self._http.server_address[1])
        self.address = f"http://127.0.0.1:{self.port}/{self.key}/"
        threading.Thread(target=self._http.serve_forever, daemon=True).start()

    def answer(self, what: str, body: bytes) -> dict[str, Any]:
        """What the page is told: the figure of a layout, or that the layout has been taken; or, as ``error``, why
        the layout is refused. A layout that the page cannot have sent is refused before it is drawn."""
        try:
            graph = Graph.from_dict(json.loads(body, parse_float=_finite, parse_constant=_finite))
            if not math.isclose(graph.bond_length, self._bond_length, rel_tol=1e-9, abs_tol=0.0):
                raise GraphError(f"the layout has a bond length of {graph.bond_length:g}, and the figure on the page "
                                 f"one of {self._bond_length:g}, which the page does not change")
            if what == "done":
                self.graph = graph
                return {}
            with self._drawing:
                made = self.diagram.render(graph=graph, **{**self.options, "on_unreadable": "render"})
        except RecursionError:
            return {"error": "This cannot be drawn: the layout is nested deeper than a layout is"}
        except (BindviewError, ValueError, TypeError, LookupError, ArithmeticError) as exc:
            return {"error": f"This cannot be drawn: {exc}"}
        q = made.quality
        wrong = [] if q is None else [f"{n} {word}{'s' if n != 1 else ''}" for n, word in (
            (q.overlaps, "overlap"), (q.link_hits, "line across a residue")) if n]
        return {"svg": made.svg, "graph": made.graph.to_dict() if made.graph is not None else None,
                "said": f"Drawn all the same: {', '.join(wrong)}." if wrong else ""}

    def close(self) -> None:
        self._http.shutdown()
        self._http.server_close()


def serve(diagram: "Diagram", title: "str | None" = None, *, port: int = 0, browser: bool = True, **options: Any
          ) -> "Graph | None":
    """Open the page with bindview behind it, and hand back the layout when it is done (None if it is stopped with
    Ctrl-C). ``options``: those of ``render``."""
    server = Server(diagram, title, port, **options)
    try:
        print(f"arrange the figure at {server.address}", file=sys.stderr, flush=True)
        if browser:
            webbrowser.open(server.address)
        server.over.wait()
    except KeyboardInterrupt:
        pass
    finally:
        server.close()
    return server.graph
