"""A figure arranged by hand (``bindview.arrange``), and the layout by its words (``Graph.move``, ``rotate``, ``flip``)."""

from __future__ import annotations

import contextlib
import io
import json
import math
import re
import signal
import threading
import types

import pytest

from conftest import CASE_SPECS, TESTDATA, chain_diagram, diagram_of, fresh_diagram, meta_of, model_of
from bindview import Diagram, Graph, GraphError, load_interactions
from bindview import arrange
from bindview.arrange import Server, page, serve
from bindview.cli import main
from bindview.errors import RenderError

#: c-Kit with imatinib (1T46): Cys673 is drawn with its atoms, Leu595 and Val603 as eyelashes.
CASE = "kit_imatinib"
PORT = 50123


class NoServer:
    """In the place of ``ThreadingHTTPServer``: it binds nothing and keeps the handler, which a test calls by hand."""

    made: "list[NoServer]" = []

    def __init__(self, address, handler):
        self.bound_to, self.handler, self.closed = address, handler, False
        self.server_address = (address[0], PORT)
        NoServer.made.append(self)

    def serve_forever(self):
        raise AssertionError("nothing serves here")

    def shutdown(self):
        pass

    def server_close(self):
        self.closed = True


class NoThread:
    def __init__(self, *args, **kwargs):
        pass

    def start(self):
        pass


@pytest.fixture(autouse=True)
def nothing_outside(monkeypatch):
    """No test of this module but one binds a socket or starts a thread, and none opens a browser: the server is the
    stand-in above, and what would have been opened is handed to the test. The one that binds a socket says so."""
    opened: "list[str]" = []
    monkeypatch.setattr(NoServer, "made", [])
    monkeypatch.setattr(arrange, "ThreadingHTTPServer", NoServer)
    monkeypatch.setattr(arrange, "threading",
                        types.SimpleNamespace(Event=threading.Event, Lock=threading.Lock, Thread=NoThread))
    monkeypatch.setattr(arrange, "webbrowser", types.SimpleNamespace(open=opened.append))
    return opened


def graph_of(style: str = "detailed") -> Graph:
    return diagram_of(CASE).render(style=style).graph


# ---------------------------------------------------------------------------
# The layout, in Python
# ---------------------------------------------------------------------------

def test_a_residue_is_found_by_what_the_figure_calls_it():
    g = graph_of()
    ident = "R:A:CYS673"
    for name in (ident, "Cys673", "cys673", "A:Cys673", "A:CYS673", "673", "A:673"):
        assert g.node(name).id == ident
    assert g.node("legend").kind == "legend" and g.node("ligand").kind == "ligand_atom"
    with pytest.raises(GraphError, match="no node 'Cys674'"):
        g.node("Cys674")
    twin = g.node(ident).__class__.from_dict({**g.node(ident).to_dict(), "id": "R:B:CYS673", "chain": "B"})
    g.nodes.append(twin)
    with pytest.raises(GraphError, match="is 2 nodes of this graph"):
        g.move("Cys673", 1, 1)
    assert g.move("B:Cys673", 1, 1) == ["R:B:CYS673"]


def test_a_residue_is_turned_and_flipped_where_it_is_drawn_with_atoms():
    dia = diagram_of(CASE)
    g = graph_of()
    was = (g.node("Cys673").x, g.node("Cys673").y, g.node("Cys673").angle, g.node("Cys673").mirror)
    assert g.rotate("Cys673", 30) == ["R:A:CYS673"] and g.flip("Cys673") == ["R:A:CYS673"]
    n = g.node("Cys673")
    assert (n.x, n.y) == was[:2] and n.angle == pytest.approx(was[2] + math.pi / 6) and n.mirror is not was[3]
    drawn = dia.render(graph=g, on_unreadable="render")
    assert drawn.svg != dia.render().svg and drawn.graph.node("Cys673").mirror is n.mirror
    g.rotate("Cys673", -30)
    g.flip("Cys673")
    assert dia.render(graph=g).svg == dia.render().svg                           # turned back it is the figure it was
    for ident in ("Leu595", "legend"):                                           # an eyelash; the key
        with pytest.raises(GraphError, match="is drawn as"):
            g.rotate(ident, 10)
        with pytest.raises(GraphError, match="nothing to flip"):
            g.flip(ident)
    with pytest.raises(GraphError, match="'bubble'"):
        graph_of("schematic").rotate("Cys673", 10)


def test_the_ligand_is_turned_alone_or_with_everything_around_it():
    """Turned alone, the ligand leaves its residues where they are; turned with them, every place keeps its distance
    from the ligand's middle, and the figure drawn from the layout has every residue where the layout puts it. (Words
    stay upright: after a turn two of them may meet, which the figure's audit counts.)"""
    g = graph_of()
    apart = {n.id: (n.x, n.y) for n in g.nodes}
    assert set(g.rotate("ligand", 90)) == set(g.group("ligand").members)
    assert all((n.x, n.y) == apart[n.id] for n in g.of_kind("residue"))
    g = graph_of()
    atoms = [n for n in g.of_kind("ligand_atom")]
    middle = (sum(n.x for n in atoms) / len(atoms), sum(n.y for n in atoms) / len(atoms))
    far = {n.id: math.dist((n.x, n.y), middle) for n in g.nodes if n.kind != "legend"}
    g.rotate("figure", 35)
    for n in g.nodes:
        if n.kind != "legend":
            assert math.dist((n.x, n.y), middle) == pytest.approx(far[n.id])
    res = diagram_of(CASE).render(graph=g, on_unreadable="render")
    assert res.svg is not None and res.quality.unplaced == 0
    assert all((res.graph.node(n.id).x, res.graph.node(n.id).y) == (n.x, n.y) for n in g.of_kind("residue"))
    with pytest.raises(GraphError, match="no ligand that is one body"):                # hirudin, a chain ligand
        chain_diagram("thrombin_hirudin").render().graph.rotate("figure", 10)


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------

def given(html: str) -> dict:
    return json.loads(re.search(r'<script type="application/json" id="graph">(.*?)</script>', html, flags=re.S).group(1))


@pytest.mark.parametrize("style", ("detailed", "schematic"))
def test_the_page_is_one_file_with_the_figure_and_its_layout(style, tmp_path):
    dia = diagram_of(CASE)
    out = dia.arrange(tmp_path / "arrange.html", style=style)
    text = out.read_text(encoding="utf-8")
    res = dia.render(style=style)
    assert res.svg in text and "window.EXACT = null" in text and 'id="done"' not in text
    assert not re.search(r'(src|href)="https?://', text)                          # nothing from the network
    data = given(text)
    assert Graph.from_dict(data["graph"]).to_json() == res.graph.to_json() and data["svg"] == res.svg
    # What the script goes by: the groups of the figure, and a node for each.
    ids = {n["id"] for n in data["graph"]["nodes"]}
    residues = re.findall(r"<g class=\"residue\"[^>]*? data-residue=\"([^\"]+)\"", res.svg)
    assert residues and {f"R:{r}" for r in residues} <= ids and 'class="legend"' in res.svg
    bonded = data["bonds"]
    assert set(bonded) == {f"L:{n}" for n in dia.ligand.names} and all(b in bonded for v in bonded.values() for b in v)


def test_a_figure_that_cannot_be_arranged_says_so(tmp_path):
    """Carbon monoxide in haemoglobin with a table that has only its header: a figure with nothing in it to move."""
    empty = tmp_path / "interactions.tsv"
    empty.write_text((TESTDATA / "hb_carbon_monoxide" / "interactions.tsv").read_text().splitlines()[0] + "\n")
    m = meta_of("hb_carbon_monoxide")["ligand"]
    dia = Diagram.from_structure(model_of("hb_carbon_monoxide"), m["residue_name_in_cif"], smiles=m["smiles"],
                                 interactions=load_interactions(empty))
    with pytest.raises(RenderError, match="nothing to arrange: the figure is empty"):
        page(dia, tmp_path / "x.html")
    assert not (tmp_path / "x.html").exists()


class Wire:
    """A socket as the handler uses one: the request to read, and what is sent back."""

    def __init__(self, request: bytes):
        self.request, self.sent = io.BytesIO(request), io.BytesIO()

    def makefile(self, mode, bufsize=-1):
        return self.request

    def sendall(self, data):
        self.sent.write(data)


def ask(server: Server, method: str, path: str, body: "bytes | None" = None, **headers) -> tuple[int, str]:
    """One request, handed to the handler as bytes; the status and the body of its answer. An exception that the
    handler lets through comes out of here: on a socket the request would go unanswered."""
    given = {"Host": f"127.0.0.1:{server.port}", "Content-Type": "application/json", **headers}
    if body is not None:
        given.setdefault("Content-Length", str(len(body)))
    lines = [f"{method} {path} HTTP/1.1"] + [f"{name}: {value}" for name, value in given.items()]
    wire = Wire(("\r\n".join(lines) + "\r\n\r\n").encode("latin-1") + (body or b""))
    server._http.handler(wire, ("127.0.0.1", 40000), server._http)
    head, _, text = wire.sent.getvalue().partition(b"\r\n\r\n")
    return int(head.split()[1]), text.decode("utf-8")


def test_what_is_let_go_is_drawn_by_the_library_and_done_hands_the_layout_back():
    dia = diagram_of(CASE)
    server = Server(dia)
    try:
        status, text = ask(server, "GET", f"/{server.key}/")
        assert status == 200 and json.loads(re.search(r"window.EXACT = (.*?);", text).group(1)) == f"/{server.key}/"
        g = graph_of()
        atoms = g.of_kind("ligand_atom")
        middle = (sum(n.x for n in atoms) / len(atoms), sum(n.y for n in atoms) / len(atoms))
        leu = g.node("Leu595")
        away = math.dist((leu.x, leu.y), middle)
        g.move("Leu595", 40.0 * (leu.x - middle[0]) / away, 40.0 * (leu.y - middle[1]) / away)   # further out
        g.rotate("Cys673", 20)
        status, text = ask(server, "POST", f"/{server.key}/render", json.dumps(g.to_dict()).encode())
        body = json.loads(text)
        exact = dia.render(graph=g, on_unreadable="render")
        assert status == 200 and body["svg"] == exact.svg and body["said"] == ""
        assert Graph.from_dict(body["graph"]).to_json() == exact.graph.to_json()
        # An eyelash faces its atoms from where it now is: the drawing is the library's, not the page's.
        assert body["svg"] != dia.render().svg
        a, b = g.node("Val603"), g.node("Leu595")
        g.move("Val603", b.x - a.x, b.y - a.y)                                   # onto another residue
        assert "Drawn all the same" in json.loads(ask(server, "POST", f"/{server.key}/render",
                                                      json.dumps(g.to_dict()).encode())[1])["said"]
        assert "cannot be drawn" in json.loads(ask(server, "POST", f"/{server.key}/render", b"[1]")[1])["error"]
        assert not server.over.is_set()
        ask(server, "POST", f"/{server.key}/done", json.dumps(g.to_dict()).encode())
        assert server.over.wait(5) and server.graph.to_json() == g.to_json()
    finally:
        server.close()


def test_the_page_answers_itself_and_nobody_else():
    server = Server(diagram_of(CASE))
    body = json.dumps(graph_of().to_dict()).encode()
    try:
        for path in ("/", "/render", f"/{server.key}/x", f"/{server.key[:-1]}/", f"/{server.key}/../"):
            assert ask(server, "GET", path)[0] == 404
        for headers in ({"Origin": "http://example.com"}, {"Host": "example.com"}, {"Host": f"localhost:{server.port}"}):
            assert ask(server, "POST", f"/{server.key}/render", body, **headers)[0] == 404
        assert ask(server, "POST", "/render", body)[0] == 404 and ask(server, "POST", f"/{server.key}/quit", body)[0] == 404
        assert ask(server, "POST", f"/{server.key}/render", b"", **{"Content-Length": "999999999"})[0] == 404
        assert ask(server, "POST", f"/{server.key}/render", body, Origin=f"http://127.0.0.1:{server.port}")[0] == 200
        assert server._http.server_address[0] == "127.0.0.1"
    finally:
        server.close()


def test_the_command_line_writes_the_page(tmp_path):
    out, figure = tmp_path / "arrange.html", tmp_path / "figure.svg"
    ligand = meta_of(CASE)["ligand"]["residue_name_in_cif"]
    assert CASE in CASE_SPECS
    assert main(["render", str(model_of(CASE)), "--ligand", ligand, "--smiles-from", "meta.json", "-q",
                 "-o", str(figure), "--arrange-page", str(out)]) == 0
    assert not figure.exists() and "window.EXACT = null" in out.read_text()


# ---------------------------------------------------------------------------
# What the page's server refuses, and where its words go
# ---------------------------------------------------------------------------

class Counted:
    """A diagram that counts how often it is drawn."""

    def __init__(self, diagram):
        self._diagram, self.drawn = diagram, 0

    def render(self, *args, **options):
        self.drawn += 1
        return self._diagram.render(*args, **options)

    def __getattr__(self, name):
        return getattr(self._diagram, name)


@contextlib.contextmanager
def within(seconds: int):
    """Stop what does not end: a layout that is drawn for ever fails its test and does not hold the others up."""
    def stop(*_):
        raise AssertionError(f"no answer in {seconds} s")
    before = signal.signal(signal.SIGALRM, stop)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, before)


def layout(**changes) -> dict:
    """The layout of the first figure, as the page has it, with something of it changed."""
    return {**diagram_of(CASE).render().graph.to_dict(), **changes}


def moved(**node) -> dict:
    """... with its first residue changed."""
    made = layout()
    made["nodes"] = [dict(n) for n in made["nodes"]]
    next(n for n in made["nodes"] if n["id"].startswith("R:")).update(node)
    return made


#: Requests that have the key and come from the page's own address, and still cannot be answered: the body, as
#: text, and a header that is not as it should be.
UNANSWERABLE = {
    "a length that is no number": (lambda: json.dumps(layout()), {"Content-Length": "\u00b2"}),
    "a length of five thousand digits": (lambda: "", {"Content-Length": "9" * 5000}),
    "a body nested 100,000 deep": (lambda: "[" * 100_000 + "]" * 100_000, {}),
    "a bond length of 0": (lambda: json.dumps(layout(bond_length=0)), {}),
    "a bond length of 1e9": (lambda: json.dumps(layout(bond_length=1e9)), {}),
    "a bond length of -30": (lambda: json.dumps(layout(bond_length=-30)), {}),
    "a bond length that is no number": (lambda: json.dumps(layout(bond_length=float("nan"))), {}),
    "a node at Infinity": (lambda: json.dumps(moved(x=float("inf"))), {}),
    "a node at 1e999": (lambda: json.dumps(moved(x=12345.5)).replace("12345.5", "1e999"), {}),
    "a node at a whole number of 400 digits": (lambda: json.dumps(moved(x=10 ** 400)), {}),
}


@pytest.mark.parametrize("what", list(UNANSWERABLE))
def test_a_request_that_cannot_be_answered_is_refused_before_anything_is_drawn(what):
    text, headers = UNANSWERABLE[what]
    body = text().encode()
    counted = Counted(diagram_of(CASE))
    server = Server(counted)
    assert counted.drawn == 1                                              # the first figure, for the page
    try:
        for word in ("render", "done"):
            with within(60):
                status, text = ask(server, "POST", f"/{server.key}/{word}", body, **headers)
            assert status in (400, 404) and counted.drawn == 1, word
            assert status == 404 or json.loads(text)["error"].startswith("This cannot be drawn: ")
            assert not server._drawing.locked() and server.graph is None and not server.over.is_set()
        # ... and the page goes on: the next layout is drawn.
        status, text = ask(server, "POST", f"/{server.key}/render", json.dumps(layout()).encode())
        assert status == 200 and json.loads(text)["svg"] == diagram_of(CASE).render().svg and counted.drawn == 2
    finally:
        server.close()


def test_a_layout_keeps_the_bond_length_of_its_figure_through_the_page():
    """What the page sends back after a figure was drawn for it is taken: the bond length is the first figure's."""
    server = Server(diagram_of(CASE))
    try:
        first = layout()
        status, text = ask(server, "POST", f"/{server.key}/render", json.dumps(first).encode())
        again = json.loads(text)["graph"]
        assert status == 200 and again["bond_length"] == first["bond_length"]
        assert ask(server, "POST", f"/{server.key}/render", json.dumps(again).encode())[0] == 200
    finally:
        server.close()


def test_a_post_is_taken_under_its_own_path_and_no_other():
    server = Server(diagram_of(CASE))
    body = json.dumps(layout()).encode()
    try:
        for path in (f"/{server.key}/a/b/render", f"/{server.key}/../../render", f"/{server.key}//render",
                     f"/x/{server.key}/render", f"/{server.key}/render/", f"/{server.key}/render?x=1"):
            assert ask(server, "POST", path, body)[0] == 404, path
        assert ask(server, "POST", f"/{server.key}/render", body)[0] == 200
    finally:
        server.close()


def test_the_key_is_compared_in_constant_time(monkeypatch):
    import hmac

    compared = []

    def compare(a, b):
        compared.append((a, b))
        return hmac.compare_digest(a, b)

    monkeypatch.setattr(arrange, "hmac", types.SimpleNamespace(compare_digest=compare))
    server = Server(diagram_of(CASE))
    try:
        wrong = "x" * len(server.key)
        assert ask(server, "GET", f"/{wrong}/")[0] == 404 and ask(server, "GET", f"/{server.key}/")[0] == 200
        assert [a for a, _ in compared] == [wrong.encode(), server.key.encode()]
        assert all(b == server.key.encode() for _, b in compared)
        assert ask(server, "GET", "/k\u00e9y/")[0] == 404                 # (a key that is not ASCII is compared too)
    finally:
        server.close()


def test_the_data_of_the_page_cannot_end_its_own_block(tmp_path):
    """A name goes into the page inside a script block of data. ``<!--<script>`` there would keep the block open
    past its end tag, as HTML reads it: no ``<`` is written into the block."""
    name = "<!--<script>"
    dia = fresh_diagram(CASE, name=name)
    text = page(dia, tmp_path / "arrange.html").read_text(encoding="utf-8")
    block = text.split('<script type="application/json" id="graph">')[1].split("</script>")[0]
    assert "<" not in block
    data = json.loads(block)
    assert data["graph"]["meta"]["ligand"] == name and data["svg"] == dia.render().svg


def test_the_page_is_built_from_a_diagram_a_title_and_its_options(tmp_path):
    """What ``page()`` writes is what the builder of the page hands back for the same figure: the page of the
    package's own documentation is made by calling it."""
    dia = diagram_of(CASE)
    written = page(dia, tmp_path / "arrange.html", "A title", style="schematic").read_text(encoding="utf-8")
    assert arrange._html(dia, "A title", None, {"style": "schematic"}) == written
    assert dia.render(style="schematic").svg in written and "<h1>A title</h1>" in written


def test_the_server_answers_on_a_socket_of_its_own(monkeypatch):
    """The page's own server, as it runs: on 127.0.0.1, on a port the system gives it, in a thread of its own. A
    layout posted under its key is answered with the figure of it; one posted beside the key is refused. The server is
    closed whatever happens."""
    import http.client
    import http.server

    monkeypatch.setattr(arrange, "ThreadingHTTPServer", http.server.ThreadingHTTPServer)    # (the real ones, here)
    monkeypatch.setattr(arrange, "threading", threading)
    dia = diagram_of(CASE)
    server = Server(dia)
    try:
        assert server._http.server_address[0] == "127.0.0.1" and server.port > 0 and not NoServer.made
        body = json.dumps(layout()).encode()

        def post(path: str) -> "tuple[int, str]":
            connection = http.client.HTTPConnection("127.0.0.1", server.port, timeout=120)
            try:
                connection.request("POST", path, body, {"Content-Type": "application/json"})
                answer = connection.getresponse()
                return answer.status, answer.read().decode("utf-8")
            finally:
                connection.close()

        status, text = post(f"/{server.key}/render")
        assert status == 200 and json.loads(text)["svg"] == dia.render().svg
        assert post("/render") == (404, "not here")
    finally:
        server.close()


def test_the_address_is_said_on_standard_error_and_the_server_is_closed_whatever_happens(capsys, monkeypatch,
                                                                                        nothing_outside):
    dia = diagram_of(CASE)
    started = Server.__init__

    def done_at_once(self, *args, **options):
        """The page as if Done were clicked on the figure as it is."""
        started(self, *args, **options)
        ask(self, "POST", f"/{self.key}/done", json.dumps(layout()).encode())

    monkeypatch.setattr(Server, "__init__", done_at_once)
    graph = serve(dia)
    captured = capsys.readouterr()
    (http,) = NoServer.made
    address = f"http://127.0.0.1:{PORT}/"
    assert graph.to_json() == dia.render().graph.to_json() and http.closed
    assert captured.out == "" and captured.err.startswith(f"arrange the figure at {address}")
    assert len(nothing_outside) == 1 and nothing_outside[0].startswith(address)
    # A browser that cannot be opened is an error, and Ctrl-C while it opens is the page not done: closed both times.
    monkeypatch.setattr(Server, "__init__", started)
    for raised in (RuntimeError("no browser here"), KeyboardInterrupt()):
        def fails(address, raised=raised):
            raise raised
        monkeypatch.setattr(arrange, "webbrowser", types.SimpleNamespace(open=fails))
        try:
            got = serve(dia)
        except (RuntimeError, KeyboardInterrupt) as exc:
            got = exc
        assert NoServer.made[-1].closed, raised
        assert got is None if isinstance(raised, KeyboardInterrupt) else got is raised


def test_arranging_from_the_command_line_leaves_standard_output_to_json(tmp_path, capsys, monkeypatch):
    started = Server.__init__

    def done_at_once(self, *args, **options):
        started(self, *args, **options)
        ask(self, "POST", f"/{self.key}/done", json.dumps(layout()).encode())

    monkeypatch.setattr(Server, "__init__", done_at_once)
    ligand = meta_of(CASE)["ligand"]
    out = tmp_path / "arranged.svg"
    assert main(["render", str(model_of(CASE)), "--ligand", ligand["residue_name_in_cif"], "--smiles", ligand["smiles"],
                 "--interactions", str(TESTDATA / CASE / "interactions.tsv"), "--arrange", "--json", "-q",
                 "-o", str(out)]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["output"] == str(out) and out.read_text().startswith("<svg ")
    assert captured.err.startswith("arrange the figure at http://127.0.0.1:")
