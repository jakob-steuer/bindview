"""An interface: the contacts between two polymer participants, drawn around a spine, with a site.

The figures are of deposited structures in the repository: the light and the heavy chain of thrombin
(2UUF, ``thrombin_hirudin``), the antibody HyHEL-10 on lysozyme (3D9A, ``fab_lysozyme``) and HLA-A2 with
its peptide (3MRE, ``mhc_peptide``); and of the predicted TNF trimer where a model's own numbers matter.
Contacts come from the plain distance table (``bindview.distance_table``). A crystal structure has no
pLDDT: where one is drawn here it is made up (``conftest.invented_plddt``). The tests that take their
contacts from PLIP run where PLIP is installed.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import replace

import numpy as np
import pytest

from conftest import chains_interface, invented_plddt, model_of, needs_plip, structure_of
from bindview import Interface, Participant, load_interface_contacts
from bindview.confidence import band_of
from bindview.distance_table import interface_contacts
from bindview.errors import BindviewError
from bindview.interface import MAX_SPINE, choose_site, dump_interface_contacts
from bindview.theme import Metrics
from scenetools import groups, words


def thrombin(**options) -> Interface:
    """The light chain (A) and the heavy chain (B) of thrombin, contacts from the distance table."""
    return chains_interface("thrombin_hirudin", "A", "B", **options)


def numbers(res) -> list[str]:
    """What the boxes of the column say, top to bottom."""
    units = sorted(groups(res, "unit"), key=lambda g: int(g.data["position"]))
    return [g.title.split(" of ")[0] for g in units]          # "Glu 23 of chain B (numbered as in ...)"


def name(structure, key, gap: str = "") -> str:
    """A residue as a figure writes it: ``Asp14L`` as a label, ``Asp 14L`` in a box."""
    return f"{structure.residue_name(key).title()}{gap}{key.number}{key.icode}"


def heavy_atoms(structure, key) -> np.ndarray:
    return np.array([a.xyz for a in structure.residue_atoms(key) if a.element not in ("H", "D")], dtype=float)


# ---------------------------------------------------------------------------
# The object
# ---------------------------------------------------------------------------

def test_participants_are_named_by_their_chains_or_by_the_caller():
    assert Participant.parse("A") == Participant("chain A", ("A",))
    assert Participant.parse("D+E") == Participant("chains D, E", ("D", "E"))
    assert Participant.parse("receptor=D+E") == Participant("receptor", ("D", "E"))
    assert Participant.parse(["D", "E"]).chains == ("D", "E")
    with pytest.raises(BindviewError):
        Participant.parse("receptor=")


def test_an_interface_is_a_spine_and_what_touches_it():
    """The two chains of thrombin: the heavy chain has fewer residues at the interface and is the spine.
    Both chains are numbered with insertion codes, and every label and box keeps them."""
    face = thrombin()
    st = face.structure
    table = interface_contacts(st, ["A"], ["B"])
    light, heavy = {c.residue_a for c in table}, {c.residue_b for c in table}
    assert len(heavy) < len(light)
    assert face.spine.name == "chain B" and [p.name for p in face.others] == ["chain A"]
    assert len(face.contacts) == len(table) and all(c.chain_a == "A" and c.chain_b == "B" for c in face.contacts)
    site = face.site
    assert site.whole and (len(site.residues), site.of_residues, site.contacts, site.of_contacts) == (
        len(heavy), len(heavy), len(table), len(table))
    res = face.render(style="schematic")
    assert res.status == "ok" and res.figure == "diagram" and res.quality.valid
    assert len(groups(res, "unit")) == len(heavy) and res.quality.n_residues == len(light)
    text = words(res)
    assert "An interface has no centre." in text
    assert "The residues of chain B that touch chain A stand in a column" in text
    assert f"All {len(heavy)} residues of chain B at the interface are shown, with {len(table)} contacts." in text
    assert "Residue of chain B, numbered as in the structure file" in text
    assert "N-terminus" not in text and "of the ligand it touches" not in text
    meta = res.graph.meta["interface"]
    assert meta["participants"] == [{"name": "chain A", "chains": ["A"]}, {"name": "chain B", "chains": ["B"]}]
    assert (meta["spine"], meta["contacts_from"]) == ("chain B", "supplied")
    assert meta["site"]["kind"] == "whole" and res.data["interface"] == meta
    assert res.graph.to_dict()["version"] == "1.2"
    # Insertion codes on both sides (14L of the light chain, 184A of the heavy chain), kept in labels and boxes.
    assert any(k.icode for k in light) and any(k.icode for k in heavy)
    assert {n.label for n in res.graph.nodes if n.kind == "residue"} == {name(st, k) for k in light}
    assert sorted(numbers(res)) == sorted(name(st, k, " ") for k in heavy)
    assert face.render(style="detailed").status == "ok"


def test_a_line_stands_for_a_pair_of_residues():
    """Several atoms of one residue on one other residue are one contact between two residues, drawn once."""
    face = thrombin()
    graph = face.render(style="schematic").graph
    lines = [(c.residue, c.ligand, c.type) for c in graph.contacts]
    assert len(lines) == len(set(lines)) and all(c.drawn for c in graph.contacts)
    n = len(face.contacts)
    assert len(lines) < n and sorted(i for c in graph.contacts for i in c.records) == list(range(n))
    several = [c for c in graph.contacts if len(c.records) > 1]
    assert several
    by_index = face.diagram.interactions
    for c in several:
        shortest = min(by_index[i].record.distance for i in c.records)
        assert c.distance == shortest and (c.label is None or c.label.startswith(f"{shortest:.2f}"))
    assert "A line is a pair of residues, at the shortest of its contacts of one kind" in words(face.render(style="schematic"))


def test_the_spine_is_the_side_with_fewer_residues_unless_told():
    face = thrombin()
    light = {c.residue_a for c in face.contacts}
    assert face.spine.chains == ("B",) and face.site.of_residues < len(light)
    other = thrombin(spine="A")
    assert other.spine.chains == ("A",) and other.site.of_residues == len(light) and other.site.whole
    assert other.render(style="schematic").status == "ok"
    with pytest.raises(BindviewError, match="no participant"):
        thrombin(spine="Q")


def test_a_spine_larger_than_a_figure_holds_narrows_to_a_site_and_says_so():
    """The heavy chain of HLA-A2 against beta-2 microglobulin and the peptide together: more residues at
    the interface than a column holds. The figure shows whole contact clusters, as many as fit."""
    face = chains_interface("mhc_peptide", "A", ["B", "P"])
    site = face.site
    assert face.spine.chains == ("A",) and site.of_residues > MAX_SPINE
    assert site.kind == "cluster" and len(site.residues) <= MAX_SPINE and site.contacts < site.of_contacts
    assert re.fullmatch(r"the \d+ of its \d+ contact clusters with the most contacts that fit a figure", site.rule)
    per = Counter(c.residue_a for c in face._oriented)
    assert site.contacts == sum(per[k] for k in site.residues) and site.of_contacts == len(face.contacts)
    res = face.render(style="schematic")
    assert res.status == "ok"
    assert (f"A site, not the whole interface: {len(site.residues)} of the {site.of_residues} residues of chain A at "
            f"the interface ({site.rule}), with {site.contacts} of {site.of_contacts} contacts.") in words(res)


def test_an_interface_figure_names_its_sides_and_no_ligand():
    """Tooltips, footnotes, the description and the message that a figure cannot be drawn: an
    interface has neither ligand nor pocket, and what a reader sees or is read does not say so."""
    face = thrombin()
    res = face.render(style="schematic")
    said = [t.strip() for t in re.findall(r">([^<>]+)<", res.svg)] + list(res.warnings)
    assert len(said) > 200 and not [t for t in said if re.search(r"ligand|pocket", t, re.I)]
    titles = [g.title for g in groups(res, "interaction")]
    assert titles and all(" of chain B (" in t for t in titles)
    assert groups(res, "unit")[0].title.endswith(" of chain B (numbered as in the structure file)")
    # The names of the data attributes are the schema every figure shares, and stay.
    assert all("ligand-residue" in g.data for g in groups(res, "interaction"))
    # A buried fraction is a ligand's number, with a cut calibrated on ligands: an interface reports
    # none, warns of none and prints none.
    site = thrombin(site="A:14")
    plain = site.render(style="schematic")
    assert plain.burial is None and "burial" not in plain.graph.meta
    assert not [w for w in plain.warnings if "buried" in w]
    off = site.render(style="schematic", burial=False)                       # on is the default: the same file
    assert off.svg == plain.svg and off.warnings == plain.warnings
    assert "buried" not in words(plain) and not groups(plain, "burial-warning")
    small = thrombin(site="A:14").render(style="schematic", figure_width_mm=60)
    assert small.figure == "summary"
    assert small.warnings[0].startswith(f"an interface diagram is not readable: {len(site.site.residues)} residues "
                                        f"in the column (")
    assert not [t for t in re.findall(r">([^<>]+)<", small.svg) + small.warnings if re.search(r"ligand|pocket", t, re.I)]


def test_a_site_is_chosen_by_a_stated_rule():
    whole = thrombin()
    st = whole.structure
    assert whole.site.rule == "the whole interface" and len(whole.site.residues) <= MAX_SPINE
    per = Counter(c.residue_a for c in whole._oriented)
    window = thrombin(max_residues=12)
    chosen = window.site.residues
    assert (window.site.kind, len(chosen)) == ("window", 12) and window.site.contacts == sum(per[k] for k in chosen)
    assert window.site.rule == (f"the 12 consecutive residues of the interface with the most contacts, residues "
                                f"{chosen[0].number}{chosen[0].icode} to {chosen[-1].number}{chosen[-1].icode}")
    assert window.render(style="schematic").status == "ok"
    # Around a residue of the other side: every residue of the spine at the interface with an atom within 8 Å of it.
    seeded = thrombin(site="A:14")
    seed = st.find_residue("A", 14, "")
    near = [k for k in whole._order
            if np.linalg.norm(heavy_atoms(st, k)[:, None] - heavy_atoms(st, seed)[None], axis=2).min() <= 8.0]
    assert seeded.site.kind == "seed" and list(seeded.site.residues) == near
    assert seeded.site.contacts == sum(per[k] for k in near)
    assert seeded.site.rule == f"within 8 Å of {name(st, seed)} of chain A"
    assert set(seeded.site.residues) < set(whole.site.residues)
    assert (f"A site, not the whole interface: {len(near)} of the {len(whole.site.residues)} residues"
            in words(seeded.render(style="schematic")))
    with pytest.raises(BindviewError, match="within 8"):
        thrombin(site="B:97")                                    # a residue far from the interface
    with pytest.raises(BindviewError, match="chain:number"):
        thrombin(site="nonsense")
    # The same choice every time, whatever order the contacts come in.
    again = choose_site(list(reversed(whole._oriented)), whole._order, whole._index, limit=12)
    assert again == window.site


def test_the_order_of_the_column_is_the_one_that_spreads_contacts_least():
    """Sequence, or along the long axis of the interface: whichever leaves the residues around the
    column reaching less far along it; where the two are level, the sequence. That is known before
    anything is laid out."""
    face = thrombin()
    by_sequence = list(face.site.residues)
    by_position = face._by_position(by_sequence)
    assert by_position != by_sequence and sorted(by_position, key=by_sequence.index) == by_sequence
    assert face.reach(by_position) < face.reach(by_sequence)
    res = face.render(style="schematic")
    assert res.graph.meta["interface"]["spine_order"] == "interface"
    assert numbers(res) == [name(face.structure, k, " ") for k in by_position]
    dep = face.diagram.backend.depict(Metrics())
    assert not dep.sequential and not any(b.kind == "gap" for b in dep.bonds)
    assert "ordered along the interface and not by their sequence" in words(res)
    # Asked for, the sequence: a line joins residues that follow each other in the chain, a dotted one the others.
    in_order = thrombin(order="sequence")
    res_seq = in_order.render(style="schematic")
    assert numbers(res_seq) == [name(face.structure, k, " ") for k in by_sequence]
    place = [in_order._index[k] for k in in_order.site.residues]
    neighbours = sum(1 for a, b in zip(place, place[1:]) if b - a == 1)
    dep = in_order.diagram.backend.depict(Metrics())
    assert dep.sequential and (sum(b.kind == "gap" for b in dep.bonds), sum(b.kind == "chain" for b in dep.bonds)) == (
        len(place) - 1 - neighbours, neighbours)
    assert "in the order of the sequence; a dotted line stands for residues in between" in words(res_seq)
    assert res_seq.graph.meta["interface"]["spine_order"] == "sequence"
    # The peptide of HLA-A2: the two orders reach equally far, and the sequence is kept.
    peptide = chains_interface("mhc_peptide", "A", "P")
    residues = list(peptide.site.residues)
    assert peptide.spine.chains == ("P",) and peptide.reach(residues) == peptide.reach(peptide._by_position(residues))
    assert peptide.render(style="schematic").graph.meta["interface"]["spine_order"] == "sequence"
    with pytest.raises(BindviewError):
        thrombin(order="alphabetical")


def test_three_participants_are_drawn_around_the_middle_one():
    """The heavy and the light chain of the antibody on lysozyme, given in that order with lysozyme
    between them: the middle one touches both and is the spine. Only the contacts that are given are drawn."""
    st = structure_of("fab_lysozyme")
    contacts = interface_contacts(st, ["H"], ["C"]) + interface_contacts(st, ["C"], ["L"])
    face = Interface(st, ["H", "C", "L"], contacts)
    assert face.spine.chains == ("C",) and [p.chains for p in face.others] == [("H",), ("L",)]
    assert {c.chain_a for c in face.contacts} | {c.chain_b for c in face.contacts} == {"H", "C", "L"}
    assert not any({c.chain_a, c.chain_b} == {"H", "L"} for c in face.contacts)
    res = face.render(style="schematic")
    assert res.status == "ok"
    assert {n.attrs["chain"] for n in res.graph.nodes if n.kind == "residue"} == {"H", "L"}
    assert "that touch chain H and chain L stand in a column" in words(res)
    labels = {n.label for n in res.graph.nodes if n.kind == "residue"}
    assert any(label.endswith("(H)") for label in labels) and any(label.endswith("(L)") for label in labels)


def test_the_spine_of_three_cannot_be_one_at_an_end():
    """Of three participants in a row the middle one touches both others, and it is the spine. One at
    an end is refused for that reason, and not for a residue that is said not to be in the structure."""
    from conftest import trimer_contacts, trimer_structure

    st = trimer_structure()
    contacts = trimer_contacts("A", "B")[:6] + trimer_contacts("B", "C")[:6]
    for end, other in (("A", "chain C"), ("C", "chain A")):
        with pytest.raises(BindviewError) as told:
            Interface(st, ["A", "B", "C"], contacts, spine=end, model_confidence=None)
        said = str(told.value)
        assert f"chain {end} is at an end" in said and f"does not touch {other}" in said and "chain B" in said
        assert "not in the structure" not in said
    middle = Interface(st, ["A", "B", "C"], contacts, spine="B", model_confidence=None)
    assert middle.spine.chains == ("B",) and [p.name for p in middle.others] == ["chain A", "chain C"]


def test_what_is_not_an_interface():
    st = structure_of("thrombin_hirudin")
    one = interface_contacts(st, ["A"], ["B"])
    with pytest.raises(BindviewError, match="two participants"):
        Interface(st, ["A"], one)
    with pytest.raises(BindviewError, match="one participant only"):
        Interface(st, ["A", "A"], one)
    with pytest.raises(BindviewError, match="no chain Q"):
        Interface(st, ["A", "Q"], one)
    with pytest.raises(BindviewError, match="no contact between"):
        Interface(st, ["A", "B"], [])
    with pytest.raises(BindviewError, match="the spine has to be one chain"):
        chains_interface("fab_lysozyme", ["H", "L"], "C", spine="H+L")
    # A chain that long is not a ligand, and the error says where to go instead.
    with pytest.raises(BindviewError, match="one side of an interface"):
        st.resolve_ligand("B")


def test_a_figure_of_no_residue_is_refused_in_words():
    """``max_residues`` is how many residues of the spine a figure shows at most. None, or fewer, is no
    figure, and is refused when the interface is made."""
    from conftest import trimer_contacts, trimer_structure

    for asked in (0, -3, 2.5, "ten"):
        with pytest.raises(BindviewError, match="max_residues is a whole number of residues, 1 or more"):
            Interface(trimer_structure(), ["A", "B"], trimer_contacts()[:6], max_residues=asked, model_confidence=None)
    one = Interface(trimer_structure(), ["A", "B"], trimer_contacts()[:6], max_residues=1, model_confidence=None)
    assert len(one.site.residues) == 1
    with pytest.raises(BindviewError, match="a site shows 1 residue or more, not 0"):
        choose_site([], [], {}, limit=0)


def test_a_contact_carries_its_confidence_into_the_graph(tmp_path):
    st = structure_of("thrombin_hirudin")
    contacts = interface_contacts(st, ["A"], ["B"])
    marked = [replace(c, pae=3.25, confidence=81.0) if n < 5 else c for n, c in enumerate(contacts)]
    graph = Interface(st, ["A", "B"], marked).render(style="schematic").graph
    carried = [c for c in graph.contacts if "pae" in c.attrs]
    assert carried and all(c.attrs["pae"] == 3.25 for c in carried)
    # The table round-trips, in either orientation, with everything that rides on a contact.
    path = tmp_path / "contacts.tsv"
    path.write_text(dump_interface_contacts(marked))
    assert load_interface_contacts(path) == marked
    flipped = [c.flipped() for c in marked]
    assert [c.flipped() for c in flipped] == marked
    assert Interface(st, ["A", "B"], flipped).contacts == Interface(st, ["A", "B"], marked).contacts


def test_a_table_error_names_the_line_of_the_file_and_a_byte_order_mark_hides_no_header(tmp_path):
    head = "\t".join(["interaction_type", "chain_a", "residue_number_a", "residue_name_a", "chain_b",
                      "residue_number_b", "residue_name_b", "atoms_a", "atoms_b"])
    good = "hbond\tA\t10\tASP\tB\t6\tARG\tO\tNH1"
    table = tmp_path / "contacts.tsv"
    # Comments and empty lines are no rows, and they are lines: the sixth line is the sixth.
    table.write_text("# a table by hand\n# second comment\n\n" + head + "\n" + good + "\n"
                     + good.replace("\t10\t", "\tten\t") + "\n")
    with pytest.raises(BindviewError, match=r"contacts\.tsv, line 6: .*'ten'"):
        load_interface_contacts(table)
    table.write_text(head + "\n" + good + "\n\n# a comment\nhbond\tA\t10\tASP\n")
    with pytest.raises(BindviewError, match=r"contacts\.tsv, line 5: the row ends before the column 'chain_b'"):
        load_interface_contacts(table)
    # A table saved by a spreadsheet may begin with a byte order mark, which is no part of its first column's name.
    table.write_bytes(b"\xef\xbb\xbf" + (head + "\n" + good + "\n").encode("utf-8"))
    assert [(c.chain_a, c.number_a, c.atoms_b) for c in load_interface_contacts(table)] == [("A", 10, ("NH1",))]


def test_the_model_s_confidence_reaches_the_boxes():
    """A box is drawn at the pLDDT of its residue, the mean of its atoms', as for any chain, and the
    figure says what the strokes mean. The pLDDT is made up for the test (``invented_plddt``)."""
    st = invented_plddt(structure_of("thrombin_hirudin"))
    face = chains_interface("thrombin_hirudin", "A", "B", structure=st)
    res = face.render(style="schematic")
    conf = res.confidence
    assert conf is not None and conf["origin"] == "declared"
    units = groups(res, "unit")
    assert sorted(g.data["confidence"] for g in units) == sorted(
        band_of(float(np.mean([a.b_iso for a in st.residue_atoms(k)]))) for k in face.site.residues)
    text = words(res)
    assert "Confidence:" in text and "a contact is drawn at the lower of its two ends' values" in text
    column = [a.b_iso for k in face.site.residues for a in st.residue_atoms(k)]
    assert conf["ligand_plddt"] == pytest.approx(float(np.mean(column)), abs=0.01)
    assert res.scene.description == (f"Predicted model: mean pLDDT {conf['pocket_plddt']:.0f} of the residues around "
                                     f"the column, mean pLDDT {conf['ligand_plddt']:.0f} of the column.")
    # The crystal structure itself has no pLDDT, and its figure says nothing of one.
    plain = thrombin().render(style="schematic")
    assert "Confidence:" not in words(plain) and not [g for g in groups(plain, "unit") if "confidence" in g.data]


def test_the_same_bytes_and_the_same_graph_twice():
    one = thrombin().render(style="schematic")
    st = structure_of("thrombin_hirudin")
    two = Interface(st, ["A", "B"], list(reversed(interface_contacts(st, ["A"], ["B"])))).render(style="schematic")
    assert one.svg == two.svg and one.graph.to_json() == two.graph.to_json()


def test_what_one_figure_narrows_to_does_not_decide_what_the_next_is_given():
    """A figure that cannot be drawn legibly narrows to a smaller site and says so. That is of the
    figure: the next render starts from the site the interface was built with, as an object that was
    never narrowed does. ``site`` and ``diagram`` are those of the figure that was drawn last."""
    from conftest import trimer_contacts, trimer_structure

    def new() -> Interface:
        return Interface(trimer_structure(), ["A", "B"], trimer_contacts()[:20], model_confidence=None)

    untouched = new()
    plain = untouched.render(style="schematic")
    assert plain.status == "ok" and len(untouched.site.residues) == 15 and "narrowed" not in " ".join(plain.warnings)
    face = new()
    narrow = face.render(style="schematic", figure_width_mm=60)          # too narrow a page for fifteen boxes
    assert len([w for w in narrow.warnings if "narrowed to 8 residues of chain A" in w]) == 1
    assert len(face.site.residues) == 8 and narrow.data["interface"]["site"]["residues"] == 8
    after = face.render(style="schematic")
    assert after.svg == plain.svg and after.warnings == plain.warnings
    assert face.site == untouched.site and (face.layout, face.layout_rule) == (untouched.layout, untouched.layout_rule)
    # The figure that has to narrow does so every time, and says so every time.
    again = face.render(style="schematic", figure_width_mm=60)
    assert again.svg == narrow.svg and again.warnings == narrow.warnings
    # A graph is drawn at the site it was made for, whatever was drawn last: here the narrow figure.
    assert len(face.site.residues) == 8
    assert face.render(style="schematic", graph=plain.graph).svg == plain.svg and len(face.site.residues) == 15
    # ... and so is the graph of a figure that narrowed, on a page too short for fifteen boxes: by the
    # object that drew it, after another figure, and by one that never drew it.
    from bindview.graph import Graph

    short = face.render(style="schematic", figure_height_mm=135)
    assert short.status == "ok" and len(face.site.residues) == 9 and "narrowed to 9 residues" in " ".join(short.warnings)
    assert face.render(style="schematic").svg == plain.svg
    for one in (face, new()):
        back = one.render(style="schematic", figure_height_mm=135, graph=Graph.from_json(short.graph.to_json()))
        assert back.svg == short.svg and len(one.site.residues) == 9


def test_command_line(tmp_path, capsys):
    from bindview.cli import main as cli

    st = structure_of("thrombin_hirudin")
    tsv = tmp_path / "contacts.tsv"
    tsv.write_text(dump_interface_contacts(interface_contacts(st, ["A"], ["B"])))
    n = thrombin().site.of_residues
    out, graph = tmp_path / "out.svg", tmp_path / "graph.json"
    code = cli(["interface", str(model_of("thrombin_hirudin")), "--between", "A", "B", "--contacts", str(tsv),
                "-o", str(out), "--style", "schematic", "--graph", str(graph), "--json"])
    assert code == 0 and "An interface has no centre." in out.read_text()
    printed = capsys.readouterr()
    assert '"spine": "chain B"' in printed.out and f"{n} of {n} residues at the interface" in printed.err
    assert '"version": "1.2"' in graph.read_text()
    assert cli(["interface", str(model_of("thrombin_hirudin")), "--between", "A", "B", "--contacts", str(tsv),
                "-o", str(out), "-q", "--style", "schematic", "--site", "A:14", "--max-residues", "12",
                "--order", "sequence"]) == 0


def test_a_peptide_against_a_receptor_of_two_chains():
    """The peptide of HLA-A2 against the two chains of the receptor taken as one participant, named by
    the caller: the peptide is the spine, its residues in the order of the sequence."""
    st = structure_of("mhc_peptide")
    contacts = interface_contacts(st, ["P"], ["A", "B"])
    face = Interface(st, ["P", "receptor=A+B"], contacts)
    touched = sorted({c.residue_a for c in contacts}, key=lambda k: (k.number, k.icode))
    assert face.spine.chains == ("P",) and face.site.whole and list(face.site.residues) == touched
    res = face.render(style="schematic")
    assert res.status == "ok" and numbers(res) == [name(st, k, " ") for k in touched]
    assert {n.attrs["chain"] for n in res.graph.nodes if n.kind == "residue"} == {c.chain_b for c in contacts}
    assert "that touch receptor stand in a column, as boxes, in the order of the sequence" in words(res)


# ---------------------------------------------------------------------------
# PLIP as the source of contacts (only where it is installed)
# ---------------------------------------------------------------------------

@needs_plip
def test_plip_profiles_an_interface_through_its_peptide_mode():
    from bindview.detect import detect_interface

    st = structure_of("thrombin_hirudin")
    contacts = detect_interface(st, ["A"], ["B"])
    assert contacts and all(c.chain_a == "A" and c.chain_b == "B" for c in contacts)
    back = detect_interface(st, ["B"], ["A"])
    assert sorted(c.flipped().sort_key for c in back) == sorted(c.sort_key for c in contacts)
    face = Interface(st, ["A", "B"])                              # no table: detected
    assert face.source == "detected" and face.contacts == sorted(contacts, key=lambda c: c.sort_key)
    assert face.render(style="schematic").status == "ok"
    with pytest.raises(BindviewError):
        detect_interface(st, ["A"], ["A"])


@needs_plip
def test_plip_keeps_a_modified_residue_and_insertion_codes_when_asked_carefully():
    from bindview import Diagram
    from bindview.detect import detect_interface

    st = structure_of("thrombin_hirudin")
    # The sulfotyrosine PLIP would drop from the peptide: its two close hydrogen bonds are there.
    hirudin = detect_interface(st, ["H"], ["B"])
    on_tys = {(c.resname_b, c.number_b, c.atoms_a[0]) for c in hirudin if c.number_a == 63 and c.type.value == "hbond"}
    assert {("TYR", 76, "O3"), ("ILE", 82, "O2")} <= on_tys and {c.resname_a for c in hirudin if c.number_a == 63} == {"TYS"}
    # Residues with insertion codes keep them, on both sides.
    chains = detect_interface(st, ["A"], ["B"])
    assert {c.icode_a for c in chains} >= {"", "L"} and any(c.icode_b for c in chains)
    assert len({c.residue_a for c in chains}) == 18
    # And a chain as ligand, with no table, is drawn from it.
    dia = Diagram.from_structure(st, "H")
    assert dia.source == "detected" and dia.backend.kind == "chain" and not dia.unresolved
    assert dia.render(style="schematic").status == "ok"


def test_an_interface_is_built_by_the_name_the_other_figures_are_built_by():
    """``Interface.from_structure`` is the class under the name ``Diagram.from_structure`` has: the same figure."""
    from conftest import trimer_contacts, trimer_structure

    from bindview import Interface as Figure

    by_class = Figure(trimer_structure(), ["A", "B"], trimer_contacts("A", "B"))
    by_name = Figure.from_structure(trimer_structure(), ["A", "B"], contacts=trimer_contacts("A", "B"))
    assert by_name.render(style="schematic").svg == by_class.render(style="schematic").svg


def _without(structure, chain: str, first: int, last: int):
    """The same structure with the residues from ``first`` to ``last`` of a chain taken out of the file."""
    from bindview.structure import Structure

    kept = [a for a in structure.atoms if not (a.chain == chain and first <= a.resnum <= last)]
    return Structure(atoms=[replace(a, index=i) for i, a in enumerate(kept)], source="with a loop not modelled",
                     plddt_scale=structure.plddt_scale, qa_metrics=structure.qa_metrics)


def test_residues_that_the_file_leaves_out_lie_between_two_that_it_has():
    """A loop of the spine that is not at the interface and is not modelled: the interface has the stretches it has
    with the loop, and the two ends of the loop are no neighbours. They are counted by their places among the
    residues of the file, where the second end follows the first."""
    from conftest import trimer_structure

    model = trimer_structure()
    whole = Interface(model, ["A", "B"], interface_contacts(model, ["A"], ["B"]), layout="column")
    shown = [k.number for k in whole.site.residues]
    ends = [(shown[last], shown[first]) for (_, last), (first, _) in zip(whole._runs, whole._runs[1:])]
    lo, hi = max(ends, key=lambda e: e[1] - e[0])                       # the longest loop between two stretches
    assert hi - lo > 6
    gapped = _without(model, whole.spine.chains[0], lo + 1, hi - 1)
    face = Interface(gapped, ["A", "B"], interface_contacts(gapped, ["A"], ["B"]), layout="column")
    assert [k.number for k in face.site.residues] == shown
    assert len(face._runs) == len(whole._runs) > 1
    a, b = (next(k for k in face._order if k.number == n) for n in (lo, hi))
    assert face._index[b] - face._index[a] == hi - lo
    told = [figure.render(style="schematic").graph.meta["interface"]["stretches"] for figure in (face, whole)]
    assert told == [len(whole._runs)] * 2


def test_numbers_that_jump_between_two_bonded_residues_leave_nothing_out():
    """Thrombin is numbered after chymotrypsin: its numbers jump where it has a residue less, and stand still
    (60A to 60I) where it has more. What counts is the peptide bond: only residues that are not modelled are a gap."""
    from bindview.interface import _joined, _places

    st = structure_of("thrombin_hirudin")
    keys = [k for k in st.polymer_residues("B")]
    places = _places(st, keys)
    jumps = [(a, b) for a, b in zip(keys, keys[1:]) if abs(b.number - a.number) > 1]
    bonded = [(a, b) for a, b in jumps if _joined(st, a, b)]
    loops = [(a, b) for a, b in jumps if not _joined(st, a, b)]
    assert bonded and all(places[b] - places[a] == 1 for a, b in bonded)
    assert all(places[b] - places[a] == b.number - a.number for a, b in loops)
    inserted = [(a, b) for a, b in zip(keys, keys[1:]) if a.number == b.number]
    assert inserted and all(places[b] - places[a] == 1 for a, b in inserted)
