"""The predicted aligned error of a contact, drawn apart from pLDDT.

Three models of the AlphaFold database in ``testdata/interface_pae``, each one chain cut at a domain
boundary into two participants, with the database's own PAE file: one whose two domains the model places
confidently against each other (FAH), one where it does not though the residues are confident (APOBEC3D),
and one where part of the interface is pinned down and part is not (TECTB). This is PAE between two parts
of one predicted chain, the same matrix and the same quantity as between two chains, and not a multimer
prediction.
"""

from __future__ import annotations

import gzip
import json
import re

import numpy as np
import pytest

from conftest import PAE_CASES, TESTDATA, pae_case
from bindview import Diagram, load_interactions
from bindview.errors import BindviewError, InteractionError, RenderError
from bindview.interactions import Interaction
from bindview.pae import PAE_BANDS, PAE_SHOWN, PAE_STRONG, Pae, pae_band, pae_shown, pae_tag
from bindview.scene import Line, Text, text_box
from bindview.structure import ResidueKey, read_structure
from scenetools import groups, key, overlap, tags, texts_with_owner, walk, words

ROOT = TESTDATA / "interface_pae"


def pae_file(slug: str):
    return ROOT / slug / "predicted_aligned_error.json.gz"


def raw_matrix(slug: str) -> np.ndarray:
    """The database's matrix as the file holds it: row ``i`` is the residue numbered ``i + 1``, on which the
    model is aligned."""
    with gzip.open(pae_file(slug), "rt", encoding="utf-8") as handle:
        return np.array(json.load(handle)[0]["predicted_aligned_error"], dtype=float)


def by_pair(face) -> dict:
    """The PAE of each pair of residues in contact: that of its contacts, which are of one pair of residues."""
    out: dict = {}
    for c in face.contacts:
        out.setdefault((c.residue_a, c.residue_b), set()).add(c.pae)
    assert all(len(v) == 1 for v in out.values())
    return {k: v.pop() for k, v in out.items()}


def span(values) -> str:
    """A range of PAE as a figure writes it."""
    lo, hi = min(values), max(values)
    return f"{lo:.1f} Å" if lo == hi else f"{lo:.1f}–{hi:.1f} Å"


def described(res, pae: str, said: str = "") -> str:
    """What the description of an interface figure of a model with pLDDT says: the two means and the PAE."""
    conf = res.confidence
    return (f"Predicted model: mean pLDDT {conf['pocket_plddt']:.0f} of the residues around the column, mean pLDDT "
            f"{conf['ligand_plddt']:.0f} of the column, contact PAE {pae}." + (f" {said}" if said else ""))


# ---------------------------------------------------------------------------
# The number
# ---------------------------------------------------------------------------

def test_the_matrix_is_read_by_residue_and_a_pair_takes_the_smaller_direction():
    st = read_structure(ROOT / "afdb_tectb" / "model.cif.gz")
    pae = Pae.read(pae_file("afdb_tectb"), st)
    raw = raw_matrix("afdb_tectb")
    assert pae.matrix.shape == raw.shape == (len(st.residues()),) * 2
    a, b = ResidueKey("A", 29), ResidueKey("A", 145)
    there, back = raw[28, 144], raw[144, 28]
    assert there != back and pae.between(a, b) == pae.between(b, a) == min(there, back)
    assert pae.between(a, ResidueKey("A", 9999)) is None
    # The chain cut into participants keeps its numbers and not its name: the file is read for it by number.
    cut = pae_case("afdb_tectb").structure
    assert {k.chain for k in cut.residues()} >= {"X", "Y"}
    assert Pae.read(pae_file("afdb_tectb"), cut).between(ResidueKey("X", 29), ResidueKey("Y", 145)) == min(there, back)
    with pytest.raises(BindviewError, match="token_res_ids"):
        Pae.from_mapping({"pae": [[0.0]]})
    with pytest.raises(BindviewError, match="do not fit"):
        Pae.from_mapping({"pae": [[0.0, 1.0], [1.0, 0.0]], "token_res_ids": [1, 2, 3]})
    two = Pae.from_mapping({"pae": [[0, 4, 9], [5, 0, 9], [9, 9, 0]], "token_res_ids": [1, 1, 2],
                            "token_chain_ids": ["A", "B", "B"]})
    assert two.between(ResidueKey("A", 1), ResidueKey("B", 1)) == 4.0          # by chain where there are several
    assert two.between(ResidueKey("X", 1), ResidueKey("B", 2)) is None
    # A hole in the matrix is no value: the other direction is taken, and a pair of two holes has none.
    holes = Pae.from_mapping({"pae": [[0, None, None], [6, 0, None], [None, None, 0]], "token_res_ids": [1, 2, 3]})
    assert holes.between(ResidueKey("A", 1), ResidueKey("A", 2)) == 6.0
    assert holes.between(ResidueKey("A", 1), ResidueKey("A", 3)) is None


def test_when_it_is_written_and_how():
    assert (PAE_SHOWN, PAE_STRONG) == (5, 10)
    assert [pae_shown(v) for v in (0.8, 4.49, 4.5, 9.49, 9.5, 21.3)] == [1, 4, 5, 9, 10, 21]
    assert pae_tag(None) is None and pae_tag(0.8) is None and pae_tag(4.49) is None
    assert pae_tag(4.5) == ("PAE 5", False) and pae_tag(9.49) == ("PAE 9", False)
    assert pae_tag(9.5) == ("PAE 10", True) and pae_tag(21.3) == ("PAE 21", True)
    with pytest.raises(InteractionError, match="cannot be negative"):
        Interaction(type="hbond", residue_number=1, residue_name="ALA", pae=-0.1)


@pytest.mark.parametrize("slug", PAE_CASES)
def test_every_contact_has_the_value_of_the_database_s_matrix(slug):
    """The smaller of the two directions of the file's matrix, for the two residues of the contact."""
    face = pae_case(slug)
    raw = raw_matrix(slug)
    assert face.contacts and all(c.pae is not None for c in face.contacts)
    assert all(c.pae == min(raw[c.number_a - 1, c.number_b - 1], raw[c.number_b - 1, c.number_a - 1])
               for c in face.contacts)


# ---------------------------------------------------------------------------
# The three figures
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("style", ("schematic", "detailed"))
def test_a_confident_interface_is_quiet_and_says_that_it_is(style):
    """FAH: every contact below 5 Å. No tag, and the silence is a statement, not an absence."""
    face = pae_case("afdb_fah")
    values = [c.pae for c in face.contacts]
    assert all(pae_tag(v) is None for v in values)
    res = face.render(style=style)
    assert res.status == "ok" and not tags(res)
    text = words(res)
    assert f"contact PAE {span(values)}" in text
    assert "No contact here reaches 5 Å, so none carries a PAE on its line." in text
    assert "PAE (predicted aligned error) is the model's expected error" in text
    assert "Predicted aligned error (PAE) of a contact" not in text            # nothing to explain in the legend
    assert not [w for w in res.warnings if "PAE" in w]
    meta = res.graph.meta["interface"]["pae"]
    assert (meta["min"], meta["max"], meta["pairs_written"], meta["with_a_value"]) == (min(values), max(values), 0,
                                                                                         len(values))
    assert res.scene.description == described(res, span(values))


@pytest.mark.parametrize("style", ("schematic", "detailed"))
def test_residues_that_look_fine_and_a_placement_the_model_does_not_know(style):
    """APOBEC3D: confident residues on both sides on average, and every contact at 10 Å or more. The trap
    the channel is for."""
    face = pae_case("afdb_apobec3d")
    pairs = by_pair(face)
    assert min(pairs.values()) >= PAE_STRONG
    res = face.render(style=style)
    assert res.status == "ok"
    written = tags(res)
    assert written and all(t.bold for t in written)
    assert sorted((t.content, t.bold) for t in written) == sorted(pae_tag(c.attrs["pae"]) for c in res.graph.contacts)
    text = words(res)
    conf = res.confidence
    assert f"contact PAE {span(pairs.values())}" in text and f"residues around pLDDT {conf['pocket_plddt']:.0f}" in text
    said = ("Every contact in this figure has a PAE of 10 Å or more: the model does not place the two sides "
            "against each other, whatever the pLDDT of their residues.")
    assert text.count(said) == 1 and said in res.warnings
    # Said where a poorly buried ligand is said, and in that colour: under the line of numbers, above the figure.
    told = groups(res, "pae-warning")
    assert len(told) == 1 and all(isinstance(t, Text) and t.fill == "warning" for t in told[0].items)
    assert " ".join(t.content for t in told[0].items) == said
    assert max(t.y for t in told[0].items) < min(f.box()[1] for f in walk(res.scene)
                                                 if f.in_class("diagram") is not None and isinstance(f.item, Text))
    assert f"All {len(pairs)} residue pairs in contact have a PAE of 5 Å or more, written on their lines." in text
    # What a screen reader gets says it too.
    assert res.scene.description == described(res, span(pairs.values()), said)
    # ... and both sides are confident on average, which is exactly why the tag is needed.
    assert conf["pocket_plddt"] >= 70 and conf["ligand_plddt"] >= 70


def test_one_interface_part_pinned_down_and_part_not():
    """TECTB: some pairs of residues placed against each other, some not, some in between. A caption
    would be wrong for both halves."""
    face = pae_case("afdb_tectb")
    pairs = by_pair(face)
    quiet = {k for k, v in pairs.items() if pae_tag(v) is None}
    written = {k for k, v in pairs.items() if pae_tag(v) is not None}
    bold = {k for k in written if pae_tag(pairs[k])[1]}
    assert quiet and bold and written - bold
    lo, hi = min(pairs.values()), max(pairs.values())
    st = face.structure
    for style in ("schematic", "detailed"):
        res = face.render(style=style)
        assert res.status == "ok"
        lines = res.graph.contacts
        # A tag on every line of a pair that is written, bold on those of a pair at 10 Å or more.
        on = [pae_tag(c.attrs["pae"]) for c in lines]
        assert sorted((t.content, t.bold) for t in tags(res)) == sorted(t for t in on if t is not None)
        calm = [c for c in lines if pae_tag(c.attrs["pae"]) is None]
        assert {(res.graph.node(c.ligand).label, res.graph.node(c.residue).label) for c in calm} == {
            (f"{st.residue_name(a).title()}{a.number}", f"{st.residue_name(b).title()}{b.number}") for a, b in quiet}
        text = words(res)
        assert f"contact PAE {lo:.1f}–{hi:.1f} Å" in text and "Every contact in this figure" not in text
        assert not groups(res, "pae-warning")
        # Counted as a PAE is defined, by pair of residues.
        assert (f"{len(written)} of the {len(pairs)} residue pairs in contact have a PAE of 5 Å or more, written on "
                f"their lines.") in text
        assert "Predicted aligned error (PAE) of a contact in Å: written from 5 Å, bold from 10 Å. Not pLDDT" in text
        assert not [w for w in res.warnings if "Every contact" in w]
        # Both channels are live and say different things: a pair the model places is drawn at a lower
        # pLDDT than a pair it does not place.
        assert min(c.confidence for c in calm) < max(c.confidence for c in lines if c.attrs["pae"] >= PAE_STRONG)
        meta = res.graph.meta["interface"]["pae"]
        assert (meta["min"], meta["max"], meta["written_from"], meta["bold_from"]) == (lo, hi, 5, 10)
        n = len(face.contacts)
        assert (meta["with_a_value"], meta["of_contacts"], meta["pairs"], meta["of_pairs"]) == (n, n, len(pairs),
                                                                                                 len(pairs))
        assert meta["direction"] == "the smaller of the two directions of the matrix"
        # The line under the title carries the two ends of the range as numbers a page can read.
        (line,) = groups(res, "confidence")
        assert (line.data["contact-pae-min"], line.data["contact-pae-max"]) == (f"{lo:.1f}", f"{hi:.1f}")
        assert meta["pairs_written"] == len(written)
        assert res.graph.to_dict()["version"] == "1.2"


@pytest.mark.parametrize("slug", PAE_CASES)
def test_pae_changes_nothing_about_a_stroke(slug):
    """Weight, opacity and dash are pLDDT's. The same interface drawn without its PAE has the same
    strokes, contact for contact; only the tags and what they need of room are different."""
    def strokes(res):
        out = {}
        for g in groups(res, "interaction"):
            paints = sorted({(item.paint.stroke, round(item.paint.width, 4), item.paint.dash, round(item.paint.opacity, 4))
                             for item in g.items if isinstance(item, Line)})
            out[(g.data["residue"], g.data.get("ligand-residue"), g.data["type"])] = paints
        return out

    with_pae = pae_case(slug).render(style="detailed")
    without = pae_case(slug, with_pae=False).render(style="detailed")
    assert strokes(with_pae) == strokes(without) and strokes(with_pae)
    assert all(strokes(with_pae).values())                  # every contact still has a line under its tag
    assert not tags(without) and "PAE" not in words(without)
    assert "pae" not in without.graph.meta["interface"] and without.graph.to_dict()["version"] == "1.2"


@pytest.mark.parametrize("slug", PAE_CASES)
def test_a_tag_is_text_of_its_own_and_runs_into_nothing(slug):
    for style in ("schematic", "detailed"):
        res = pae_case(slug).render(style=style)
        texts = texts_with_owner(res)
        clashes = [(a[2].item.content, b[2].item.content) for i, a in enumerate(texts) for b in texts[i + 1:]
                   if a[1] != b[1] and overlap(a[0], b[0])]
        assert not clashes
        assert all(t.fill == "text" and t.opacity == 1.0 for t in tags(res))       # never faded with its line


def test_tooltips_name_the_side_and_the_error():
    face = pae_case("afdb_tectb")
    st = face.structure
    res = face.render(style="detailed")
    lines = groups(res, "interaction")
    titles = [g.title for g in lines] + [g.title for g in groups(res, "unit")]
    assert titles and not [t for t in titles if "of the ligand" in t]
    assert all(f"{st.residue_name(k).title()} {k.number} of the N-terminal domain (numbered as in the structure file)"
               in titles for k in face.site.residues)
    assert lines and all("of the N-terminal domain (" in g.title and g.title.endswith(f", PAE {g.data['pae']} Å")
                         for g in lines)
    assert {g.data["pae"] for g in lines} == {f"{c.attrs['pae']:.1f}" for c in res.graph.contacts}
    # A chain drawn as a ligand is a ligand's figure: its tooltips say so, and it has no PAE to speak of.
    from conftest import chain_diagram

    peptide = chain_diagram("thrombin_hirudin").render(style="schematic")
    assert any("of the ligand" in g.title for g in groups(peptide, "unit")) and "PAE" not in words(peptide)


def test_command_line_reads_the_database_s_file(tmp_path, capsys):
    from bindview.cli import main as cli
    from bindview.interface import dump_interface_contacts

    face = pae_case("afdb_tectb", with_pae=False)
    pairs = by_pair(pae_case("afdb_tectb"))
    st = face.structure
    lines = ["data_x", "loop_"] + ["_atom_site." + c for c in (
        "group_PDB", "type_symbol", "label_atom_id", "label_comp_id", "label_asym_id", "label_seq_id",
        "Cartn_x", "Cartn_y", "Cartn_z", "pdbx_PDB_model_num")]
    for a in st.atoms:
        if a.element not in ("H", "D"):
            lines.append(f"ATOM {a.element} {a.name} {a.resname} {a.chain} {a.resnum} {a.x:.3f} {a.y:.3f} {a.z:.3f} 1")
    cif, tsv, out = tmp_path / "cut.cif", tmp_path / "contacts.tsv", tmp_path / "out.svg"
    cif.write_text("\n".join(lines) + "\n")
    tsv.write_text(dump_interface_contacts(face.contacts))
    code = cli(["interface", str(cif), "--between", "X", "Y", "--contacts", str(tsv), "-o", str(out), "-q",
                "--style", "schematic", "--pae", str(pae_file("afdb_tectb")), "--json"])
    assert code == 0 and f">PAE {pae_shown(max(pairs.values()))}<" in out.read_text()
    written = sum(1 for v in pairs.values() if pae_tag(v) is not None)
    assert f'"pairs_written": {written}' in capsys.readouterr().out
    # A model without pLDDT still has its PAE in the description of the figure.
    assert f"<desc>Predicted model: contact PAE {span(pairs.values())}.</desc>" in out.read_text()


def test_a_table_in_place_of_the_figure_keeps_the_warning():
    """Too narrow a page: the per-residue table stands in, and no line is left to carry a PAE. What
    PAE is, its range, and the warning that every contact is unplaced must not go with the lines."""
    said = ("Every contact in this figure has a PAE of 10 Å or more: the model does not place the two sides "
            "against each other, whatever the pLDDT of their residues.")
    face = pae_case("afdb_apobec3d", fresh=True)
    res = face.render(style="schematic", figure_width_mm=60)
    assert (res.status, res.figure) == ("unreadable", "summary") and not tags(res)
    text = words(res)
    assert f"contact PAE {span([c.pae for c in face.contacts])}" in text
    assert text.count(said) == 1 and said in res.warnings
    assert len(groups(res, "pae-warning")) == 1
    assert "PAE (predicted aligned error) is the model's expected error" in text
    assert "this table does not give it contact by contact" in text
    assert "line weight, dashes" not in text                                    # there are no lines to speak of
    assert said in res.scene.description
    calm = pae_case("afdb_fah", fresh=True)
    quiet = calm.render(style="schematic", figure_width_mm=60)
    assert quiet.figure == "summary" and f"contact PAE {span([c.pae for c in calm.contacts])}" in words(quiet)
    assert "Every contact" not in words(quiet)


def test_between_five_and_ten_the_tag_is_there_and_not_bold():
    """None of the contacts of FAH is in this band, so one pair of residues is given a value in it."""
    from dataclasses import replace

    from bindview.interface import Interface

    face = pae_case("afdb_fah")
    first = face.contacts[0]
    pair = (first.residue_a, first.residue_b)
    contacts = [replace(c, pae=7.4) if (c.residue_a, c.residue_b) == pair else c for c in face.contacts]
    res = Interface(face.structure, ["the N-terminal domain=X", "the catalytic domain=Y"], contacts).render(style="detailed")
    written = tags(res)
    assert [(t.content, t.bold, t.fill, t.opacity) for t in written] == [("PAE 7", False, "text", 1.0)]
    text = words(res)
    rest = [c.pae for c in contacts if (c.residue_a, c.residue_b) != pair]
    assert f"contact PAE {min(rest):.1f}–7.4 Å" in text and "Every contact in this figure" not in text
    assert (f"1 of the {len(by_pair(face))} residue pairs in contact has a PAE of 5 Å or more, written on its line."
            in text)


@pytest.mark.parametrize("slug", PAE_CASES)
def test_a_figure_with_tags_is_redrawn_from_its_graph_as_it_was(slug):
    """A tag has a place along its line, and the graph keeps it also where the line has no distance
    to print (every line of a schematic figure)."""
    from bindview.graph import Graph

    face = pae_case(slug)
    for style in ("schematic", "detailed"):
        res = face.render(style=style)
        again = face.diagram.render(style=style, graph=Graph.from_json(res.graph.to_json()))
        assert again.svg == res.svg
        for c in res.graph.contacts:
            assert (c.label_t is not None) == bool(c.label or pae_tag(c.attrs["pae"]))


def test_a_ligand_diagram_carries_a_pae_and_does_not_draw_it():
    """The tag is explained in the figure of an interface and nowhere else, so it is written there
    and nowhere else. A record of a ligand diagram keeps its value in the graph and in the tooltip."""
    from dataclasses import replace

    from conftest import diagram_of, meta_of, model_of
    from bindview import Diagram

    case = "kit_imatinib"
    plain = diagram_of(case)
    records = [replace(r.record, pae=14.2) for r in plain.interactions]
    dia = Diagram.from_structure(model_of(case), ligand=meta_of(case)["ligand"]["residue_name_in_cif"],
                                 smiles=meta_of(case)["ligand"]["smiles"], interactions=records,
                                 name=meta_of(case)["ligand"]["name"])
    for style in ("schematic", "detailed"):
        res, was = dia.render(style=style), plain.render(style=style)
        assert not tags(res) and "PAE" not in words(res)
        assert all(c.attrs["pae"] == 14.2 for c in res.graph.contacts)
        assert any(", PAE 14.2 Å" in g.title for g in groups(res, "interaction"))
        # ... and its lines and labels are where and what they were without it.
        strip = lambda svg: __import__("re").sub(r'(, PAE 14\.2 Å| data-pae="14\.2")', "", svg)   # noqa: E731
        assert strip(res.svg) == was.svg


def test_a_matrix_that_knows_none_of_the_contacts_is_refused():
    """Not drawn without a word: a figure with no PAE in it reads as confident, not as unmeasured."""
    from bindview.interface import Interface

    face = pae_case("afdb_tectb", with_pae=False)
    elsewhere = Pae.from_mapping({"pae": np.ones((3, 3)).tolist(), "token_res_ids": [901, 902, 903],
                                  "token_chain_ids": ["Q", "Q", "R"]}, source="other.json")
    with pytest.raises(BindviewError, match=r"other.json has no value for any contact of this interface.*chains 'Q', 'R', "
                                          r"residue numbers 901 to 903"):
        Interface(face.structure, ["X", "Y"], face.contacts, pae=elsewhere)


def test_a_pair_without_a_value_is_counted_as_such():
    """Contacts that bring their own PAE, and one that brings none: said, not passed over."""
    from dataclasses import replace

    from bindview.interface import Interface

    face = pae_case("afdb_tectb")
    pairs = by_pair(face)
    written = sum(1 for v in pairs.values() if pae_tag(v) is not None)
    without = next(c for c in face.contacts if c.pae is not None and c.pae > PAE_STRONG)
    contacts = [replace(c, pae=None) if (c.residue_a, c.residue_b) == (without.residue_a, without.residue_b) else c
                for c in face.contacts]
    res = Interface(face.structure, ["the N-terminal domain=X", "the ZP domain=Y"], contacts).render(style="schematic")
    text = words(res)
    assert (f"{written - 1} of the {len(pairs) - 1} residue pairs in contact have a PAE of 5 Å or more, written on "
            f"their lines. For 1 more pair there is no PAE.") in text
    meta = res.graph.meta["interface"]["pae"]
    assert (meta["pairs"], meta["of_pairs"], meta["pairs_written"]) == (len(pairs) - 1, len(pairs), written - 1)
    assert len(tags(res)) == sum(1 for c in res.graph.contacts if pae_tag(c.attrs.get("pae")) is not None)
    assert "direction" not in meta                    # the values came with the contacts: no matrix was read here
    # "Every contact" is said only where every contact has a value.
    trap = pae_case("afdb_apobec3d")
    gone = trap.contacts[0]
    some = [replace(c, pae=None) if (c.residue_a, c.residue_b) == (gone.residue_a, gone.residue_b) else c
            for c in trap.contacts]
    res = Interface(trap.structure, ["the N-terminal domain=X", "the C-terminal domain=Y"], some).render(
        style="schematic")
    said = ("Every contact in this figure that has a PAE has one of 10 Å or more: the model does not place the two "
            "sides against each other, whatever the pLDDT of their residues.")
    assert said in res.warnings and said in words(res) and "Every contact in this figure has a PAE" not in words(res)


@pytest.mark.parametrize("slug", PAE_CASES)
def test_an_interface_figure_never_speaks_of_a_ligand_or_a_pocket(slug):
    """Not in a label, a footnote, a tooltip, the description or a warning: it has neither. (The
    ``data-ligand-*`` attributes are names of the schema it shares with every figure, not words.)"""
    import re

    for res in (pae_case(slug).render(style="schematic"), pae_case(slug).render(style="detailed"),
                pae_case(slug, fresh=True).render(style="schematic", figure_width_mm=60)):
        said = [t.strip() for t in re.findall(r">([^<>]+)<", res.svg)] + list(res.warnings)
        assert not [t for t in said if re.search(r"ligand|pocket", t, re.I)]
        assert len(said) > 40


def test_the_summary_of_some_values():
    pae = Pae.from_mapping({"pae": np.zeros((2, 2)).tolist(), "token_res_ids": [1, 2]})
    assert pae.summary([3.0, 1.0, 20.0]) == {"min": 1.0, "median": 3.0, "max": 20.0}


# ---------------------------------------------------------------------------
# The legend's key of a tag
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("style", ("schematic", "detailed"))
def test_the_key_of_a_tag_is_written_as_the_tags_of_the_figure_are(style):
    """A key shows a thing the way the figure draws it: a bold sample where every tag is bold, a plain one
    where every tag is plain, one of each where there are both, and none where no tag is written. A bold
    sample in a figure of plain tags would read as a value that no contact has."""
    from dataclasses import replace

    from conftest import trimer, trimer_structure
    from bindview.interface import Interface

    def said(res) -> list[tuple[str, bool]]:
        return [(t.content, t.bold) for t in key(res)]

    # Every tag bold: the key is the bold one.
    res = pae_case("afdb_apobec3d").render(style=style)
    assert tags(res) and all(t.bold for t in tags(res)) and said(res) == [("PAE 11", True)]
    # Every tag plain (the trimer over the two draws that are in the repository): a plain one.
    face = trimer("A", "B")
    res = face.render(style=style)
    assert tags(res) and not any(t.bold for t in tags(res)) and said(res) == [("PAE 7", False)]
    assert not [t for t in tags(res) + key(res) if t.bold]             # nothing bold that says "PAE" is left
    # Both: the same contacts with one of them at 12 Å. One sample of each, the plain one first, side by side.
    contacts = [replace(c, pae=12.0 if (c.number_a, c.number_b) == (53, 2) else c.pae, pae_draws=())
                for c in face.contacts]
    res = Interface(trimer_structure(), ["A", "B"], contacts).render(style=style)
    assert sorted({t.bold for t in tags(res)}) == [False, True] and said(res) == [("PAE 7", False), ("PAE 11", True)]
    plain, bold = key(res)
    assert plain.y == bold.y and plain.x < bold.x and not overlap(text_box(plain), text_box(bold), slack=0.0)
    # No tag, no line about tags in the legend, no key.
    assert key(pae_case("afdb_fah").render(style=style)) == []


def test_the_samples_of_the_key_are_tags_and_not_values():
    from bindview.pae import pae_key

    assert pae_key([True, True]) == (pae_tag(11.0),) == (("PAE 11", True),)
    assert pae_key([False]) == (pae_tag(7.0),) == (("PAE 7", False),)
    assert pae_key([True, False, False]) == (("PAE 7", False), ("PAE 11", True))
    assert pae_key(iter([False, True])) == (("PAE 7", False), ("PAE 11", True))
    # A legend line about tags in a figure that wrote none has the bold sample.
    assert pae_key([]) == (("PAE 11", True),)


# ---------------------------------------------------------------------------
# The glow: pae_colors
# ---------------------------------------------------------------------------

def glow_of(res, group) -> "str | None":
    """The colour of the line that a contact's group starts with, where that is a glow."""
    first = next((f.item for f in walk(res.scene) if isinstance(f.item, Line) and group in f.path), None)
    token = first.paint.stroke if first is not None else None
    return token if token and token.startswith("plddt_") else None


def test_the_bands_of_a_pae():
    assert [pae_band(v) for v in (0.3, 4.99, 5.0, 9.9, 10.0, 14.9, 15.0, 31.0)] == [
        "very_high", "very_high", "confident", "confident", "low", "low", "very_low", "very_low"]
    assert [name for name, _ in PAE_BANDS] == ["very_high", "confident", "low", "very_low"]


@pytest.mark.parametrize("style", ("detailed", "schematic"))
def test_a_glow_takes_the_place_of_the_numbers(style):
    face = pae_case("afdb_tectb")
    plain, res = face.render(style=style), face.render(style=style, pae_colors=True)
    assert res.status == "ok" and res.quality.valid
    assert tags(plain) and not tags(res) and not key(res)
    drawn = [g for g in groups(res, "interaction") if g.data.get("pae")]
    assert len(drawn) == len([g for g in groups(plain, "interaction") if g.data.get("pae")])
    for g in drawn:
        assert glow_of(res, g) == f"plddt_{pae_band(float(g.data['pae']))}", (g.data["residue"], g.data["pae"])
    # Part of it placed, part not: as many colours as bands among the values.
    assert len({glow_of(res, g) for g in drawn}) == len({pae_band(c.pae) for c in face.contacts}) > 2
    text = words(res)
    assert "PAE of the contact: < 5 Å 5–10 Å 10–15 Å ≥ 15 Å" in text
    assert "stands on a glow in the colour of its PAE" in text and "written on their lines" not in text
    assert "contact PAE" in text                                                # the range under the title stays
    # Off is what it was, and what was asked for and has nothing to draw says so.
    assert face.render(style=style, pae_colors=False).svg == plain.svg
    bare = pae_case("afdb_tectb", with_pae=False).render(style=style, pae_colors=True)
    assert "pae_colors was asked for, and no contact of this figure has a predicted aligned error" in bare.warnings


def a_matrix(dia, value):
    """An AlphaFold 3 file made up for the model of a ligand diagram: its tokens as the model has them (one for
    a residue, one for each heavy atom of anything else, the ligand among them); ``value`` gives the error of
    residue number and ligand atom, and every other pair is at 2 Å."""
    from bindview.pae import tokens_of

    st, key = dia.structure, dia.ligand_key
    chains, numbers = tokens_of(st)
    heavy = [a.name for a in st.residue_atoms(key) if a.element not in ("H", "D")]
    rows = [i for i, (c, n) in enumerate(zip(chains, numbers)) if (c, n) == (key.chain, key.number)]
    assert len(rows) == len(heavy)
    m = np.full((len(chains), len(chains)), 2.0)
    for i, n in enumerate(numbers):
        if i in rows:
            continue
        for j, name in zip(rows, heavy):
            m[i, j] = value(n, name)
            m[j, i] = value(n, name) + 3.0                         # the other direction, and not the smaller
    return {"pae": m.tolist(), "token_res_ids": list(numbers), "token_chain_ids": list(chains)}


def test_a_ligand_diagram_reads_the_error_of_the_atoms_that_make_a_contact(tmp_path):
    """Imatinib in c-Kit, with a matrix made up for its model. Cys673 makes one contact, a hydrogen bond
    to N3: its column of that atom is at 12 Å and every other atom's at 2 Å, and the contact has 12."""
    from conftest import diagram_of, meta_of, model_of

    case = "kit_imatinib"
    given = diagram_of(case)
    ligand = meta_of(case)["ligand"]
    table = load_interactions(TESTDATA / case / "interactions.tsv")
    assert {(r.type.value, r.ligand_atom) for r in table if r.residue_number == 673} == {("hbond", "N3")}
    path = tmp_path / "confidences.json"
    chosen = {673: 12.0, 790: 7.0, 640: 1.5, 811: 22.0}
    path.write_text(json.dumps(a_matrix(given, lambda n, name: 2.0 if (n == 673 and name != "N3")
                                        else chosen.get(n, 4.0))))
    dia = Diagram.from_structure(model_of(case), ligand["residue_name_in_cif"], ligand["smiles"], table, pae=path)
    by = {r.record.residue_number: r.record.pae for r in dia.interactions}
    assert {n: by[n] for n in chosen} == chosen
    for style in ("detailed", "schematic"):
        res = dia.render(style=style)
        assert res.quality.valid
        seen = {int(re.search(r"(\d+)$", g.data["residue"]).group(1)): glow_of(res, g)
                for g in groups(res, "interaction")}
        assert {n: seen[n] for n in chosen} == {673: "plddt_low", 790: "plddt_confident", 640: "plddt_very_high",
                                                811: "plddt_very_low"}
        assert "PAE of the contact:" in words(res) and not tags(res)
        assert "the residue and the ligand to be placed against each other" in words(res)
        off = dia.render(style=style, pae_colors=False)
        assert off.svg.count("plddt_") == 0 and "PAE of the contact" not in words(off)
    # ... and without a matrix a ligand's diagram has no PAE.
    assert "PAE" not in words(given.render())
    # The file of another model is refused when it is read, and a matrix that was read without a model and
    # has none of the contacts when they are looked up in it.
    from conftest import trimer_file

    elsewhere = Pae.from_mapping({"pae": np.ones((3, 3)).tolist(), "token_res_ids": [901, 902, 903],
                                  "token_chain_ids": ["Q", "Q", "R"]}, source="other.json")
    for other, error, said in ((trimer_file("pae"), BindviewError, "the two are not of the same model"),
                               (elsewhere, RenderError, "has no value for any contact of this ligand")):
        with pytest.raises(error, match=said):
            Diagram.from_structure(model_of(case), ligand["residue_name_in_cif"], interactions=table, pae=other)


def test_the_command_line_takes_the_glow(tmp_path):
    from bindview.cli import main
    from conftest import diagram_of, meta_of, model_of

    case = "kit_imatinib"
    ligand = meta_of(case)["ligand"]
    path, out = tmp_path / "confidences.json", tmp_path / "f.svg"
    path.write_text(json.dumps(a_matrix(diagram_of(case), lambda n, name: 8.0)))
    assert main(["render", str(model_of(case)), "--ligand", ligand["residue_name_in_cif"], "--smiles", ligand["smiles"],
                 "--interactions", str(TESTDATA / case / "interactions.tsv"), "--pae", str(path), "-o", str(out),
                 "-q"]) == 0
    assert "PAE of the contact:" in out.read_text()


# ---------------------------------------------------------------------------
# A matrix is read with the model it belongs to
# ---------------------------------------------------------------------------

def made_model(folder, residues, name: str = "made.pdb"):
    """A model made here, as a PDB file: ``(residue name, chain, number, insertion code)`` for each
    residue, which gets the four atoms of its backbone, 3.8 Å along x from the residue before it."""
    from bindview.structure import read_structure

    lines, serial = [], 1
    for k, (resname, chain, number, icode) in enumerate(residues):
        for j, atom in enumerate(("N", "CA", "C", "O")):
            lines.append("ATOM  %5d  %-3s %3s %s%4d%1s   %8.3f%8.3f%8.3f  1.00 80.00          %2s"
                         % (serial, atom, resname, chain, number, icode or " ", 3.8 * k, 1.4 * j, 0.0, atom[0]))
            serial += 1
    (folder / name).write_text("\n".join(lines) + "\nEND\n")
    return read_structure(folder / name)


def as_json(path, matrix, chains, numbers):
    """A matrix as AlphaFold 3 writes it, with the chain and the residue number of every row."""
    path.write_text(json.dumps({"pae": np.asarray(matrix, dtype=float).tolist(), "token_res_ids": list(numbers),
                                "token_chain_ids": list(chains)}))
    return path


def test_a_json_of_another_model_is_refused_with_both_token_counts(tmp_path):
    """The chains of the trimer by name, and 20 residues each where the model has 157."""
    from conftest import trimer_contacts, trimer_structure
    from bindview.interface import Interface, InterfaceContact
    from bindview.pae import read_pae

    st, n = trimer_structure(), 20
    other = as_json(tmp_path / "other_confidences.json", np.full((3 * n, 3 * n), 25.0),
                    ["A"] * n + ["B"] * n + ["C"] * n, list(range(1, n + 1)) * 3)
    with pytest.raises(BindviewError, match=r"other_confidences\.json has 60 tokens \(chain A: 20, chain B: 20, "
                                          r"chain C: 20\), and .* has 471 \(chain A: 157, chain B: 157, chain C: "
                                          r"157\): the two are not of the same model"):
        Pae.read(other, st)
    with pytest.raises(BindviewError, match="60 tokens.* has 471 "):
        Interface(st, ["A", "B"], trimer_contacts()[:6], pae=other, model_confidence=None)
    # A matrix of one chain is not read by residue number alone for a model of three: residues 94 of
    # chain A and 151 of chain B would get the value of residues 94 and 151 of that one chain.
    one = as_json(tmp_path / "one_confidences.json", np.full((157, 157), 15.8), ["A"] * 157, range(1, 158))
    with pytest.raises(BindviewError, match=r"157 tokens \(chain A: 157\), and .* has 471 "):
        read_pae(one, st)
    contact = InterfaceContact("hbond", "A", 94, "SER", ("O",), "B", 151, "TYR", ("N",), distance=3.0)
    with pytest.raises(BindviewError, match="157 tokens.* has 471 "):
        Interface(st, ["A", "B"], [contact], pae=one, model_confidence=None)


def test_a_json_of_the_model_is_read_whatever_the_order_of_its_chains(tmp_path):
    st = made_model(tmp_path, [("ALA", "A", 1, ""), ("GLY", "A", 2, ""), ("SER", "B", 1, ""), ("TRP", "B", 2, "")])
    m = np.arange(16.0).reshape(4, 4)
    order = [2, 3, 0, 1]                                    # chain B first in the matrix
    path = as_json(tmp_path / "confidences.json", m[np.ix_(order, order)], "BBAA", (1, 2, 1, 2))
    pae = Pae.read(path, st)
    assert pae.between(ResidueKey("A", 1), ResidueKey("B", 2)) == min(m[0, 3], m[3, 0]) == 3.0
    assert pae.between(ResidueKey("A", 2), ResidueKey("B", 1)) == min(m[1, 2], m[2, 1]) == 6.0
    assert pae.resnames == ("SER", "TRP", "ALA", "GLY")       # the names of its rows are those of the model


def test_a_chain_cut_in_two_is_read_by_number_where_its_residues_map_one_to_one(tmp_path):
    """A model of one chain whose parts were given chain names of their own keeps its numbers. A
    matrix of one chain is read for it by number alone, and for no model whose numbers it does not have."""
    cut = made_model(tmp_path, [("ALA", "X", 1, ""), ("GLY", "X", 2, ""), ("SER", "Y", 3, ""), ("TRP", "Y", 4, "")])
    m = np.arange(16.0).reshape(4, 4)
    whole = as_json(tmp_path / "confidences.json", m, "AAAA", (1, 2, 3, 4))
    assert Pae.read(whole, cut).between(ResidueKey("X", 2), ResidueKey("Y", 4)) == min(m[1, 3], m[3, 1]) == 7.0
    alike = made_model(tmp_path, [("ALA", "X", 1, ""), ("GLY", "X", 2, ""), ("SER", "Y", 1, ""), ("TRP", "Y", 2, "")],
                       "alike.pdb")
    with pytest.raises(BindviewError, match=r"confidences\.json has 4 tokens \(chain A: 4\), and .* has 4 \(chain X: "
                                          r"2, chain Y: 2\): the two are not of the same model"):
        Pae.read(whole, alike)
    longer = as_json(tmp_path / "longer.json", np.zeros((5, 5)), "AAAAA", range(1, 6))
    with pytest.raises(BindviewError, match=r"longer\.json has 5 tokens \(chain A: 5\), and .* has 4 "):
        Pae.read(longer, cut)


def test_a_matrix_made_without_a_model_is_read_by_number_only_for_a_model_that_has_its_numbers(tmp_path):
    """A matrix that is handed in as an object was not checked against a model when it was made. For
    the model it is used with, a residue under another chain name is found by its number alone only
    where the matrix has one chain and that model has its residues, number for number."""
    from conftest import trimer_structure
    from bindview.interface import Interface, InterfaceContact
    from bindview.pae import read_pae

    one = Pae.from_mapping({"pae": np.full((157, 157), 15.8).tolist(), "token_res_ids": list(range(1, 158)),
                            "token_chain_ids": ["A"] * 157}, source="one.json")
    contact = InterfaceContact("hbond", "A", 94, "SER", ("O",), "B", 151, "TYR", ("N",), distance=3.0)
    with pytest.raises(BindviewError, match=r"one\.json has no value for any contact of this interface"):
        Interface(trimer_structure(), ["A", "B"], [contact], pae=one, model_confidence=None)
    assert read_pae(one, trimer_structure()).between(ResidueKey("A", 94), ResidueKey("B", 151)) is None
    # The chain cut in two, of which it has every number: read, as from the file.
    cut = made_model(tmp_path, [("ALA", "X", 1, ""), ("GLY", "X", 2, ""), ("SER", "Y", 3, ""), ("TRP", "Y", 4, "")])
    m = np.arange(16.0).reshape(4, 4)
    whole = Pae.from_mapping({"pae": m.tolist(), "token_res_ids": [1, 2, 3, 4], "token_chain_ids": list("AAAA")})
    assert read_pae(whole, cut).between(ResidueKey("X", 2), ResidueKey("Y", 4)) == 7.0
    assert read_pae([whole, Pae(m + 2.0, whole.chains, whole.numbers)], cut).between(
        ResidueKey("X", 2), ResidueKey("Y", 4)) == 8.0


def test_residues_that_differ_by_their_insertion_code_have_rows_of_their_own(tmp_path):
    """An antibody numbered after it was predicted: H:100, H:100A and H:100B are three residues, and
    three rows of a matrix that comes without the names of its rows."""
    from bindview.pae import ipsae

    st = made_model(tmp_path, [("ALA", "H", 100, ""), ("GLY", "H", 100, "A"), ("SER", "H", 100, "B"),
                               ("ALA", "L", 50, "")])
    m = np.array([[0.0, 9.0, 9.0, 20.0], [9.0, 0.0, 9.0, 3.0], [9.0, 9.0, 0.0, 25.0], [20.0, 3.0, 25.0, 0.0]])
    np.savez(tmp_path / "pae_made.npz", pae=m)
    pae = Pae.read(tmp_path / "pae_made.npz", st)
    light = ResidueKey("L", 50)
    assert [pae.between(ResidueKey("H", 100, icode), light) for icode in ("", "A", "B")] == [20.0, 3.0, 25.0]
    assert pae.between(ResidueKey("H", 100, "C"), light) is None
    # ... and each enters an ipSAE by its own row: 100B alone is 25 Å from the light chain's residue.
    score = ipsae(pae, [ResidueKey("H", 100, "B")], [light], cutoff=30.0)
    assert score.residues == (1, 1) and score.forward.value == pytest.approx(1.0 / (1.0 + 25.0 ** 2))


def test_a_pae_that_is_no_number_of_angstrom_is_refused_in_words():
    from bindview.interface import InterfaceContact

    for bad in (float("nan"), -1.0):
        with pytest.raises(BindviewError, match="a PAE is a number of ångström from 0"):
            pae_band(bad)
    for bad in (float("nan"), float("inf")):
        with pytest.raises(BindviewError, match="a PAE is a number of ångström from 0"):
            pae_shown(bad)
        with pytest.raises(BindviewError, match=r"the contact of A:1 with B:2 has a PAE of (nan|inf)"):
            InterfaceContact("hbond", "A", 1, "ALA", ("O",), "B", 2, "GLY", ("N",), pae=bad)
    assert pae_shown(None) is None and pae_band(31.75) == "very_low"


def test_a_table_cell_that_is_no_pae_is_refused_with_its_line(tmp_path):
    """``nan`` is a number to Python and no PAE: a table that holds it is refused when it is read."""
    from bindview.interface import COLUMNS, load_interface_contacts

    row = {"interaction_type": "hbond", "chain_a": "A", "residue_number_a": "1", "residue_name_a": "ALA",
           "atoms_a": "O", "chain_b": "B", "residue_number_b": "2", "residue_name_b": "GLY", "atoms_b": "N",
           "distance": "3.0", "pae": "nan"}
    table = tmp_path / "contacts.tsv"
    table.write_text("\t".join(COLUMNS) + "\n" + "\t".join(row.get(c, "") for c in COLUMNS) + "\n")
    with pytest.raises(BindviewError, match=r"contacts\.tsv, line 2: the contact of A:1 with B:2 has a PAE of nan"):
        load_interface_contacts(table)
    table.write_text(table.read_text().replace("nan", "4.5"))
    assert [c.pae for c in load_interface_contacts(table)] == [4.5]


# ---------------------------------------------------------------------------
# The PAE file of the AlphaFold database
# ---------------------------------------------------------------------------

AFDB_NAME = "AF-MADE-F1-predicted_aligned_error_v4.json"


def afdb_matrix(n: int = 20) -> np.ndarray:
    """A matrix every cell of which says where it stands: the row, counted from 1, before the point and
    the column after it (``3.008``: aligned on the third residue, the error of the eighth)."""
    m = np.array([[i + 1 + (j + 1) / 1000.0 for j in range(n)] for i in range(n)])
    np.fill_diagonal(m, 0.0)
    return m


def afdb_file(folder, matrix, form: str = "list", gz: bool = False, name: str = AFDB_NAME):
    """The matrix as the AlphaFold database writes it: a ``list`` with one object, that ``object`` alone
    (some older files), or the three ``flat`` lists of the first files, residues counted from 1."""
    n = len(matrix)
    if form == "flat":
        held: object = [{"residue1": [i + 1 for i in range(n) for _ in range(n)],
                         "residue2": [j + 1 for _ in range(n) for j in range(n)],
                         "distance": [float(matrix[i][j]) for i in range(n) for j in range(n)],
                         "max_predicted_aligned_error": 31.75}]
    elif form == "object":
        held = {"predicted_aligned_error": np.asarray(matrix).tolist()}
    else:
        held = [{"predicted_aligned_error": np.asarray(matrix).tolist(), "max_predicted_aligned_error": 31.75}]
    path = folder / (name + (".gz" if gz else ""))
    if gz:
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            handle.write(json.dumps(held))
    else:
        path.write_text(json.dumps(held))
    return path


def afdb_model(folder, n: int = 20, first: int = 1, name: str = "AF-MADE-F1-model_v4.pdb"):
    """A made-up model of one chain, A, of ``n`` residues numbered from ``first``."""
    names = ("ALA", "GLY", "SER", "LEU", "LYS")
    return made_model(folder, [(names[k % 5], "A", first + k, "") for k in range(n)], name)


@pytest.mark.parametrize("gz", (False, True))
@pytest.mark.parametrize("form", ("list", "object", "flat"))
def test_the_file_of_the_alphafold_database_is_read_with_the_residues_of_its_model(tmp_path, form, gz):
    from bindview.pae import read_pae

    st = afdb_model(tmp_path, first=5)                  # numbered from 5: a row is named by the model, not counted
    path = afdb_file(tmp_path, afdb_matrix(), form, gz)
    for pae in (Pae.read(path, st), read_pae(path, st)):
        assert pae.source == path.name and pae.matrix.shape == (20, 20)
        assert pae.chains == ("A",) * 20 and pae.numbers == tuple(range(5, 25))
        assert pae.resnames[:5] == ("ALA", "GLY", "SER", "LEU", "LYS") and pae.places.shape == (20, 3)
        # A row is the residue the model is aligned on: residue1 of the flat lists, the outer list of the matrix.
        assert np.array_equal(pae.matrix, afdb_matrix())
        assert (pae.matrix[2, 7], pae.matrix[7, 2]) == (3.008, 8.003)
        # Residues 7 and 12 of the model are its third and its eighth: the smaller of the two directions.
        assert pae.between(ResidueKey("A", 7), ResidueKey("A", 12)) == 3.008
        assert pae.between(ResidueKey("A", 3), ResidueKey("A", 12)) is None      # the model has no residue 3
        assert pae.between(ResidueKey("B", 7), ResidueKey("A", 12)) is None      # and no chain B


@pytest.mark.parametrize("form", ("list", "flat"))
def test_a_row_of_the_database_s_matrix_is_the_residue_the_model_is_aligned_on(tmp_path, form):
    """A matrix that is sure in one direction only: aligned on a residue of the first half, the second
    half is 2 Å off; aligned on one of the second half, the first is 20 Å off. ipSAE has a direction and
    reads rows as aligned residues, so the two directions of the score tell which index is which."""
    from bindview.pae import ipsae

    m = np.where(np.arange(20)[:, None] < np.arange(20)[None, :], 2.0, 20.0)
    np.fill_diagonal(m, 0.0)
    st = afdb_model(tmp_path)
    pae = Pae.read(afdb_file(tmp_path, m, form), st)
    first, second = ([ResidueKey("A", k) for k in span] for span in (range(1, 11), range(11, 21)))
    score = ipsae(pae, first, second)
    assert score.forward.value == pytest.approx(1.0 / (1.0 + 2.0 ** 2)) and score.forward.n0 == 10
    assert score.backward.value == 0.0 and score.same_chain


def test_the_database_s_file_is_refused_for_another_model_with_both_sizes(tmp_path):
    from conftest import trimer_contacts, trimer_structure
    from bindview.interface import Interface

    path = afdb_file(tmp_path, afdb_matrix())
    with pytest.raises(BindviewError, match=r"PAE file of the AlphaFold database, a matrix without the names of its "
                                          r"rows: the model it belongs to has to be given"):
        Pae.read(path)
    longer = afdb_model(tmp_path, n=23, name="longer.pdb")
    with pytest.raises(BindviewError, match=r"_v4\.json is a matrix of 20 x 20, and .*longer\.pdb has 23 residues "
                                          r"\(chain A: 23\): the two are not of the same model"):
        Pae.read(path, longer)
    # Two chains of ten residues are twenty residues, and no model of one chain: each is numbered from 1.
    two = made_model(tmp_path, [("ALA", "A", k, "") for k in range(1, 11)] + [("GLY", "B", k, "") for k in range(1, 11)],
                     "two.pdb")
    with pytest.raises(BindviewError, match=r"a matrix of 20 x 20, for the residues of one chain .*two\.pdb has 2 "
                                          r"polymer chains \(chain A: 10, chain B: 10\)"):
        Pae.read(path, two)
    with pytest.raises(BindviewError, match=r"3 polymer chains \(chain A: 157, chain B: 157, chain C: 157\)"):
        Interface(trimer_structure(), ["A", "B"], trimer_contacts()[:6], pae=path, model_confidence=None)
    # What is written as such a file and holds no matrix is refused by the name of the file.
    for held in ([{"predicted_aligned_error": [[0, 1, 2], [1, 0, 2]]}],
                 [{"predicted_aligned_error": [[0, "far"], [1, 0]]}],
                 [{"residue1": [1, 2], "residue2": [1], "distance": [0.0, 1.0]}],
                 [{"residue1": [0, 1], "residue2": [1, 1], "distance": [0.0, 1.0]}]):
        (tmp_path / "broken.json").write_text(json.dumps(held))
        with pytest.raises(BindviewError, match=r"broken\.json is written as the PAE file of the AlphaFold database"):
            Pae.read(tmp_path / "broken.json", afdb_model(tmp_path))
    (tmp_path / "list.json").write_text("[1, 2]")
    with pytest.raises(BindviewError, match=r"list\.json is no PAE file that is read here.*a JSON list"):
        Pae.read(tmp_path / "list.json", afdb_model(tmp_path))


def one_chain_of_the_trimer():
    """Chain A of the TNF trimer as a model of its own: one chain of 157 residues, as the AlphaFold
    database has its models, and a matrix made up for it."""
    from conftest import trimer_structure
    from bindview.structure import Structure

    return Structure(atoms=[a for a in trimer_structure().atoms if a.chain == "A"], source="one chain of the trimer")


def test_the_database_s_file_is_read_for_two_parts_of_its_chain_as_an_interface(tmp_path):
    """An interface needs two participants, and a model of the database has one chain: its parts are
    given chain names of their own (``Structure.split``) and keep their numbers. The matrix is then
    read for the chain as a whole, by number, and the score says that both sides are one chain."""
    from bindview.distance_table import interface_contacts
    from bindview.interface import Interface

    model = one_chain_of_the_trimer().split("A", {"X": (1, 75), "Y": (80, 157)})
    path = afdb_file(tmp_path, afdb_matrix(157), gz=True)
    contacts = interface_contacts(model, ["X"], ["Y"])[:6]
    face = Interface(model, ["X", "Y"], contacts, pae=path, ipsae=True, model_confidence=None)
    assert face.contacts and all(c.chain_a == "X" and c.chain_b == "Y" for c in face.contacts)
    # The smaller direction of this matrix is that of the smaller residue number: its own value.
    assert all(c.pae == pytest.approx(c.number_a + c.number_b / 1000.0) for c in face.contacts)
    assert face.pae.numbers == tuple(range(1, 158)) and face.ipsae[0].draws[0].same_chain
    res = face.render(style="schematic")
    assert res.status == "ok" and "Both sides are parts of one chain of this model" in words(res)
    low, high = min(c.pae for c in face.contacts), max(c.pae for c in face.contacts)
    assert f"contact PAE {low:.1f}–{high:.1f} Å" in words(res)


def test_the_database_s_file_is_read_for_a_stretch_of_the_chain_taken_as_the_ligand(tmp_path):
    from bindview.distance_table import contact_table

    model = one_chain_of_the_trimer().split("A", {"Y": (150, 157)})
    path = afdb_file(tmp_path, afdb_matrix(157))
    dia = Diagram.from_structure(model, "chain:Y", interactions=contact_table(model, "Y"), pae=path)
    known = [r.record for r in dia.interactions if r.record.pae is not None]
    assert known and len(known) == len(dia.interactions)
    # Each contact is of a residue before the stretch with one or more of the stretch: the row of the
    # first, and the smallest column among those of the stretch that the contact names.
    for record in known:
        row, column = divmod(round(record.pae * 1000.0), 1000)
        assert row == record.residue_number < 150 <= column <= 157
