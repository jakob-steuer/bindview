"""The predicted aligned error between chains that were predicted together.

``testdata/interface_multimer/tnfa_trimer`` is a Boltz prediction of human TNF-alpha as a homotrimer. Two
of its draws are in the repository, draws 0 and 1: a pair in which the chains are named in another order.

Three things fail quietly on a prediction like this one, and each has its tests here:

* pLDDT on the 0-1 scale (Boltz's ``plddt_*.npz``) handed in where 0-100 is read;
* the predicted *distance* error taken for a second PAE;
* the copies of a chain named in another order from one draw to the next.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from conftest import (MULTIMER_KEPT, TESTDATA, invented_plddt, meta_of, model_of, needs_plip, structure_of, trimer,
                      trimer_contacts, trimer_file, trimer_structure)
from bindview.confidence import Confidence, band_of, plddt_factor
from bindview.errors import InteractionError, BindviewError
from bindview.interface import Interface, dump_interface_contacts, load_interface_contacts
from bindview.pae import (ONE_TOKEN, Pae, PaeDraws, match_chains, model_beside, pae_tag, read_pae, tokens_of)
from bindview.structure import ResidueKey, Structure, read_structure
from scenetools import groups, tags, words

PAIRS = (("A", "B"), ("A", "C"), ("B", "C"))
AS_NAMED = {"A": "A", "B": "B", "C": "C"}
B_AND_C_EXCHANGED = {"A": "A", "B": "C", "C": "B"}


def matrix(draw: int = 0, kind: str = "pae") -> np.ndarray:
    with np.load(trimer_file(kind, draw), allow_pickle=False) as held:
        return np.array(held[kind], dtype=float)


def both() -> PaeDraws:
    """The two draws that are in the repository."""
    made = read_pae([trimer_file("pae", k) for k in range(MULTIMER_KEPT)], trimer_structure())
    assert isinstance(made, PaeDraws)
    return made


# ---------------------------------------------------------------------------
# Which row of the matrix is which residue
# ---------------------------------------------------------------------------

def test_the_rows_of_the_matrix_are_the_residues_of_the_model_in_file_order():
    st = trimer_structure()
    chains, numbers = tokens_of(st)
    assert len(chains) == 471 and [chains.count(c) for c in "ABC"] == [157, 157, 157]
    assert numbers == tuple(list(range(1, 158)) * 3)
    # Not from 471 = 3 x 157: the per-token pLDDT of the model, times 100, is the B-factor of the
    # residue in that row, and with the chains in another order it is not.
    with np.load(trimer_file("plddt"), allow_pickle=False) as held:
        per_token = np.array(held["plddt"], dtype=float) * 100.0
    per_residue = np.array([np.mean([a.b_iso for a in st.residue_atoms(k)]) for k in st.residues()])
    assert np.abs(per_token - per_residue).max() < 1e-3
    assert np.abs(per_token - np.r_[per_residue[157:], per_residue[:157]]).max() > 1.0


def test_a_pair_between_two_chains_is_read_from_the_block_between_them():
    st = trimer_structure()
    pae, raw = Pae.read(trimer_file("pae"), st), matrix()
    assert pae.matrix.shape == (471, 471) and pae.source == trimer_file("pae").name
    a, b = ResidueKey("A", 94), ResidueKey("B", 151)             # row 93, row 157 + 150
    there, back = raw[93, 307], raw[307, 93]
    assert there != back and pae.between(a, b) == pae.between(b, a) == min(there, back)
    # The same two numbers in another pair of chains are another pair of residues.
    assert pae.between(ResidueKey("A", 94), ResidueKey("C", 151)) == min(raw[93, 464], raw[464, 93]) != pae.between(a, b)
    assert pae.between(a, ResidueKey("D", 1)) is None


def test_a_matrix_without_row_names_needs_its_model_and_only_its_own():
    with pytest.raises(BindviewError, match="without the names of its rows"):
        Pae.read(trimer_file("pae"))
    other = read_structure(TESTDATA / "interface_pae" / "afdb_fah" / "model.cif.gz")
    with pytest.raises(BindviewError, match=r"471 x 471.*has \d+ tokens.*not of the same model"):
        Pae.read(trimer_file("pae"), other)


def test_a_residue_that_is_not_a_standard_one_is_a_token_per_atom(tmp_path):
    """The published tokenisation, on a model with a ligand (imatinib in c-Kit, and its waters): a token for
    each residue of the twenty, one for each heavy atom of anything else. A matrix made up for it is read
    by those tokens; the size of a matrix has to equal the number of tokens counted, or it is not taken."""
    st = structure_of("kit_imatinib")
    chains, numbers = tokens_of(st)
    other = [k for k in st.residues() if st.residue_name(k) not in ONE_TOKEN]
    heavy = sum(1 for k in other for a in st.residue_atoms(k) if a.element != "H")
    ligand = next(k for k in other if st.residue_name(k) == meta_of("kit_imatinib")["ligand"]["residue_name_in_cif"])
    assert heavy > len(other)
    assert len(chains) == len(st.residues()) - len(other) + heavy
    n = len(chains)
    made = np.arange(n * n, dtype=float).reshape(n, n)
    np.savez_compressed(tmp_path / "pae_x.npz", pae=made)
    pae = Pae.read(tmp_path / "pae_x.npz", st)
    rows = pae._rows(ligand)
    assert len(rows) == sum(1 for a in st.residue_atoms(ligand) if a.element != "H") > 1
    protein = next(k for k in st.residues() if st.residue_name(k) in ONE_TOKEN)
    assert pae.between(protein, ligand) == min(made[np.ix_(pae._rows(protein), rows)].min(),
                                              made[np.ix_(rows, pae._rows(protein))].min())


# ---------------------------------------------------------------------------
# What is not a PAE
# ---------------------------------------------------------------------------

def test_a_distance_error_is_not_taken_for_an_aligned_error(tmp_path):
    """PDE is a third matrix, not a second opinion on PAE: refused by name where a PAE is expected."""
    np.savez_compressed(tmp_path / "pde_x.npz", pde=np.zeros((471, 471), dtype=np.float32))
    with pytest.raises(BindviewError, match=r"predicted distance error \(PDE\), not a predicted aligned error.*"
                                          r"does not draw PDE"):
        Pae.read(tmp_path / "pde_x.npz", trimer_structure())
    with pytest.raises(BindviewError, match="predicted distance error"):
        Interface(trimer_structure(), ["A", "B"], trimer_contacts(), pae=tmp_path / "pde_x.npz")


def test_a_plddt_file_is_not_a_pae_either(tmp_path):
    with pytest.raises(BindviewError, match=r"per-residue pLDDT, not a predicted aligned error.*0-100"):
        Pae.read(trimer_file("plddt"), trimer_structure())
    np.savez_compressed(tmp_path / "pae_y.npz", contact_probs=np.zeros((3, 3)))
    with pytest.raises(BindviewError, match="no array named 'pae' .*contact_probs"):
        Pae.read(tmp_path / "pae_y.npz", trimer_structure())
    (tmp_path / "pae_z.npz").write_text("not a zip file")
    with pytest.raises(BindviewError, match="cannot read a PAE matrix"):
        Pae.read(tmp_path / "pae_z.npz", trimer_structure())


# ---------------------------------------------------------------------------
# Several draws: the median, and the copies of a chain
# ---------------------------------------------------------------------------

def test_the_draws_do_not_name_the_copies_alike():
    """Draw 1 has the three chains in the opposite order round the axis. Read by chain name, a pair
    that touches in the structure drawn is two residues on opposite sides of the trimer there."""
    draws = both()
    assert draws.copies == [["A", "B", "C"]]
    assert [d["draw"] for d in draws.renamed] == [1]
    odd = draws.renamed[0]
    assert odd["chains"] == B_AND_C_EXCHANGED and odd["source"] == trimer_file("pae", 1).name
    assert odd["rmsd"] < 1.5 and odd["rmsd_as_named"] > 20.0
    assert list(draws.assignments) == [AS_NAMED, B_AND_C_EXCHANGED]
    assert draws.fits[0] == (0.0, 0.0)

    def apart(st: Structure, a: ResidueKey, b: ResidueKey) -> float:
        pa = np.array([x.xyz for x in st.residue_atoms(a)])
        pb = np.array([x.xyz for x in st.residue_atoms(b)])
        return float(np.linalg.norm(pa[:, None, :] - pb[None, :, :], axis=2).min())

    a, b = ResidueKey("A", 94), ResidueKey("B", 151)
    assert apart(trimer_structure(0), a, b) < 4.0 < 10.0 < apart(trimer_structure(1), a, b)
    assert apart(trimer_structure(1), a, ResidueKey("C", 151)) < 4.0
    # ... and that is the pair whose value is read from draw 1.
    raw = matrix(1)
    assert draws.each(a, b)[1] == min(raw[93, 464], raw[464, 93]) != min(raw[93, 307], raw[307, 93])


def test_the_value_of_a_pair_is_the_median_of_its_draws_and_not_the_largest():
    draws = both()
    st = trimer_structure()
    for c in trimer_contacts("A", "B"):
        each = draws.each(c.residue_a, c.residue_b)
        assert len(each) == MULTIMER_KEPT and draws.between(c.residue_a, c.residue_b) == float(np.median(each))
    # The salt bridge of the disordered N-terminus: one draw of the two puts it past 10, the median does not.
    a, b = ResidueKey("A", 53), ResidueKey("B", 2)
    assert st.residue_name(a) == "GLU" and st.residue_name(b) == "ARG"
    each = draws.each(a, b)
    assert min(each) < 7.5 and max(each) > 10.5 and draws.between(a, b) == sum(each) / 2.0 < 9.5
    assert pae_tag(max(each)) == ("PAE 11", True) and pae_tag(draws.between(a, b)) == ("PAE 9", False)
    # The median of two values is their mean, so two draws cannot tell the one from the other. Three made-up
    # draws of a model of two chains can: 2, 4 and 30 Å have a median of 4, a mean of 12 and a largest of 30.
    made = tuple(Pae(np.array([[0.0, v], [v + 1.0, 0.0]]), ("H", "L"), (1, 1), f"draw {k}", ("ALA", "GLY"))
                 for k, v in enumerate((2.0, 30.0, 4.0)))
    assert PaeDraws(made).each(ResidueKey("H", 1), ResidueKey("L", 1)) == (2.0, 30.0, 4.0)
    assert PaeDraws(made).between(ResidueKey("H", 1), ResidueKey("L", 1)) == 4.0
    # One of several is not a set of draws, and no draws are none.
    assert isinstance(read_pae([trimer_file("pae")], st), Pae) and read_pae([], st) is None
    with pytest.raises(BindviewError, match="a matrix file, a Pae, or a pair"):
        read_pae([3.5], st)


def test_copies_cannot_be_told_apart_without_the_model_of_every_draw(tmp_path):
    """A matrix alone cannot say which of two identical chains is which: refused, not read by name."""
    st = trimer_structure()
    for k in range(2):                                  # the matrices without their models beside them
        np.savez_compressed(tmp_path / f"pae_draw_{k}.npz", pae=matrix(k))
    assert model_beside(tmp_path / "pae_draw_0.npz") is None
    assert model_beside(trimer_file("pae", 1)) == trimer_file("model", 1)
    with pytest.raises(BindviewError, match="chains A, B, C are copies of one another.*model file of every draw"):
        read_pae([tmp_path / "pae_draw_0.npz", tmp_path / "pae_draw_1.npz"], st)
    with pytest.raises(BindviewError, match="copies of one another"):
        Interface(st, ["A", "B"], trimer_contacts(), pae=[tmp_path / "pae_draw_0.npz", tmp_path / "pae_draw_1.npz"])
    # Given with its model, as a pair, each draw is matched.
    paired = read_pae([(tmp_path / f"pae_draw_{k}.npz", trimer_file("model", k)) for k in range(2)], st)
    assert isinstance(paired, PaeDraws) and paired.assignments[1] == B_AND_C_EXCHANGED
    # A model without copies needs no coordinates: its chains are who they are by what they are made of.
    one = Pae(np.array([[0.0, 2.0], [4.0, 0.0]]), ("H", "L"), (1, 1), "first", ("ALA", "GLY"))
    two = Pae(np.array([[0.0, 6.0], [8.0, 0.0]]), ("H", "L"), (1, 1), "second", ("ALA", "GLY"))
    assert match_chains(one, two) == ({"H": "H", "L": "L"}, 0.0, 0.0)
    assert PaeDraws((one, two)).between(ResidueKey("H", 1), ResidueKey("L", 1)) == 4.0
    with pytest.raises(BindviewError, match="not draws of one model"):
        PaeDraws((one, Pae(np.zeros((2, 2)), ("H", "L"), (1, 2), "third")))


def test_chains_numbered_alike_are_not_called_copies_where_their_residues_are_not_known():
    """Two draws that came without a model: chains A and B have the same residue numbers, and whether
    they have the same residues is not known. The refusal says that, and not that they are copies."""
    def draw(k: int, names: tuple = ()) -> Pae:
        return Pae(np.full((4, 4), float(k)), ("A", "A", "B", "B"), (1, 2, 1, 2), f"draw_{k}.json", names)

    with pytest.raises(BindviewError) as told:
        PaeDraws((draw(0), draw(1)))
    said = str(told.value)
    assert "chains A, B have the same residue numbers and may be copies of one another" in said
    assert "are copies of one another" not in said and "model file of every draw" in said
    # Where the names are known and differ, the two are two chains and are read by name.
    names = ("ALA", "GLY", "SER", "TRP")
    assert PaeDraws((draw(0, names), draw(1, names))).copies == []
    # ... and where they are the same, they are copies, and are called that.
    with pytest.raises(BindviewError, match="chains A, B are copies of one another"):
        PaeDraws((draw(0, ("ALA", "GLY") * 2), draw(1, ("ALA", "GLY") * 2)))


def test_an_assignment_that_fits_no_better_does_not_rename():
    """A symmetric trimer fits equally well in each of its rotations: the chains stay as they are named
    unless another assignment fits better by more than half an ångström."""
    ref = Pae.read(trimer_file("pae"), trimer_structure())
    names, fit, as_named = match_chains(ref, ref)
    assert ref.places is not None and names == AS_NAMED and fit == as_named < 1e-5
    # Draw 0 with its names turned round the axis: what it calls A lies where C does here, and so on. As
    # named it fits to 0.2 Å, with the names turned back exactly. That is within half an ångström, and it
    # keeps its names: a rotation of the names of a symmetric trimer cannot be told from none.
    turned = replace(ref, places=np.roll(ref.places, 157, axis=0), source="draw 0, its names turned")
    names, fit, as_named = match_chains(ref, turned)
    assert names == AS_NAMED and 0.05 < fit == as_named < 0.3
    # The opposite order round the axis can be told, and is: draw 1.
    names, fit, as_named = match_chains(ref, Pae.read(trimer_file("pae", 1), trimer_structure(1)))
    assert names == B_AND_C_EXCHANGED and fit < 1.0 and as_named > 21.0


# ---------------------------------------------------------------------------
# The figure
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pair", PAIRS)
@pytest.mark.parametrize("style", ("schematic", "detailed"))
def test_each_trimer_interface_is_confident_but_for_its_tail(pair, style):
    """A confident trimer: most contacts below 5 Å, a few between 5 and 10 Å at the N-terminus, no
    interface with the figure-level warning."""
    face = trimer(*pair)
    res = face.render(style=style)
    assert res.status == "ok" and face.site.whole and not res.warnings
    written = tags(res)
    assert written and all(t.fill == "text" and t.opacity == 1.0 for t in written)
    pairs = {(c.residue_a, c.residue_b): c.pae for c in face.contacts}
    meta = res.graph.meta["interface"]["pae"]
    assert (meta["pairs"], meta["of_pairs"]) == (len(pairs), len(pairs))
    assert meta["pairs_written"] == sum(1 for v in pairs.values() if pae_tag(v) is not None) == 5
    assert len(written) == sum(1 for c in res.graph.contacts if pae_tag(c.attrs["pae"]) is not None)
    # Over the two draws no pair reaches 10 Å.
    assert not any(t.bold for t in written)
    assert sorted(t.content for t in written)[0] == "PAE 7" and sorted(t.content for t in written)[-1] == "PAE 9"
    # Every pair with a tag has one of the first six residues of a chain on one side: the disordered tail.
    tagged = {(c.residue_a.number, c.residue_b.number) for c in face.contacts if pae_tag(c.pae) is not None}
    assert tagged and all(min(p) <= 6 for p in tagged)
    assert all(pae_tag(c.pae) is None for c in face.contacts if min(c.number_a, c.number_b) > 6)
    text = words(res)
    values = [c.pae for c in face.contacts]
    assert f"contact PAE, median of 2 draws {min(values):.1f}–{max(values):.1f} Å" in text
    assert "Every contact in this figure" not in text and not groups(res, "pae-warning")
    assert "5 of the" in text and "residue pairs in contact have a PAE of 5 Å or more, written on their lines." in text


def test_the_figure_says_that_it_shows_a_median_and_of_what():
    res = trimer("A", "B").render(style="schematic")
    text = words(res)
    # Said of the draws that were read: how often the model was sampled in all, a figure cannot know.
    assert ("The PAE comes from 2 samples of the model (2 draws). The PAE of a contact is the median of its 2 "
            "values, not the largest, which would rise with the number of draws whatever the model knows. Draw by "
            "draw the contacts shown here range over 0.4–7.4 and 0.4–11.1 Å.") in text
    assert ("Chains A, B and C are copies of one another, and the model names them in another order in 1 of these "
            "2 draws: the chains of every draw were matched to those of the structure shown here, by "
            "superposition, before its values were read.") in text
    assert "Predicted aligned error (PAE) of a contact in Å, median of 2 draws: written from 5 Å, bold from 10 Å. " \
           "Not pLDDT" in text
    assert "contact PAE, median of 2 draws" in res.scene.description and "was sampled" not in text
    meta = res.graph.meta["interface"]["pae"]
    assert (meta["draws"], meta["over_draws"], len(meta["by_draw"])) == (2, "median", 2)
    assert meta["direction"] == "the smaller of the two directions of the matrix"
    assert meta["copies"] == [["A", "B", "C"]] and meta["chains_matched_by"] == "superposition"
    assert [d["draw"] for d in meta["draws_named_otherwise"]] == [1]
    assert [d["max"] for d in meta["by_draw"]] == [7.41, 11.05]
    strip = groups(res, "confidence")[0]
    assert strip.data["contact-pae-draws"] == "2" and strip.data["contact-pae-max"] == "9.2"


def test_every_draw_s_value_travels_with_the_contact():
    face = trimer("A", "B")
    draws = both()
    for c in face.contacts:
        assert c.pae_draws == draws.each(c.residue_a, c.residue_b) and c.pae == float(np.median(c.pae_draws))
    res = face.render(style="detailed")
    lines = groups(res, "interaction")
    assert lines and all(len(g.data["pae-draws"].split()) == 2 for g in lines)
    one = next(g for g in lines if g.data["residue"].endswith("GLU53") and g.data["ligand-residue"] == "Arg2")
    its = next(c.pae_draws for c in face.contacts if (c.number_a, c.number_b) == (53, 2))
    assert one.data["pae"] == "9.1" and one.data["pae-draws"] == " ".join(f"{v:.1f}" for v in its) == "7.3 11.0"
    assert ", PAE 9.1 Å (median of 2 draws, 7.3 to 11.0)" in one.title
    for c in res.graph.contacts:
        assert len(c.attrs["pae_draws"]) == 2 and abs(c.attrs["pae"] - float(np.median(c.attrs["pae_draws"]))) < 0.011
    # The table keeps them, in a column that is there only where there are draws.
    table = dump_interface_contacts(face.contacts)
    assert table.splitlines()[0].endswith("\tpae\twater\tpae_draws")
    assert not dump_interface_contacts(trimer_contacts()).splitlines()[0].endswith("pae_draws")


def test_the_table_of_contacts_brings_its_draws_back(tmp_path):
    face = trimer("A", "B")
    path = tmp_path / "contacts.tsv"
    path.write_text(dump_interface_contacts(face.contacts), encoding="utf-8")
    back = load_interface_contacts(path)
    assert [(c.pae, c.pae_draws) for c in back] == [(c.pae, c.pae_draws) for c in face.contacts]
    again = Interface(trimer_structure(), ["A", "B"], back)             # no matrix: the values come with the table
    res, ref = again.render(style="schematic"), face.render(style="schematic")
    text = words(res)
    assert "contact PAE, median of 2 draws 0.4–9.2 Å" in text
    assert "The PAE comes from 2 samples of the model (2 draws)." in text
    assert "matched to those of the structure" not in text              # a table does not say what was matched
    assert [t.content for t in tags(res)] == [t.content for t in tags(ref)]


def test_the_same_matrix_twice_is_not_two_draws():
    """A figure of draws says how many samples of the model its values are a median of. The same file
    given twice is one sample and not two: refused, not counted."""
    st = trimer_structure()
    twice = [trimer_file("pae"), trimer_file("pae")]
    name = trimer_file("pae").name.replace(".", r"\.")
    with pytest.raises(BindviewError, match=rf"draws 1 and 2 \({name} and {name}\) are the same matrix"):
        read_pae(twice, st)
    with pytest.raises(BindviewError, match="are the same matrix: a draw is given once"):
        Interface(st, ["A", "B"], trimer_contacts()[:6], pae=twice, model_confidence=None)
    with pytest.raises(BindviewError, match=r"draws 1 and 3 \(first and third\) are the same matrix"):
        PaeDraws(tuple(Pae(np.array([[0.0, v], [v + 1.0, 0.0]]), ("H", "L"), (1, 1), source, ("ALA", "GLY"))
                       for source, v in (("first", 2.0), ("second", 4.0), ("third", 2.0))))
    # Two draws that are two draws are read as they were.
    assert len(read_pae([trimer_file("pae", k) for k in range(2)], st).draws) == 2


def test_the_direction_of_the_matrix_is_said_of_values_that_were_read_from_it():
    """A contact that brings its own PAE keeps it, whatever the matrix says. Of such values it is not
    known which direction of a matrix they are, and the graph does not say that it is."""
    st = trimer_structure()
    own = [replace(c, pae=1.0) for c in trimer_contacts()[:5]]
    face = Interface(st, ["A", "B"], own, pae=trimer_file("pae"), model_confidence=None)
    meta = face.render(style="schematic").graph.meta["interface"]["pae"]
    assert (meta["min"], meta["max"], meta["with_a_value"]) == (1.0, 1.0, 5) and "direction" not in meta
    # One contact without a value of its own gets the matrix's, and then the direction is said.
    mixed = [replace(c, pae=None) if k == 0 else c for k, c in enumerate(own)]
    face = Interface(st, ["A", "B"], mixed, pae=trimer_file("pae"), model_confidence=None)
    meta = face.render(style="schematic").graph.meta["interface"]["pae"]
    assert meta["max"] > 1.0 and meta["direction"] == "the smaller of the two directions of the matrix"


def test_one_draw_is_a_figure_of_one_draw():
    one, two = trimer("A", "B", draws=1), trimer("A", "B")
    res = one.render(style="schematic")
    text = words(res)
    assert "median" not in text and "draws" not in text and "contact PAE 0.4–7.4 Å" in text
    assert all(c.pae_draws == () for c in one.contacts)
    assert [c.pae for c in one.contacts] == [c.pae_draws[0] for c in two.contacts]
    assert "draws" not in res.graph.meta["interface"]["pae"]
    assert all("pae-draws" not in g.data for g in groups(res, "interaction"))
    # Without a PAE the figure says nothing of one, and has the same strokes as with it.
    bare = trimer("A", "B", draws=0).render(style="schematic")
    assert "PAE" not in words(bare)


@pytest.mark.parametrize("style", ("schematic", "detailed"))
def test_a_figure_of_draws_is_redrawn_from_its_graph_as_it_was(style):
    from bindview.graph import Graph

    face = trimer("B", "C")
    res = face.render(style=style)
    again = face.diagram.render(style=style, graph=Graph.from_json(res.graph.to_json()))
    assert again.svg == res.svg


def test_both_channels_are_live_and_say_the_same_of_the_tail():
    """pLDDT on the things and the lines, PAE as a tag: on this trimer they agree, each in its own way."""
    face = trimer("A", "B")
    res = face.render(style="schematic")
    lines = groups(res, "interaction")
    low = [g for g in lines if g.data.get("confidence") == "very_low"]
    with_tag = [g for g in lines if pae_tag(float(g.data["pae"])) is not None]
    assert low and {id(g) for g in with_tag} <= {id(g) for g in low}
    assert all(g.data["ligand-residue"] in ("Arg2", "Ser3", "Ser4", "Arg6") for g in with_tag)
    # The strokes are those of the figure without any PAE.
    bare = trimer("A", "B", draws=0).render(style="schematic")
    assert sorted((g.data["residue"], g.data["type"], g.data.get("confidence")) for g in lines) == \
        sorted((g.data["residue"], g.data["type"], g.data.get("confidence")) for g in groups(bare, "interaction"))


def test_command_line_takes_the_draws_of_a_model(tmp_path, capsys):
    from bindview.cli import main as cli

    tsv, out = tmp_path / "contacts.tsv", tmp_path / "out.svg"
    tsv.write_text(dump_interface_contacts(trimer_contacts("A", "B")), encoding="utf-8")
    base = ["interface", str(trimer_file("model")), "--between", "A", "B", "--contacts", str(tsv), "-o", str(out),
            "-q", "--style", "schematic", "--json"]
    args = list(base)
    for k in range(MULTIMER_KEPT):
        args += ["--pae", str(trimer_file("pae", k))]
    assert cli(args) == 0
    said = capsys.readouterr().out
    assert '"draws": 2' in said and '"over_draws": "median"' in said and '"pairs_written": 5' in said
    assert out.read_text(encoding="utf-8") == trimer("A", "B").render(style="schematic").svg
    # One matrix, of the model that is drawn.
    assert cli(base + ["--pae", str(trimer_file("pae"))]) == 0
    assert "median of" not in out.read_text(encoding="utf-8") and ">PAE 7<" in out.read_text(encoding="utf-8")


@needs_plip
def test_plip_reads_the_trimer_the_same_way():
    """PLIP's contacts in place of the distance table, over the two draws: more contacts than the distance
    table, about as many residue pairs, and the same picture. For each interface the numbers of contacts,
    residue pairs, pairs with a tag and bold ones; every pair with a tag has one of the first six residues
    of a chain on one side, and no pair without one has. One pair of the tail lies just above the first
    threshold (5.3 Å, written as ``PAE 5``)."""
    n, counts, first_tag_above = MULTIMER_KEPT, ((30, 24, 3, 0), (31, 24, 4, 0), (34, 27, 4, 0)), 5.0
    files = [trimer_file("pae", k) for k in range(n)]
    for (a, b), counted in zip(PAIRS, counts):
        face = Interface(trimer_structure(), [a, b], pae=files)
        pairs = {(c.residue_a, c.residue_b): c.pae for c in face.contacts}
        tagged = {k: v for k, v in pairs.items() if pae_tag(v) is not None}
        bold = sum(1 for v in tagged.values() if pae_tag(v)[1])
        assert face.source == "detected" and (len(face.contacts), len(pairs), len(tagged), bold) == counted
        assert all(min(x.number, y.number) <= 6 for x, y in tagged)
        assert not [k for k in pairs if k not in tagged and min(k[0].number, k[1].number) <= 6]
        assert max(v for k, v in pairs.items() if k not in tagged) < 4.5 < first_tag_above < min(tagged.values())
        res = face.render(style="schematic")
        assert res.status == "ok" and f"contact PAE, median of {n} draws" in words(res)


# ---------------------------------------------------------------------------
# pLDDT on the 0-1 scale
# ---------------------------------------------------------------------------

def raw_plddt() -> dict:
    with np.load(trimer_file("plddt"), allow_pickle=False) as held:
        values = np.array(held["plddt"], dtype=float)
    return {(k.chain, k.number): float(v) for k, v in zip(trimer_structure().residues(), values)}


def test_boltz_writes_plddt_on_two_scales_and_the_model_file_is_on_the_one_that_is_read():
    st = trimer_structure()
    assert st.declares_plddt and st.plddt_scale == 1.0
    raw = raw_plddt()
    assert 0.4 < min(raw.values()) < max(raw.values()) < 1.0
    res = trimer("A", "B", draws=0).render(style="schematic")
    assert res.confidence["origin"] == "declared" and 80 < res.confidence["pocket_plddt"] < 95
    assert res.confidence["contacts_by_band"] == {"very_high": 16, "confident": 4, "low": 1, "very_low": 6}


def test_contact_confidences_on_the_0_1_scale_are_refused_and_not_drawn_as_doubt():
    """Taken as they are, every line would be drawn as pLDDT below 50 with its ``?`` while the boxes and the
    numbers under the title stay as they are: a figure that looks like a result. They are refused, with what
    to do instead."""
    st, raw = trimer_structure(), raw_plddt()
    contacts = [replace(c, confidence=min(raw[(c.chain_a, c.number_a)], raw[(c.chain_b, c.number_b)]))
                for c in trimer_contacts("A", "B")]
    assert all(band_of(c.confidence) == "very_low" for c in contacts)             # how they would have been read
    with pytest.raises(InteractionError, match=r"contacts of this interface carry a confidence of 0\.433 to 0\.975: "
                                               r"nothing above 1.*0-1 scale.*nothing is multiplied for you"):
        Interface(st, ["A", "B"], contacts)
    # Stated by the caller, which here means: given on the scale that is read.
    right = Interface(st, ["A", "B"], [replace(c, confidence=100.0 * c.confidence) for c in contacts])
    res, ref = right.render(style="schematic"), trimer("A", "B", draws=0).render(style="schematic")
    bands = lambda r: sorted((g.data["residue"], g.data["type"], g.data["confidence"])        # noqa: E731
                             for g in groups(r, "interaction"))
    assert bands(res) == bands(ref) and len({b[2] for b in bands(res)}) == 4
    # A ligand diagram has the same guard, on the records it is given.
    from bindview import Diagram, load_interactions

    m = meta_of("kit_imatinib")
    rows = [r.with_(confidence=0.9) for r in load_interactions(TESTDATA / "kit_imatinib" / "interactions.tsv")]
    with pytest.raises(InteractionError, match="The contacts carry a confidence of 0.9: nothing above 1"):
        Diagram.from_structure(model_of("kit_imatinib"), ligand=m["ligand"]["residue_name_in_cif"],
                               smiles=m["ligand"]["smiles"], interactions=rows)


def test_values_handed_in_state_their_scale_or_are_on_0_100():
    raw = raw_plddt()
    with pytest.raises(BindviewError, match=r"are 0\.427 to 0\.984: nothing above 1.*not guessed.*"
                                          r"scale='0-1' reads them as 43 to 98; scale='0-100' takes them"):
        Confidence.from_values(raw)
    stated = Confidence.from_values(raw, scale="0-1")
    st = trimer_structure()
    declared = {(k.chain, k.number, k.icode): float(np.mean([a.b_iso for a in st.residue_atoms(k)]))
                for k in st.residues()}
    assert max(abs(stated.residues[k] - v) for k, v in declared.items()) < 1e-3
    assert stated.given_on == "0-1" and stated.summary()["given_on"].startswith("the 0-1 scale, as stated")
    # On 0-100 nothing is said of a scale, stated or not.
    assert "given_on" not in Confidence.from_values({"A:5": 88.0}).summary()
    assert "given_on" not in Confidence.from_values({"A:5": 88.0}, scale="0-100").summary()
    # A stated scale is the caller's word, and is taken as far as the values allow it.
    assert Confidence.from_values({"A:5": 0.9}, scale="0-100").residues == {("A", 5, ""): 0.9}
    with pytest.raises(BindviewError, match="scale stated is 0-1, on which nothing exceeds 1.*Nothing is divided"):
        Confidence.from_values({"A:5": 88.0, "A:6": 0.5}, scale="0-1")
    with pytest.raises(BindviewError, match="nothing exceeds 100"):
        Confidence.from_values({"A:5": 4300.0})                     # a hundredfold applied twice
    with pytest.raises(BindviewError, match="not negative"):
        Confidence.from_values({"A:5": -3.0})
    with pytest.raises(ValueError, match="'0-100' or '0-1', not 'percent'"):
        Confidence.from_values({"A:5": 50.0}, scale="percent")
    assert Confidence.from_values({"A:5": 0.0}).residues == {("A", 5, ""): 0.0}     # zero is zero on both
    assert plddt_factor([]) == 1.0 and plddt_factor([], scale="0-1") == 100.0
    assert plddt_factor([0.2, 0.9], scale="0-1") == 100.0 and plddt_factor([20.0, 90.0]) == 1.0


def test_a_b_factor_column_on_the_other_scale_than_it_is_said_to_be_on_is_refused():
    """On imatinib in c-Kit with a pLDDT on 0-100 that is made up for the test (``invented_plddt``), and the
    same values divided by a hundred."""
    from bindview import Diagram, load_interactions

    m = meta_of("kit_imatinib")
    st = invented_plddt(structure_of("kit_imatinib"))
    assert st.declares_plddt and st.plddt_scale == 1.0

    def build(structure, **how):
        return Diagram.from_structure(structure, ligand=m["ligand"]["residue_name_in_cif"], smiles=m["ligand"]["smiles"],
                                      interactions=load_interactions(TESTDATA / "kit_imatinib" / "interactions.tsv"),
                                      **how)

    unit = Structure(atoms=[replace(a, b_iso=a.b_iso / 100.0) for a in st.atoms], source="on 0-1",
                     chem_comp_smiles=st.chem_comp_smiles, plddt_scale=st.plddt_scale, qa_metrics=st.qa_metrics)
    # The file says pLDDT, on 0-100, and holds nothing above 1: refused, and not drawn as a model of pure doubt.
    with pytest.raises(BindviewError, match=r"B-factor column of on 0-1, declared a pLDDT on the 0-100 scale, holds "
                                          r"0\.\d+ to 0\.\d+: nothing above 1.*plddt='bfactor01'"):
        build(unit)
    with pytest.raises(BindviewError, match="asserted to be pLDDT on the 0-100 scale, holds"):
        build(unit, plddt="bfactor")
    said = build(unit, plddt="bfactor01")
    ref = build(st)
    assert said.confidence.origin == "asserted"
    assert max(abs(said.confidence.residues[k] - v) for k, v in ref.confidence.residues.items()) < 1e-9
    # And the other way round: a column on 0-100 that is said to be on 0-1 is not multiplied by a hundred.
    with pytest.raises(BindviewError, match="asserted to be pLDDT on the 0-1 scale, holds .*nothing exceeds 1"):
        build(st, plddt="bfactor01")
    wrong = Structure(atoms=list(st.atoms), source="declared in [0,1]", chem_comp_smiles=st.chem_comp_smiles,
                      plddt_scale=100.0, qa_metrics=st.qa_metrics)
    with pytest.raises(BindviewError, match="declared a pLDDT on the 0-1 scale, holds .*nothing exceeds 1"):
        build(wrong)
    with pytest.raises(ValueError, match="'auto', 'bfactor', 'bfactor01', None or a Confidence"):
        build(st, plddt="guess")
