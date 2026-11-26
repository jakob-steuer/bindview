"""ipSAE: a score of an interface as a whole, worked out from the PAE matrix where it is asked for.

The definition is Dunbrack's (2025) and that of its reference script, which is not run here. What is
tested: the values of d0 that its authors publish, scores worked out by hand on small matrices, and the
definition written out as loops and run on the matrices in the repository: three models of the AlphaFold
database, each cut between two domains, and the predicted TNF trimer.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from conftest import PAE_CASES, TESTDATA, pae_case, trimer, trimer_contacts, trimer_file, trimer_structure
from bindview.errors import BindviewError
from bindview.interface import Interface, InterfaceIpsae
from bindview.pae import IPSAE_CUTOFF, Pae, PaeDraws, ipsae, ipsae_d0, ipsae_of, read_pae
from bindview.structure import ResidueKey
from scenetools import words

def side(chain: str, n: int) -> list[ResidueKey]:
    return [ResidueKey(chain, k) for k in range(1, n + 1)]


def two_chains(a_to_b, b_to_a=None, far: float = 30.0) -> Pae:
    """A matrix of two chains, A and B: ``a_to_b`` are the rows of A over the columns of B, ``b_to_a`` the
    rows of B over the columns of A (as far apart as ``far`` where it is left out)."""
    a_to_b = np.array(a_to_b, dtype=float)
    n_a, n_b = a_to_b.shape
    b_to_a = np.full((n_b, n_a), far) if b_to_a is None else np.array(b_to_a, dtype=float)
    matrix = np.zeros((n_a + n_b, n_a + n_b))
    matrix[:n_a, n_a:], matrix[n_a:, :n_a] = a_to_b, b_to_a
    return Pae(matrix, ("A",) * n_a + ("B",) * n_b, tuple(range(1, n_a + 1)) + tuple(range(1, n_b + 1)), "made up")


def best_row(matrix: np.ndarray, aligned: list[int], scored: list[int], cutoff: float) -> tuple:
    """One direction of ipSAE as its definition reads, residue by residue and pair by pair: the value, and the
    row it comes from with its number of residues within reach and its d0 (the first of equal rows)."""
    best: tuple = (0.0, None, 0, None)
    for i in aligned:
        below = [float(matrix[i][j]) for j in scored if matrix[i][j] < cutoff]
        if not below:
            continue
        d0 = max(1.0, 1.24 * (max(len(below), 26) - 15) ** (1.0 / 3.0) - 1.8)
        value = sum(1.0 / (1.0 + (x / d0) ** 2) for x in below) / len(below)
        if value > best[0]:
            best = (value, i, len(below), d0)
    return best


def by_the_book(matrix: np.ndarray, aligned: list[int], scored: list[int], cutoff: float) -> float:
    """One direction of ipSAE as its definition reads: the value alone."""
    return best_row(matrix, aligned, scored, cutoff)[0]


def as_the_book_says(way, matrix: np.ndarray, aligned: list[int], scored: list[int], cutoff: float) -> None:
    """One direction of a score is the definition's, with the residue it is aligned on, its residues within
    reach and its d0."""
    value, row, n0, d0 = best_row(matrix, aligned, scored, cutoff)
    assert way.value == pytest.approx(value, abs=1e-12) and way.n0 == n0
    if row is not None:
        assert way.d0 == pytest.approx(d0, abs=1e-12)


def domains(slug: str, structure=None) -> tuple[Pae, list[ResidueKey], list[ResidueKey]]:
    """The matrix of one of the models of ``testdata/interface_pae``, read for its model (or for ``structure``),
    and the residues of its two domains, as chain A."""
    from bindview.structure import read_structure

    folder = TESTDATA / "interface_pae" / slug
    (a0, a1), (b0, b1) = (p["residues"] for p in json.loads((folder / "meta.json").read_text())["participants"])
    model = structure if structure is not None else read_structure(folder / "model.cif.gz")
    pae = Pae.read(folder / "predicted_aligned_error.json.gz", model)
    numbers = sorted(set(pae.numbers))
    return (pae, [ResidueKey("A", n) for n in numbers if a0 <= n <= a1],
            [ResidueKey("A", n) for n in numbers if b0 <= n <= b1])


# -- the score ------------------------------------------------------------------------------------------

@pytest.mark.parametrize("residues, d0", ((290, 6.26), (1571, 12.57), (553, 8.29), (75, 3.05), (40, 1.83),
                                          (268, 6.04)))
def test_d0_is_the_one_its_authors_publish(residues, d0):
    """Three pairs from the sample output of the reference script's README, one from the paper, and two
    from the output the script's repository holds for an AlphaFold 3 model (its ``Example`` folder)."""
    assert round(ipsae_d0(residues), 2) == d0


def test_d0_is_never_below_one():
    assert [ipsae_d0(n) for n in (0, 1, 15, 26)] == [1.0] * 4
    assert ipsae_d0(27) == pytest.approx(1.0389, abs=1e-4) and ipsae_d0(100) == pytest.approx(3.6521, abs=1e-4)


def test_a_score_worked_out_by_hand():
    """Two residues against three. With so few residues d0 is 1, and a term is 1 / (1 + PAE squared)."""
    pae = two_chains([[1.0, 2.0, 20.0], [3.0, 9.99, 10.0]], [[4.0, 30.0], [30.0, 30.0], [10.0, 12.0]])
    score = ipsae(pae, side("A", 2), side("B", 3))
    # Aligned on A1: 1 and 2 Å are below 10, so (1/2 + 1/5) / 2. Aligned on A2: 3 and 9.99 are, 10.0 is not.
    assert score.forward.value == pytest.approx((1 / 2 + 1 / 5) / 2)
    assert (score.forward.aligned, score.forward.n0, score.forward.d0) == (("A", 1), 2, 1.0)
    second = (1 / (1 + 3.0 ** 2) + 1 / (1 + 9.99 ** 2)) / 2
    assert second < score.forward.value and by_the_book(pae.matrix, [1], [2, 3, 4], 10.0) == pytest.approx(second)
    # The other direction is another number: aligned on B1 one residue of A is within reach, at 4 Å.
    assert score.backward.value == pytest.approx(1 / 17)
    assert (score.backward.aligned, score.backward.n0) == (("B", 1), 1)
    assert score.values == (score.forward.value, score.backward.value)
    assert (score.cutoff, score.residues, score.left_out, score.same_chain) == (IPSAE_CUTOFF, (2, 3), 0, False)
    # The rows are the aligned residues: with the two sides exchanged, the two directions change places.
    turned = ipsae(pae, side("B", 3), side("A", 2))
    assert turned.values == (score.backward.value, score.forward.value)


def test_the_cut_off_is_strict_and_can_be_another():
    pae = two_chains([[9.0, 12.0]], [[9.0], [12.0]])
    assert ipsae(pae, side("A", 1), side("B", 2)).forward.value == pytest.approx(1 / 82)
    wider = ipsae(pae, side("A", 1), side("B", 2), cutoff=15.0)
    assert wider.cutoff == 15.0 and wider.forward.value == pytest.approx((1 / 82 + 1 / 145) / 2)
    assert ipsae(pae, side("A", 1), side("B", 2), cutoff=9.0).values == (0.0, 0.0)
    for bad in (0.0, -5.0):
        with pytest.raises(BindviewError, match="positive number"):
            ipsae(pae, side("A", 1), side("B", 2), cutoff=bad)


def test_no_pair_within_reach_is_a_score_of_zero():
    score = ipsae(two_chains([[25.0, 31.0], [12.0, 10.0]]), side("A", 2), side("B", 2))
    assert score.values == (0.0, 0.0)
    assert score.forward.aligned is None and score.forward.n0 == 0


def test_d0_follows_the_number_of_residues_within_reach():
    """One residue with a hundred residues at 2 Å: d0 is that of a hundred, and every term the same."""
    score = ipsae(two_chains([[2.0] * 100]), side("A", 1), side("B", 100))
    assert score.forward.n0 == 100 and score.forward.d0 == pytest.approx(ipsae_d0(100))
    assert score.forward.value == pytest.approx(1.0 / (1.0 + (2.0 / ipsae_d0(100)) ** 2))
    assert score.forward.value == pytest.approx(0.7693, abs=1e-4)
    # With thirty of them at 2 Å and seventy out of reach the same residue scores lower: d0 is of thirty.
    fewer = ipsae(two_chains([[2.0] * 30 + [20.0] * 70]), side("A", 1), side("B", 100))
    assert fewer.forward.n0 == 30 and fewer.forward.value < score.forward.value


def test_a_missing_value_is_within_reach_of_nothing():
    pae = two_chains([[float("nan"), 2.0]], [[float("nan")], [float("nan")]])
    score = ipsae(pae, side("A", 1), side("B", 2))
    assert score.forward.value == pytest.approx(1 / 5) and score.forward.n0 == 1 and score.backward.value == 0.0


def with_a_modified_residue(tmp_path):
    """A made-up model and its matrix. Chain A: a glycine, a modified residue of three heavy atoms (the
    model takes it atom by atom: three rows), a ligand of two atoms and an ion. Chain B: two alanines."""
    from bindview.structure import read_structure

    atoms = [("ATOM", "N", "N", "GLY", "A", 1), ("ATOM", "C", "CA", "GLY", "A", 1), ("ATOM", "C", "C", "GLY", "A", 1),
             ("ATOM", "N", "N", "TPO", "A", 2), ("ATOM", "C", "CA", "TPO", "A", 2), ("ATOM", "C", "CB", "TPO", "A", 2),
             ("HETATM", "C", "C1", "LIG", "A", 3), ("HETATM", "C", "C2", "LIG", "A", 3),
             ("HETATM", "MG", "MG", "MG", "A", 4),
             ("ATOM", "N", "N", "ALA", "B", 1), ("ATOM", "C", "CA", "ALA", "B", 1),
             ("ATOM", "N", "N", "ALA", "B", 2), ("ATOM", "C", "CA", "ALA", "B", 2)]
    lines = ["data_x", "loop_"] + ["_atom_site." + c for c in (
        "group_PDB", "type_symbol", "label_atom_id", "label_comp_id", "label_asym_id", "label_seq_id",
        "Cartn_x", "Cartn_y", "Cartn_z", "pdbx_PDB_model_num")]
    lines += [f"{g} {e} {n} {r} {c} {k} {1.5 * i:.3f} 0.000 0.000 1" for i, (g, e, n, r, c, k) in enumerate(atoms)]
    (tmp_path / "made.cif").write_text("\n".join(lines) + "\n")
    structure = read_structure(tmp_path / "made.cif")
    # Rows: A1; N, CA and CB of A2; the two atoms of the ligand; the ion; B1; B2.
    chains, numbers = tuple("AAAAAAABB"), (1, 2, 2, 2, 3, 3, 4, 1, 2)
    matrix = np.full((9, 9), 20.0)
    for row, value in ((0, 8.0), (1, 3.0), (2, 1.0), (3, 5.0), (4, 0.5), (5, 0.5), (6, 0.5)):
        matrix[row, 7:] = value
    return structure, Pae(matrix, chains, numbers, "made up"), side("A", 4), side("B", 2)


def test_a_modified_residue_enters_by_its_alpha_carbon(tmp_path):
    """As in the reference script. The model is needed to say which of the residue's rows that is."""
    structure, pae, first, second = with_a_modified_residue(tmp_path)
    score = ipsae(pae, first, second, structure=structure)
    # Its CA is the second of its three heavy atoms, so the second of its rows, at 1 Å from both alanines.
    assert score.forward.value == pytest.approx(1 / 2) and score.forward.aligned == ("A", 2)
    assert score.by_alpha_carbon == 1 and score.residues == (2, 2)
    # With the row of its N it would have been 1/10, with that of its CB 1/26.
    assert by_the_book(pae.matrix, [1], [7, 8], 10.0) == pytest.approx(1 / 10)


def test_what_has_no_alpha_carbon_does_not_enter(tmp_path):
    """The ligand and the ion of chain A are nearer to chain B than any residue, by the matrix: they would
    give the score if they entered."""
    structure, pae, first, second = with_a_modified_residue(tmp_path)
    score = ipsae(pae, first, second, structure=structure)
    assert score.left_out == 2 and score.forward.value == pytest.approx(1 / 2)
    of_the_ion = by_the_book(pae.matrix, [6], [7, 8], 10.0)
    assert of_the_ion == pytest.approx(1 / 1.25) and of_the_ion > score.forward.value
    # A side of nothing but those is no side of an ipSAE.
    with pytest.raises(BindviewError, match="alpha carbon on one side"):
        ipsae(pae, first[2:], second, structure=structure)


def test_without_the_model_a_residue_of_several_rows_is_left_out_and_counted(tmp_path):
    """Which of its rows is its alpha carbon cannot be told from the matrix. A residue of one row is taken
    for a residue, which an ion is not: the model is what tells them apart."""
    _, pae, first, second = with_a_modified_residue(tmp_path)
    score = ipsae(pae, first, second)
    assert score.left_out == 2 and score.by_alpha_carbon == 0 and score.residues == (2, 2)
    assert score.forward.aligned == ("A", 4) and score.forward.value == pytest.approx(1 / 1.25)


def test_what_cannot_be_scored_is_refused():
    pae = two_chains([[2.0, 3.0]])
    with pytest.raises(BindviewError, match="alpha carbon on one side"):
        ipsae(pae, side("A", 1), side("C", 2))
    with pytest.raises(BindviewError, match="alpha carbon on either side"):
        ipsae(pae, side("C", 1), side("D", 2))
    with pytest.raises(BindviewError, match="share residues"):
        ipsae(pae, side("A", 1) + side("B", 1), side("B", 2))


@pytest.mark.parametrize("slug", PAE_CASES)
def test_the_definition_written_out_as_loops_gives_the_same_on_a_real_matrix(slug):
    pae, first, second = domains(slug)
    rows = {n: i for i, n in enumerate(pae.numbers)}
    a, b = [rows[k.number] for k in first], [rows[k.number] for k in second]
    for cutoff in (10.0, 15.0):
        score = ipsae(pae, first, second, cutoff)
        as_the_book_says(score.forward, pae.matrix, a, b, cutoff)
        as_the_book_says(score.backward, pae.matrix, b, a, cutoff)
        assert 0.0 <= min(score.values) <= max(score.values) <= 1.0


def test_the_same_between_two_chains_of_a_real_matrix():
    st = trimer_structure()
    pae = Pae.read(trimer_file("pae"), st)
    first, second = ([k for k in st.residues() if k.chain == c] for c in "AB")
    a, b = ([i for i, c in enumerate(pae.chains) if c == chain] for chain in "AB")
    score = ipsae(pae, first, second)
    as_the_book_says(score.forward, pae.matrix, a, b, 10.0)
    as_the_book_says(score.backward, pae.matrix, b, a, 10.0)
    assert score.residues == (157, 157) and not score.same_chain and score.left_out == 0
    # No residue of this model is taken atom by atom: with the model or without it, the same.
    assert ipsae(pae, first, second, structure=st) == score and score.by_alpha_carbon == 0


def test_two_parts_of_one_chain_are_found_by_number_and_said_to_be_of_one_chain():
    """The models of ``testdata/interface_pae`` are one chain each. Cut in two, its parts carry other chain
    names in the structure than in the matrix, as the contacts of such a figure do."""
    from conftest import pae_case

    whole, first, second = domains("afdb_fah")
    pae, _, _ = domains("afdb_fah", pae_case("afdb_fah").structure)
    renamed = ipsae(pae, [ResidueKey("X", k.number) for k in first], [ResidueKey("Y", k.number) for k in second])
    named = ipsae(whole, first, second)
    assert renamed.same_chain and renamed.values == named.values and renamed.residues == named.residues
    assert [(w.aligned[1], w.n0, w.d0) for w in (renamed.forward, renamed.backward)] == [
        (w.aligned[1], w.n0, w.d0) for w in (named.forward, named.backward)]


# -- the interfaces of the repository ---------------------------------------------------------------------

def test_a_placed_interface_scores_high_and_one_that_is_not_placed_low():
    fah = ipsae(*domains("afdb_fah"))
    a3d = ipsae(*domains("afdb_apobec3d"))
    tectb = ipsae(*domains("afdb_tectb"))
    # A placed interface scores high, one that is not placed low, in both directions.
    assert min(fah.values) >= 0.8 and max(a3d.values) <= 0.3
    # Not zero where nothing is placed: residues near the cut follow each other in the chain.
    assert min(a3d.values) > 0.0
    # The one that is placed in part lies between the two, and is low: a few pairs of residues placed do
    # not make an interface placed.
    assert max(a3d.values) < min(tectb.values) and max(tectb.values) < min(fah.values)
    assert max(tectb.values) < 0.3
    st = trimer_structure()
    first, second = ([k for k in st.residues() if k.chain == c] for c in "AB")
    between = ipsae(Pae.read(trimer_file("pae"), st), first, second)
    assert min(between.values) >= 0.7 and abs(between.forward.value - between.backward.value) <= 0.05


def test_the_draws_of_the_trimer_agree_and_are_read_under_their_own_chain_names():
    st = trimer_structure()
    draws = read_pae([trimer_file("pae", k) for k in range(2)], st)
    assert isinstance(draws, PaeDraws) and draws.assignments[1] == {"A": "A", "B": "C", "C": "B"}
    first, second = ([k for k in st.residues() if k.chain == c] for c in "AB")
    each = ipsae_of(draws, first, second)
    assert len(each) == 2
    for way in ("forward", "backward"):
        values = [getattr(e, way).value for e in each]
        assert abs(values[0] - values[1]) <= 0.1
    # The second draw calls chain B "C": its score is that of its chains A and C.
    theirs = [ResidueKey("C", k.number) for k in second]
    assert each[1] == ipsae(draws.draws[1], first, theirs)
    assert ipsae_of(draws.draws[0], first, second) == (each[0],)


def test_a_draw_that_names_its_copies_otherwise_is_scored_as_the_chains_it_means():
    """Made up, so that reading by name would show: chain A against its neighbour, which the second draw
    calls C. Read by name, that draw would give the score of two chains that are far apart."""
    chains, numbers = tuple("AAABBBCCC"), (1, 2, 3) * 3
    names = ("GLY",) * 3 + ("ALA",) * 6                  # B and C are copies of one another, A is not
    base = np.array([[0.0, 0.0, 0.0], [3.8, 0.0, 0.0], [7.6, 1.0, 0.0]])
    near, far = base + np.array([0.0, 6.0, 0.0]), base + np.array([0.0, 0.0, 40.0])

    def draw(b_at, c_at, a_with: str, source: str) -> Pae:
        matrix = np.full((9, 9), 28.0)
        rows = {c: [i for i, x in enumerate(chains) if x == c] for c in "ABC"}
        matrix[np.ix_(rows["A"], rows[a_with])] = 2.0
        matrix[np.ix_(rows[a_with], rows["A"])] = 3.0
        return Pae(matrix, chains, numbers, source, names, np.vstack([base, b_at, c_at]))

    one, two = draw(near, far, "B", "draw 0"), draw(far, near, "C", "draw 1")
    draws = PaeDraws((one, two))
    assert draws.assignments == ({"A": "A", "B": "B", "C": "C"}, {"A": "A", "B": "C", "C": "B"})
    each = ipsae_of(draws, side("A", 3), side("B", 3))
    assert each[0].values == each[1].values == (pytest.approx(1 / 5), pytest.approx(1 / 10))
    assert ipsae(two, side("A", 3), side("B", 3)).values == (0.0, 0.0)       # by name: the wrong chain


# -- in a figure -----------------------------------------------------------------------------------------

def test_left_alone_nothing_is_computed_and_nothing_is_said():
    face = trimer(draws=1)
    res = face.render(style="schematic")
    assert face.ipsae == [] and "ipSAE" not in words(res) and "ipsae" not in res.svg
    assert "ipsae" not in res.graph.meta["interface"]
    assert "ipTM" not in words(res) and res.confidence["iptm"] is None


def test_asked_for_it_stands_under_the_title_with_what_it_is():
    face = trimer(draws=1, ipsae=True)
    (score,) = face.ipsae
    assert isinstance(score, InterfaceIpsae) and (score.first, score.second) == ("chain B", "chain A")
    assert score.cutoff == IPSAE_CUTOFF and len(score.draws) == 1
    st = trimer_structure()
    first, second = ([k for k in st.residues() if k.chain == c] for c in "BA")
    direct = ipsae(Pae.read(trimer_file("pae"), st), first, second)
    assert (score.forward, score.backward) == direct.values
    for style in ("schematic", "detailed"):
        res = face.render(style=style)
        said = words(res)
        assert res.status == "ok"
        assert f"ipSAE {score.forward:.2f} / {score.backward:.2f}" in said
        assert "ipSAE (Dunbrack 2025) is a score from 0 to 1" in said and "below 10 Å" in said
        assert "not from the contacts shown here" in said
        assert ("the first value is with the model aligned on a residue of chain B and scored over chain A, "
                "the second the reverse") in said
        assert "parts of one chain" not in said and "median" not in said.split("ipSAE (Dunbrack")[1]
        assert f'data-ipsae="{score.forward:.2f} {score.backward:.2f}"' in res.svg
        assert 'data-ipsae-cutoff="10"' in res.svg
        meta = res.graph.meta["interface"]["ipsae"]
        assert meta["cutoff"] == 10.0 and "draws" not in meta
        assert meta["scores"] == [
            {"aligned_on": "chain B", "scored_over": "chain A", "value": round(score.forward, 3)},
            {"aligned_on": "chain A", "scored_over": "chain B", "value": round(score.backward, 3)}]


def test_it_changes_nothing_else_about_the_figure():
    """The contacts, their strokes and their PAE tags are those of the figure without the score."""
    from scenetools import tags

    plain, scored = trimer(draws=1).render(style="schematic"), trimer(draws=1, ipsae=True).render(style="schematic")
    assert [t.content for t in tags(plain)] == [t.content for t in tags(scored)]
    assert plain.graph.to_dict()["contacts"] == scored.graph.to_dict()["contacts"]
    assert [w for w in scored.warnings] == [w for w in plain.warnings]


def test_it_needs_the_matrix():
    with pytest.raises(BindviewError, match="ipSAE is computed from the model's PAE matrix, and none was given"):
        Interface(trimer_structure(), ["A", "B"], trimer_contacts(), ipsae=True)
    with pytest.raises(BindviewError, match="positive number"):
        Interface(trimer_structure(), ["A", "B"], trimer_contacts(), pae=trimer_file("pae"), ipsae=-1.0)


def test_a_whole_number_is_not_taken_for_yes_or_for_a_cut_off():
    """``ipsae=True`` asks for the usual cut-off of 10 Å, and ``1`` is ``True`` almost everywhere in
    Python: it is taken neither for yes nor for a cut-off of 1 Å. A cut-off is written with a point."""
    def face(asked):
        return Interface(trimer_structure(), ["A", "B"], trimer_contacts()[:6], pae=trimer_file("pae"), ipsae=asked,
                         model_confidence=None)

    for asked in (1, 0, 10):
        with pytest.raises(BindviewError, match=rf"ipsae={asked}: .*True.*or the cut-off in ångström as a number "
                                              rf"with a point \({asked}\.0\)"):
            face(asked)
    assert face(True).ipsae[0].cutoff == 10.0 and face(1.0).ipsae[0].cutoff == 1.0 and face(False).ipsae == []


def test_another_cut_off_is_said_in_the_figure():
    face = trimer(draws=1, ipsae=15.0)
    res = face.render(style="schematic")
    assert face.ipsae[0].cutoff == 15.0
    assert "below 15 Å" in words(res) and 'data-ipsae-cutoff="15"' in res.svg
    assert res.graph.meta["interface"]["ipsae"]["cutoff"] == 15.0


def test_of_several_draws_it_is_the_median_and_the_figure_gives_each():
    face = trimer(ipsae=True)                           # the two draws that are in the repository
    (score,) = face.ipsae
    assert len(score.draws) == 2
    assert score.forward == pytest.approx(np.median([d.forward.value for d in score.draws]))
    assert score.backward == pytest.approx(np.median([d.backward.value for d in score.draws]))
    res = face.render(style="schematic")
    said = words(res)
    each = ", ".join(f"{d.forward.value:.2f} / {d.backward.value:.2f}" for d in score.draws)
    assert f"Each value is the median of 2 draws of the model (draw by draw: {each})." in said
    meta = res.graph.meta["interface"]["ipsae"]
    assert (meta["draws"], meta["over_draws"]) == (2, "median")
    assert meta["scores"][0]["by_draw"] == [round(d.forward.value, 3) for d in score.draws]


def test_two_domains_of_one_chain_are_scored_and_the_figure_says_what_they_are():
    face = pae_case("afdb_apobec3d", fresh=True, ipsae=True)
    (score,) = face.ipsae
    assert max(score.forward, score.backward) <= 0.3 and score.draws[0].same_chain
    res = face.render(style="schematic")
    said = words(res)
    assert f"ipSAE {score.forward:.2f} / {score.backward:.2f}" in said
    assert ("Both sides are parts of one chain of this model. The score was defined for two chains, and residues "
            "that follow each other in a chain have a low PAE whatever its domains do.") in said
    # The warning of the PAE is the PAE's; the score brings no verdict of its own.
    assert len(res.warnings) == len(pae_case("afdb_apobec3d", fresh=True).render(style="schematic").warnings)


def test_three_in_a_row_have_two_scores_and_the_figure_says_which_is_which():
    st = trimer_structure()
    contacts = trimer_contacts("B", "A") + trimer_contacts("A", "C")
    face = Interface(st, ["B", "A", "C"], contacts, pae=trimer_file("pae"), ipsae=True)
    assert [(s.first, s.second) for s in face.ipsae] == [("chain A", "chain B"), ("chain A", "chain C")]
    res = face.render(style="schematic")
    said = words(res)
    one, two = face.ipsae
    assert f"ipSAE {one.forward:.2f} / {one.backward:.2f} · {two.forward:.2f} / {two.backward:.2f}" in said
    assert "The pairs are chain A with chain B and chain A with chain C, in that order." in said
    assert len(res.graph.meta["interface"]["ipsae"]["scores"]) == 4


def test_the_command_line_asks_for_it(tmp_path, capsys):
    from bindview.cli import main as cli
    from bindview.interface import dump_interface_contacts

    tsv, out = tmp_path / "contacts.tsv", tmp_path / "out.svg"
    tsv.write_text(dump_interface_contacts(trimer_contacts()))
    common = ["interface", str(trimer_file("model")), "--between", "A", "B", "--contacts", str(tsv), "-o", str(out),
              "-q", "--style", "schematic", "--pae", str(trimer_file("pae"))]
    assert cli(common) == 0 and "ipsae" not in out.read_text() and "data-iptm" not in out.read_text()
    assert cli(common + ["--ipsae", "--json", "--model-confidence", str(trimer_file("confidence"))]) == 0
    svg = out.read_text()
    assert 'data-ipsae="' in svg and 'data-ipsae-cutoff="10"' in svg and 'data-iptm="0.96"' in svg
    assert '"ipsae": {' in capsys.readouterr().out
    assert cli(common + ["--ipsae", "15"]) == 0 and 'data-ipsae-cutoff="15"' in out.read_text()


# -- what the model says of itself -----------------------------------------------------------------------

def test_an_interface_takes_what_the_model_says_of_itself():
    """Boltz writes ipTM into ``confidence_<name>.json``, which is not found by itself: handed in, it
    stands under the title as it does in a ligand diagram."""
    given = json.loads(trimer_file("confidence").read_text())
    for handed in (str(trimer_file("confidence")), {"iptm": given["iptm"]}):
        face = Interface(trimer_structure(), ["A", "B"], trimer_contacts(), pae=trimer_file("pae"),
                         model_confidence=handed)
        res = face.render(style="schematic")
        assert res.confidence["iptm"] == pytest.approx(given["iptm"])
        assert f"ipTM {given['iptm']:.2f}" in words(res) and f'data-iptm="{given["iptm"]:.2f}"' in res.svg


def test_the_residue_a_score_is_aligned_on_is_named_with_its_chain_where_a_chain_was_cut():
    """The matrix of a chain cut into two parts names no chain for its rows; the aligned residue was given an empty
    chain name. It is named by the chain of the residue it stands for."""
    face = pae_case("afdb_fah", ipsae=True)
    (score,) = face.ipsae
    ways = [w for draw in score.draws for w in (draw.forward, draw.backward)]
    spine = face.spine.chains[0]                            # the forward score is aligned on the spine
    assert [w.aligned[0] for w in ways if w.aligned is not None] == [spine, ({"X", "Y"} - {spine}).pop()]
