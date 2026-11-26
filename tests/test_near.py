"""Distances between atoms a block of rows at a time: the numbers of the array of every pair, without the array."""

from __future__ import annotations

import tracemalloc

import numpy as np
import pytest

from bindview import near


@pytest.mark.parametrize("block", [near.BLOCK, 3, 50])
def test_the_blocks_give_the_numbers_of_the_array_of_every_pair(monkeypatch, block):
    """To the last bit, whatever the size of a block: one row at a time (3 numbers) or all at once."""
    monkeypatch.setattr(near, "BLOCK", block)
    rng = np.random.default_rng(7)
    a, b = rng.normal(scale=6.0, size=(37, 3)), rng.normal(scale=6.0, size=(23, 3))
    plain = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    assert np.array_equal(near.distance_matrix(a, b), plain)
    assert np.array_equal(near.nearest_distance(a, b), plain.min(axis=1))
    first, second, apart = near.pairs_within(a, b, 5.0)
    want = np.nonzero(plain <= 5.0)
    assert len(first) > 20
    assert np.array_equal(first, want[0]) and np.array_equal(second, want[1]) and np.array_equal(apart, plain[want])


def test_no_points_are_no_pairs():
    none, some = np.zeros((0, 3)), np.ones((4, 3))
    assert near.distance_matrix(none, some).shape == (0, 4) and near.distance_matrix(some, none).shape == (4, 0)
    assert np.isinf(near.nearest_distance(some, none)).all() and len(near.nearest_distance(none, some)) == 0
    assert all(len(part) == 0 for part in near.pairs_within(none, some, 9.0) + near.pairs_within(some, none, 9.0))


def test_the_atoms_of_a_long_chain_do_not_take_the_memory_of_every_pair():
    """Five thousand atoms against five thousand: the differences of every pair are 600 MB, which is what the
    table of a chain, the chain as a ligand and the report's look at a model asked for. A block is 8 MB."""
    rng = np.random.default_rng(11)
    a, b = rng.normal(scale=30.0, size=(5000, 3)), rng.normal(scale=30.0, size=(5000, 3))
    tracemalloc.start()
    try:
        nearest = near.nearest_distance(a, b)
        first, _, _ = near.pairs_within(a, a, 1.9)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert len(nearest) == 5000 and len(first) >= 5000          # (each atom is within reach of itself)
    assert peak < 60e6, f"{peak / 1e6:.0f} MB"


def test_the_table_of_a_chain_does_not_take_the_memory_of_every_pair():
    """An antibody on lysozyme, 4,936 atoms: for the table of the heavy chain against the others, the chain as a
    ligand and the table itself each held the differences of every pair of atoms."""
    from conftest import structure_of

    from bindview.distance_table import contact_table

    st = structure_of("fab_lysozyme")
    tracemalloc.start()
    try:
        rows = contact_table(st, "H", stacking=True)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert len(rows) > 20
    assert peak < 100e6, f"{peak / 1e6:.0f} MB"
