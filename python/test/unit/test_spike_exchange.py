# -*- coding: utf-8 -*-
#
# test_spike_exchange.py

import unittest
import arbor as A
from arbor import units as U
from .. import fixtures

"""
tests for neighbourhood spike exchange; these use all MPI ranks if run with MPI
"""


def ring(rank, offset, num_ranks):
    return (rank + offset) % num_ranks


def neighborhood_shapes(rank, num_ranks):
    return {
        "none": [],
        # Nested: the next {-1, +1}, then the next {-2, ..., +2}.
        "nested": [
            [ring(rank, -1, num_ranks), ring(rank, 1, num_ranks)],
            [ring(rank, d, num_ranks) for d in (-2, -1, 1, 2)],
        ],
        # Asymmetric: we receive from our successor, but it does not receive from us.
        "successor": [[ring(rank, 1, num_ranks)]],
        "everyone": [list(range(num_ranks))],
    }


def run(context, spike_exchange=None):
    rec = fixtures.ring_distance_recipe(context.ranks)
    dd = A.partition_load_balance(rec, context)
    sim = A.simulation(rec, context, dd, spike_exchange=spike_exchange)
    sim.record(A.spike_recording.local)
    # Stops are not multiples of the epoch length; held back spikes must be flushed at each.
    for t in [3.3, 10, 31.7, 50]:
        sim.run(t * U.ms, 0.025 * U.ms)
    return sim.spikes().tolist()


class TestSpikeExchange(unittest.TestCase):
    def test_options(self):
        self.assertEqual([], A.neighborhood_spike_exchange().neighborhoods)

        ex = A.neighborhood_spike_exchange([[1, 2], [3]])
        self.assertEqual([[1, 2], [3]], ex.neighborhoods)
        self.assertEqual(
            "<arbor.neighborhood_spike_exchange: neighborhoods [[1, 2], [3]]>", repr(ex)
        )

        ex.neighborhoods = [[4]]
        self.assertEqual([[4]], ex.neighborhoods)

        ex = A.neighborhood_spike_exchange(neighborhoods=[[5]])
        self.assertEqual([[5]], ex.neighborhoods)
        self.assertFalse(ex.automatic)

        ex = A.neighborhood_spike_exchange(automatic=True)
        self.assertTrue(ex.automatic)
        self.assertEqual([], ex.neighborhoods)
        self.assertEqual("<arbor.neighborhood_spike_exchange: automatic>", repr(ex))

    # The local spikes of every rank must be the same as with the default exchange.
    @fixtures.context()
    def test_same_spikes(self, context):
        expected = run(context)
        self.assertGreater(len(expected), 0)
        shapes = neighborhood_shapes(context.rank, context.ranks)
        for name, levels in shapes.items():
            with self.subTest(neighborhoods=name):
                result = run(context, A.neighborhood_spike_exchange(levels))
                self.assertEqual(expected, result)
        with self.subTest(neighborhoods="automatic"):
            result = run(context, A.neighborhood_spike_exchange(automatic=True))
            self.assertEqual(expected, result)

    @fixtures.context()
    def test_levels(self, context):
        rec = fixtures.ring_distance_recipe(context.ranks)
        dd = A.partition_load_balance(rec, context)

        sim = A.simulation(rec, context, dd)
        self.assertEqual([], sim.spike_exchange_levels())
        # The shortest delay is 1 ms.
        self.assertAlmostEqual(0.5, sim.max_epoch_length())

        shapes = neighborhood_shapes(context.rank, context.ranks)
        options = [A.neighborhood_spike_exchange(levels) for levels in shapes.values()]
        options.append(A.neighborhood_spike_exchange(automatic=True))
        for ex in options:
            with self.subTest(spike_exchange=repr(ex)):
                sim = A.simulation(rec, context, dd, spike_exchange=ex)
                levels = sim.spike_exchange_levels()
                # Neighbourhoods, then the far level.
                self.assertEqual(
                    [False] * (len(levels) - 1) + [True], [lvl.far for lvl in levels]
                )
                # Every other rank is in exactly one level.
                ranks = sorted(r for lvl in levels for r in lvl.ranks)
                others = [r for r in range(context.ranks) if r != context.rank]
                self.assertEqual(others, ranks)

    @fixtures.context()
    def test_invalid_options(self, context):
        # Checked on all ranks, so that all of them raise.
        with self.assertRaises(RuntimeError):
            run(context, A.neighborhood_spike_exchange([[context.ranks]]))
        with self.assertRaises(RuntimeError):
            run(context, A.neighborhood_spike_exchange([[0]], automatic=True))
