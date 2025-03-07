# -*- coding: utf-8 -*-
#
# test_spikes.py

import unittest
import arbor as A
from arbor import units as U
from .. import fixtures

"""
all tests for the simulator wrapper
"""


class TestSpikes(unittest.TestCase):
    # test that all spikes are sorted by time then by gid
    @fixtures.single_context()
    def test_spikes_sorted(self, single_context):
        if A.config()["profiling"]:
            A.profiler_clear()
        rec = fixtures.art_spiker_recipe()
        dd = A.partition_load_balance(rec, single_context)
        sim = A.simulation(rec, single_context, dd)
        sim.record(A.spike_recording.all)
        # run simulation in 5 steps, forcing 5 epochs
        sim.run(1 * U.ms, 0.01 * U.ms)
        sim.run(2 * U.ms, 0.01 * U.ms)
        sim.run(3 * U.ms, 0.01 * U.ms)
        sim.run(4 * U.ms, 0.01 * U.ms)
        sim.run(5 * U.ms, 0.01 * U.ms)

        spikes = sim.spikes()
        times = spikes["time"].tolist()
        gids = spikes["source"]["gid"].tolist()

        self.assertEqual([2, 1, 0, 0, 1, 2, 0, 1, 2, 0, 2, 1, 1], gids)
        self.assertEqual(
            [0.2, 0.4, 0.8, 2.0, 2.0, 2.0, 2.1, 2.2, 2.8, 3.0, 3.0, 3.1, 4.5], times
        )
        if A.config()["profiling"]:
            A.profiler_clear()
