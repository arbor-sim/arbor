# -*- coding: utf-8 -*-
#
# test_spikes.py

import unittest
import arbor as A
import numpy as np

from .. import fixtures
from .. import cases

"""
all tests for the simulator wrapper
"""


@cases.skipIfDistributed()
class TestClearSamplers(unittest.TestCase):
    # test that all spikes are sorted by time then by gid
    @fixtures.art_spiking_sim()
    def test_spike_clearing(self, art_spiking_sim):
        sim = art_spiking_sim
        sim.record(A.spike_recording.all)
        handle = sim.sample((3, 0), A.regular_schedule(0.1))

        # baseline to test against Run in exactly the same stepping to make sure there are no rounding differences
        sim.run(3, 0.01)
        sim.run(5, 0.01)
        spikes = sim.spikes()
        times = spikes["time"].tolist()
        gids = spikes["source"]["gid"].tolist()
        data, meta = sim.samples(handle)[0]
        # reset the simulator
        sim.reset()

        # simulated with clearing the memory inbetween the steppings
        sim.run(3, 0.01)
        spikes = sim.spikes()
        times_t = spikes["time"].tolist()
        gids_t = spikes["source"]["gid"].tolist()
        data_t, meta_t = sim.samples(handle)[0]

        # clear the samplers memory
        sim.clear_samplers()

        # Check if the memory is cleared
        spikes = sim.spikes()
        self.assertEqual(0, len(spikes["time"].tolist()))
        self.assertEqual(0, len(spikes["source"]["gid"].tolist()))
        data_test, meta_test = sim.samples(handle)[0]
        self.assertEqual(0, data_test.size)

        # run the next part of the simulation
        sim.run(5, 0.01)
        spikes = sim.spikes()
        times_t.extend(spikes["time"].tolist())
        gids_t.extend(spikes["source"]["gid"].tolist())
        data_temp, meta_temp = sim.samples(handle)[0]
        data_t = np.concatenate((data_t, data_temp), 0)

        # check if results are the same
        self.assertEqual(gids, gids_t)
        self.assertEqual(times_t, times)
        self.assertEqual(list(data[:, 0]), list(data_t[:, 0]))
        self.assertEqual(list(data[:, 1]), list(data_t[:, 1]))
