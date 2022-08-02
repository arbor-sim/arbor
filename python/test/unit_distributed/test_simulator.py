# -*- coding: utf-8 -*-
#
# test_simulator.py

import unittest
import arbor as A
from .. import cases

mpi_enabled = A.__config__["mpi"]

"""
test for MPI distribution of spike recording
"""


class lifN_recipe(A.recipe):
    def __init__(self, n_cell):
        A.recipe.__init__(self)
        self.n_cell = n_cell
        self.props = A.neuron_cable_properties()

    def num_cells(self):
        return self.n_cell

    def cell_kind(self, gid):
        return A.cell_kind.lif

    def connections_on(self, gid):
        return []

    def event_generators(self, gid):
        sched_dt = 0.25
        weight = 400
        return [
            A.event_generator("tgt", weight, A.regular_schedule(sched_dt))
            for gid in range(0, self.num_cells())
        ]

    def probes(self, gid):
        return []

    def global_properties(self, kind):
        return self.props

    def cell_description(self, gid):
        c = A.lif_cell("src", "tgt")
        if gid % 2 == 0:
            c.t_ref = 2
        else:
            c.t_ref = 4
        return c


@cases.skipIfNotDistributed()
class TestSimulator(unittest.TestCase):
    def init_sim(self):
        comm = A.mpi_comm()
        context = A.context(threads=1, gpu_id=None, mpi=comm)
        self.rank = context.rank
        self.ranks = context.ranks

        recipe = lifN_recipe(context.ranks)
        dd = A.partition_load_balance(recipe, context)

        # Confirm decomposition has gid 0 on rank 0, ..., gid N-1 on rank N-1.
        self.assertEqual(1, dd.num_local_cells)
        local_groups = dd.groups
        self.assertEqual(1, len(local_groups))
        self.assertEqual([self.rank], local_groups[0].gids)

        return A.simulation(recipe, context, dd)

    def test_local_spikes(self):
        sim = self.init_sim()
        sim.record(A.spike_recording.local)
        sim.run(9, 0.01)
        spikes = sim.spikes().tolist()

        # Everything should come from the one cell, gid == rank.
        self.assertEqual({(self.rank, 0)}, {s for s, t in spikes})

        times = sorted([t for s, t in spikes])
        if self.rank % 2 == 0:
            self.assertEqual([0, 2, 4, 6, 8], times)
        else:
            self.assertEqual([0, 4, 8], times)

    def test_global_spikes(self):
        sim = self.init_sim()
        sim.record(A.spike_recording.all)
        sim.run(9, 0.01)
        spikes = sim.spikes().tolist()

        expected = [
            ((s, 0), t)
            for s in range(0, self.ranks)
            for t in ([0, 2, 4, 6, 8] if s % 2 == 0 else [0, 4, 8])
        ]
        self.assertEqual(expected, sorted(spikes))
