# -*- coding: utf-8 -*-
#
# test_spikes.py

import unittest
import arbor as A

# to be able to run .py file from child directory
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

try:
    import options
except ModuleNotFoundError:
    from test import options

"""
all tests for the simulator wrapper
"""

# Test recipe art_spiker_recipe comprises three artificial spiking cells 

class art_spiker_recipe(A.recipe):
    def __init__(self):
        A.recipe.__init__(self)
        self.props = A.neuron_cable_properties()
        self.trains = [
                [0.8, 2, 2.1, 3],
                [0.4, 2, 2.2, 3.1, 4.5],
                [0.2, 2, 2.8, 3]]

    def num_cells(self):
        return 3

    def cell_kind(self, gid):
        return A.cell_kind.spike_source

    def connections_on(self, gid):
        return []

    def event_generators(self, gid):
        return []

    def global_properties(self, kind):
        return self.the_props

    def probes(self, gid):
        return []

    def cell_description(self, gid):
        return A.spike_source_cell("src", A.explicit_schedule(self.trains[gid]))


class Spikes(unittest.TestCase):
    # Helper for constructing a simulation from a recipe using default context and domain decomposition.
    def init_sim(self, recipe):
        context = A.context()
        dd = A.partition_load_balance(recipe, context)
        return A.simulation(recipe, dd, context)

    # test that all spikes are sorted by time then by gid
    def test_spikes_sorted(self):
        sim = self.init_sim(art_spiker_recipe())
        sim.record(A.spike_recording.all)
        # run simulation in 5 steps, forcing 5 epochs
        sim.run(1, 0.01)
        sim.run(2, 0.01)
        sim.run(3, 0.01)
        sim.run(4, 0.01)
        sim.run(5, 0.01)

        spikes = sim.spikes()
        times = spikes["time"].tolist()
        gids = spikes["source"]["gid"].tolist()

        self.assertEqual([2, 1, 0, 0, 1, 2, 0, 1, 2, 0, 2, 1, 1], gids)
        self.assertEqual([0.2, 0.4, 0.8, 2., 2., 2., 2.1, 2.2, 2.8, 3., 3., 3.1, 4.5], times)

def suite():
    # specify class and test functions in tuple (here: all tests starting with 'test' from class Contexts
    suite = unittest.makeSuite(Spikes, ('test'))
    return suite

def run():
    v = options.parse_arguments().verbosity
    runner = unittest.TextTestRunner(verbosity = v)
    runner.run(suite())

if __name__ == "__main__":
    run()
