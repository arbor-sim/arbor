# -*- coding: utf-8 -*-
#
# test_spikes.py

import unittest
import arbor as A
import numpy as np

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

def make_cable_cell():
    # (1) Create a morphology with a single (cylindrical) segment of length=diameter=6 μm
    tree = A.segment_tree()
    tree.append(A.mnpos, A.mpoint(-3, 0, 0, 3), A.mpoint(3, 0, 0, 3), tag=1)

    # (2) Define the soma and its midpoint
    labels = A.label_dict({'soma':   '(tag 1)',
                               'midpoint': '(location 0 0.5)'})

    # (3) Create cell and set properties
    decor = A.decor()
    decor.set_property(Vm=-40)
    decor.paint('"soma"', 'hh')
    decor.place('"midpoint"', A.iclamp( 10, 2, 0.8), "iclamp")
    decor.place('"midpoint"', A.spike_detector(-10), "detector")
    return A.cable_cell(tree, labels, decor)

# Test recipe art_spiker_recipe comprises three artificial spiking cells
class art_spiker_recipe(A.recipe):
    def __init__(self):
        A.recipe.__init__(self)
        self.the_props = A.neuron_cable_properties()
        self.trains = [
                [0.8, 2, 2.1, 3],
                [0.4, 2, 2.2, 3.1, 4.5],
                [0.2, 2, 2.8, 3]]

    def num_cells(self):
        return 4

    def cell_kind(self, gid):
        if gid < 3:
            return A.cell_kind.spike_source
        else:
            return A.cell_kind.cable

    def connections_on(self, gid):
        return []

    def event_generators(self, gid):
        return []

    def global_properties(self, kind):
        return self.the_props

    def probes(self, gid):
        if gid < 3:
            return []
        else:
            return [A.cable_probe_membrane_voltage('"midpoint"')]

    def cell_description(self, gid):
        if gid < 3:
            return A.spike_source_cell("src", A.explicit_schedule(self.trains[gid]))
        else:
            return make_cable_cell()



class Clear_samplers(unittest.TestCase):
    # Helper for constructing a simulation from a recipe using default context and domain decomposition.
    def init_sim(self, recipe):
        context = A.context()
        dd = A.partition_load_balance(recipe, context)
        return A.simulation(recipe, dd, context)

    # test that all spikes are sorted by time then by gid
    def test_spike_clearing(self):

        sim = self.init_sim(art_spiker_recipe())
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
        sim.run(3,0.01)
        spikes = sim.spikes()
        times_t  = spikes["time"].tolist()
        gids_t   = spikes["source"]["gid"].tolist()
        data_t, meta_t = sim.samples(handle)[0]

        # clear the samplers memory
        sim.clear_samplers()

        # Check if the memory is cleared
        spikes = sim.spikes()
        self.assertEqual(0, len(spikes["time"].tolist()))
        self.assertEqual(0, len(spikes["source"]["gid"].tolist()))
        data_test, meta_test = sim.samples(handle)[0]
        self.assertEqual(0,data_test.size)

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

def suite():
    # specify class and test functions in tuple (here: all tests starting with 'test' from class Contexts
    suite = unittest.makeSuite(Clear_samplers, ('test'))
    return suite

def run():
    v = options.parse_arguments().verbosity
    runner = unittest.TextTestRunner(verbosity = v)
    runner.run(suite())

if __name__ == "__main__":
    run()
