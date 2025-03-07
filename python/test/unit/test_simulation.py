import arbor as A
from arbor import units as U
import unittest

from .. import fixtures

"""
End to end tests at simulation level.
"""


class DelayRecipe(A.recipe):
    """
    Construct a simple network with configurable axonal delay.
    """

    def __init__(self, delay):
        A.recipe.__init__(self)
        self.delay = delay

    def num_cells(self):
        return 2

    def cell_kind(self, _):
        return A.cell_kind.cable

    def cell_description(self, _):
        # morphology
        tree = A.segment_tree()
        tree.append(A.mnpos, A.mpoint(-1, 0, 0, 1), A.mpoint(1, 0, 0, 1), tag=1)

        decor = (
            A.decor()
            .place("(location 0 0.5)", A.synapse("expsyn"), "syn")
            .place("(location 0 0.5)", A.threshold_detector(-15 * U.mV), "det")
            .paint("(all)", A.density("hh"))
        )

        return A.cable_cell(tree, decor, A.label_dict())

    def connections_on(self, gid):
        src = (gid + 1) % self.num_cells()
        return [A.connection((src, "det"), "syn", 0.42, self.delay)]

    def probes(self, _):
        return [A.cable_probe_membrane_voltage("(location 0 0.5)", "Um")]

    def global_properties(self, _):
        return A.neuron_cable_properties()


class TestDelayNetwork(unittest.TestCase):
    @fixtures.single_context()
    def test_zero_delay(self, single_context):
        if A.config()["profiling"]:
            A.profiler_clear()
        rec = DelayRecipe(0.0 * U.ms)
        self.assertRaises(ValueError, A.simulation, rec, single_context)
        if A.config()["profiling"]:
            A.profiler_clear()

    @fixtures.single_context()
    def test_dt_half_delay(self, single_context):
        T = 1 * U.ms
        dt = 0.01 * U.ms
        rec = DelayRecipe(2 * dt)
        sim = A.simulation(rec, single_context)
        sim.run(T, dt)

    def dt_must_be_finite(self):
        T = 1 * U.ms
        dt = 0.01 * U.ms
        rec = DelayRecipe(2 * dt)
        sim = A.simulation(rec)
        self.assertRaises(ValueError, sim.run, T, 0 * U.ms)
        if A.config()["profiling"]:
            A.profiler_clear()
        self.assertRaises(ValueError, sim.run, T, 1.0 / 0.0 * U.ms)
        if A.config()["profiling"]:
            A.profiler_clear()
        self.assertRaises(ValueError, sim.run, 1.0 / 0.0 * U.ms, dt)
        if A.config()["profiling"]:
            A.profiler_clear()
