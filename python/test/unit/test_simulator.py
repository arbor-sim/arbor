# -*- coding: utf-8 -*-
#
# test_simulator.py

import unittest
import numpy as np
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

# Test recipe cc2 comprises two cable cells and some probes.

class cc2_recipe(A.recipe):
    def __init__(self):
        A.recipe.__init__(self)
        st = A.segment_tree()
        i = st.append(A.mnpos, (0, 0, 0, 10), (1, 0, 0, 10), 1)
        st.append(i, (1, 3, 0, 5), 1)
        st.append(i, (1, -4, 0, 3), 1)
        self.the_morphology = A.morphology(st)
        self.the_cat = A.default_catalogue()
        self.the_props = A.neuron_cable_properties()
        self.the_props.register(self.the_cat)

    def num_cells(self):
        return 2

    def cell_kind(self, gid):
        return A.cell_kind.cable

    def connections_on(self, gid):
        return []

    def event_generators(self, gid):
        return []

    def global_properties(self, kind):
        return self.the_props

    def probes(self, gid):
        # Cell 0 has three voltage probes:
        #     0, 0: end of branch 1
        #     0, 1: end of branch 2
        #     0, 2: all terminal points
        # Values sampled from (0, 0) and (0, 1) should correspond
        # to the values sampled from (0, 2).

        # Cell 1 has whole cell probes:
        #     0, 0: all membrane voltages
        #     0, 1: all expsyn state variable 'g'

        if gid==0:
            return [A.cable_probe_membrane_voltage('(location 1 1)'),
                    A.cable_probe_membrane_voltage('(location 2 1)'),
                    A.cable_probe_membrane_voltage('(terminal)')]
        elif gid==1:
            return [A.cable_probe_membrane_voltage_cell(),
                    A.cable_probe_point_state_cell('expsyn', 'g')]
        else:
            return []

    def cell_description(self, gid):
        c = A.cable_cell(self.the_morphology, A.label_dict())
        c.set_properties(Vm=0.0, cm=0.01, rL=30, tempK=300)
        c.paint('(all)', "pas")
        c.place('(location 0 0)', A.iclamp(current=10 if gid==0 else 20))
        c.place('(sum (on-branches 0.3) (location 0 0.6))', "expsyn")
        return c

# Test recipe lif2 comprises two independent LIF cells driven by a regular, rapid
# sequence of incoming spikes. The cells have differing refactory periods.

class lif2_recipe(A.recipe):
    def __init__(self):
        A.recipe.__init__(self)

    def num_cells(self):
        return 2

    def cell_kind(self, gid):
        return A.cell_kind.lif

    def connections_on(self, gid):
        return []

    def event_generators(self, gid):
        sched_dt = 0.25
        weight = 400
        return [A.event_generator((gid,0), weight, A.regular_schedule(sched_dt)) for gid in range(0, self.num_cells())]

    def probes(self, gid):
        return []

    def cell_description(self, gid):
        c = A.lif_cell()
        if gid==0:
            c.t_ref = 2
        if gid==1:
            c.t_ref = 4
        return c

class Simulator(unittest.TestCase):
    def init_sim(self, recipe):
        context = A.context()
        dd = A.partition_load_balance(recipe, context)
        return A.simulation(recipe, dd, context)

    def test_simple_run(self):
        sim = self.init_sim(cc2_recipe())
        sim.run(1.0, 0.01)

    def test_probe_meta(self):
        sim = self.init_sim(cc2_recipe())

        self.assertEqual([A.location(1, 1)], sim.probe_metadata((0, 0)))
        self.assertEqual([A.location(2, 1)], sim.probe_metadata((0, 1)))
        self.assertEqual([A.location(1, 1), A.location(2, 1)], sorted(sim.probe_metadata((0, 2)), key=lambda x:(x.branch, x.pos)))

        # Default CV policy is one per branch, which also gives a tivial CV over the branch point.
        # Expect metadata cables to be one for each full branch, plus three length-zero cables corresponding to the branch point.
        self.assertEqual([A.cable(0, 0, 1), A.cable(0, 1, 1), A.cable(1, 0, 0), A.cable(1, 0, 1), A.cable(2, 0, 0), A.cable(2, 0, 1)],
                sorted(sim.probe_metadata((1,0))[0], key=lambda x:(x.branch, x.prox, x.dist)))

        # Four expsyn synapses; the two on branch zero should be coalesced, giving a multiplicity of 2.
        # Expect entries to be in target index order.
        m11 = sim.probe_metadata((1,1))[0]
        self.assertEqual(4, len(m11))
        self.assertEqual([0, 1, 2, 3], [x.target for x in m11])
        self.assertEqual([2, 2, 1, 1], [x.multiplicity for x in m11])
        self.assertEqual([A.location(0, 0.3), A.location(0, 0.6), A.location(1, 0.3), A.location(2, 0.3)], [x.location for x in m11])

    def test_probe_scalar_recorders(self):
        sim = self.init_sim(cc2_recipe())
        ts = [0, 0.1, 0.3, 0.7]
        h = sim.sample((0, 0), A.explicit_schedule(ts))
        dt = 0.01
        sim.run(10., dt)
        s, meta = sim.samples(h)[0]
        self.assertEqual(A.location(1, 1), meta)
        for i, t in enumerate(s[:,0]):
            self.assertLess(abs(t-ts[i]), dt)

        sim.remove_sampler(h)
        sim.reset()
        h = sim.sample(A.cell_member(0, 0), A.explicit_schedule(ts), A.sampling_policy.exact)
        sim.run(10., dt)
        s, meta = sim.samples(h)[0]
        for i, t in enumerate(s[:,0]):
            self.assertEqual(t, ts[i])


    def test_probe_multi_scalar_recorders(self):
        sim = self.init_sim(cc2_recipe())
        ts = [0, 0.1, 0.3, 0.7]
        h0 = sim.sample((0, 0), A.explicit_schedule(ts))
        h1 = sim.sample((0, 1), A.explicit_schedule(ts))
        h2 = sim.sample((0, 2), A.explicit_schedule(ts))

        dt = 0.01
        sim.run(10., dt)

        r0 = sim.samples(h0)
        self.assertEqual(1, len(r0))
        s0, meta0 = r0[0]

        r1 = sim.samples(h1)
        self.assertEqual(1, len(r1))
        s1, meta1 = r1[0]

        r2 = sim.samples(h2)
        self.assertEqual(2, len(r2))
        s20, meta20 = r2[0]
        s21, meta21 = r2[1]

        # Probe id (0, 2) has probes over the two locations that correspond to probes (0, 0) and (0, 1).

        # (order is not guaranteed to line up though)
        if meta20==meta0:
            self.assertEqual(meta1, meta21)
            self.assertTrue((s0[:,1]==s20[:,1]).all())
            self.assertTrue((s1[:,1]==s21[:,1]).all())
        else:
            self.assertEqual(meta1, meta20)
            self.assertTrue((s1[:,1]==s20[:,1]).all())
            self.assertEqual(meta0, meta21)
            self.assertTrue((s0[:,1]==s21[:,1]).all())

    def test_probe_vector_recorders(self):
        sim = self.init_sim(cc2_recipe())
        ts = [0, 0.1, 0.3, 0.7]
        h0 = sim.sample((1, 0), A.explicit_schedule(ts), A.sampling_policy.exact)
        h1 = sim.sample((1, 1), A.explicit_schedule(ts), A.sampling_policy.exact)
        sim.run(10., 0.01)

        # probe (1, 0) is the whole cell voltage; expect time + 6 sample values per row in returned data (see test_probe_meta above).

        s0, meta0 = sim.samples(h0)[0]
        self.assertEqual(6, len(meta0))
        self.assertEqual((len(ts), 7), s0.shape)
        for i, t in enumerate(s0[:,0]):
            self.assertEqual(t, ts[i])

        # probe (1, 1) is the 'g' state for all expsyn synapses.
        # With the default descretization, expect two synapses with multiplicity 2 and two with multiplicity 1.

        s1, meta1 = sim.samples(h1)[0]
        self.assertEqual(4, len(meta1))
        self.assertEqual((len(ts), 5), s1.shape)
        for i, t in enumerate(s1[:,0]):
            self.assertEqual(t, ts[i])

        meta1_mult = {(m.location.branch, m.location.pos): m.multiplicity for m in meta1}
        self.assertEqual(2, meta1_mult[(0, 0.3)])
        self.assertEqual(2, meta1_mult[(0, 0.6)])
        self.assertEqual(1, meta1_mult[(1, 0.3)])
        self.assertEqual(1, meta1_mult[(2, 0.3)])

    def test_spikes(self):
        sim = self.init_sim(lif2_recipe())
        sim.record(A.spike_recording.all)
        sim.run(21, 0.01)

        spikes = sim.spikes().tolist()
        s0 = sorted([t for s, t in spikes if s==(0, 0)])
        s1 = sorted([t for s, t in spikes if s==(1, 0)])

        self.assertEqual([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20], s0)
        self.assertEqual([0, 4, 8, 12, 16, 20], s1)

def suite():
    # specify class and test functions in tuple (here: all tests starting with 'test' from class Contexts
    suite = unittest.makeSuite(Simulator, ('test'))
    return suite

def run():
    v = options.parse_arguments().verbosity
    runner = unittest.TextTestRunner(verbosity = v)
    runner.run(suite())

if __name__ == "__main__":
    run()
