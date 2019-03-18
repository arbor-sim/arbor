# -*- coding: utf-8 -*-
#
# test_event_generators.py

import unittest
import numpy as np

import arbor as arb

# to be able to run .py file from child directory
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

try:
    import options
except ModuleNotFoundError:
    from test import options

"""
all tests for event generators (regular, explicit, poisson)
"""

class RegularSchedule(unittest.TestCase):
    def test_default(self):
        rs = arb.regular_schedule()
        self.assertEqual(rs.tstart, np.finfo(np.float32).max)
        self.assertEqual(rs.dt, 0)
        self.assertEqual(rs.tstop, np.finfo(np.float32).max)
        self.assertEqual(rs.tstart, rs.tstop)

    def test_dt_contor(self):
        rs = arb.regular_schedule(0.025)
        self.assertEqual(rs.tstart, 0)
        self.assertEqual(float("{0:.3f}".format(rs.dt)), 0.025)
        self.assertEqual(rs.tstop, np.finfo(np.float32).max)

    def test_tstart_dt_tstop_contor(self):
        rs = arb.regular_schedule(10,1,20)
        self.assertEqual(rs.tstart, 10)
        self.assertEqual(rs.dt, 1)
        self.assertEqual(rs.tstop, 20)

    def test_set_tstart_dt_tstop(self):
        rs = arb.regular_schedule()
        rs.tstart = 17
        rs.dt = 0.5
        rs.tstop = 42
        self.assertEqual(rs.tstart, 17)
        self.assertEqual(rs.dt, 0.5)
        self.assertEqual(rs.tstop, 42)

    def test_event_generator(self):
        cm = arb.cell_member()
        cm.gid = 0
        cm.index = 23
        rs = arb.regular_schedule(17,1,42)
        rg = arb.event_generator(cm, 0.01, rs)
        self.assertEqual(rg.target.gid, 0)
        self.assertEqual(rg.target.index, 23)
        self.assertEqual(rg.weight, 0.01)

class ExplicitSchedule(unittest.TestCase):
    def test_default(self):
        es = arb.explicit_schedule()
        self.assertEqual(es.times, [])

    def test_times_contor(self):
        es = arb.explicit_schedule([1, 2, 3, 4.5])
        self.assertEqual(es.times, [1, 2, 3, 4.5])

    def test_set_times(self):
        es = arb.explicit_schedule()
        es.times = [42, 43, 44, 55.5, 100]
        self.assertEqual(es.times, [42, 43, 44, 55.5, 100])

    def test_event_generator(self):
        cm = arb.cell_member()
        cm.gid = 0
        cm.index = 42
        es = arb.explicit_schedule([0,1,2,3,4.4])
        rg = arb.event_generator(cm, -0.01, es)
        self.assertEqual(rg.target.gid, 0)
        self.assertEqual(rg.target.index, 42)
        self.assertEqual(rg.weight, -0.01)

class PoissonSchedule(unittest.TestCase):
    def test_default(self):
        ps = arb.poisson_schedule()
        self.assertEqual(ps.tstart, 0)
        self.assertEqual(ps.freq, 10)
        self.assertEqual(ps.seed, 0)

    def test_freq_seed_contor(self):
        ps = arb.poisson_schedule(5, 42)
        self.assertEqual(ps.tstart, 0)
        self.assertEqual(ps.freq, 5)
        self.assertEqual(ps.seed, 42)

    def test_tstart_freq_seed_contor(self):
       ps = arb.poisson_schedule(10, 100, 1000)
       self.assertEqual(ps.tstart, 10)
       self.assertEqual(ps.freq, 100)
       self.assertEqual(ps.seed, 1000)

    def test_set_tstart_freq_seed(self):
        ps = arb.poisson_schedule()
        ps.tstart = 4.5
        ps.freq = 5.5
        ps.seed = 83
        self.assertEqual(ps.tstart, 4.5)
        self.assertEqual(ps.freq, 5.5)
        self.assertEqual(ps.seed, 83)

    def test_event_generator(self):
        cm = arb.cell_member()
        cm.gid = 42
        cm.index = 42
        es = arb.explicit_schedule([0,1,2,3,4.4])
        rg = arb.event_generator(cm, -0.01, es)
        self.assertEqual(rg.target.gid, 42)
        self.assertEqual(rg.target.index, 42)
        self.assertEqual(rg.weight, -0.01)

def suite():
    # specify class and test functions in tuple (here: all tests starting with 'test' from classes RegularSchedule, ExplicitSchedule and PoissonSchedule
    suite = unittest.TestSuite()
    suite.addTests(unittest.makeSuite(RegularSchedule, ('test')))
    suite.addTests(unittest.makeSuite(ExplicitSchedule, ('test')))
    suite.addTests(unittest.makeSuite(PoissonSchedule, ('test')))
    return suite

def run():
    v = options.parse_arguments().verbosity
    runner = unittest.TextTestRunner(verbosity = v)
    runner.run(suite())

if __name__ == "__main__":
    run()
