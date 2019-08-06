# -*- coding: utf-8 -*-
#
# test_schedules.py

import unittest

import arbor as arb

# to be able to run .py file from child directory
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

try:
    import options
except ModuleNotFoundError:
    from test import options

"""
all tests for schedules (regular, explicit, poisson)
"""

class RegularSchedule(unittest.TestCase):
    def test_none_contor_regular_schedule(self):
        rs = arb.regular_schedule(tstart=None, tstop=None)

    def test_tstart_dt_tstop_contor_regular_schedule(self):
        rs = arb.regular_schedule(10., 1., 20.)
        self.assertEqual(rs.tstart, 10.)
        self.assertEqual(rs.dt, 1.)
        self.assertEqual(rs.tstop, 20.)

    def test_set_tstart_dt_tstop_regular_schedule(self):
        rs = arb.regular_schedule()
        rs.tstart = 17.
        rs.dt = 0.5
        rs.tstop = 42.
        self.assertEqual(rs.tstart, 17.)
        self.assertAlmostEqual(rs.dt, 0.5, places = 1)
        self.assertEqual(rs.tstop, 42.)

    def test_events_regular_schedule(self):
        expected = [0, 0.25, 0.5, 0.75, 1.0]
        rs = arb.regular_schedule(tstart = 0., dt = 0.25, tstop = 1.25)
        self.assertEqual(expected, rs.events(0., 1.25))
        self.assertEqual(expected, rs.events(0., 5.))
        self.assertEqual([], rs.events(5., 10.))

    def test_exceptions_regular_schedule(self):
        with self.assertRaisesRegex(RuntimeError,
            "tstart must be a non-negative number, or None"):
            arb.regular_schedule(tstart = -1.)
        with self.assertRaisesRegex(RuntimeError,
            "dt must be a non-negative number"):
            arb.regular_schedule(dt = -0.1)
        with self.assertRaises(TypeError):
            arb.regular_schedule(dt = None)
        with self.assertRaises(TypeError):
            arb.regular_schedule(dt = 'dt')
        with self.assertRaisesRegex(RuntimeError,
            "tstop must be a non-negative number, or None"):
            arb.regular_schedule(tstop = 'tstop')
        with self.assertRaisesRegex(RuntimeError,
            "t0 must be a non-negative number"):
            rs = arb.regular_schedule(0., 1., 10.)
            rs.events(-1, 0)
        with self.assertRaisesRegex(RuntimeError,
            "t1 must be a non-negative number"):
            rs = arb.regular_schedule(0., 1., 10.)
            rs.events(0, -10)

class ExplicitSchedule(unittest.TestCase):
    def test_times_contor_explicit_schedule(self):
        es = arb.explicit_schedule([1, 2, 3, 4.5])
        self.assertEqual(es.times, [1, 2, 3, 4.5])

    def test_set_times_explicit_schedule(self):
        es = arb.explicit_schedule()
        es.times = [42, 43, 44, 55.5, 100]
        self.assertEqual(es.times, [42, 43, 44, 55.5, 100])

    def test_events_explicit_schedule(self):
        times = [0.1, 0.3, 1.0, 2.2, 1.25, 1.7]
        expected = [0.1, 0.3, 1.0]
        es = arb.explicit_schedule(times)
        for i in range(len(expected)):
            self.assertAlmostEqual(expected[i], es.events(0., 1.25)[i], places = 2)
        expected = [0.3, 1.0, 1.25, 1.7]
        for i in range(len(expected)):
            self.assertAlmostEqual(expected[i], es.events(0.3, 1.71)[i], places = 2)

    def test_exceptions_explicit_schedule(self):
        with self.assertRaisesRegex(RuntimeError,
            "explicit time schedule cannot contain negative values"):
            arb.explicit_schedule([-1])
        with self.assertRaises(TypeError):
            arb.explicit_schedule(['times'])
        with self.assertRaises(TypeError):
            arb.explicit_schedule([None])
        with self.assertRaises(TypeError):
            arb.explicit_schedule([[1,2,3]])
        with self.assertRaisesRegex(RuntimeError,
            "t0 must be a non-negative number"):
            rs = arb.regular_schedule()
            rs.events(-1., 1.)
        with self.assertRaisesRegex(RuntimeError,
            "t1 must be a non-negative number"):
            rs = arb.regular_schedule()
            rs.events(1., -1.)

class PoissonSchedule(unittest.TestCase):
    def test_freq_seed_contor_poisson_schedule(self):
        ps = arb.poisson_schedule(freq = 5., seed = 42)
        self.assertEqual(ps.freq, 5.)
        self.assertEqual(ps.seed, 42)

    def test_tstart_freq_seed_contor_poisson_schedule(self):
        ps = arb.poisson_schedule(10., 100., 1000)
        self.assertEqual(ps.tstart, 10.)
        self.assertEqual(ps.freq, 100.)
        self.assertEqual(ps.seed, 1000)

    def test_set_tstart_freq_seed_poisson_schedule(self):
        ps = arb.poisson_schedule()
        ps.tstart = 4.5
        ps.freq = 5.5
        ps.seed = 83
        self.assertAlmostEqual(ps.tstart, 4.5)
        self.assertAlmostEqual(ps.freq, 5.5)
        self.assertEqual(ps.seed, 83)

    def test_events_poisson_schedule(self):
        expected = [17.4107, 502.074, 506.111, 597.116]
        ps = arb.poisson_schedule(0., 10., 0)
        for i in range(len(expected)):
            self.assertAlmostEqual(expected[i], ps.events(0., 600.)[i], places = 3)
        expected = [5030.22, 5045.75, 5069.84, 5091.56, 5182.17, 5367.3, 5566.73, 5642.13, 5719.85, 5796, 5808.33]
        for i in range(len(expected)):
            self.assertAlmostEqual(expected[i], ps.events(5000., 6000.)[i], places = 2)

    def test_exceptions_poisson_schedule(self):
        with self.assertRaisesRegex(RuntimeError,
            "tstart must be a non-negative number"):
            arb.poisson_schedule(tstart = -10.)
        with self.assertRaises(TypeError):
            arb.poisson_schedule(tstart = None)
        with self.assertRaises(TypeError):
            arb.poisson_schedule(tstart = 'tstart')
        with self.assertRaisesRegex(RuntimeError,
            "frequency must be a non-negative number"):
            arb.poisson_schedule(freq = -100.)
        with self.assertRaises(TypeError):
            arb.poisson_schedule(freq = 'freq')
        with self.assertRaises(TypeError):
            arb.poisson_schedule(seed = -1)
        with self.assertRaises(TypeError):
            arb.poisson_schedule(seed = 10.)
        with self.assertRaises(TypeError):
            arb.poisson_schedule(seed = 'seed')
        with self.assertRaises(TypeError):
            arb.poisson_schedule(seed = None)
        with self.assertRaisesRegex(RuntimeError,
            "t0 must be a non-negative number"):
            ps = arb.poisson_schedule()
            ps.events(-1., 1.)
        with self.assertRaisesRegex(RuntimeError,
            "t1 must be a non-negative number"):
            ps = arb.poisson_schedule()
            ps.events(1., -1.)

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
