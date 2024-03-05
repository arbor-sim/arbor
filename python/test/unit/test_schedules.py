# -*- coding: utf-8 -*-
#
# test_schedules.py

import unittest

import arbor as A
from arbor import units as U

"""
all tests for schedules (regular, explicit, poisson)
"""


class TestRegularSchedule(unittest.TestCase):
    def test_none_ctor_regular_schedule(self):
        rs = A.regular_schedule(tstart=0 * U.ms, dt=0.1 * U.ms, tstop=None)
        self.assertEqual(rs.dt, 0.1 * U.ms)

    def test_tstart_dt_tstop_ctor_regular_schedule(self):
        rs = A.regular_schedule(10.0 * U.ms, 1.0 * U.ms, 20.0 * U.ms)
        self.assertEqual(rs.tstart, 10.0 * U.ms)
        self.assertEqual(rs.dt, 1.0 * U.ms)
        self.assertEqual(rs.tstop, 20.0 * U.ms)

    def test_set_tstart_dt_tstop_regular_schedule(self):
        rs = A.regular_schedule(0.1 * U.ms)
        self.assertAlmostEqual(rs.dt.value_as(U.ms), 0.1, places=1)
        rs.tstart = 17.0 * U.ms
        rs.dt = 0.5 * U.ms
        rs.tstop = 42.0 * U.ms
        self.assertEqual(rs.tstart, 17.0 * U.ms)
        self.assertAlmostEqual(rs.dt.value_as(U.ms), 0.5, places=1)
        self.assertEqual(rs.tstop, 42.0 * U.ms)

    def test_events_regular_schedule(self):
        expected = [0, 0.25, 0.5, 0.75, 1.0]
        rs = A.regular_schedule(tstart=0.0 * U.ms, dt=0.25 * U.ms, tstop=1.25 * U.ms)
        self.assertEqual(expected, rs.events(0.0, 1.25))
        self.assertEqual(expected, rs.events(0.0, 5.0))
        self.assertEqual([], rs.events(5.0, 10.0))

    def test_exceptions_regular_schedule(self):
        with self.assertRaisesRegex(
            RuntimeError, "tstart must be a non-negative number"
        ):
            A.regular_schedule(tstart=-1.0 * U.ms, dt=0.1 * U.ms)
        with self.assertRaisesRegex(RuntimeError, "dt must be a positive number"):
            A.regular_schedule(dt=-0.1 * U.ms)
        with self.assertRaisesRegex(RuntimeError, "dt must be a positive number"):
            A.regular_schedule(dt=0 * U.ms)
        with self.assertRaises(TypeError):
            A.regular_schedule(dt=None)
        with self.assertRaises(TypeError):
            A.regular_schedule(dt="dt")
        with self.assertRaises(TypeError):
            A.regular_schedule(tstart=0 * U.ms, dt=0.1 * U.ms, tstop="tstop")
        with self.assertRaisesRegex(RuntimeError, "t0 must be a non-negative number"):
            rs = A.regular_schedule(0.0 * U.ms, 1.0 * U.ms, 10.0 * U.ms)
            rs.events(-1, 0)
        with self.assertRaisesRegex(RuntimeError, "t1 must be a non-negative number"):
            rs = A.regular_schedule(0.0 * U.ms, 1.0 * U.ms, 10.0 * U.ms)
            rs.events(0, -10)


class TestExplicitSchedule(unittest.TestCase):
    def test_times_contor_explicit_schedule(self):
        es = A.explicit_schedule([t * U.ms for t in range(1, 6)])
        self.assertEqual(es.events(0, 1000000), [1, 2, 3, 4, 5])

    def test_events_explicit_schedule(self):
        times = [0.1, 0.3, 1.0, 2.2, 1.25, 1.7]
        expected = [0.1, 0.3, 1.0]
        es = A.explicit_schedule([t * U.ms for t in times])
        for i in range(len(expected)):
            self.assertAlmostEqual(expected[i], es.events(0.0, 1.25)[i], places=2)
        expected = [0.3, 1.0, 1.25, 1.7]
        for i in range(len(expected)):
            self.assertAlmostEqual(expected[i], es.events(0.3, 1.71)[i], places=2)

    def test_exceptions_explicit_schedule(self):
        with self.assertRaises(RuntimeError):
            A.explicit_schedule([-1 * U.ms])
        with self.assertRaises(TypeError):
            A.explicit_schedule(["times"])
        with self.assertRaises(TypeError):
            A.explicit_schedule([None])
        with self.assertRaises(TypeError):
            A.explicit_schedule([[1, 2, 3]])
        with self.assertRaisesRegex(RuntimeError, "t1 must be a non-negative number"):
            rs = A.regular_schedule(0.1 * U.ms)
            rs.events(1.0, -1.0)


class TestPoissonSchedule(unittest.TestCase):
    def test_freq_poisson_schedule(self):
        ps = A.poisson_schedule(42.0 * U.kHz)
        self.assertEqual(ps.freq, 42.0 * U.kHz)

    def test_freq_tstart_contor_poisson_schedule(self):
        ps = A.poisson_schedule(freq=5.0 * U.kHz, tstart=4.3 * U.ms)
        self.assertEqual(ps.freq, 5.0 * U.kHz)
        self.assertEqual(ps.tstart, 4.3 * U.ms)

    def test_freq_seed_contor_poisson_schedule(self):
        ps = A.poisson_schedule(freq=5.0 * U.kHz, seed=42)
        self.assertEqual(ps.freq, 5.0 * U.kHz)
        self.assertEqual(ps.seed, 42)

    def test_tstart_freq_seed_contor_poisson_schedule(self):
        ps = A.poisson_schedule(tstart=10.0 * U.ms, freq=100.0 * U.kHz, seed=1000)
        self.assertEqual(ps.tstart, 10.0 * U.ms)
        self.assertEqual(ps.freq, 100.0 * U.kHz)
        self.assertEqual(ps.seed, 1000)

    def test_events_poisson_schedule(self):
        expected = [17.4107, 502.074, 506.111, 597.116]
        ps = A.poisson_schedule(tstart=0.0 * U.ms, freq=0.01 * U.kHz, seed=0)
        for i in range(len(expected)):
            self.assertAlmostEqual(
                expected[i], ps.events(0.0 * U.ms, 600.0 * U.ms)[i], places=3
            )
        expected = [
            5030.22,
            5045.75,
            5069.84,
            5091.56,
            5182.17,
            5367.3,
            5566.73,
            5642.13,
            5719.85,
            5796,
            5808.33,
        ]
        for i in range(len(expected)):
            self.assertAlmostEqual(
                expected[i], ps.events(5000.0 * U.ms, 6000.0 * U.ms)[i], places=2
            )

    def test_exceptions_poisson_schedule(self):
        with self.assertRaises(TypeError):
            A.poisson_schedule()
        with self.assertRaises(TypeError):
            A.poisson_schedule(tstart=10.0 * U.ms)
        with self.assertRaises(TypeError):
            A.poisson_schedule(seed=1432)
        with self.assertRaisesRegex(
            RuntimeError, "tstart must be a non-negative number"
        ):
            A.poisson_schedule(freq=34.0 * U.kHz, tstart=-10.0 * U.ms)
        with self.assertRaises(TypeError):
            A.poisson_schedule(freq=34.0 * U.kHz, tstart=None)
        with self.assertRaises(TypeError):
            A.poisson_schedule(freq=34.0, tstart="tstart")
        with self.assertRaisesRegex(
            RuntimeError, "frequency must be a non-negative number"
        ):
            A.poisson_schedule(freq=-100.0 * U.kHz)
        with self.assertRaises(TypeError):
            A.poisson_schedule(freq="freq")
        with self.assertRaises(TypeError):
            A.poisson_schedule(freq=34.0 * U.kHz, seed=-1)
        with self.assertRaises(TypeError):
            A.poisson_schedule(freq=34.0 * U.kHz, seed=10.0)
        with self.assertRaises(TypeError):
            A.poisson_schedule(freq=34.0 * U.kHz, seed="seed")
        with self.assertRaises(TypeError):
            A.poisson_schedule(freq=34.0 * U.kHz, seed=None)
        with self.assertRaisesRegex(RuntimeError, "t0 must be a non-negative number"):
            ps = A.poisson_schedule(tstart=0 * U.ms, freq=0.01 * U.kHz)
            ps.events(-1.0 * U.ms, 1.0 * U.ms)
        with self.assertRaisesRegex(RuntimeError, "t1 must be a non-negative number"):
            ps = A.poisson_schedule(tstart=0 * U.ms, freq=0.01 * U.kHz)
            ps.events(1.0 * U.ms, -1.0 * U.ms)
        with self.assertRaises(TypeError):
            ps = A.poisson_schedule(tstart=0 * U.ms, freq=0.1 * U.kHz, tstop="tstop")
            ps.events(1.0 * U.ms, -1.0 * U.ms)

    def test_tstop_poisson_schedule(self):
        tstop = 50
        events = A.poisson_schedule(
            tstart=0.0 * U.ms, freq=1 * U.kHz, seed=0, tstop=tstop * U.ms
        ).events(0 * U.ms, 100 * U.ms)
        self.assertTrue(max(events) < tstop)
