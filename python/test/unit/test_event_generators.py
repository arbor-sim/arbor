# -*- coding: utf-8 -*-
#
# test_event_generators.py

import unittest

import arbor as arb
from arbor import units as U
"""
all tests for event generators (regular, explicit, poisson)
"""


class TestEventGenerator(unittest.TestCase):
    def test_event_generator_regular_schedule(self):
        cm = arb.cell_local_label("tgt0")
        rs = arb.regular_schedule(2.0*U.ms, 1.0*U.ms, 100.0*U.ms)
        rg = arb.event_generator(cm, 3.14, rs)
        self.assertEqual(rg.target.label, "tgt0")
        self.assertEqual(rg.target.policy, arb.selection_policy.univalent)
        self.assertAlmostEqual(rg.weight, 3.14)

    def test_event_generator_explicit_schedule(self):
        cm = arb.cell_local_label("tgt1", arb.selection_policy.round_robin)
        es = arb.explicit_schedule([0*U.ms, 1*U.ms, 2*U.ms, 3*U.ms, 4.4*U.ms])
        eg = arb.event_generator(cm, -0.01, es)
        self.assertEqual(eg.target.label, "tgt1")
        self.assertEqual(eg.target.policy, arb.selection_policy.round_robin)
        self.assertAlmostEqual(eg.weight, -0.01)

    def test_event_generator_poisson_schedule(self):
        ps = arb.poisson_schedule(freq=10.0*U.kHz, seed=0)
        pg = arb.event_generator("tgt2", 42.0, ps)
        self.assertEqual(pg.target.label, "tgt2")
        self.assertEqual(pg.target.policy, arb.selection_policy.univalent)
        self.assertEqual(pg.weight, 42.0)
