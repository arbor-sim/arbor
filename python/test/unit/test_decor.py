# -*- coding: utf-8 -*-

import unittest
import arbor as A
from arbor import units as U

"""
Tests for decor and decoration wrappers.
TODO: Coverage for more than just iclamp.
"""


class TestDecorClasses(unittest.TestCase):
    def test_iclamp(self):
        # Constant amplitude iclamp:
        clamp = A.iclamp(10 * U.nA)
        self.assertEqual(0, clamp.frequency)
        self.assertEqual([(0, 10)], clamp.envelope)

        clamp = A.iclamp(current=10 * U.nA, frequency=20 * U.kHz)
        self.assertEqual(20, clamp.frequency)
        self.assertEqual([(0, 10)], clamp.envelope)

        # Square pulse:
        clamp = A.iclamp(100 * U.ms, 20 * U.ms, 3 * U.nA)
        self.assertEqual(0, clamp.frequency)
        self.assertEqual([(100, 3), (120, 3), (120, 0)], clamp.envelope)

        clamp = A.iclamp(100 * U.ms, 20 * U.ms, 3 * U.nA, frequency=7 * U.kHz)
        self.assertEqual(7, clamp.frequency)
        self.assertEqual([(100, 3), (120, 3), (120, 0)], clamp.envelope)

        # Explicit envelope:
        envelope = [(1, 10), (3, 30), (5, 50), (7, 0)]
        clamp = A.iclamp([(t * U.ms, i * U.nA) for t, i in envelope])
        self.assertEqual(0, clamp.frequency)
        self.assertEqual(envelope, clamp.envelope)

        clamp = A.iclamp(
            [(t * U.ms, i * U.nA) for t, i in envelope], frequency=7 * U.kHz
        )
        self.assertEqual(7, clamp.frequency)
        self.assertEqual(envelope, clamp.envelope)
