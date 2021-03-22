# -*- coding: utf-8 -*-

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
Tests for decor and decoration wrappers.
TODO: Coverage for more than just iclamp.
"""

class DecorClasses(unittest.TestCase):
    def test_iclamp(self):
        # Constant amplitude iclamp:
        clamp = A.iclamp(10);
        self.assertEqual(0, clamp.frequency)
        self.assertEqual([(0, 10)], clamp.envelope)

        clamp = A.iclamp(10, 20);
        self.assertEqual(20, clamp.frequency)
        self.assertEqual([(0, 10)], clamp.envelope)

        # Square pulse:
        clamp = A.iclamp(100, 20, 3);
        self.assertEqual(0, clamp.frequency)
        self.assertEqual([(100, 3), (120, 3), (120, 0)], clamp.envelope)

        clamp = A.iclamp(100, 20, 3, 7);
        self.assertEqual(7, clamp.frequency)
        self.assertEqual([(100, 3), (120, 3), (120, 0)], clamp.envelope)

        # Explicit envelope:
        envelope = [(1, 10), (3, 30), (5, 50), (7, 0)]
        clamp = A.iclamp(envelope);
        self.assertEqual(0, clamp.frequency)
        self.assertEqual(envelope, clamp.envelope)

        clamp = A.iclamp(envelope, 7);
        self.assertEqual(7, clamp.frequency)
        self.assertEqual(envelope, clamp.envelope)

def suite():
    # specify class and test functions in tuple (here: all tests starting with 'test' from class Contexts
    suite = unittest.makeSuite(DecorClasses, ('test'))
    return suite

def run():
    v = options.parse_arguments().verbosity
    runner = unittest.TextTestRunner(verbosity = v)
    runner.run(suite())

if __name__ == "__main__":
    run()
