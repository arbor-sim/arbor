# -*- coding: utf-8 -*-
#
# test_identifiers.py

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
all tests for identifiers, indexes, kinds
"""

class CellMembers(unittest.TestCase):

    def test_gid_index_contor_cell_member(self):
        cm = arb.cell_member(17,42)
        self.assertEqual(cm.gid, 17)
        self.assertEqual(cm.index, 42)

    def test_set_git_index_cell_member(self):
        cm = arb.cell_member(0,0)
        cm.gid = 13
        cm.index = 23
        self.assertEqual(cm.gid, 13)
        self.assertEqual(cm.index, 23)

def suite():
    # specify class and test functions in tuple (here: all tests starting with 'test' from class Contexts
    suite = unittest.makeSuite(CellMembers, ('test'))
    return suite

def run():
    v = options.parse_arguments().verbosity
    runner = unittest.TextTestRunner(verbosity = v)
    runner.run(suite())

if __name__ == "__main__":
    run()
