# -*- coding: utf-8 -*-
#
# test_identifiers.py

import unittest

import arbor as arb


"""
all tests for identifiers, indexes, kinds
"""


class TestCellMembers(unittest.TestCase):
    def test_gid_index_ctor_cell_member(self):
        cm = arb.cell_member(17, 42)
        self.assertEqual(cm.gid, 17)
        self.assertEqual(cm.index, 42)

    def test_set_gid_index_cell_member(self):
        cm = arb.cell_member(0, 0)
        cm.gid = 13
        cm.index = 23
        self.assertEqual(cm.gid, 13)
        self.assertEqual(cm.index, 23)
