import unittest

import pyarb as arb

class test_cell_member(unittest.TestCase):
    def test_gid_index(self):
        i = arb.cell_member(17, 42)
        self.assertEqual(i.gid, 17)
        self.assertEqual(i.index, 42)

        i.gid = 13
        i.index = 23
        self.assertEqual(i.gid, 13)
        self.assertEqual(i.index, 23)

if __name__ == '__main__':
    unittest.main()
