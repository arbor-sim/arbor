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
tests for (dynamically loaded) catalogues
"""

class Catalogues(unittest.TestCase):
    def test_nonexistent(self):
        with self.assertRaises(RuntimeError):
            arb.load_catalogue("_NO_EXIST_.cat")

    def test_shared_catalogue(self):
        try:
            cat = arb.load_catalogue("lib/bbp.cat")
        except:
            print("BBP catalogue not found. Are you running from build directory?")
            raise
        nms = cat.keys()
        nms.sort()
        exp = arb.bbp_catalogue().keys()
        exp.sort()
        self.assertEqual(nms, exp, "Expected equal names.")
        mch = cat['Im']
        prm = list(mch.parameters.keys())
        self.assertEqual(prm, ['gImbar'], "Expected equal parameters on mechanism 'Im'.")

def suite():
    # specify class and test functions in tuple (here: all tests starting with 'test' from class Contexts
    suite = unittest.makeSuite(Catalogues, ('test'))
    return suite

def run():
    v = options.parse_arguments().verbosity
    runner = unittest.TextTestRunner(verbosity = v)
    runner.run(suite())

if __name__ == "__main__":
    run()
