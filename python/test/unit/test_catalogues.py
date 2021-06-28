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

class recipe(arb.recipe):
    def __init__(self):
        arb.recipe.__init__(self)
        self.tree = arb.segment_tree()
        self.tree.append(arb.mnpos, (0, 0, 0, 10), (1, 0, 0, 10), 1)
        self.props = arb.neuron_cable_properties()
        try:
            self.cat = arb.default_catalogue()
            self.props.register(self.cat)
        except:
            print("Catalogue not found. Are you running from build directory?")
            raise

        d = arb.decor()
        d.paint('(all)', 'pas')
        d.set_property(Vm=0.0)
        self.cell = arb.cable_cell(self.tree, arb.label_dict(), d)

    def global_properties(self, _):
        return self.props

    def num_cells(self):
        return 1

    def cell_kind(self, gid):
        return arb.cell_kind.cable

    def cell_description(self, gid):
        return self.cell


class Catalogues(unittest.TestCase):
    def test_nonexistent(self):
        with self.assertRaises(RuntimeError):
            arb.load_catalogue("_NO_EXIST_.so")

    def test_shared_catalogue(self):
        try:
            cat = arb.load_catalogue("lib/dummy-catalogue.so")
        except:
            print("BBP catalogue not found. Are you running from build directory?")
            raise
        nms = [m for m in cat]
        self.assertEqual(nms, ['dummy'], "Expected equal names.")
        for nm in nms:
            prm = list(cat[nm].parameters.keys())
            self.assertEqual(prm, ['gImbar'], "Expected equal parameters on mechanism '{}'.".format(nm))

    def test_simulation(self):
        rcp = recipe()
        ctx = arb.context()
        dom = arb.partition_load_balance(rcp, ctx)
        sim = arb.simulation(rcp, dom, ctx)
        sim.run(tfinal=30)


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
