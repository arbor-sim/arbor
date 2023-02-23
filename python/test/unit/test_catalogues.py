from .. import fixtures
import unittest
import arbor as arb

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
            self.props.catalogue = arb.load_catalogue("dummy-catalogue.so")
        except Exception:
            print("Catalogue not found. Are you running from build directory?")
            raise
        self.props.catalogue = arb.default_catalogue()

        d = arb.decor()
        d.paint("(all)", arb.density("pas"))
        d.set_property(Vm=0.0)
        self.cell = arb.cable_cell(self.tree, d)

    def global_properties(self, _):
        return self.props

    def num_cells(self):
        return 1

    def cell_kind(self, gid):
        return arb.cell_kind.cable

    def cell_description(self, gid):
        return self.cell


class TestCatalogues(unittest.TestCase):
    def test_nonexistent(self):
        with self.assertRaises(FileNotFoundError):
            arb.load_catalogue("_NO_EXIST_.so")

    @fixtures.dummy_catalogue()
    def test_shared_catalogue(self, dummy_catalogue):
        cat = dummy_catalogue
        nms = [m for m in cat]
        self.assertEqual(nms, ["dummy"], "Expected equal names.")
        for nm in nms:
            prm = list(cat[nm].parameters.keys())
            self.assertEqual(
                prm,
                ["gImbar"],
                "Expected equal parameters on mechanism '{}'.".format(nm),
            )

    def test_simulation(self):
        rcp = recipe()
        ctx = arb.context()
        dom = arb.partition_load_balance(rcp, ctx)
        sim = arb.simulation(rcp, ctx, dom)
        sim.run(tfinal=30)

    def test_empty(self):
        def len(cat):
            return sum(1 for _ in cat)

        def hash_(cat):
            return hash(" ".join(sorted(cat)))

        cat = arb.catalogue()
        ref = arb.default_catalogue()
        other = arb.default_catalogue()
        # Test empty constructor
        self.assertEqual(0, len(cat), "Expected no mechanisms in `arbor.catalogue()`.")
        # Test empty extend
        other.extend(cat, "")
        self.assertEqual(
            hash_(ref), hash_(other), "Extending cat with empty should not change cat."
        )
        self.assertEqual(
            0, len(cat), "Extending cat with empty should not change empty."
        )
        other.extend(cat, "prefix/")
        self.assertEqual(
            hash_(ref),
            hash_(other),
            "Extending cat with prefixed empty should not change cat.",
        )
        self.assertEqual(
            0, len(cat), "Extending cat with prefixed empty should not change empty."
        )
        cat.extend(other, "")
        self.assertEqual(
            hash_(other),
            hash_(cat),
            "Extending empty with cat should turn empty into cat.",
        )
        cat = arb.catalogue()
        cat.extend(other, "prefix/")
        self.assertNotEqual(
            hash_(other),
            hash_(cat),
            "Extending empty with prefixed cat should not yield cat",
        )
