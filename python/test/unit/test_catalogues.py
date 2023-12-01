from .. import fixtures
import unittest
import arbor as A
from arbor import units as U

"""
tests for (dynamically loaded) catalogues
"""


class recipe(A.recipe):
    def __init__(self):
        A.recipe.__init__(self)
        self.tree = A.segment_tree()
        self.tree.append(A.mnpos, (0, 0, 0, 10), (1, 0, 0, 10), 1)
        self.props = A.neuron_cable_properties()
        try:
            self.props.catalogue = A.load_catalogue("dummy-catalogue.so")
        except Exception:
            print("Catalogue not found. Are you running from build directory?")
            raise
        self.props.catalogue = A.default_catalogue()

        d = A.decor()
        d.paint("(all)", A.density("pas"))
        d.set_property(Vm=0.0)
        self.cell = A.cable_cell(self.tree, d)

    def global_properties(self, _):
        return self.props

    def num_cells(self):
        return 1

    def cell_kind(self, gid):
        return A.cell_kind.cable

    def cell_description(self, gid):
        return self.cell


class TestCatalogues(unittest.TestCase):
    def test_nonexistent(self):
        with self.assertRaises(FileNotFoundError):
            A.load_catalogue("_NO_EXIST_.so")

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
        ctx = A.context()
        dom = A.partition_load_balance(rcp, ctx)
        sim = A.simulation(rcp, ctx, dom)
        sim.run(tfinal=30 * U.ms)

    def test_empty(self):
        def len(cat):
            return sum(1 for _ in cat)

        def hash_(cat):
            return hash(" ".join(sorted(cat)))

        cat = A.catalogue()
        ref = A.default_catalogue()
        other = A.default_catalogue()
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
        cat = A.catalogue()
        cat.extend(other, "prefix/")
        self.assertNotEqual(
            hash_(other),
            hash_(cat),
            "Extending empty with prefixed cat should not yield cat",
        )
