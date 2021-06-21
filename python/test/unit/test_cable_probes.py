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
tests for cable probe wrappers
"""

# Test recipe cc comprises one simple cable cell and mechanisms on it
# sufficient to test cable cell probe wrappers wrap correctly.

class cc_recipe(A.recipe):
    def __init__(self):
        A.recipe.__init__(self)
        st = A.segment_tree()
        st.append(A.mnpos, (0, 0, 0, 10), (1, 0, 0, 10), 1)

        dec = A.decor()

        dec.place('(location 0 0.08)', "expsyn", "syn0")
        dec.place('(location 0 0.09)', "exp2syn", "syn1")
        dec.place('(location 0 0.1)', A.iclamp(20.), "iclamp")
        dec.paint('(all)', "hh")

        self.cell = A.cable_cell(st, A.label_dict(), dec)

        self.cat = A.default_catalogue()
        self.props = A.neuron_cable_properties()
        self.props.register(self.cat)

    def num_cells(self):
        return 1

    def cell_kind(self, gid):
        return A.cell_kind.cable

    def global_properties(self, kind):
        return self.props

    def probes(self, gid):
        # Use keyword arguments to check that the wrappers have actually declared keyword arguments correctly.
        # Place single-location probes at (location 0 0.01*j) where j is the index of the probe address in
        # the returned list.
        return [
            # probe id (0, 0)
            A.cable_probe_membrane_voltage(where='(location 0 0.00)'),
            # probe id (0, 1)
            A.cable_probe_membrane_voltage_cell(),
            # probe id (0, 2)
            A.cable_probe_axial_current(where='(location 0 0.02)'),
            # probe id (0, 3)
            A.cable_probe_total_ion_current_density(where='(location 0 0.03)'),
            # probe id (0, 4)
            A.cable_probe_total_ion_current_cell(),
            # probe id (0, 5)
            A.cable_probe_total_current_cell(),
            # probe id (0, 6)
            A.cable_probe_density_state(where='(location 0 0.06)', mechanism='hh', state='m'),
            # probe id (0, 7)
            A.cable_probe_density_state_cell(mechanism='hh', state='n'),
            # probe id (0, 8)
            A.cable_probe_point_state(target=0, mechanism='expsyn', state='g'),
            # probe id (0, 9)
            A.cable_probe_point_state_cell(mechanism='exp2syn', state='B'),
            # probe id (0, 10)
            A.cable_probe_ion_current_density(where='(location 0 0.10)', ion='na'),
            # probe id (0, 11)
            A.cable_probe_ion_current_cell(ion='na'),
            # probe id (0, 12)
            A.cable_probe_ion_int_concentration(where='(location 0 0.12)', ion='na'),
            # probe id (0, 13)
            A.cable_probe_ion_int_concentration_cell(ion='na'),
            # probe id (0, 14)
            A.cable_probe_ion_ext_concentration(where='(location 0 0.14)', ion='na'),
            # probe id (0, 15)
            A.cable_probe_ion_ext_concentration_cell(ion='na'),
            # probe id (0, 15)
            A.cable_probe_stimulus_current_cell()
        ]

    def cell_description(self, gid):
        return self.cell

class CableProbes(unittest.TestCase):
    def test_probe_addr_metadata(self):
        recipe = cc_recipe()
        context = A.context()
        dd = A.partition_load_balance(recipe, context)
        sim = A.simulation(recipe, dd, context)

        all_cv_cables = [A.cable(0, 0, 1)]

        m = sim.probe_metadata((0, 0))
        self.assertEqual(1, len(m))
        self.assertEqual(A.location(0, 0.0), m[0])

        m = sim.probe_metadata((0, 1))
        self.assertEqual(1, len(m))
        self.assertEqual(all_cv_cables, m[0])

        m = sim.probe_metadata((0, 2))
        self.assertEqual(1, len(m))
        self.assertEqual(A.location(0, 0.02), m[0])

        m = sim.probe_metadata((0, 3))
        self.assertEqual(1, len(m))
        self.assertEqual(A.location(0, 0.03), m[0])

        m = sim.probe_metadata((0, 4))
        self.assertEqual(1, len(m))
        self.assertEqual(all_cv_cables, m[0])

        m = sim.probe_metadata((0, 5))
        self.assertEqual(1, len(m))
        self.assertEqual(all_cv_cables, m[0])

        m = sim.probe_metadata((0, 6))
        self.assertEqual(1, len(m))
        self.assertEqual(A.location(0, 0.06), m[0])

        m = sim.probe_metadata((0, 7))
        self.assertEqual(1, len(m))
        self.assertEqual(all_cv_cables, m[0])

        m = sim.probe_metadata((0, 8))
        self.assertEqual(1, len(m))
        self.assertEqual(A.location(0, 0.08), m[0].location)
        self.assertEqual(1, m[0].multiplicity)
        self.assertEqual(0, m[0].target)

        m = sim.probe_metadata((0, 9))
        self.assertEqual(1, len(m))
        self.assertEqual(1, len(m[0]))
        self.assertEqual(A.location(0, 0.09), m[0][0].location)
        self.assertEqual(1, m[0][0].multiplicity)
        self.assertEqual(1, m[0][0].target)

        m = sim.probe_metadata((0, 10))
        self.assertEqual(1, len(m))
        self.assertEqual(A.location(0, 0.10), m[0])

        m = sim.probe_metadata((0, 11))
        self.assertEqual(1, len(m))
        self.assertEqual(all_cv_cables, m[0])

        m = sim.probe_metadata((0, 12))
        self.assertEqual(1, len(m))
        self.assertEqual(A.location(0, 0.12), m[0])

        m = sim.probe_metadata((0, 13))
        self.assertEqual(1, len(m))
        self.assertEqual(all_cv_cables, m[0])

        m = sim.probe_metadata((0, 14))
        self.assertEqual(1, len(m))
        self.assertEqual(A.location(0, 0.14), m[0])

        m = sim.probe_metadata((0, 15))
        self.assertEqual(1, len(m))
        self.assertEqual(all_cv_cables, m[0])

        m = sim.probe_metadata((0, 16))
        self.assertEqual(1, len(m))
        self.assertEqual(all_cv_cables, m[0])

def suite():
    # specify class and test functions in tuple (here: all tests starting with 'test' from class Contexts
    suite = unittest.makeSuite(CableProbes, ('test'))
    return suite

def run():
    v = options.parse_arguments().verbosity
    runner = unittest.TextTestRunner(verbosity = v)
    runner.run(suite())

if __name__ == "__main__":
    run()
