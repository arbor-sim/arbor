# -*- coding: utf-8 -*-

import unittest
import arbor as A
import numpy as np

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

        dec.place("(location 0 0.08)", A.synapse("expsyn"), "syn0")
        dec.place("(location 0 0.09)", A.synapse("exp2syn"), "syn1")
        dec.place("(location 0 0.1)", A.iclamp(20.0), "iclamp")
        dec.paint("(all)", A.density("hh"))

        self.cell = A.cable_cell(st, dec)

        self.props = A.neuron_cable_properties()
        self.props.catalogue = A.default_catalogue()

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
            A.cable_probe_membrane_voltage(where="(location 0 0.00)"),
            # probe id (0, 1)
            A.cable_probe_membrane_voltage_cell(),
            # probe id (0, 2)
            A.cable_probe_axial_current(where="(location 0 0.02)"),
            # probe id (0, 3)
            A.cable_probe_total_ion_current_density(where="(location 0 0.03)"),
            # probe id (0, 4)
            A.cable_probe_total_ion_current_cell(),
            # probe id (0, 5)
            A.cable_probe_total_current_cell(),
            # probe id (0, 6)
            A.cable_probe_density_state(
                where="(location 0 0.06)", mechanism="hh", state="m"
            ),
            # probe id (0, 7)
            A.cable_probe_density_state_cell(mechanism="hh", state="n"),
            # probe id (0, 8)
            A.cable_probe_point_state(target=0, mechanism="expsyn", state="g"),
            # probe id (0, 9)
            A.cable_probe_point_state_cell(mechanism="exp2syn", state="B"),
            # probe id (0, 10)
            A.cable_probe_ion_current_density(where="(location 0 0.10)", ion="na"),
            # probe id (0, 11)
            A.cable_probe_ion_current_cell(ion="na"),
            # probe id (0, 12)
            A.cable_probe_ion_int_concentration(where="(location 0 0.12)", ion="na"),
            # probe id (0, 13)
            A.cable_probe_ion_int_concentration_cell(ion="na"),
            # probe id (0, 14)
            A.cable_probe_ion_ext_concentration(where="(location 0 0.14)", ion="na"),
            # probe id (0, 15)
            A.cable_probe_ion_ext_concentration_cell(ion="na"),
            # probe id (0, 15)
            A.cable_probe_stimulus_current_cell(),
        ]

    def cell_description(self, gid):
        return self.cell


class TestCableProbes(unittest.TestCase):
    def test_probe_addr_metadata(self):
        recipe = cc_recipe()
        context = A.context()
        dd = A.partition_load_balance(recipe, context)
        sim = A.simulation(recipe, context, dd)

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


class lif_recipe(A.recipe):
    def __init__(self):
        A.recipe.__init__(self)

    def num_cells(self):
        return 1

    def cell_kind(self, gid):
        return A.cell_kind.lif

    def global_properties(self, kind):
        return None

    def probes(self, gid):
        return [
            # probe id (0, 0)
            A.lif_probe_voltage(),
        ]

    def cell_description(self, gid):
        cell = A.lif_cell("src", "tgt")
        cell.E_L = -42
        cell.V_m = -23
        cell.t_ref = 0.2
        return cell


class TestLifProbes(unittest.TestCase):
    def test_probe_addr_metadata(self):
        rec = lif_recipe()
        sim = A.simulation(rec)

        m = sim.probe_metadata((0, 0))
        self.assertEqual(1, len(m))
        self.assertTrue(all(isinstance(i, A.lif_probe_metadata) for i in m))

    def test_probe_result(self):
        rec = lif_recipe()
        sim = A.simulation(rec)
        hdl = sim.sample((0, 0), A.regular_schedule(0.1))
        sim.run(1.0, 0.05)
        smp = sim.samples(hdl)
        exp = np.array(
            [
                [0.0, -23.0],
                [0.1, -23.18905316],
                [0.2, -23.37622521],
                [0.3, -23.56153486],
                [0.4, -23.74500066],
                [0.5, -23.92664093],
                [0.6, -24.10647386],
                [0.7, -24.28451742],
                [0.8, -24.46078942],
                [0.9, -24.63530748],
            ]
        )
        for d, _ in smp:
            np.testing.assert_allclose(d, exp)
