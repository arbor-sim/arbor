# -*- coding: utf-8 -*-

import unittest
import arbor as A
from arbor import units as U
import numpy as np
from .. import fixtures

"""
Tests for the concentration and amount of diffusive particles across time and morphology.
Four different morphological structures are considered: 1 segment ("soma only"), 2 segments
("soma with dendrite"), 3 segments with one branching point ("soma with two dendrites"), and
4 segments with one branching point ("dendrite with one spine").

NOTE: Internally, Arbor only knows concentrations. Thus, particle amounts have to be computed
      from concentrations by integrating over the volume of the morphology. The total amount
      of particles should be conserved unless there is deliberate injection or removal of
      particles.
"""


class recipe(A.recipe):
    # Constructor
    # - cat: catalogue of custom mechanisms
    # - cell: cell description
    # - probes: list of probes
    # - inject_remove: list of dictionaries of the form [ {"time" : <time>, "synapse" : <synapse>, "change" : <change>} ],
    #                  where <time> is the time of the event in milliseconds, <synapse> is a label, and <change> is the
    #                  change in total particle amount in 1e-18 mol
    def __init__(self, cat, cell, probes, inject_remove):
        A.recipe.__init__(self)
        self.the_cell = cell
        self.the_probes = probes
        self.the_props = A.neuron_cable_properties()
        # use the provided catalogue of diffusion mechanisms
        self.the_props.catalogue = cat
        # use diffusive particles "s"
        self.the_props.set_ion(
            "s", valence=1, int_con=0 * U.mM, ext_con=0 * U.mM, rev_pot=0 * U.mV
        )
        self.inject_remove = inject_remove

    def num_cells(self):
        return 1

    def cell_kind(self, _):
        return A.cell_kind.cable

    def cell_description(self, _):
        return self.the_cell

    def probes(self, _):
        return self.the_probes

    def global_properties(self, _):
        return self.the_props

    def event_generators(self, _):
        event_gens = []
        for event in self.inject_remove:
            event_gens.append(
                A.event_generator(
                    event["synapse"],
                    event["change"],
                    A.explicit_schedule([event["time"] * U.ms]),
                )
            )
        return event_gens


class TestDiffusion(unittest.TestCase):
    # Constructor (overridden)
    # - args: arguments that are passed to the super class
    def __init__(self, args):
        super(TestDiffusion, self).__init__(args)

        self.runtime = 5.00 * U.ms  # runtime of the whole simulation in ms
        self.dt = 0.01 * U.ms  # duration of one timestep in ms
        self.dev = 0.01  # accepted relative deviation for `assertAlmostEqual`

    # get_morph_and_decor_1_seg
    # Method that sets up and returns a morphology and decoration for one segment with the given parameters
    # (one segment => there'll be one branch)
    # - length_1: axial length of the first segment in µm
    # - radius_1: radius of the first segment in µm
    def get_morph_and_decor_1_seg(self, length_1, radius_1):
        # set up the morphology
        tree = A.segment_tree()
        _ = tree.append(
            A.mnpos,
            A.mpoint(-length_1, 0, 0, radius_1),
            A.mpoint(0, 0, 0, radius_1),
            tag=0,
        )
        labels = A.label_dict(
            {
                "soma-region": "(tag 0)",
                "soma-start": '(on-components 0.0 (region "soma-region"))',
                "soma-center": '(on-components 0.5 (region "soma-region"))',
                "soma-end": '(on-components 1.0 (region "soma-region"))',
            }
        )
        morph = A.morphology(tree)
        # decorate the morphology with mechanisms
        dec = (
            A.decor()
            .place('"soma-end"', A.synapse("synapse_with_diffusion"), "syn_exc_A")
            .place('"soma-end"', A.synapse("synapse_with_diffusion"), "syn_exc_B")
            .place('"soma-end"', A.synapse("synapse_with_diffusion"), "syn_inh")
            .paint("(all)", A.density("neuron_with_diffusion"))
        )

        return morph, dec, labels

    # get_morph_and_decor_2_seg
    # Method that sets up and returns a morphology and decoration for two segments with the given parameters
    # (two segments => there'll be one branch)
    # - length_1: axial length of the first segment in µm
    # - length_2: axial length of the second segment in µm
    # - radius_1: radius of the first segment in µm
    # - radius_2: radius of the second segment in µm
    def get_morph_and_decor_2_seg(self, length_1, length_2, radius_1, radius_2):
        tree = A.segment_tree()
        s = tree.append(
            A.mnpos,
            A.mpoint(-length_1, 0, 0, radius_1),
            A.mpoint(0, 0, 0, radius_1),
            tag=0,
        )
        _ = tree.append(
            s,
            A.mpoint(0, 0, 0, radius_2),
            A.mpoint(+length_2, 0, 0, radius_2),
            tag=1,
        )
        labels = A.label_dict(
            {
                "soma-region": "(tag 0)",
                "dendriteA-region": "(tag 1)",
                "soma-start": '(on-components 0.0 (region "soma-region"))',
                "soma-center": '(on-components 0.5 (region "soma-region"))',
                "soma-end": '(on-components 1.0 (region "soma-region"))',
                "dendriteA-center": '(on-components 0.5 (region "dendriteA-region"))',
            }
        )
        morph = A.morphology(tree)
        # decorate the morphology with mechanisms
        dec = (
            A.decor()
            .place(
                '"dendriteA-center"', A.synapse("synapse_with_diffusion"), "syn_exc_A"
            )
            .place('"soma-end"', A.synapse("synapse_with_diffusion"), "syn_exc_B")
            .place('"soma-end"', A.synapse("synapse_with_diffusion"), "syn_inh")
            .paint("(all)", A.density("neuron_with_diffusion"))
        )

        return morph, dec, labels

    # get_morph_and_decor_3_seg
    # Method that sets up and returns a morphology and decoration for three segments with one branching
    # point with the given parameters (three segments => there'll be three branches)
    # - length_1: axial length of the first segment in µm
    # - length_2: axial length of the second segment in µm
    # - length_3: axial length of the third segment in µm
    # - radius_1: radius of the first segment in µm
    # - radius_2: radius of the second segment in µm
    # - radius_3: radius of the third segment in µm
    def get_morph_and_decor_3_seg(
        self,
        length_1,
        length_2,
        length_3,
        radius_1,
        radius_2,
        radius_3,
    ):
        tree = A.segment_tree()
        s = tree.append(
            A.mnpos,
            A.mpoint(-length_1, 0, 0, radius_1),
            A.mpoint(0, 0, 0, radius_1),
            tag=0,
        )
        _ = tree.append(
            s,
            A.mpoint(0, 0, 0, radius_2),
            A.mpoint(+length_2, 0, 0, radius_2),
            tag=1,
        )
        _ = tree.append(
            s,
            A.mpoint(0, 0, 0, radius_3),
            A.mpoint(+length_3, 0, 0, radius_3),
            tag=2,
        )
        labels = A.label_dict(
            {
                "soma-region": "(tag 0)",
                "dendriteA-region": "(tag 1)",
                "dendriteB-region": "(tag 2)",
                "soma-start": '(on-components 0.0 (region "soma-region"))',
                "soma-center": '(on-components 0.5 (region "soma-region"))',
                "soma-end": '(on-components 1.0 (region "soma-region"))',
                "dendriteA-center": '(on-components 0.5 (region "dendriteA-region"))',
                "dendriteB-center": '(on-components 0.5 (region "dendriteB-region"))',
            }
        )
        morph = A.morphology(tree)

        # decorate the morphology with mechanisms
        dec = (
            A.decor()
            .place(
                '"dendriteA-center"', A.synapse("synapse_with_diffusion"), "syn_exc_A"
            )
            .place(
                '"dendriteB-center"', A.synapse("synapse_with_diffusion"), "syn_exc_B"
            )
            .place('"soma-end"', A.synapse("synapse_with_diffusion"), "syn_inh")
            .paint("(all)", A.density("neuron_with_diffusion"))
        )

        return morph, dec, labels

    # get_morph_and_decor_4_seg
    # Method that sets up and returns a morphology and decoration for four segments with one branching
    # point with the given parameters (three segments => there'll be three branches)
    # - length_1: axial length of the first segment in µm
    # - length_2: axial length of the second segment in µm
    # - length_3: axial length of the third segment in µm
    # - length_3: axial length of the fourth segment in µm
    # - radius_1: radius of the first segment in µm
    # - radius_2: radius of the second segment in µm
    # - radius_3: radius of the third segment in µm
    # - radius_3: radius of the fourth segment in µm
    def get_morph_and_decor_4_seg(
        self,
        length_1,
        length_2,
        length_3,
        length_4,
        radius_1,
        radius_2,
        radius_3,
        radius_4,
    ):
        # ---------------------------------------------------------------------------------------
        # set up the morphology
        tree = A.segment_tree()
        s = tree.append(
            A.mnpos,
            A.mpoint(-length_1, 0, 0, radius_1),
            A.mpoint(0, 0, 0, radius_1),
            tag=0,
        )
        s2 = tree.append(
            s,
            A.mpoint(0, 0, 0, radius_2),
            A.mpoint(+length_2, 0, 0, radius_2),
            tag=1,
        )
        _ = tree.append(
            s,
            A.mpoint(0, 0, 0, radius_3),
            A.mpoint(0, +length_3, 0, radius_3),
            tag=2,
        )
        _ = tree.append(
            s2,
            A.mpoint(+length_2, 0, 0, radius_4),
            A.mpoint(+length_2 + length_4, 0, 0, radius_4),
            tag=3,
        )
        labels = A.label_dict(
            {
                "soma-region": "(tag 0)",
                "dendriteA-region": "(tag 1)",
                "dendriteB-region": "(tag 2)",
                "soma-start": '(on-components 0.0 (region "soma-region"))',
                "soma-center": '(on-components 0.5 (region "soma-region"))',
                "soma-end": '(on-components 1.0 (region "soma-region"))',
                "dendriteA-center": '(on-components 0.5 (region "dendriteA-region"))',
                "dendriteB-center": '(on-components 0.5 (region "dendriteB-region"))',
            }
        )
        morph = A.morphology(tree)

        # ---------------------------------------------------------------------------------------
        # decorate the morphology with mechanisms
        dec = (
            A.decor()
            .place(
                '"dendriteA-center"', A.synapse("synapse_with_diffusion"), "syn_exc_A"
            )
            .place(
                '"dendriteB-center"', A.synapse("synapse_with_diffusion"), "syn_exc_B"
            )
            .place('"soma-end"', A.synapse("synapse_with_diffusion"), "syn_inh")
            .paint("(all)", A.density("neuron_with_diffusion"))
        )

        return morph, dec, labels

    # simulate_and_test_diffusion
    # Method that runs an Arbor simulation with diffusion across different segments and subsequently
    # performs tests on the results
    # - cat: catalogue of custom mechanisms
    # - num_segs: number of segments (1, 2, or 3)
    # - num_cvs_per_seg: number of CVs per segment
    # - num_bps [optional]: number of branching points (1 or 2)
    # - l_1 [optional]: axial length of the first segment in µm
    # - l_2 [optional]: axial length of the second segment in µm
    # - l_3 [optional]: axial length of the third segment in µm
    # - r_1 [optional]: radius of the first segment in µm
    # - r_2 [optional]: radius of the second segment in µm
    # - r_3 [optional]: radius of the third segment in µm
    # - test_max [optional]: test the value of the maximum (only makes sense if it has had time to equilibrate)
    def simulate_and_test_diffusion(
        self,
        ctx,
        cat,
        num_segs,
        num_cvs_per_seg,
        l_1=5.0,
        l_2=5.0,
        l_3=5.0,
        l_4=5.0,
        r_1=4.0,
        r_2=4.0,
        r_3=4.0,
        r_4=4.0,
        test_max=True,
    ):
        # set parameters
        inject_remove = [
            {"time": 0.1, "synapse": "syn_exc_A", "change": 600},
            {"time": 0.5, "synapse": "syn_exc_B", "change": 1200},
            {"time": 1.5, "synapse": "syn_inh", "change": -1400},
        ]  # changes in particle amount (in 1e-18 mol)
        diffusivity = 1  # diffusivity (in m^2/s)

        # get morphology, decoration, and labels, and calculate geometrical measures
        if num_segs == 1:
            r_2 = l_2 = 0  # set radius and length of second segment to zero
            r_3 = l_3 = 0  # set radius and length of third segment to zero
            r_4 = l_4 = 0  # set radius and length of fourth segment to zero
            morph, dec, labels = self.get_morph_and_decor_1_seg(l_1, r_1)
            length_soma_cv = (
                l_1 / num_cvs_per_seg
            )  # consider 'fixed-per-branch' policy for one segment, which forms one branch
            cvp = A.cv_policy(f"(fixed-per-branch {num_cvs_per_seg})")
        elif num_segs == 2:
            r_3 = l_3 = 0  # set radius and length of third segment to zero
            r_4 = l_4 = 0  # set radius and length of fourth segment to zero
            morph, dec, labels = self.get_morph_and_decor_2_seg(l_1, l_2, r_1, r_2)
            length_soma_cv = (
                (l_1 + l_2) / (2 * num_cvs_per_seg)
            )  # consider 'fixed-per-branch' policy for two segments, which only form one branch
            # use 'fixed-per-branch' policy to obtain exact number of CVs; there's one branch here
            cvp = A.cv_policy(f"(fixed-per-branch {2 * num_cvs_per_seg})")
        elif num_segs == 3:
            r_4 = l_4 = 0  # set radius and length of fourth segment to zero
            morph, dec, labels = self.get_morph_and_decor_3_seg(
                l_1, l_2, l_3, r_1, r_2, r_3
            )  # get morphology, decoration, and labels
            length_soma_cv = (
                l_1 / num_cvs_per_seg
            )  # consider 'fixed-per-branch' policy for three segments, which form three branches
            # use 'fixed-per-branch' policy to obtain exact number of CVs; there are three branches here
            cvp = A.cv_policy(f"(fixed-per-branch {num_cvs_per_seg})")
        elif num_segs == 4:
            morph, dec, labels = self.get_morph_and_decor_4_seg(
                l_1, l_2, l_3, l_4, r_1, r_2, r_3, r_4
            )
            length_soma_cv = (
                l_1 / num_cvs_per_seg
            )  # consider 'fixed-per-branch' policy for three segments, which form three branches
            cvp = A.cv_policy(f"(fixed-per-branch {num_cvs_per_seg})")
        else:
            raise ValueError(
                f"Specified numbers of segments ({num_segs}) not supported."
            )
        volume_soma_cv = np.pi * (
            r_1**2 * length_soma_cv
        )  # volume of one cylindrical CV of the first segment in µm^3
        volume_tot = np.pi * (
            r_1**2 * l_1 + r_2**2 * l_2 + r_3**2 * l_3 + r_4**2 * l_4
        )  # volume of the whole setup in µm^3

        # add the diffusive particle species 's'
        dec.set_ion("s", int_con=0.0 * U.mM, diff=diffusivity * U.m2 / U.s)

        # set probes
        prb = [
            A.cable_probe_ion_diff_concentration('"soma-start"', "s", "X"),
            A.cable_probe_density_state(
                '"soma-start"', "neuron_with_diffusion", "sV", "XV"
            ),
            A.cable_probe_density_state_cell("neuron_with_diffusion", "sV", "XVs"),
        ]

        # prepare the simulation
        cel = A.cable_cell(morph, dec, labels, cvp)
        # A.write_component(cel, "morpho.txt"
        rec = recipe(cat, cel, prb, inject_remove)
        sim = A.simulation(rec, ctx)

        # set handles
        sched = A.regular_schedule(self.dt)
        hdl_s = sim.sample((0, "X"), sched)  # s at "soma-start"
        hdl_sV = sim.sample((0, "XV"), sched)  # sV at "soma-start"
        hdl_sV_all = sim.sample((0, "XVs"), sched)  # sV (cell-wide array)

        # run the simulation
        sim.run(dt=self.dt, tfinal=self.runtime)

        # retrieve data and do the testing
        data_s = sim.samples(hdl_s)[0][0]
        data_sV = sim.samples(hdl_sV)[0][0]
        tmp_data = sim.samples(hdl_sV_all)[0][0]
        num_cvs = tmp_data.shape[1] - 1
        # compute the total amount of particles by summing over all CVs of the whole neuron
        data_sV_total = np.sum(tmp_data[:, 1:], axis=1)

        # final value of the total particle amount of s
        sV_tot_lim_expected = 0
        for event in inject_remove:
            sV_tot_lim_expected += event["change"]

        # final value of the concentration of s (total particle amount divided by total volume)
        s_lim_expected = sV_tot_lim_expected / volume_tot

        # maximum value of the total particle amount of s
        sV_tot_max_expected = 0
        for event in inject_remove:
            if event["change"] > 0:
                sV_tot_max_expected += event["change"]

        # maximum value of the concentration of s (total particle amount divided by total volume)
        s_max_expected = sV_tot_max_expected / volume_tot

        # tests
        if num_segs < 3:
            self.assertEqual(morph.num_branches, 1)  # number of branches (1 expected)
        else:
            self.assertEqual(
                morph.num_branches, 3
            )  # number of branches (3 expected, see https://docs.arbor-sim.org/en/latest/concepts/morphology.html)
        if num_segs < 4:
            self.assertEqual(num_cvs, num_segs * num_cvs_per_seg)  # total number of CVs
        self.assertAlmostEqual(
            data_s[-1, 1], s_lim_expected, delta=self.dev * s_lim_expected
        )  # equilibrium concentration lim_{t->inf}(s) [direct]
        self.assertAlmostEqual(
            data_sV[-1, 1] / volume_soma_cv,
            s_lim_expected,
            delta=self.dev * s_lim_expected,
        )  # equilibrium concentration lim_{t->inf}(s) [estimated]
        if test_max:
            self.assertAlmostEqual(
                np.max(data_s[:, 1]), s_max_expected, delta=self.dev * s_max_expected
            )  # maximum concentration max_{t}(s) [direct]
        self.assertAlmostEqual(
            data_sV_total[-1],
            sV_tot_lim_expected,
            delta=self.dev * sV_tot_lim_expected,
        )  # equilibrium particle amount lim_{t->inf}(s⋅V) [direct]
        self.assertAlmostEqual(
            data_sV[-1, 1] / volume_soma_cv * volume_tot,
            sV_tot_lim_expected,
            delta=self.dev * sV_tot_lim_expected,
        )  # equilibrium particle amount lim_{t->inf}(s⋅V) [estimated]
        if test_max:
            self.assertAlmostEqual(
                np.max(data_sV_total),
                sV_tot_max_expected,
                delta=self.dev * sV_tot_max_expected,
            )  # maximum particle amount max_{t}(s⋅V) [direct]

    # test_diffusion_equal_radii
    # Test: simulations with segments of equal length and equal radius
    # - diffusion_catalogue: catalogue of diffusion mechanisms
    @fixtures.single_context()
    @fixtures.diffusion_catalogue()
    def test_diffusion_equal_radii_equal_length(
        self, single_context, diffusion_catalogue
    ):
        self.simulate_and_test_diffusion(
            single_context, diffusion_catalogue, 1, 150, l_1=5, r_1=4
        )  # 1 segment with radius 4 µm
        self.simulate_and_test_diffusion(
            single_context, diffusion_catalogue, 2, 75, l_1=5, l_2=5, r_1=4, r_2=4
        )  # 2 segments with radius 4 µm
        self.simulate_and_test_diffusion(
            single_context,
            diffusion_catalogue,
            3,
            50,
            l_1=5,
            l_2=5,
            l_3=5,
            r_1=4,
            r_2=4,
            r_3=4,
        )  # 3 segments with radius 4 µm
        self.simulate_and_test_diffusion(
            single_context,
            diffusion_catalogue,
            4,
            100,
            l_1=5,
            l_2=5,
            l_3=5,
            l_4=5,
            r_1=4,
            r_2=4,
            r_3=4,
            r_4=4,
            test_max=False,
        )  # 4 segments with radius 4 µm

    # test_diffusion_different_length
    # Test: simulations with segments of different length but equal radius
    # - diffusion_catalogue: catalogue of diffusion mechanisms
    @fixtures.single_context()
    @fixtures.diffusion_catalogue()
    def test_diffusion_equal_radii_different_length(
        self, single_context, diffusion_catalogue
    ):
        self.simulate_and_test_diffusion(
            single_context, diffusion_catalogue, 1, 150, l_1=5, r_1=4
        )  # 1 segment with radius 4 µm
        self.simulate_and_test_diffusion(
            single_context, diffusion_catalogue, 2, 75, l_1=5, l_2=3, r_1=4, r_2=4
        )  # 2 segments with radius 4 µm
        self.simulate_and_test_diffusion(
            single_context,
            diffusion_catalogue,
            3,
            50,
            l_1=5,
            l_2=3,
            l_3=3,
            r_1=4,
            r_2=4,
            r_3=4,
        )  # 3 segments with radius 4 µm
        self.simulate_and_test_diffusion(
            single_context,
            diffusion_catalogue,
            4,
            100,
            l_1=5,
            l_2=3,
            l_3=1,
            l_4=3,
            r_1=4,
            r_2=4,
            r_3=4,
            r_4=4,
            test_max=False,
        )  # 4 segments with radius 4 µm

    # test_diffusion_different_radii
    # Test: simulations with segments of equal length but different radius
    # - diffusion_catalogue: catalogue of diffusion mechanisms
    @fixtures.single_context()
    @fixtures.diffusion_catalogue()
    def test_diffusion_different_radii_equal_length(
        self, single_context, diffusion_catalogue
    ):
        self.simulate_and_test_diffusion(
            single_context, diffusion_catalogue, 2, 75, l_1=5, l_2=5, r_1=4, r_2=6
        )  # 2 segments with radius 4 µm and 6 µm
        self.simulate_and_test_diffusion(
            single_context,
            diffusion_catalogue,
            3,
            50,
            l_1=5,
            l_2=5,
            l_3=5,
            r_1=4,
            r_2=6,
            r_3=6,
        )  # 3 segments with radius 4 µm and 6 µm
        self.simulate_and_test_diffusion(
            single_context,
            diffusion_catalogue,
            4,
            100,
            l_1=5,
            l_2=5,
            l_3=5,
            l_4=5,
            r_1=4,
            r_2=6,
            r_3=1,
            r_4=6,
            test_max=False,
        )  # 4 segments with radius 4 µm and 6 µm
