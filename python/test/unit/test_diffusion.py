# -*- coding: utf-8 -*-

import unittest
import arbor as A
import numpy as np
from .. import fixtures

"""
Tests for the concentration and amount of diffusive particles across time and morphology.
Three different morphological structures are considered: 1 segment ("soma only"), 2 segments
("soma with dendrite"), and 3 segments ("soma with two dendrites").

NOTE: Internally, Arbor only knows concentrations. Thus, particle amounts have to be computed
      from concentrations by integrating over the volume of the morphology. The total amount
      of particles should be conserved unless there is deliberate injection or removal of
      particles.
"""


# ---------------------------------------------------------------------------------------
# recipe class
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
        self.the_props.catalogue = (
            cat  # use the provided catalogue of diffusion mechanisms
        )
        self.the_props.set_ion("s", 1, 0, 0, 0)  # use diffusive particles "s"
        self.inject_remove = inject_remove

    # num_cells
    # Returns the total number of cells
    def num_cells(self):
        return 1

    # cell_kind
    # Returns the kind of the specified cell
    # - gid: the identifier of the cell
    def cell_kind(self, gid):
        return A.cell_kind.cable

    # cell_description
    # Returns the description object of the specified cell
    # - gid: the identifier of the cell
    def cell_description(self, gid):
        return self.the_cell

    # probes
    # Returns the list of probes for the specified cell
    # - gid: the identifier of the cell
    def probes(self, gid):
        return self.the_probes

    # global_properties
    # Returns the properties of the specified cell
    # - kind: the kind of the specified cell
    def global_properties(self, kind):
        return self.the_props

    # event_generators
    # Returns the list of event generators for the specified cell
    # - gid: the identifier of the cell
    def event_generators(self, gid):
        event_gens = []
        for event in self.inject_remove:
            event_gens.append(
                A.event_generator(
                    event["synapse"],
                    event["change"],
                    A.explicit_schedule([event["time"]]),
                )
            )
        return event_gens


# ---------------------------------------------------------------------------------------
# test class
class TestDiffusion(unittest.TestCase):
    # Constructor (overridden)
    # - args: arguments that are passed to the super class
    def __init__(self, args):
        super(TestDiffusion, self).__init__(args)

        self.runtime = 5.00  # runtime of the whole simulation in ms
        self.dt = 0.01  # duration of one timestep in ms
        self.dev = 0.01  # accepted relative deviation for `assertAlmostEqual`

    # get_morph_and_decor_1_seg
    # Method that sets up and returns a morphology and decoration for one segment with the given parameters
    # (one segment => there'll be one branch)
    # - num_cvs_per_seg: number of CVs per segment
    # - length_1: axial length of the first segment in µm
    # - radius_1: radius of the first segment in µm
    def get_morph_and_decor_1_seg(self, num_cvs_per_seg, length_1, radius_1):
        # ---------------------------------------------------------------------------------------
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

        # ---------------------------------------------------------------------------------------
        # decorate the morphology with mechanisms
        dec = A.decor()
        dec.discretization(
            A.cv_policy(f"(fixed-per-branch {num_cvs_per_seg})")
        )  # use 'fixed-per-branch' policy to obtain exact number of CVs; there's one branch here
        dec.place('"soma-end"', A.synapse("synapse_with_diffusion"), "syn_exc_A")
        dec.place('"soma-end"', A.synapse("synapse_with_diffusion"), "syn_exc_B")
        dec.place('"soma-end"', A.synapse("synapse_with_diffusion"), "syn_inh")
        dec.paint("(all)", A.density("neuron_with_diffusion"))

        return morph, dec, labels

    # get_morph_and_decor_2_seg
    # Method that sets up and returns a morphology and decoration for two segments with the given parameters
    # (two segments => there'll be one branch)
    # - num_cvs_per_seg: number of CVs per segment
    # - length_1: axial length of the first segment in µm
    # - length_2: axial length of the second segment in µm
    # - radius_1: radius of the first segment in µm
    # - radius_2: radius of the second segment in µm
    def get_morph_and_decor_2_seg(
        self, num_cvs_per_seg, length_1, length_2, radius_1, radius_2
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

        # ---------------------------------------------------------------------------------------
        # decorate the morphology with mechanisms
        dec = A.decor()
        dec.discretization(
            A.cv_policy(f"(fixed-per-branch {2*num_cvs_per_seg})")
        )  # use 'fixed-per-branch' policy to obtain exact number of CVs; there's one branch here
        dec.place(
            '"dendriteA-center"', A.synapse("synapse_with_diffusion"), "syn_exc_A"
        )
        dec.place('"soma-end"', A.synapse("synapse_with_diffusion"), "syn_exc_B")
        dec.place('"soma-end"', A.synapse("synapse_with_diffusion"), "syn_inh")
        dec.paint("(all)", A.density("neuron_with_diffusion"))

        return morph, dec, labels

    # get_morph_and_decor_3_seg
    # Method that sets up and returns a morphology and decoration for three segments with the given parameters
    # (three segments => there'll be three branches)
    # - num_cvs_per_seg: number of CVs per segment
    # - length_1: axial length of the first segment in µm
    # - length_2: axial length of the second segment in µm
    # - length_3: axial length of the third segment in µm
    # - radius_1: radius of the first segment in µm
    # - radius_2: radius of the second segment in µm
    # - radius_3: radius of the third segment in µm
    def get_morph_and_decor_3_seg(
        self,
        num_cvs_per_seg,
        length_1,
        length_2,
        length_3,
        radius_1,
        radius_2,
        radius_3,
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

        # ---------------------------------------------------------------------------------------
        # decorate the morphology with mechanisms
        dec = A.decor()
        dec.discretization(
            A.cv_policy(f"(fixed-per-branch {num_cvs_per_seg})")
        )  # use 'fixed-per-branch' policy to obtain exact number of CVs; there are three branches here
        dec.place(
            '"dendriteA-center"', A.synapse("synapse_with_diffusion"), "syn_exc_A"
        )
        dec.place(
            '"dendriteB-center"', A.synapse("synapse_with_diffusion"), "syn_exc_B"
        )
        dec.place('"soma-end"', A.synapse("synapse_with_diffusion"), "syn_inh")
        dec.paint("(all)", A.density("neuron_with_diffusion"))

        return morph, dec, labels

    # simulate_and_test_diffusion
    # Method that runs an Arbor simulation with diffusion across different segments and subsequently
    # performs tests on the results
    # - cat: catalogue of custom mechanisms
    # - num_segs: number of segments (1, 2, or 3)
    # - num_cvs_per_seg: number of CVs per segment
    # - l_1 [optional]: axial length of the first segment in µm
    # - l_2 [optional]: axial length of the second segment in µm
    # - l_3 [optional]: axial length of the third segment in µm
    # - r_1 [optional]: radius of the first segment in µm
    # - r_2 [optional]: radius of the second segment in µm
    # - r_3 [optional]: radius of the third segment in µm
    def simulate_and_test_diffusion(
        self,
        cat,
        num_segs,
        num_cvs_per_seg,
        l_1=5.0,
        l_2=5.0,
        l_3=5.0,
        r_1=4.0,
        r_2=4.0,
        r_3=4.0,
    ):
        # ---------------------------------------------------------------------------------------
        # set parameters
        inject_remove = [
            {"time": 0.1, "synapse": "syn_exc_A", "change": 600},
            {"time": 0.5, "synapse": "syn_exc_B", "change": 1200},
            {"time": 1.5, "synapse": "syn_inh", "change": -1400},
        ]  # changes in particle amount (in 1e-18 mol)
        diffusivity = 1  # diffusivity (in m^2/s)

        # ---------------------------------------------------------------------------------------
        # get morphology, decoration, and labels, and calculate geometrical measures
        if num_segs == 1:
            r_2 = l_2 = 0  # set radius and length of second segment to zero
            r_3 = l_3 = 0  # set radius and length of third segment to zero
            morph, dec, labels = self.get_morph_and_decor_1_seg(
                num_cvs_per_seg, l_1, r_1
            )  # get morphology, decoration, and labels
            length_soma_cv = (
                l_1 / num_cvs_per_seg
            )  # consider 'fixed-per-branch' policy for one segment, which forms one branch
        elif num_segs == 2:
            r_3 = l_3 = 0  # set radius and length of third segment to zero
            morph, dec, labels = self.get_morph_and_decor_2_seg(
                num_cvs_per_seg, l_1, l_2, r_1, r_2
            )  # get morphology, decoration, and labels
            length_soma_cv = (l_1 + l_2) / (
                2 * num_cvs_per_seg
            )  # consider 'fixed-per-branch' policy for two segments, which only form one branch
        elif num_segs == 3:
            morph, dec, labels = self.get_morph_and_decor_3_seg(
                num_cvs_per_seg, l_1, l_2, l_3, r_1, r_2, r_3
            )  # get morphology, decoration, and labels
            length_soma_cv = (
                l_1 / num_cvs_per_seg
            )  # consider 'fixed-per-branch' policy for three segments, which form three branches
        else:
            raise ValueError(
                f"Specified number of segments ({num_segs}) not supported."
            )
        volume_soma_cv = np.pi * (
            r_1**2 * length_soma_cv
        )  # volume of one cylindrical CV of the first segment in µm^3
        volume_tot = np.pi * (
            r_1**2 * l_1 + r_2**2 * l_2 + r_3**2 * l_3
        )  # volume of the whole setup in µm^3

        # ---------------------------------------------------------------------------------------
        # add the diffusive particle species 's'
        dec.set_ion("s", int_con=0.0, diff=diffusivity)

        # ---------------------------------------------------------------------------------------
        # set probes
        prb = [
            A.cable_probe_ion_diff_concentration('"soma-start"', "s"),
            A.cable_probe_density_state('"soma-start"', "neuron_with_diffusion", "sV"),
            A.cable_probe_density_state_cell("neuron_with_diffusion", "sV"),
        ]

        # ---------------------------------------------------------------------------------------
        # prepare the simulation
        cel = A.cable_cell(morph, dec, labels)
        rec = recipe(cat, cel, prb, inject_remove)
        sim = A.simulation(rec)

        # ---------------------------------------------------------------------------------------
        # set handles
        hdl_s = sim.sample((0, 0), A.regular_schedule(self.dt))  # s at "soma-start"
        hdl_sV = sim.sample((0, 1), A.regular_schedule(self.dt))  # sV at "soma-start"
        hdl_sV_all = sim.sample(
            (0, 2), A.regular_schedule(self.dt)
        )  # sV (cell-wide array)

        # ---------------------------------------------------------------------------------------
        # run the simulation
        sim.run(dt=self.dt, tfinal=self.runtime)

        # ---------------------------------------------------------------------------------------
        # retrieve data and do the testing
        data_s = sim.samples(hdl_s)[0][0]
        data_sV = sim.samples(hdl_sV)[0][0]
        tmp_data = sim.samples(hdl_sV_all)[0][0]
        data_sV_total = np.zeros_like(tmp_data[:, 0])
        num_cvs = len(tmp_data[0, :]) - 1
        for i in range(
            len(tmp_data[0, :]) - 1
        ):  # compute the total amount of particles by summing over all CVs of the whole neuron
            data_sV_total += tmp_data[:, i + 1]

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
        self.assertEqual(num_cvs, num_segs * num_cvs_per_seg)  # total number of CVs
        self.assertAlmostEqual(
            data_s[-1, 1], s_lim_expected, delta=self.dev * s_lim_expected
        )  # equilibrium concentration lim_{t->inf}(s) [direct]
        self.assertAlmostEqual(
            data_sV[-1, 1] / volume_soma_cv,
            s_lim_expected,
            delta=self.dev * s_lim_expected,
        )  # equilibrium concentration lim_{t->inf}(s) [estimated]
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
        self.assertAlmostEqual(
            np.max(data_sV_total),
            sV_tot_max_expected,
            delta=self.dev * sV_tot_max_expected,
        )  # maximum particle amount max_{t}(s⋅V) [direct]

    # test_diffusion_equal_radii
    # Test: simulations with segments of equal length and equal radius
    # - diffusion_catalogue: catalogue of diffusion mechanisms
    @fixtures.diffusion_catalogue()
    def test_diffusion_equal_radii(self, diffusion_catalogue):
        self.simulate_and_test_diffusion(
            diffusion_catalogue, 1, 150, l_1=5, r_1=4
        )  # 1 segment with radius 4 µm
        self.simulate_and_test_diffusion(
            diffusion_catalogue, 2, 75, l_1=5, l_2=5, r_1=4, r_2=4
        )  # 2 segments with radius 4 µm
        self.simulate_and_test_diffusion(
            diffusion_catalogue, 3, 50, l_1=5, l_2=5, l_3=5, r_1=4, r_2=4, r_3=4
        )  # 3 segments with radius 4 µm

    """ TODO: not succeeding as of Arbor v0.9.0:
    # test_diffusion_different_length
    # Test: simulations with segments of different length but equal radius
    # - diffusion_catalogue: catalogue of diffusion mechanisms
    @fixtures.diffusion_catalogue()
    def test_diffusion_different_length(self, diffusion_catalogue):
        self.simulate_and_test_diffusion(
            diffusion_catalogue, 1, 150, l_1=5, r_1=4
        )  # 1 segment with radius 4 µm
        self.simulate_and_test_diffusion(
            diffusion_catalogue, 2, 75, l_1=5, l_2=3, r_1=4, r_2=4
        )  # 2 segments with radius 4 µm
        self.simulate_and_test_diffusion(
            diffusion_catalogue, 3, 50, l_1=5, l_2=3, l_3=3, r_1=4, r_2=4, r_3=4
        )  # 3 segments with radius 4 µm

    # test_diffusion_different_radii
    # Test: simulations with segments of equal length but different radius
    # - diffusion_catalogue: catalogue of diffusion mechanisms
    @fixtures.diffusion_catalogue()
    def test_diffusion_different_radii(self, diffusion_catalogue):
        self.simulate_and_test_diffusion(
            diffusion_catalogue, 2, 75, l_1=5, l_2=5, r_1=4, r_2=6
        )  # 2 segments with radius 4 µm and 6 µm
        self.simulate_and_test_diffusion(
            diffusion_catalogue, 3, 50, l_1=5, l_2=5, l_3=5, r_1=4, r_2=6, r_3=6
        )  # 3 segments with radius 4 µm and 6 µm
    """
