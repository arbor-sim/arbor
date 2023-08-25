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

        self.runtime = 3.00  # runtime of the whole simulation in ms
        self.dt = 0.01  # duration of one timestep in ms
        self.dev = 0.01  # accepted relative deviation for `assertAlmostEqual`

    # get_morph_and_decor
    # Method that sets up and returns a morphology and decoration for given parameters
    # - num_segs: number of segments
    # - num_cvs_per_seg: number of CVs per segment
    # - length: length of the whole setup (in case of 1 or 2 segments, one branch) in µm
    # - radius_1: radius of the first segment in µm
    # - radius_2: radius of the second segment in µm
    # - radius_3: radius of the third segment in µm
    def get_morph_and_decor(
        self, num_segs, num_cvs_per_seg, length, radius_1, radius_2, radius_3
    ):
        # ---------------------------------------------------------------------------------------
        # set up the morphology
        tree = A.segment_tree()
        if num_segs == 1:
            _ = tree.append(
                A.mnpos,
                A.mpoint(-length / 2, 0, 0, radius_1),
                A.mpoint(+length / 2, 0, 0, radius_1),
                tag=0,
            )

            labels = A.label_dict(
                {
                    "soma-region": "(tag 0)",
                    "soma-center": '(on-components 0.5 (region "soma-region"))',
                    "soma-end": '(on-components 1.0 (region "soma-region"))',
                }
            )
        elif num_segs == 2:
            s = tree.append(
                A.mnpos,
                A.mpoint(-length / 2, 0, 0, radius_1),
                A.mpoint(0, 0, 0, radius_1),
                tag=0,
            )
            _ = tree.append(
                s,
                A.mpoint(0, 0, 0, radius_2),
                A.mpoint(+length / 2, 0, 0, radius_2),
                tag=1,
            )

            labels = A.label_dict(
                {
                    "soma-region": "(tag 0)",
                    "dendriteA-region": "(tag 1)",
                    "soma-center": '(on-components 0.5 (region "soma-region"))',
                    "soma-end": '(on-components 1.0 (region "soma-region"))',
                    "dendriteA-center": '(on-components 0.5 (region "dendriteA-region"))',
                }
            )
        elif num_segs == 3:
            s = tree.append(
                A.mnpos,
                A.mpoint(-1 / 3 * length, 0, 0, radius_1),
                A.mpoint(0, 0, 0, radius_1),
                tag=0,
            )
            _ = tree.append(
                s,
                A.mpoint(0, 0, 0, radius_2),
                A.mpoint(+1 / 3 * length, 0, 0, radius_2),
                tag=1,
            )
            _ = tree.append(
                s,
                A.mpoint(-1 / 3 * length, 0, 0, radius_3),
                A.mpoint(-2 / 3 * length, 0, 0, radius_3),
                tag=2,
            )

            labels = A.label_dict(
                {
                    "soma-region": "(tag 0)",
                    "dendriteA-region": "(tag 1)",
                    "dendriteB-region": "(tag 2)",
                    "soma-center": '(on-components 0.5 (region "soma-region"))',
                    "soma-end": '(on-components 1.0 (region "soma-region"))',
                    "dendriteA-center": '(on-components 0.5 (region "dendriteA-region"))',
                    "dendriteB-center": '(on-components 0.5 (region "dendriteB-region"))',
                }
            )
        else:
            raise ValueError(
                f"Specified number of segments ({num_segs}) not supported."
            )
        morph = A.morphology(tree)

        # ---------------------------------------------------------------------------------------
        # decorate the morphology with mechanisms
        dec = A.decor()
        if num_segs < 3:
            dec.discretization(
                A.cv_policy(f"(fixed-per-branch {num_segs*num_cvs_per_seg})")
            )  # there is only branch for less than three segments
        elif num_segs == 3:
            dec.discretization(A.cv_policy(f"(fixed-per-branch {num_cvs_per_seg})"))
        if num_segs == 1:
            dec.place('"soma-end"', A.synapse("synapse_with_diffusion"), "syn_exc_A")
            dec.place('"soma-end"', A.synapse("synapse_with_diffusion"), "syn_exc_B")
            dec.place('"soma-end"', A.synapse("synapse_with_diffusion"), "syn_inh")
        elif num_segs == 2:
            dec.place(
                '"dendriteA-center"', A.synapse("synapse_with_diffusion"), "syn_exc_A"
            )
            dec.place('"soma-end"', A.synapse("synapse_with_diffusion"), "syn_exc_B")
            dec.place('"soma-end"', A.synapse("synapse_with_diffusion"), "syn_inh")
        elif num_segs == 3:
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
    # - num_segs: number of segments
    # - num_cvs_per_seg: number of CVs per segment
    # - length: length of the whole setup (in case of 1 or 2 segments, one branch) in µm
    # - r_1 [optional]: radius of the first segment in µm
    # - r_2 [optional]: radius of the second segment in µm
    # - r_3 [optional]: radius of the third segment in µm
    def simulate_and_test_diffusion(
        self, cat, num_segs, num_cvs_per_seg, length, r_1, r_2=0.0, r_3=0.0
    ):
        # ---------------------------------------------------------------------------------------
        # set parameters and calculate geometrical measures
        radius_1 = r_1
        if num_segs > 1:
            radius_2 = r_2
        else:
            radius_2 = 0
        if num_segs > 2:
            radius_3 = r_3
        else:
            radius_3 = 0

        length_per_seg = length / num_segs  # axial length of a segment in µm
        volume_tot = (
            np.pi * (radius_1**2 + radius_2**2 + radius_3**2) * length_per_seg
        )  # volume of the whole setup in µm^3
        volume_per_cv = volume_tot / (
            num_segs * num_cvs_per_seg
        )  # volume of one cylindrical CV in µm^3

        inject_remove = [
            {"time": 0.1, "synapse": "syn_exc_A", "change": 600},
            {"time": 0.5, "synapse": "syn_exc_B", "change": 1200},
            {"time": 1.5, "synapse": "syn_inh", "change": -1400},
        ]  # changes in particle amount (in 1e-18 mol)
        diffusivity = 1  # diffusivity (in m^2/s)

        # ---------------------------------------------------------------------------------------
        # get morphology, decoration, and labels, and add the diffusive particle species 's'
        morph, dec, labels = self.get_morph_and_decor(
            num_segs, num_cvs_per_seg, length, radius_1, radius_2, radius_3
        )
        dec.set_ion("s", int_con=0.0, diff=diffusivity)

        # ---------------------------------------------------------------------------------------
        # set probes
        prb = [
            A.cable_probe_ion_diff_concentration('"soma-center"', "s"),
            A.cable_probe_density_state('"soma-center"', "neuron_with_diffusion", "sV"),
            A.cable_probe_density_state_cell("neuron_with_diffusion", "sV"),
        ]

        # ---------------------------------------------------------------------------------------
        # prepare the simulation
        cel = A.cable_cell(morph, dec, labels)
        rec = recipe(cat, cel, prb, inject_remove)
        sim = A.simulation(rec)

        # ---------------------------------------------------------------------------------------
        # set handles
        hdl_s = sim.sample((0, 0), A.regular_schedule(self.dt))  # s at "soma-center"
        hdl_sV = sim.sample((0, 1), A.regular_schedule(self.dt))  # sV at "soma-center"
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

        if num_segs < 3:
            self.assertEqual(morph.num_branches, 1)  # expected number of branches: 1
        else:
            self.assertEqual(morph.num_branches, 3)  # expected number of branches: 3
        self.assertEqual(
            num_cvs, num_segs * num_cvs_per_seg
        )  # expected total number of CVs
        self.assertAlmostEqual(
            data_sV[-1, 1] / volume_per_cv,
            s_lim_expected,
            delta=self.dev * s_lim_expected,
        )  # lim_{t->inf}(s) [estimated]
        self.assertAlmostEqual(
            data_s[-1, 1], s_lim_expected, delta=self.dev * s_lim_expected
        )  # lim_{t->inf}(s) [direct]
        self.assertAlmostEqual(
            np.max(data_s[:, 1]), s_max_expected, delta=self.dev * s_max_expected
        )  # max_{t}(s) [direct]
        self.assertAlmostEqual(
            data_sV[-1, 1] * num_segs * num_cvs_per_seg,
            sV_tot_lim_expected,
            delta=self.dev * sV_tot_lim_expected,
        )  # lim_{t->inf}(s⋅V) [estimated]
        self.assertAlmostEqual(
            data_sV_total[-1],
            sV_tot_lim_expected,
            delta=self.dev * sV_tot_lim_expected,
        )  # lim_{t->inf}(s⋅V) [direct]
        self.assertAlmostEqual(
            np.max(data_sV_total),
            sV_tot_max_expected,
            delta=self.dev * sV_tot_max_expected,
        )  # max_{t}(s⋅V) [direct]

    # test_diffusion_equal_radii
    # Test: simulations with segments of equal radius
    # - diffusion_catalogue: catalogue of diffusion mechanisms
    @fixtures.diffusion_catalogue()
    def test_diffusion_equal_radii(self, diffusion_catalogue):
        self.simulate_and_test_diffusion(
            diffusion_catalogue, 1, 600, 10, 4
        )  # 1 segment with radius 4 µm
        self.simulate_and_test_diffusion(
            diffusion_catalogue, 2, 300, 10, 4, 4
        )  # 2 segments with radius 4 µm
        self.simulate_and_test_diffusion(
            diffusion_catalogue, 3, 200, 10, 4, 4, 4
        )  # 3 segments with radius 4 µm

    """ TODO: not succeeding as of Arbor v0.9.0:
    # test_diffusion_different_radii
    # Test: simulations with segments of different radius
    # - diffusion_catalogue: catalogue of diffusion mechanisms
    @fixtures.diffusion_catalogue()
    def test_diffusion_different_radii(self, diffusion_catalogue):
        self.simulate_and_test_diffusion(
            diffusion_catalogue, 2, 300, 10, 4, 6
        )  # 2 segments with radius 4 µm and 6 µm
        self.simulate_and_test_diffusion(
            diffusion_catalogue, 3, 200, 10, 4, 6, 6
        )  # 3 segments with radius 4 µm and 6 µm
    """
