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
    def __init__(self, cat, cell, probes, inc_0, inc_1, dec_0):
        A.recipe.__init__(self)
        self.the_cell = cell
        self.the_probes = probes
        self.the_props = A.neuron_cable_properties()
        self.the_props.catalogue = (
            cat  # use the provided catalogue of diffusion mechanisms
        )
        self.the_props.set_ion("s", 1, 0, 0, 0)  # use diffusive particles "s"
        self.inc_0 = inc_0  # increase in particle amount at 0.1 s (in 1e-18 mol)
        self.inc_1 = inc_1  # increase in particle amount at 0.5 s (in 1e-18 mol)
        self.dec_0 = dec_0  # decrease in particle amount at 1.5 s (in 1e-18 mol)

    def num_cells(self):
        return 1

    def cell_kind(self, gid):
        return A.cell_kind.cable

    def cell_description(self, gid):
        return self.the_cell

    def probes(self, gid):
        return self.the_probes

    def global_properties(self, kind):
        return self.the_props

    def event_generators(self, gid):
        g = [
            A.event_generator("syn_exc_A", self.inc_0, A.explicit_schedule([0.1])),
            A.event_generator("syn_exc_B", self.inc_1, A.explicit_schedule([0.5])),
            A.event_generator("syn_inh", -self.dec_0, A.explicit_schedule([1.5])),
        ]
        return g


# ---------------------------------------------------------------------------------------
# test class
class TestDiffusion(unittest.TestCase):

    # Constructor (overridden)
    def __init__(self, args):
        super(TestDiffusion, self).__init__(args)

        self.runtime = 3.00  # runtime of the whole simulation in ms
        self.dt = 0.01  # duration of one timestep in ms
        self.dev = 0.01  # accepted relative deviation for `assertAlmostEqual`

    # Method to run an Arbor simulation with diffusion across different segments
    def simulate_diffusion(
        self, cat, _num_segs, _num_cvs_per_seg, _length, _r_1, _r_2=0.0, _r_3=0.0
    ):

        # ---------------------------------------------------------------------------------------
        # set the main parameters and calculate geometrical measures
        num_segs = _num_segs  # number of segments
        num_cvs_per_seg = _num_cvs_per_seg  # number of CVs per segment

        length = _length  # length of the whole setup (in case of 1 or 2 segments, one branch) in µm
        radius_1 = _r_1  # radius of the first segment in µm
        if num_segs > 1:
            radius_2 = _r_2  # radius of the second segment in µm
        else:
            radius_2 = 0
        if num_segs > 2:
            radius_3 = _r_3  # radius of the third segment in µm
        else:
            radius_3 = 0

        length_per_seg = length / num_segs  # axial length of a segment in µm
        area_tot = (
            2 * np.pi * (radius_1 + radius_2 + radius_3) * length_per_seg
        )  # surface area of the whole setup in µm^2 (excluding the circle-shaped ends, since Arbor does not consider current flux there)
        area_per_cv = area_tot / (
            num_segs * num_cvs_per_seg
        )  # surface area of one cylindrical CV in µm^2 (excluding the circle-shaped ends, since Arbor does not consider current flux there)
        volume_tot = (
            np.pi * (radius_1 ** 2 + radius_2 ** 2 + radius_3 ** 2) * length_per_seg
        )  # volume of the whole setup in µm^3
        volume_per_cv = volume_tot / (
            num_segs * num_cvs_per_seg
        )  # volume of one cylindrical CV in µm^3

        inc_0 = 600  # first increase in particle amount (in 1e-18 mol)
        inc_1 = 1200  # second increase in particle amount (in 1e-18 mol)
        dec_0 = 1400  # decrease in particle amount (in 1e-18 mol)
        diffusivity = 1  # diffusivity (in m^2/s)

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
            raise ValueError(f"Specified number of segments not supported.")
        morph = A.morphology(tree)

        # ---------------------------------------------------------------------------------------
        # decorate the morphology with mechanisms
        dec = A.decor()
        if num_segs < 3:
            dec.discretization(
                A.cv_policy(f"(fixed-per-branch {num_segs*num_cvs_per_seg} (branch 0))")
            )
        elif num_segs == 3:
            dec.discretization(
                A.cv_policy(
                    f"(replace (fixed-per-branch {num_cvs_per_seg} (branch 0)) "
                    + f"(fixed-per-branch {num_cvs_per_seg} (branch 1)) "
                    + f"(fixed-per-branch {num_cvs_per_seg} (branch 2)))"
                )
            )
        dec.set_ion("s", int_con=0.0, diff=diffusivity)
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

        # ---------------------------------------------------------------------------------------
        # set probes
        prb = [
            A.cable_probe_ion_diff_concentration('"soma-center"', "s"),
            A.cable_probe_density_state('"soma-center"', "neuron_with_diffusion", "sV"),
            A.cable_probe_density_state_cell("neuron_with_diffusion", "sV"),
        ]

        # ---------------------------------------------------------------------------------------
        # prepare the simulation
        cel = A.cable_cell(tree, dec, labels)
        rec = recipe(cat, cel, prb, inc_0, inc_1, dec_0)
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
        times = data_s[:, 0]
        data_sV = sim.samples(hdl_sV)[0][0]
        tmp_data = sim.samples(hdl_sV_all)[0][0]
        data_sV_total = np.zeros_like(tmp_data[:, 0])
        num_cvs = len(tmp_data[0, :]) - 1
        for i in range(
            len(tmp_data[0, :]) - 1
        ):  # compute the total amount of particles by summing over all CVs of the whole neuron
            data_sV_total += tmp_data[:, i + 1]

        s_lim_expected = (
            inc_0 + inc_1 - dec_0
        ) / volume_tot  # total particle amount of s divided by total volume
        s_max_expected = (
            inc_0 + inc_1
        ) / volume_tot  # total particle amount of s divided by total volume

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
            inc_0 + inc_1 - dec_0,
            delta=self.dev * (inc_0 + inc_1 - dec_0),
        )  # lim_{t->inf}(s⋅V) [estimated]
        self.assertAlmostEqual(
            data_sV_total[-1],
            inc_0 + inc_1 - dec_0,
            delta=self.dev * (inc_0 + inc_1 - dec_0),
        )  # lim_{t->inf}(s⋅V) [direct]
        self.assertAlmostEqual(
            np.max(data_sV_total), inc_0 + inc_1, delta=self.dev * (inc_0 + inc_1)
        )  # max_{t}(s⋅V) [direct]

    # Test: simulations with equal radii
    @fixtures.diffusion_catalogue()
    def test_diffusion_equal_radii(self, diffusion_catalogue):

        self.simulate_diffusion(
            diffusion_catalogue, 1, 600, 10, 4
        )  # 1 segment with radius 4 µm
        self.simulate_diffusion(
            diffusion_catalogue, 2, 300, 10, 4, 4
        )  # 2 segments with radius 4 µm
        self.simulate_diffusion(
            diffusion_catalogue, 3, 200, 10, 4, 4, 4
        )  # 3 segments with radius 4 µm

    """ TODO: not succeeding as of Arbor v0.9.0:
    # Test: simulations with different radii
    @fixtures.diffusion_catalogue()
    def test_diffusion_different_radii(self, diffusion_catalogue):

        self.simulate_diffusion(diffusion_catalogue, 2, 300, 10, 4, 6) # 2 segments with radius 4 µm and 6 µm
        self.simulate_diffusion(diffusion_catalogue, 3, 200, 10, 4, 6, 6) # 3 segments with radius 4 µm and 6 µm
    """
