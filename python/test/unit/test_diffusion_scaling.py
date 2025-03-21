# -*- coding: utf-8 -*-

import unittest
import arbor as A
from arbor import units as U
import numpy as np

try:
    from scipy.sparse import diags
    from scipy.sparse.linalg import spsolve

    scipy_found = True
except ModuleNotFoundError:
    scipy_found = False

"""
Tests for the correct scaling of the amount _and_ concentration of diffusive
particles in Arbor, as compared to an independent SciPy implementation that
solves the diffusion equation via the Crank-Nicolson method.
"""


class recipe(A.recipe):
    # Constructor
    # - cell: cell description
    # - probes: list of probes
    # - stim: the stimulation protocol
    def __init__(self, cell, probes, stim):
        A.recipe.__init__(self)
        self.the_cell = cell
        self.the_probes = probes
        self.the_props = A.neuron_cable_properties()
        self.stim = stim
        self.the_props.catalogue = A.default_catalogue()
        self.the_props.set_ion(
            "my_ion", valence=1, int_con=0 * U.mM, ext_con=0 * U.mM, rev_pot=0 * U.mV
        )

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
        return [
            A.event_generator(
                f"syn_{i}", 1.0, A.explicit_schedule(np.array([0]) * U.ms)
            )
            for i in self.stim
        ]


class TestDiffusionScaling(unittest.TestCase):
    # Constructor (overridden)
    # - args: arguments that are passed to the super class
    def __init__(self, args):
        super(TestDiffusionScaling, self).__init__(args)

        # spatial parameters
        self.L = 10.0  # length of the domain, in µm
        self.Nx = 100  # number of spatial points
        self.dx = self.L / self.Nx  # spatial step size, in µm
        self.dendrite_radius = 1  # in µm
        self.cv_area = 2 * np.pi * self.dendrite_radius * self.dx  # in µm^2
        self.cv_volume = np.pi * self.dendrite_radius**2 * self.dx  # in µm^3

        # temporal parameters
        self.T = 50.1  # runtime of the whole simulation in ms
        self.dt = 0.01  # duration of one timestep in ms
        self.Nt = int(self.T / self.dt)  # number of time steps

        # diffusion constant
        self.D = 1e-9  # diffusion constant in m^2/s
        # stimulated points
        self.stim = range(int(0.4 * self.Nx), int(0.6 * self.Nx))
        # maximum (initially injected) particle amount in µmol/l / 1e-21 mol
        self.inject_max_amount = 1.0
        self.dev_dyn = 0.20  # accepted relative deviation from maximum particle amount / concentration (in dynamic regime)
        self.dev_ss = 0.05  # accepted relative deviation from maximum particle amount/concentration (in steady state)

    # simulate_arbor
    # Method that sets up and simulates the diffusion of particles in Arbor.
    # - mech_name: the name of the injection point mechanism to be used
    # - return: the distribution of the particle amount/concentration over space and time
    def simulate_arbor(self, mech_name):
        # set up the morphology
        tree = A.segment_tree()
        labels = A.label_dict()
        # the first segment: the root
        parent = A.mnpos
        for i in range(self.Nx):
            x = -self.L / 2 + i * self.dx
            parent = tree.append(
                parent,
                A.mpoint(x, 0, 0, self.dendrite_radius),
                A.mpoint(x + self.dx, 0, 0, self.dendrite_radius),
                tag=i,
            )
            labels[f"dendrite_{i}"] = f"(tag {i})"
        morph = A.morphology(tree)

        # set up decor
        decor = A.decor()
        decor.set_ion("my_ion", int_con=0.0 * U.mM, diff=self.D * U.m2 / U.s)

        # set up particle/ion injection
        # NOTE 'inject_norm_concentration' mechanism will automatically normalize by the volume
        for i in self.stim:
            decor.place(
                f'(on-components 0.5 (region "dendrite_{i}"))',
                A.synapse(f"{mech_name}/x=my_ion", alpha=self.inject_max_amount),
                f"syn_{i}",
            )

        # set up probes
        probes = [A.cable_probe_ion_diff_concentration_cell("my_ion", "tag_my_ion")]

        # create simulation and handles objects
        cv_policy = A.cv_policy(f"(max-extent {self.dx})")
        # cv_policy = A.cv_policy(f'(fixed-per-branch {self.Nx} (branch 0))')
        cel = A.cable_cell(morph, decor, labels, discretization=cv_policy)
        rec = recipe(cel, probes, self.stim)
        sim = A.simulation(rec)
        ion_probe_handle = sim.sample(
            0, "tag_my_ion", A.regular_schedule(self.dt * U.ms)
        )

        # run the Arbor simulation and retrieve the data
        sim.run(tfinal=self.T * U.ms, dt=self.dt * U.ms)
        data, _ = sim.samples(ion_probe_handle)[0]

        # strip off time column and FIXME scale
        return data[:, 1:] / self.cv_area * 1000.0

    # simulate_independent
    # Method that sets up and simulates the diffusion of particles with an independent SciPy implementation that
    # solves the diffusion equation via the Crank-Nicolson method.
    # - concentration: flag that specifies if particle concentrations are considered (instead of amounts)
    # - return: the distribution of the particle amount/concentration over space and time
    def simulate_independent(self, concentration):
        # set Crank-Nicolson coefficient, initial conditions, stimulation
        alpha = self.D * (self.dt * 1e-3) / (2 * (self.dx * 1e-6) ** 2)
        # x = np.linspace(0, self.L*1e-6, self.Nx)
        u = np.zeros(self.Nx)
        if concentration:
            u[self.stim] = self.inject_max_amount / self.cv_volume
        else:
            u[self.stim] = self.inject_max_amount
        data = np.zeros((self.Nt, self.Nx))

        # start to construct the matrices for the Crank-Nicolson method
        main_diag = (1 + 2 * alpha) * np.ones(self.Nx)
        off_diag = -alpha * np.ones(self.Nx - 1)
        main_diag_B = (1 - 2 * alpha) * np.ones(self.Nx)
        off_diag_B = alpha * np.ones(self.Nx - 1)

        # add Neumann boundary conditions
        main_diag[0] = 1 + alpha
        main_diag[-1] = 1 + alpha
        main_diag_B[0] = 1 - alpha
        main_diag_B[-1] = 1 - alpha

        # obtain the final matrices
        A = diags([main_diag, off_diag, off_diag], [0, -1, 1], format="csr")
        B = diags([main_diag_B, off_diag_B, off_diag_B], [0, -1, 1], format="csr")

        # simulation loop
        for n in range(self.Nt):
            # store the result of the previous timestep
            data[n, :] = u

            # compute the right-hand side
            b = B @ u

            # solve the linear system A @ u = b
            u = spsolve(A, b)

        return data

    # test_diffusion_scaling_amount
    # Test: compare the amount of diffusive particles in Arbor and independent implementation
    @unittest.skipIf(not scipy_found, "SciPy not found")
    def test_diffusion_scaling_amount(self):
        # perform the simulations
        data_arbor = self.simulate_arbor("inject_norm_amount")
        data_ind = self.simulate_independent(concentration=False)

        # test initial state
        self.assertTrue(
            np.allclose(
                data_arbor[0], data_ind[0], atol=self.dev_ss * self.inject_max_amount
            ),
            f"{data_arbor[0]} != {data_ind[0]}",
        )

        # test final (equilibrium) state
        self.assertTrue(
            np.allclose(
                data_arbor[-1], data_ind[-1], atol=self.dev_ss * self.inject_max_amount
            ),
            f"{data_arbor[-1]} != {data_ind[-1]}",
        )

        # test whole dynamic time course
        self.assertTrue(
            np.allclose(
                data_arbor, data_ind, atol=self.dev_dyn * self.inject_max_amount
            ),
            f"{data_arbor} != {data_ind}",
        )

    # test_diffusion_scaling_concentration
    # Test: compare the concentration of diffusive particles in Arbor and independent implementation
    @unittest.skipIf(not scipy_found, "SciPy not found")
    def test_diffusion_scaling_concentration(self):
        # perform the simulations
        data_arbor = self.simulate_arbor("inject_norm_concentration")
        data_ind = self.simulate_independent(concentration=True)

        # test initial state
        self.assertTrue(
            np.allclose(
                data_arbor[0],
                data_ind[0],
                atol=self.dev_ss * self.inject_max_amount / self.cv_volume,
            ),
            f"{data_arbor[0]} != {data_ind[0]}",
        )

        # test final (equilibrium) state
        self.assertTrue(
            np.allclose(
                data_arbor[-1],
                data_ind[-1],
                atol=self.dev_ss * self.inject_max_amount / self.cv_volume,
            ),
            f"{data_arbor[-1]} != {data_ind[-1]}",
        )

        # test whole dynamic time course
        self.assertTrue(
            np.allclose(
                data_arbor,
                data_ind,
                atol=self.dev_dyn * self.inject_max_amount / self.cv_volume,
            ),
            f"{data_arbor} != {data_ind}",
        )
