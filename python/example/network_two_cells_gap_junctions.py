#!/usr/bin/env python3

import arbor
import argparse
import numpy as np

import pandas  # You may have to pip install these.
import seaborn  # You may have to pip install these.
import matplotlib.pyplot as plt


class TwoCellsWithGapJunction(arbor.recipe):
    def __init__(
        self, probes, Vms, length, radius, cm, rL, g, gj_g, cv_policy_max_extent
    ):
        """
        probes -- list of probes

        Vms -- membrane leak potentials of the two cells
        length -- length of cable in μm
        radius -- radius of cable in μm
        cm -- membrane capacitance in F/m^2
        rL -- axial resistivity in Ω·cm
        g -- membrane conductivity in S/cm^2
        gj_g -- gap junction conductivity in μS

        cv_policy_max_extent -- maximum extent of control volume in μm
        """

        # The base C++ class constructor must be called first, to ensure that
        # all memory in the C++ class is initialized correctly.
        arbor.recipe.__init__(self)

        self.the_probes = probes

        self.Vms = Vms
        self.length = length
        self.radius = radius
        self.cm = cm
        self.rL = rL
        self.g = g
        self.gj_g = gj_g

        self.cv_policy_max_extent = cv_policy_max_extent

        self.the_props = arbor.neuron_cable_properties()

    def num_cells(self):
        return 2

    def num_sources(self, gid):
        assert gid in [0, 1]
        return 0

    def cell_kind(self, gid):
        assert gid in [0, 1]
        return arbor.cell_kind.cable

    def probes(self, gid):
        assert gid in [0, 1]
        return self.the_probes

    def global_properties(self, kind):
        return self.the_props

    def cell_description(self, gid):
        """A high level description of the cell with global identifier gid.

        For example the morphology, synapses and ion channels required
        to build a multi-compartment neuron.
        """
        assert gid in [0, 1]

        tree = arbor.segment_tree()

        tree.append(
            arbor.mnpos,
            arbor.mpoint(0, 0, 0, self.radius),
            arbor.mpoint(self.length, 0, 0, self.radius),
            tag=1,
        )

        labels = arbor.label_dict({"cell": "(tag 1)", "gj_site": "(location 0 0.5)"})

        decor = (
            arbor.decor()
            .set_property(Vm=self.Vms[gid])
            .set_property(cm=self.cm)
            .set_property(rL=self.rL)
            # add a gap junction mechanism at the "gj_site" location and label that specific mechanism on that location "gj_label"
            .place('"gj_site"', arbor.junction("gj", {"g": self.gj_g}), "gj_label")
            .paint('"cell"', arbor.density(f"pas/e={self.Vms[gid]}", {"g": self.g}))
        )

        if self.cv_policy_max_extent is not None:
            policy = arbor.cv_policy_max_extent(self.cv_policy_max_extent)
            decor.discretization(policy)
        else:
            decor.discretization(arbor.cv_policy_single())

        return arbor.cable_cell(tree, decor, labels)

    def gap_junctions_on(self, gid):
        # create a bidirectional gap junction from cell 0 at label "gj_label" to cell 1 at label "gj_label" and back.

        if gid == 0:
            tgt = 1
        elif gid == 1:
            tgt = 0
        else:
            raise RuntimeError("Invalid GID for example.")
        return [arbor.gap_junction_connection((tgt, "gj_label"), "gj_label", 1)]


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Two cells connected via a gap junction"
    )

    parser.add_argument(
        "--Vms",
        help="membrane leak potentials in mV",
        type=float,
        default=[-100, -60],
        nargs=2,
    )
    parser.add_argument("--length", help="cell length in μm", type=float, default=100)
    parser.add_argument("--radius", help="cell radius in μm", type=float, default=3)
    parser.add_argument(
        "--cm", help="membrane capacitance in F/m^2", type=float, default=0.005
    )
    parser.add_argument(
        "--rL", help="axial resistivity in Ω·cm", type=float, default=90
    )
    parser.add_argument(
        "--g", help="membrane conductivity in S/cm^2", type=float, default=0.001
    )

    parser.add_argument(
        "--gj_g", help="gap junction conductivity in μS", type=float, default=0.01
    )

    parser.add_argument(
        "--cv_policy_max_extent",
        help="maximum extent of control volume in μm",
        type=float,
    )

    # parse the command line arguments
    args = parser.parse_args()

    # set up membrane voltage probes at the position of the gap junction
    probes = [arbor.cable_probe_membrane_voltage('"gj_site"')]
    recipe = TwoCellsWithGapJunction(probes, **vars(args))

    # configure the simulation and handles for the probes
    sim = arbor.simulation(recipe)

    dt = 0.01
    handles = []
    for gid in [0, 1]:
        handles += [
            sim.sample((gid, i), arbor.regular_schedule(dt)) for i in range(len(probes))
        ]

    # run the simulation for 5 ms
    sim.run(tfinal=5, dt=dt)

    # retrieve the sampled membrane voltages and convert to a pandas DataFrame
    print("Plotting results ...")
    df_list = []
    for probe in range(len(handles)):
        samples, meta = sim.samples(handles[probe])[0]
        df_list.append(
            pandas.DataFrame(
                {"t/ms": samples[:, 0], "U/mV": samples[:, 1], "Cell": f"{probe}"}
            )
        )

    df = pandas.concat(df_list, ignore_index=True)

    fig, ax = plt.subplots()

    # plot the membrane potentials of the two cells as function of time
    seaborn.lineplot(ax=ax, data=df, x="t/ms", y="U/mV", hue="Cell", errorbar=None)

    # area of cells
    area = args.length * 1e-6 * 2 * np.pi * args.radius * 1e-6

    # total conductance and resistance
    cell_g = args.g / 1e-4 * area
    cell_R = 1 / cell_g

    # gap junction conductance and resistance in base units
    si_gj_g = args.gj_g * 1e-6
    si_gj_R = 1 / si_gj_g

    # indicate the expected equilibrium potentials
    for (i, j) in [[0, 1], [1, 0]]:
        weighted_potential = args.Vms[i] + (
            (args.Vms[j] - args.Vms[i]) * (si_gj_R + cell_R)
        ) / (2 * cell_R + si_gj_R)
        ax.axhline(weighted_potential, linestyle="dashed", color="black", alpha=0.5)

    # plot the initial/nominal resting potentials
    for gid, Vm in enumerate(args.Vms):
        ax.axhline(Vm, linestyle="dashed", color="black", alpha=0.5)

    fig.savefig("two_cell_gap_junctions_result.svg")
