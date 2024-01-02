#!/usr/bin/env python3


import arbor as A
from arbor import units as U
import argparse
import numpy as np

import pandas as pd  # You may have to pip install these.
import seaborn as sns  # You may have to pip install these.
import matplotlib.pyplot as plt


class TwoCellsWithGapJunction(A.recipe):
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
        A.recipe.__init__(self)

        self.the_probes = probes

        self.Vms = Vms
        self.length = length
        self.radius = radius
        self.cm = cm
        self.rL = rL
        self.g = g
        self.gj_g = gj_g

        self.cv_policy_max_extent = cv_policy_max_extent

        self.the_props = A.neuron_cable_properties()

    def num_cells(self):
        # NOTE: This *must* be 2 and 2 only.
        return 2

    def cell_kind(self, gid):
        return A.cell_kind.cable

    def global_properties(self, kind):
        assert kind == A.cell_kind.cable
        return self.the_props

    def cell_description(self, gid):
        """A high level description of the cell with global identifier gid.

        For example the morphology, synapses and ion channels required
        to build a multi-compartment neuron.
        """

        tree = A.segment_tree()

        tree.append(
            A.mnpos,
            A.mpoint(0, 0, 0, self.radius),
            A.mpoint(self.length, 0, 0, self.radius),
            tag=1,
        )

        labels = A.label_dict({"cell": "(tag 1)", "gj_site": "(location 0 0.5)"})

        decor = (
            A.decor()
            .set_property(
                Vm=self.Vms[gid] * U.mV,
                cm=self.cm * U.F / U.m2,
                rL=self.rL * U.Ohm * U.cm,
            )
            # add a gap junction mechanism at the "gj_site" location and label that specific mechanism on that location "gj_label"
            .place('"gj_site"', A.junction("gj", g=self.gj_g), "gj_label")
            .paint('"cell"', A.density(f"pas/e={self.Vms[gid]}", g=self.g))
        )

        if self.cv_policy_max_extent is not None:
            policy = A.cv_policy_max_extent(self.cv_policy_max_extent)
            decor.discretization(policy)
        else:
            decor.discretization(A.cv_policy_single())

        return A.cable_cell(tree, decor, labels)

    def gap_junctions_on(self, gid):
        # a bidirectional gap junction between cells 0 and 1 at label "gj_label".
        tgt = (gid + 1) % 2
        return [A.gap_junction_connection((tgt, "gj_label"), "gj_label", 1)]

    def probes(self, gid):
        assert gid in [0, 1]
        return self.the_probes


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
    probes = [A.cable_probe_membrane_voltage('"gj_site"', "Um")]
    recipe = TwoCellsWithGapJunction(probes, **vars(args))

    # configure the simulation and handles for the probes
    sim = A.simulation(recipe)

    T = 5 * U.ms
    dt = 0.01 * U.ms

    # generate handles for all probes and gids.
    # NOTE We have only one probe, so this is just a reminder.
    handles = [
        sim.sample((gid, "Um"), A.regular_schedule(dt))
        for _ in enumerate(probes)
        for gid in range(recipe.num_cells())
    ]

    # run the simulation
    sim.run(tfinal=T, dt=dt)

    # retrieve the sampled membrane voltages
    print("Plotting results ...")
    df_list = []
    for probe, handle in enumerate(handles):
        samples, meta = sim.samples(handle)[0]
        df_list.append(
            pd.DataFrame(
                {"t/ms": samples[:, 0], "U/mV": samples[:, 1], "Cell": f"{probe}"}
            )
        )

    df = pd.concat(df_list, ignore_index=True)

    fg, ax = plt.subplots()

    # plot the membrane potentials of the two cells as function of time
    sns.lineplot(ax=ax, data=df, x="t/ms", y="U/mV", hue="Cell", errorbar=None)

    # area of cells
    area = args.length * U.um * 2 * np.pi * args.radius * U.um

    # total and gap junction conductance in base units
    cell_g = area * args.g * U.S / U.cm2
    si_gj_g = args.gj_g * U.uS

    # weight
    w = (si_gj_g + cell_g) / (2 * si_gj_g + cell_g)

    assert w.units == U.nil, f"weight must be dimensionless, but has units {w.units}"

    # indicate the expected equilibrium potentials
    for i, j in [[0, 1], [1, 0]]:
        weighted_potential = args.Vms[i] + w.value * (args.Vms[j] - args.Vms[i])
        ax.axhline(weighted_potential, linestyle="dashed", color="black", alpha=0.5)
        ax.text(
            2,
            weighted_potential,
            f"$\\tilde U_{j} = U_{j} + w\\cdot(U_{j} - U_{i})$",
            va="center",
            ha="center",
            backgroundcolor="w",
        )
        ax.text(
            2, args.Vms[j], f"$U_{j}$", va="center", ha="center", backgroundcolor="w"
        )

    ax.set_xlim(0, T.value)

    # plot the initial/nominal resting potentials
    for gid, Vm in enumerate(args.Vms):
        ax.axhline(Vm, linestyle="dashed", color="black", alpha=0.5)

    fg.savefig("two_cell_gap_junctions_result.svg")
