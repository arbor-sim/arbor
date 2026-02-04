#!/usr/bin/env python3

import arbor as A
from arbor import units as U
from argparse import ArgumentParser
import numpy as np
import pandas as pd  # You may have to pip install these.
import seaborn as sns  # You may have to pip install these.
import matplotlib.pyplot as plt


class TwoCellsWithGapJunction(A.recipe):
    def __init__(self, Vms, length, radius, cm, rL, g, gj_g, max_extent):
        """
        Vms -- membrane leak potentials of the two cells
        length -- length of cable in μm
        radius -- radius of cable in μm
        cm -- membrane capacitance in F/m²
        rL -- axial resistivity in Ω·cm
        g -- membrane conductivity in S/cm²
        gj_g -- gap junction conductivity in μS
        max_extent -- maximum extent of control volume in μm
        """

        # Call base constructor first to ensure proper initialization
        A.recipe.__init__(self)

        self.Vms = [Vm * U.mV for Vm in Vms]
        self.length = length * U.um
        self.radius = radius * U.um
        self.area = self.length * 2 * np.pi * self.radius
        self.cm = cm * U.F / U.m2
        self.rL = rL * U.Ohm * U.cm
        self.g = g * U.S / U.cm2
        self.gj_g = gj_g * U.uS
        self.max_extent = max_extent
        self.the_props = A.neuron_cable_properties()

    def num_cells(self):
        return 2

    def cell_kind(self, _):
        return A.cell_kind.cable

    def global_properties(self, _):
        return self.the_props

    def cell_description(self, gid):
        tree = A.segment_tree()
        r, l = self.radius.value, self.length.value
        tree.append(A.mnpos, (0, 0, 0, r), (l, 0, 0, r), tag=1)

        labels = A.label_dict({"midpoint": "(location 0 0.5)"})

        decor = (
            A.decor()
            .set_property(Vm=self.Vms[gid], cm=self.cm, rL=self.rL)
            .place('"midpoint"', A.junction("gj", g=self.gj_g.value), "gj")
            .paint("(all)", A.density(f"pas/e={self.Vms[gid].value}", g=self.g.value))
        )

        if self.max_extent is not None:
            cvp = A.cv_policy_max_extent(self.max_extent)
        else:
            cvp = A.cv_policy_single()

        return A.cable_cell(tree, decor, labels, cvp)

    def gap_junctions_on(self, gid):
        return [A.gap_junction_connection(((gid + 1) % 2, "gj"), "gj", 1)]

    def probes(self, _):
        return [A.cable_probe_membrane_voltage('"midpoint"', "Um")]


# parse the command line arguments
parser = ArgumentParser(description="Two cells connected via a gap junction")

parser.add_argument(
    "--Vms",
    help="membrane leak potentials [mV]",
    type=float,
    default=[-100, -60],
    nargs=2,
)
parser.add_argument("--length", help="cell length [μm]", type=float, default=100)
parser.add_argument("--radius", help="cell radius [μm]", type=float, default=3)
parser.add_argument(
    "--cm", help="membrane capacitance [F/m²]", type=float, default=0.005
)
parser.add_argument("--rL", help="axial resistivity [Ω·cm]", type=float, default=90)
parser.add_argument("--g", help="leak conductivity [S/cm²]", type=float, default=0.001)
parser.add_argument(
    "--gj_g", help="gap junction conductivity [μS]", type=float, default=0.01
)
parser.add_argument("--max-extent", help="discretization length [μm]", type=float)

args = parser.parse_args()

# set up membrane voltage probes at the position of the gap junction
rec = TwoCellsWithGapJunction(**vars(args))

# configure the simulation and handles for the probes
sim = A.simulation(rec)

T = 5 * U.ms
dt = 0.01 * U.ms

# generate handles for all probes and gids.
handles = [sim.sample((gid, "Um"), A.regular_schedule(dt)) for gid in [0, 1]]

# run the simulation
sim.run(tfinal=T, dt=dt)

# retrieve the sampled membrane voltages
print("Plotting results ...")
df_list = []
for gid, handle in enumerate(handles):
    data, meta = sim.samples(handle)[0]
    df_list.append(
        pd.DataFrame({"t/ms": data[:, 0], "U/mV": data[:, 1], "Cell": f"{gid}"})
    )
df = pd.concat(df_list, ignore_index=True)

# plot the membrane potentials of the two cells as function of time
fg, ax = plt.subplots()
sns.lineplot(ax=ax, data=df, x="t/ms", y="U/mV", hue="Cell", errorbar=None)

# use total and gap junction conductance to compute weight
w = (rec.gj_g + rec.area * rec.g) / (2 * rec.gj_g + rec.area * rec.g)


# indicate the expected equilibrium potentials
def note(ax, x, y, txt):
    ax.text(x, y, txt, va="center", ha="center", backgroundcolor="w")


for i, j in [[0, 1], [1, 0]]:
    Vj, Vi = args.Vms[j], args.Vms[i]
    Vw = Vi + w.value * (Vj - Vi)
    ax.axhline(Vi, linestyle="dashed", color="black", alpha=0.5)
    ax.axhline(Vw, linestyle="dashed", color="black", alpha=0.5)
    note(ax, 2, Vw, rf"$\tilde U_{j} = U_{j} + w\cdot(U_{j} - U_{i})$")
    note(ax, 2, Vj, rf"$U_{j}$")

ax.set_xlim(0, T.value)

fg.savefig("two_cell_gap_junctions_result.svg")
