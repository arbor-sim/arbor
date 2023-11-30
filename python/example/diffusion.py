#!/usr/bin/env python3

import arbor as A
from arbor import units as U
import seaborn as sns
import matplotlib.pyplot as plt


class recipe(A.recipe):
    def __init__(self, cell, probes):
        A.recipe.__init__(self)
        self.the_cell = cell
        self.the_probes = probes
        self.the_props = A.neuron_cable_properties()

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
        return [A.event_generator("Zap", 0.005, A.explicit_schedule([0.0 * U.ms]))]


tree = A.segment_tree()
s = A.mnpos
s = tree.append(s, (-3, 0, 0, 3), (3, 0, 0, 3), tag=1)
_ = tree.append(s, (3, 0, 0, 1), (33, 0, 0, 1), tag=3)

dec = (
    A.decor()
    .set_ion("na", int_con=0.0, diff=0.005)
    .place("(location 0 0.5)", A.synapse("inject/x=na", {"alpha": 200.0}), "Zap")
    .paint("(all)", A.density("decay/x=na"))
    .discretization(A.cv_policy("(max-extent 5)"))
    # Set up ion diffusion
    .set_ion("na", int_con=1.0, ext_con=140, rev_pot=50, diff=0.005)
    .paint("(tag 1)", ion="na", int_con=100.0, diff=0.01)
)

prb = [
    A.cable_probe_ion_diff_concentration_cell("na", "nad"),
]
cel = A.cable_cell(tree, dec)
rec = recipe(cel, prb)
sim = A.simulation(rec)
hdl = (sim.sample((0, "nad"), A.regular_schedule(0.1 * U.ms)),)

sim.run(tfinal=0.5 * U.ms)

sns.set_theme()
fg, ax = plt.subplots()
for h in hdl:
    for d, m in sim.samples(h):
        # Plot
        for lbl, ix in zip(m, range(1, d.shape[1])):
            ax.plot(d[:, 0], d[:, ix], label=lbl)
        W = 8
        # Table
        print("Sodium concentration (NaD/mM)")
        print("|-" + "-+-".join("-" * W for _ in range(d.shape[1])) + "-|")
        print(f"| {'Time/ms':<{W}} | " + " | ".join(f"{l.prox:<{W}}" for l in m) + " |")
        print("|-" + "-+-".join("-" * W for _ in range(d.shape[1])) + "-|")
        for ix in range(d.shape[0]):
            print("| " + " | ".join(f"{v:>{W}.3f}" for v in d[ix, :]) + " |")
        print("|-" + "-+-".join("-" * W for _ in range(d.shape[1])) + "-|")
ax.legend()
fg.savefig("results.pdf")
