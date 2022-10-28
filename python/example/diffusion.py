#!/usr/bin/env python3

import arbor as A
import seaborn as sns
import matplotlib.pyplot as plt


class recipe(A.recipe):
    def __init__(self, cell, probes):
        A.recipe.__init__(self)
        self.the_cell = cell
        self.the_probes = probes
        self.the_props = A.neuron_cable_properties()
        self.the_props.catalogue = A.default_catalogue()

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
        return [A.event_generator("Zap", 0.005, A.explicit_schedule([0.0]))]


tree = A.segment_tree()
s = tree.append(A.mnpos, A.mpoint(-3, 0, 0, 3), A.mpoint(3, 0, 0, 3), tag=1)
_ = tree.append(s, A.mpoint(3, 0, 0, 1), A.mpoint(33, 0, 0, 1), tag=3)

dec = A.decor()
dec.set_ion("na", int_con=0.0, diff=0.005)
dec.place("(location 0 0.5)", A.synapse("inject/x=na", {"alpha": 200.0}), "Zap")
dec.paint("(all)", A.density("decay/x=na"))
dec.discretization(A.cv_policy("(max-extent 5)"))

# Set up ion diffusion
dec.set_ion("na", int_con=1.0, ext_con=140, rev_pot=50, diff=0.005)
dec.paint("(tag 1)", ion_name="na", int_con=100.0, diff=0.01)

prb = [
    A.cable_probe_ion_diff_concentration_cell("na"),
]
cel = A.cable_cell(tree, dec)
rec = recipe(cel, prb)
sim = A.simulation(rec)
hdl = (sim.sample((0, 0), A.regular_schedule(0.1)),)

sim.run(tfinal=0.5)

sns.set_theme()
fg, ax = plt.subplots()
for h in hdl:
    for d, m in sim.samples(h):
        # Plot
        for lbl, ix in zip(m, range(1, d.shape[1])):
            ax.plot(d[:, 0], d[:, ix], label=lbl)
        # Table
        print("Sodium concentration (NaD/mM)")
        print("|-" + "-+-".join("-" * 20 for _ in range(d.shape[1])) + "-|")
        print(
            "| Time (ms)            | " + " | ".join(f"{str(l):<20}" for l in m) + " |"
        )
        print("|-" + "-+-".join("-" * 20 for _ in range(d.shape[1])) + "-|")
        for ix in range(d.shape[0]):
            print("| " + " | ".join(f"{v:>20.3f}" for v in d[ix, :]) + " |")
        print("|-" + "-+-".join("-" * 20 for _ in range(d.shape[1])) + "-|")
ax.legend()
fg.savefig("results.pdf")
