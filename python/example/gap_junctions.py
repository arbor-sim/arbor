#!/usr/bin/env python3

import arbor as A
from arbor import units as U
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Construct chains of cells linked with gap junctions,
# Chains are connected by synapses.
# An event generator is attached to the first cell in the network.
#
# c --gj-- c --gj-- c --gj-- c --gj-- c
#                                     |
#                                    syn
#                                     |
# c --gj-- c --gj-- c --gj-- c --gj-- c
#
# The individual cells consist of a soma and one dendrite


def make_cable_cell(gid):
    # Build a segment tree
    tree = A.segment_tree()

    # Soma with radius 5 μm and length 2 * radius = 10 μm, (tag = 1)
    s = tree.append(A.mnpos, A.mpoint(-10, 0, 0, 5), A.mpoint(0, 0, 0, 5), tag=1)

    # Single dendrite with radius 2 μm and length 40 μm, (tag = 2)
    tree.append(s, A.mpoint(0, 0, 0, 2), A.mpoint(40, 0, 0, 2), tag=2)

    # Label dictionary for cell components
    labels = A.label_dict(
        {
            # Mark location for synapse site at midpoint of dendrite (branch 0  soma + dendrite)
            "synapse_site": "(location 0 0.6)",
            # Gap junction site at connection point of soma and dendrite
            "gj_site": "(location 0 0.2)",
            # Label root of the tree
            "root": "(root)",
        }
    ).add_swc_tags()

    # Paint dynamics onto the cell, hh on soma and passive properties on dendrite
    decor = (
        A.decor()
        .paint('"soma"', A.density("hh"))
        .paint('"dend"', A.density("pas"))
        # Attach one synapse and gap junction each on their labeled sites
        .place('"synapse_site"', A.synapse("expsyn"), "syn")
        .place('"gj_site"', A.junction("gj"), "gj")
        # Attach detector to cell root
        .place('"root"', A.threshold_detector(-10 * U.ms), "detector")
    )

    return A.cable_cell(tree, decor, labels)


# Create a recipe that generates connected chains of cells
class chain_recipe(A.recipe):
    def __init__(self, ncells_per_chain, nchains):
        A.recipe.__init__(self)
        self.nchains = nchains
        self.ncells_per_chain = ncells_per_chain
        self.props = A.neuron_cable_properties()

    def num_cells(self):
        return self.ncells_per_chain * self.nchains

    def cell_description(self, gid):
        return make_cable_cell(gid)

    def cell_kind(self, gid):
        return A.cell_kind.cable

    # Create synapse connection between last cell of one chain and first cell of following chain
    def connections_on(self, gid):
        if (gid == 0) or (gid % self.ncells_per_chain > 0):
            return []
        else:
            src = gid - 1
            return [A.connection((src, "detector"), "syn", 0.05, 10 * U.ms)]

    # Create gap junction connections between a cell within a chain and its neighbor(s)
    def gap_junctions_on(self, gid):
        conns = []

        chain_begin = int(gid / self.ncells_per_chain) * self.ncells_per_chain
        chain_end = chain_begin + self.ncells_per_chain

        next_cell = gid + 1
        prev_cell = gid - 1

        if next_cell < chain_end:
            conns.append(A.gap_junction_connection((gid + 1, "gj"), "gj", 0.015))
        if prev_cell >= chain_begin:
            conns.append(A.gap_junction_connection((gid - 1, "gj"), "gj", 0.015))

        return conns

    # Event generator at first cell
    def event_generators(self, gid):
        if gid == 0:
            sched = A.explicit_schedule([1 * U.ms])
            weight = 0.1
            return [A.event_generator("syn", weight, sched)]
        return []

    # Place a probe at the root of each cell
    def probes(self, gid):
        return [A.cable_probe_membrane_voltage('"root"', "Um")]

    def global_properties(self, kind):
        return self.props


# Number of cells per chain
ncells_per_chain = 5

# Number of chains
nchains = 3

# Total number of cells
ncells = nchains * ncells_per_chain

# Instantiate recipe
recipe = chain_recipe(ncells_per_chain, nchains)

# Create a default simulation
sim = A.simulation(recipe)

# Set spike generators to record
sim.record(A.spike_recording.all)

# Sampler
handles = [
    sim.sample((gid, "Um"), A.regular_schedule(0.1 * U.ms)) for gid in range(ncells)
]

# Run simulation
sim.run(100 * U.ms)
print("Simulation finished")

# Print spike times
print("spikes:")
for sp in sim.spikes():
    print(" ", sp)

# Plot the results
print("Plotting results ...")
df_list = []
for gid in range(ncells):
    samples, meta = sim.samples(handles[gid])[0]
    df_list.append(
        pd.DataFrame(
            {"t/ms": samples[:, 0], "U/mV": samples[:, 1], "Cell": f"cell {gid}"}
        )
    )

df = pd.concat(df_list, ignore_index=True)
sns.relplot(data=df, kind="line", x="t/ms", y="U/mV", hue="Cell", errorbar=None)
plt.show()
