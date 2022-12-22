#!/usr/bin/env python3

import arbor
import pandas
import seaborn
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
    tree = arbor.segment_tree()

    # Soma with radius 5 μm and length 2 * radius = 10 μm, (tag = 1)
    s = tree.append(
        arbor.mnpos, arbor.mpoint(-10, 0, 0, 5), arbor.mpoint(0, 0, 0, 5), tag=1
    )

    # Single dendrite with radius 2 μm and length 40 μm, (tag = 2)
    tree.append(s, arbor.mpoint(0, 0, 0, 2), arbor.mpoint(40, 0, 0, 2), tag=2)

    # Label dictionary for cell components
    labels = arbor.label_dict(
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
        arbor.decor()
        .paint('"soma"', arbor.density("hh"))
        .paint('"dend"', arbor.density("pas"))
        # Attach one synapse and gap junction each on their labeled sites
        .place('"synapse_site"', arbor.synapse("expsyn"), "syn")
        .place('"gj_site"', arbor.junction("gj"), "gj")
        # Attach detector to cell root
        .place('"root"', arbor.threshold_detector(-10), "detector")
    )

    return arbor.cable_cell(tree, decor, labels)


# Create a recipe that generates connected chains of cells
class chain_recipe(arbor.recipe):
    def __init__(self, ncells_per_chain, nchains):
        arbor.recipe.__init__(self)
        self.nchains = nchains
        self.ncells_per_chain = ncells_per_chain
        self.props = arbor.neuron_cable_properties()

    def num_cells(self):
        return self.ncells_per_chain * self.nchains

    def cell_description(self, gid):
        return make_cable_cell(gid)

    def cell_kind(self, gid):
        return arbor.cell_kind.cable

    # Create synapse connection between last cell of one chain and first cell of following chain
    def connections_on(self, gid):
        if (gid == 0) or (gid % self.ncells_per_chain > 0):
            return []
        else:
            src = gid - 1
            w = 0.05
            d = 10
            return [arbor.connection((src, "detector"), "syn", w, d)]

    # Create gap junction connections between a cell within a chain and its neighbor(s)
    def gap_junctions_on(self, gid):
        conns = []

        chain_begin = int(gid / self.ncells_per_chain) * self.ncells_per_chain
        chain_end = chain_begin + self.ncells_per_chain

        next_cell = gid + 1
        prev_cell = gid - 1

        if next_cell < chain_end:
            conns.append(arbor.gap_junction_connection((gid + 1, "gj"), "gj", 0.015))
        if prev_cell >= chain_begin:
            conns.append(arbor.gap_junction_connection((gid - 1, "gj"), "gj", 0.015))

        return conns

    # Event generator at first cell
    def event_generators(self, gid):
        if gid == 0:
            sched = arbor.explicit_schedule([1])
            weight = 0.1
            return [arbor.event_generator("syn", weight, sched)]
        return []

    # Place a probe at the root of each cell
    def probes(self, gid):
        return [arbor.cable_probe_membrane_voltage('"root"')]

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
sim = arbor.simulation(recipe)

# Set spike generators to record
sim.record(arbor.spike_recording.all)

# Sampler
handles = [sim.sample((gid, 0), arbor.regular_schedule(0.1)) for gid in range(ncells)]

# Run simulation for 100 ms
sim.run(100)
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
        pandas.DataFrame(
            {"t/ms": samples[:, 0], "U/mV": samples[:, 1], "Cell": f"cell {gid}"}
        )
    )

df = pandas.concat(df_list, ignore_index=True)
seaborn.relplot(data=df, kind="line", x="t/ms", y="U/mV", hue="Cell", errorbar=None)
plt.show()
