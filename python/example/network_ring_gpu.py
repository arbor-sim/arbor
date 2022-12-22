#!/usr/bin/env python3
# This script is included in documentation. Adapt line numbers if touched.

import arbor
import pandas  # You may have to pip install these
import seaborn  # You may have to pip install these
from math import sqrt

# Construct a cell with the following morphology.
# The soma (at the root of the tree) is marked 's', and
# the end of each branch i is marked 'bi'.
#
#         b1
#        /
# s----b0
#        \
#         b2


def make_cable_cell(gid):
    # (1) Build a segment tree
    tree = arbor.segment_tree()

    # Soma (tag=1) with radius 6 μm, modelled as cylinder of length 2*radius
    s = tree.append(
        arbor.mnpos, arbor.mpoint(-12, 0, 0, 6), arbor.mpoint(0, 0, 0, 6), tag=1
    )

    # Single dendrite (tag=3) of length 50 μm and radius 2 μm attached to soma.
    b0 = tree.append(s, arbor.mpoint(0, 0, 0, 2), arbor.mpoint(50, 0, 0, 2), tag=3)

    # Attach two dendrites (tag=3) of length 50 μm to the end of the first dendrite.
    # As there's no further use for them, we discard the returned handles.
    # (b1) Radius tapers from 2 to 0.5 μm over the length of the dendrite.
    _ = tree.append(
        b0,
        arbor.mpoint(50, 0, 0, 2),
        arbor.mpoint(50 + 50 / sqrt(2), 50 / sqrt(2), 0, 0.5),
        tag=3,
    )
    # (b2) Constant radius of 1 μm over the length of the dendrite.
    _ = tree.append(
        b0,
        arbor.mpoint(50, 0, 0, 1),
        arbor.mpoint(50 + 50 / sqrt(2), -50 / sqrt(2), 0, 1),
        tag=3,
    )

    # Associate labels to tags
    labels = arbor.label_dict()
    labels["soma"] = "(tag 1)"
    labels["dend"] = "(tag 3)"

    # (2) Mark location for synapse at the midpoint of branch 1 (the first dendrite).
    labels["synapse_site"] = "(location 1 0.5)"
    # Mark the root of the tree.
    labels["root"] = "(root)"

    # (3) Create a decor and a cable_cell
    decor = arbor.decor()

    # Put hh dynamics on soma, and passive properties on the dendrites.
    decor.paint('"soma"', arbor.density("hh"))
    decor.paint('"dend"', arbor.density("pas"))

    # (4) Attach a single synapse.
    decor.place('"synapse_site"', arbor.synapse("expsyn"), "syn")

    # Attach a detector with threshold of -10 mV.
    decor.place('"root"', arbor.threshold_detector(-10), "detector")

    cell = arbor.cable_cell(tree, decor, labels)

    return cell


# (5) Create a recipe that generates a network of connected cells.
class ring_recipe(arbor.recipe):
    def __init__(self, ncells):
        # The base C++ class constructor must be called first, to ensure that
        # all memory in the C++ class is initialized correctly.
        arbor.recipe.__init__(self)
        self.ncells = ncells
        self.props = arbor.neuron_cable_properties()

    # (6) The num_cells method that returns the total number of cells in the model
    # must be implemented.
    def num_cells(self):
        return self.ncells

    # (7) The cell_description method returns a cell
    def cell_description(self, gid):
        return make_cable_cell(gid)

    # The kind method returns the type of cell with gid.
    # Note: this must agree with the type returned by cell_description.
    def cell_kind(self, gid):
        return arbor.cell_kind.cable

    # (8) Make a ring network. For each gid, provide a list of incoming connections.
    def connections_on(self, gid):
        src = (gid - 1) % self.ncells
        w = 0.01  # 0.01 μS on expsyn
        d = 5  # ms delay
        return [arbor.connection((src, "detector"), "syn", w, d)]

    # (9) Attach a generator to the first cell in the ring.
    def event_generators(self, gid):
        if gid == 0:
            sched = arbor.explicit_schedule([1])  # one event at 1 ms
            weight = 0.1  # 0.1 μS on expsyn
            return [arbor.event_generator("syn", weight, sched)]
        return []

    # (10) Place a probe at the root of each cell.
    def probes(self, gid):
        return [arbor.cable_probe_membrane_voltage('"root"')]

    def global_properties(self, kind):
        return self.props


# (11) Set up the hardware context
# gpu_id set to None will not use a GPU.
# gpu_id=0 instructs Arbor to the first GPU present in your system
context = arbor.context(threads="avail_threads", gpu_id=None)
print(context)

# (12) Set up and start the meter manager
meters = arbor.meter_manager()
meters.start(context)

# (13) Instantiate recipe
ncells = 50
recipe = ring_recipe(ncells)
meters.checkpoint("recipe-create", context)

# (14) Define a hint at to the execution.
hint = arbor.partition_hint()
hint.prefer_gpu = True
hint.gpu_group_size = 1000
print(hint)
hints = {arbor.cell_kind.cable: hint}

# (15) Domain decomp
decomp = arbor.partition_load_balance(recipe, context, hints)
print(decomp)
meters.checkpoint("load-balance", context)

# (16) Simulation init and set spike generators to record
sim = arbor.simulation(recipe, context, decomp)
sim.record(arbor.spike_recording.all)
handles = [sim.sample((gid, 0), arbor.regular_schedule(1)) for gid in range(ncells)]
meters.checkpoint("simulation-init", context)

# (17) Run simulation
sim.run(ncells * 5)
print("Simulation finished")
meters.checkpoint("simulation-run", context)

# (18) Results
# Print profiling information
print(f"{arbor.meter_report(meters, context)}")

# Print spike times
print("spikes:")
for sp in sim.spikes():
    print(" ", sp)

# Plot the recorded voltages over time.
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
seaborn.relplot(
    data=df, kind="line", x="t/ms", y="U/mV", hue="Cell", errorbar=None
).savefig("network_ring_gpu_result.svg")
