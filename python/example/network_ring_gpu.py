#!/usr/bin/env python3
# This script is included in documentation. Adapt line numbers if touched.

import arbor as A
from arbor import units as U
import pandas as pd  # You may have to pip install these
import seaborn as sns  # You may have to pip install these
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


def make_cable_cell(_):
    # (1) Build a segment tree
    tree = A.segment_tree()

    # Soma (tag=1) with radius 6 μm, modelled as cylinder of length 2*radius
    s = A.mnpos
    s = tree.append(s, (-12, 0, 0, 6), (0, 0, 0, 6), tag=1)

    # Single dendrite (tag=3) of length 50 μm and radius 2 μm attached to soma.
    b0 = tree.append(s, (0, 0, 0, 2), (50, 0, 0, 2), tag=3)

    # Attach two dendrites (tag=3) of length 50 μm to the end of the first dendrite.
    # As there's no further use for them, we discard the returned handles.
    # (b1) Radius tapers from 2 to 0.5 μm over the length of the dendrite.
    _ = tree.append(
        b0,
        (50, 0, 0, 2),
        (50 + 50 / sqrt(2), 50 / sqrt(2), 0, 0.5),
        tag=3,
    )
    # (b2) Constant radius of 1 μm over the length of the dendrite.
    _ = tree.append(
        b0,
        (50, 0, 0, 1),
        (50 + 50 / sqrt(2), -50 / sqrt(2), 0, 1),
        tag=3,
    )

    # Associate labels to tags
    labels = A.label_dict()
    labels["soma"] = "(tag 1)"
    labels["dend"] = "(tag 3)"

    # (2) Mark location for synapse at the midpoint of branch 1 (the first dendrite).
    labels["synapse_site"] = "(location 1 0.5)"
    # Mark the root of the tree.
    labels["root"] = "(root)"

    # (3) Create a decor and a cable_cell
    decor = A.decor()

    # Put hh dynamics on soma, and passive properties on the dendrites.
    decor.paint('"soma"', A.density("hh"))
    decor.paint('"dend"', A.density("pas"))

    # (4) Attach a single synapse.
    decor.place('"synapse_site"', A.synapse("expsyn"), "syn")

    # Attach a detector with threshold of -10 mV.
    decor.place('"root"', A.threshold_detector(-10 * U.mV), "detector")

    cell = A.cable_cell(tree, decor, labels)

    return cell


# (5) Create a recipe that generates a network of connected cells.
class ring_recipe(A.recipe):
    def __init__(self, ncells):
        # The base C++ class constructor must be called first, to ensure that
        # all memory in the C++ class is initialized correctly.
        A.recipe.__init__(self)
        self.ncells = ncells
        self.props = A.neuron_cable_properties()

    # (6) The num_cells method that returns the total number of cells in the model
    # must be implemented.
    def num_cells(self):
        return self.ncells

    # (7) The cell_description method returns a cell
    def cell_description(self, gid):
        return make_cable_cell(gid)

    # The kind method returns the type of cell with gid.
    # Note: this must agree with the type returned by cell_description.
    def cell_kind(self, _):
        return A.cell_kind.cable

    # (8) Make a ring network. For each gid, provide a list of incoming connections.
    def connections_on(self, gid):
        src = (gid - 1) % self.ncells
        w = 0.01  # 0.01 μS on expsyn
        d = 5  # ms delay
        return [A.connection((src, "detector"), "syn", w, d)]

    # (9) Attach a generator to the first cell in the ring.
    def event_generators(self, gid):
        if gid == 0:
            sched = A.explicit_schedule([1 * U.ms])  # one event at 1 ms
            weight = 0.1  # 0.1 μS on expsyn
            return [A.event_generator("syn", weight, sched)]
        return []

    # (10) Place a probe at the root of each cell.
    def probes(self, _):
        return [A.cable_probe_membrane_voltage('"root"', "Um")]

    def global_properties(self, _):
        return self.props


# (11) Set up the hardware context
# gpu_id set to None will not use a GPU.
# gpu_id=0 instructs Arbor to the first GPU present in your system
context = A.context(gpu_id=None)
print(context)

# (12) Set up and start the meter manager
meters = A.meter_manager()
meters.start(context)

# (13) Instantiate recipe
ncells = 50
recipe = ring_recipe(ncells)
meters.checkpoint("recipe-create", context)

# (14) Define a hint at to the execution.
hint = A.partition_hint()
hint.prefer_gpu = True
hint.gpu_group_size = 1000
print(hint)
hints = {A.cell_kind.cable: hint}

# (15) Domain decomp
decomp = A.partition_load_balance(recipe, context, hints)
print(decomp)
meters.checkpoint("load-balance", context)

# (16) Simulation init and set spike generators to record
sim = A.simulation(recipe, context, decomp)
sim.record(A.spike_recording.all)
handles = [
    sim.sample((gid, "Um"), A.regular_schedule(1 * U.ms)) for gid in range(ncells)
]
meters.checkpoint("simulation-init", context)

# (17) Run simulation
sim.run(ncells * 5 * U.ms)
print("Simulation finished")
meters.checkpoint("simulation-run", context)

# (18) Results
# Print profiling information
print(f"{A.meter_report(meters, context)}")

# Print spike times
print("spikes:")
for (gid, lid), t in sim.spikes():
    print(f" * t={t:.3f}ms gid={gid} lid={lid}")

# Plot the recorded voltages over time.
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
sns.relplot(
    data=df, kind="line", x="t/ms", y="U/mV", hue="Cell", errorbar=None
).savefig("network_ring_gpu_result.svg")
