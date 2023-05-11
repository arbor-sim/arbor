#!/usr/bin/env python3
# This script is included in documentation. Adapt line numbers if touched.

import arbor
import pandas  # You may have to pip install these
import seaborn  # You may have to pip install these
from math import sqrt
import math

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

    # (b0) Single dendrite (tag=3) of length 50 μm and radius 2 μm attached to soma.
    b0 = tree.append(s, arbor.mpoint(0, 0, 0, 2), arbor.mpoint(0, 0, 50, 2), tag=3)

    # Attach two dendrites (tag=3) of length 50 μm to the end of the first dendrite.
    # (b1) Radius tapers from 2 to 0.5 μm over the length of the dendrite.
    tree.append(
        b0,
        arbor.mpoint(0, 0, 50, 2),
        arbor.mpoint(0, 50 / sqrt(2), 50 + 50 / sqrt(2) , 0.5),
        tag=3,
    )
    # (b2) Constant radius of 1 μm over the length of the dendrite.
    tree.append(
        b0,
        arbor.mpoint(0, 0, 50, 1),
        arbor.mpoint(0, -50 / sqrt(2), 50 + 50 / sqrt(2), 1),
        tag=3,
    )

    # Associate labels to tags
    labels = arbor.label_dict(
        {
            "soma": "(tag 1)",
            "dend": "(tag 3)",
            # (2) Mark location for synapse at the midpoint of branch 1 (the first dendrite).
            "synapse_site": "(location 1 0.5)",
            # Mark the root of the tree.
            "root": "(root)",
        }
    )

    # (3) Create a decor and a cable_cell
    decor = (
        arbor.decor()
        # Put hh dynamics on soma, and passive properties on the dendrites.
        .paint('"soma"', arbor.density("hh")).paint('"dend"', arbor.density("pas"))
        # (4) Attach a single synapse.
        .place('"synapse_site"', arbor.synapse("expsyn"), "syn")
        # Attach a detector with threshold of -10 mV.
        .place('"root"', arbor.threshold_detector(-10), "detector")
    )

    return arbor.cable_cell(tree, decor, labels)


# (5) Create a recipe that generates a network of connected cells.
class random_ring_recipe(arbor.recipe):
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

    def cell_isometry(self, gid):
        # place cells with equal distance on a circle
        radius = 500.0 # μm
        angle = 2.0 * math.pi * gid / self.ncells
        return arbor.isometry.translate(radius * math.cos(angle), radius * math.sin(angle), 0)

    def network_description(self):
        seed = 42

        # create a chain
        ring = f"(chain (gid-range 0 {self.ncells}))"
        # connect front and back of chain to form ring
        ring = f"(join {ring} (intersect (source-cell {self.ncells - 1}) (destination-cell 0)))"

        # Create random connections with probability inversely proportional to the distance within a
        # radius
        max_dist = 400.0 # μm
        probability = f"(div (sub {max_dist} (distance)) {max_dist})"
        rand = f"(intersect (random {seed} {probability}) (distance-lt {max_dist}))"

        # combine ring with random selection
        s = f"(join {ring} {rand})"
        # restrict to inter-cell connections and certain source / destination labels
        s = f"(intersect {s} (inter-cell) (source-label \"detector\") (destination-label \"syn\"))"

        # normal distributed weight with mean 0.02 μS, standard deviation 0.01 μS
        # and truncated to [0.005, 0.035]
        w = f"(truncated-normal-distribution {seed} 0.02 0.01 0.005 0.035)"
        # fixed delay
        d = "(scalar 5.0)"  # ms delay

        return arbor.network_description(s, w, d, {})

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


# (11) Instantiate recipe
ncells = 4
recipe = random_ring_recipe(ncells)

# (12) Create an execution context using all locally available threads and simulation
ctx = arbor.context("avail_threads")
sim = arbor.simulation(recipe, ctx)

# (13) Set spike generators to record
sim.record(arbor.spike_recording.all)

# (14) Attach a sampler to the voltage probe on cell 0. Sample rate of 10 sample every ms.
handles = [sim.sample((gid, 0), arbor.regular_schedule(0.1)) for gid in range(ncells)]

# (15) Run simulation for 100 ms
sim.run(100)
print("Simulation finished")

# (16) Print spike times
print("spikes:")
for sp in sim.spikes():
    print(" ", sp)

# (17) Plot the recorded voltages over time.
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
).savefig("network_ring_result.svg")
