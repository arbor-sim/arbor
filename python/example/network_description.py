#!/usr/bin/env python3
# This script is included in documentation. Adapt line numbers if touched.

import arbor as A
from arbor import units as U
import pandas  # You may have to pip install these
import seaborn  # You may have to pip install these
from math import sqrt


def make_cable_cell(gid):
    # (1) Build a segment tree
    # The dendrite (dend) attaches to the soma and has two simple segments
    # attached.
    #
    #             left
    #            /
    # soma - dend
    #            \
    #             right
    tree = A.segment_tree()
    root = A.mnpos
    # Soma (tag=1) with radius 6 μm, modelled as cylinder of length 2*radius
    soma = tree.append(root, (-12, 0, 0, 6), (0, 0, 0, 6), tag=1)
    # Single dendrite (tag=3) of length 50 μm and radius 2 μm attached to soma.
    dend = tree.append(soma, (0, 0, 0, 2), (50, 0, 0, 2), tag=3)
    # Attach two dendrites (tag=3) of length 50 μm to the end of the first dendrite.
    # Radius tapers from 2 to 0.5 μm over the length of the dendrite.
    l = 50 / sqrt(2)
    _ = tree.append(
        dend,
        (50, 0, 0, 2),
        (50 + l, l, 0, 0.5),
        tag=3,
    )
    # Constant radius of 1 μm over the length of the dendrite.
    _ = tree.append(
        dend,
        (50, 0, 0, 1),
        (50 + l, -l, 0, 1),
        tag=3,
    )

    # Associate labels to tags
    labels = A.label_dict(
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
        A.decor()
        # Put hh dynamics on soma, and passive properties on the dendrites.
        .paint('"soma"', A.density("hh"))
        .paint('"dend"', A.density("pas"))
        # (4) Attach a single synapse.
        .place('"synapse_site"', A.synapse("expsyn"), "syn")
        # Attach a detector with threshold of -10 mV.
        .place('"root"', A.threshold_detector(-10 * U.mV), "detector")
    )

    return A.cable_cell(tree, decor, labels)


# (5) Create a recipe that generates a network of connected cells.
class random_ring_recipe(A.recipe):
    def __init__(self, ncells):
        # Base class constructor must be called first for proper initialization.
        A.recipe.__init__(self)
        self.ncells = ncells
        self.props = A.neuron_cable_properties()

    # (6) Returns the total number of cells in the model; must be implemented.
    def num_cells(self):
        return self.ncells

    # (7) The cell_description method returns a cell
    def cell_description(self, gid):
        return make_cable_cell(gid)

    # Return the type of cell; must be implemented and match cell_description.
    def cell_kind(self, _):
        return A.cell_kind.cable

    # (8) Descripe network
    def network_description(self):
        seed = 42

        # create a chain
        chain = f"(chain (gid-range 0 {self.ncells}))"
        # connect front and back of chain to form ring
        ring = f"(join {chain} (intersect (source-cell {self.ncells - 1}) (target-cell 0)))"

        # Create random connections with probability inversely proportional to the distance within a
        # radius
        max_dist = 400.0  # μm
        probability = f"(div (sub {max_dist} (distance)) {max_dist})"
        rand = f"(intersect (random {seed} {probability}) (distance-lt {max_dist}))"

        # combine ring with random selection
        s = f"(join {ring} {rand})"
        # restrict to inter-cell connections and certain source / target labels
        s = f'(intersect {s} (inter-cell) (source-label "detector") (target-label "syn"))'

        # fixed weight for connections in ring
        w_ring = "(scalar 0.01)"
        # random normal distributed weight with mean 0.02 μS, standard deviation 0.01 μS
        # and truncated to [0.005, 0.035]
        w_rand = f"(truncated-normal-distribution {seed} 0.02 0.01 0.005 0.035)"

        # combine into single weight expression
        w = f"(if-else {ring} {w_ring} {w_rand})"

        # fixed delay
        d = "(scalar 5.0)"  # ms delay

        return A.network_description(s, w, d, {})

    # (9) Attach a generator to the first cell in the ring.
    def event_generators(self, gid):
        if gid == 0:
            sched = A.explicit_schedule([1 * U.ms])  # one event at 1 ms
            weight = 0.1  # 0.1 μS on expsyn
            return [A.event_generator("syn", weight, sched)]
        return []

    # (10) Place a probe at the root of each cell.
    def probes(self, gid):
        return [A.cable_probe_membrane_voltage('"root"', "Um")]

    def global_properties(self, _):
        return self.props


# (11) Instantiate recipe
ncells = 4
recipe = random_ring_recipe(ncells)

# (12) Create a simulation using the default settings:
# - Use all threads available
# - Use round-robin distribution of cells across groups with one cell per group
# - Use GPU if present
# - No MPI
# Other constructors of simulation can be used to change all of these.
sim = A.simulation(recipe)

# (13) Set spike generators to record
sim.record(A.spike_recording.all)

# (14) Attach a sampler to the voltage probe on cell 0. Sample rate of 10 sample every ms.
handles = [
    sim.sample((gid, "Um"), A.regular_schedule(0.1 * U.ms)) for gid in range(ncells)
]

# (15) Run simulation for 100 ms
sim.run(100 * U.ms)
print("Simulation finished")

# (16) Print spike times
print("spikes:")
for sp in sim.spikes():
    print(" ", sp)

# (17) Plot the recorded voltages over time.
print("Plotting results ...")
dfs = []
for gid in range(ncells):
    samples, meta = sim.samples(handles[gid])[0]
    dfs.append(
        pandas.DataFrame(
            {"t/ms": samples[:, 0], "U/mV": samples[:, 1], "Cell": f"cell {gid}"}
        )
    )
df = pandas.concat(dfs, ignore_index=True)
seaborn.relplot(
    data=df, kind="line", x="t/ms", y="U/mV", hue="Cell", errorbar=None
).savefig("network_description_result.svg")
