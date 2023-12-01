#!/usr/bin/env python3
# This script is included in documentation. Adapt line numbers if touched.

import arbor as A
from arbor import units as U
import pandas as pd
from math import sqrt

# Run with srun -n NJOBS python network_ring_mpi.py

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
        .paint('"soma"', A.density("hh")).paint('"dend"', A.density("pas"))
        # (4) Attach a single synapse.
        .place('"synapse_site"', A.synapse("expsyn"), "syn")
        # Attach a detector with threshold of -10 mV.
        .place('"root"', A.threshold_detector(-10), "detector")
    )

    return A.cable_cell(tree, decor, labels)


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
    def cell_kind(self, gid):
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
    def probes(self, gid):
        return [A.cable_probe_membrane_voltage('"root"')]

    def global_properties(self, kind):
        return self.props


# (11) Instantiate recipe
ncells = 500
recipe = ring_recipe(ncells)

# (12) Create an MPI communicator, and use it to create a hardware context
A.mpi_init()
comm = A.mpi_comm()
print(comm)
context = A.context(mpi=comm)
print(context)

# (13) Create a default domain decomposition and simulation
sim = A.simulation(recipe, context)

# (14) Set spike generators to record
sim.record(A.spike_recording.all)

# (15) Attach a sampler to the voltage probe on cell 0. Sample rate of 1 sample every ms.
# Sampling period increased w.r.t network_ring.py to reduce amount of data
handles = [sim.sample((gid, 0), A.regular_schedule(1 * U.ms)) for gid in range(ncells)]

# (16) Run simulation
sim.run(ncells * 5 * U.ms)
print("Simulation finished")

# (17) Store the recorded voltages
print("Storing results ...")
df_list = []
for gid in range(ncells):
    if len(sim.samples(handles[gid])):
        samples, meta = sim.samples(handles[gid])[0]
        df_list.append(
            pd.DataFrame(
                {"t/ms": samples[:, 0], "U/mV": samples[:, 1], "Cell": f"cell {gid}"}
            )
        )

if len(df_list):
    df = pd.concat(df_list, ignore_index=True)
    df.to_csv(f"result_mpi_{context.rank}.csv", float_format="%g")
