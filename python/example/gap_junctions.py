#!/usr/bin/env python3

import arbor
import pandas, seaborn
import matplotlib.pyplot as plt
import mpi4py.MPI as mpi

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

    # Soma with radius 5 μm and length 2 * radius = 10 m, (tag = 1)
    s = tree.append(arbor.mnpos, arbor.mpoint(-10, 0, 0, 5), arbor.mpoint(0, 0, 0, 5), tag=1)

    # Single dendrite with radius 2 μm and length 40 μm, (tag = 2)
    b = tree.append(s, arbor.mpoint(0, 0, 0, 2), arbor.mpoint(40, 0, 0, 2), tag=2)

    # Label dictionary for cell components
    labels = arbor.label_dict()
    labels['soma'] = '(tag 1)'
    labels['dend'] = '(tag 2)'

    # Mark location for synapse site at midpoint of dendrite (branch 0 = soma + dendrite)
    labels['synapse_site'] = '(location 0 0.6)'

    # Gap junction site at connection point of soma and dendrite
    labels['gj_site_0'] = '(location 0 0.2)'
    labels['gj_site_1'] = '(location 0 0.5)'

    # Label root of the tree
    labels['root'] = '(root)'

    # Paint dynamics onto the cell, hh on soma and passive properties on dendrite
    decor = arbor.decor()
    decor.paint('"soma"', arbor.density("hh"))
    decor.paint('"dend"', arbor.density("pas"))

    #split into multiple cvs
    policy = arbor.cv_policy_explicit('(location 0 0.35)')
    #policy = arbor.cv_policy_single()
    decor.discretization(policy)
    #decor.discretization("(max-extent 9)")

    # Attach one synapse and gap junction each on their labeled sites
    decor.place('"synapse_site"', arbor.synapse('expsyn'), 'syn')
    decor.place('"gj_site_0"', arbor.junction('gj'), 'gj_0')
    decor.place('"gj_site_1"', arbor.junction('gj'), 'gj_1')

    # Attach spike detector to cell root
    decor.place('"root"', arbor.spike_detector(-10), 'detector')

    cell = arbor.cable_cell(tree, labels, decor)

    return cell

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
            src = gid-1
            w   = 0.05
            d   = 10
            return [arbor.connection((src,'detector'), 'syn', w, d)]
        return []
    
    # Create gap junction connections between a cell within a chain and its neighbor(s)
    def gap_junctions_on(self, gid):
        conns = []

        chain_begin = int(gid/self.ncells_per_chain) * self.ncells_per_chain
        chain_end   = chain_begin + self.ncells_per_chain

        next_cell = gid + 1
        prev_cell = gid - 1

        if next_cell < chain_end:
            conns.append(arbor.gap_junction_connection((gid+1, 'gj_0'), 'gj_0', 0.2))
            #conns.append(arbor.gap_junction_connection((gid+1, 'gj_0'), 'gj_0', 0.1*(gid+1)))
        if prev_cell >= chain_begin:
            conns.append(arbor.gap_junction_connection((gid-1, 'gj_0'), 'gj_0', 0.2))
            #conns.append(arbor.gap_junction_connection((gid-1, 'gj_0'), 'gj_0', 0.1*(gid+1)))

        return conns

    # Event generator at first cell
    def event_generators(self, gid):
        if (gid == 0):
            sched = arbor.explicit_schedule([1])
            weight = 0.1
            return [arbor.event_generator('syn', weight, sched)]
        return []

    # Place a probe at the root of each cell
    def probes(self, gid):
        return []
        return [arbor.cable_probe_membrane_voltage('"root"')]

    def global_properties(self, kind):
        return self.props

# Number of cells per chain
ncells_per_chain = 5

# Number of chains
nchains = 1

# Total number of cells
ncells = nchains * ncells_per_chain

#Instantiate recipe
recipe = chain_recipe(ncells_per_chain, nchains)

#context = arbor.context()
#decomp = arbor.partition_load_balance(recipe, context)
#sim = arbor.simulation(recipe, decomp, context)

#context = arbor.context(gpu_id = None)
#groups = [arbor.group_description(arbor.cell_kind.cable, [0, 1], arbor.backend.multicore), arbor.group_description(arbor.cell_kind.cable, [2, 3], arbor.backend.multicore)]
#decomp = arbor.partition_by_group(recipe, context, groups)
#sim = arbor.simulation(recipe, decomp, context)

alloc   = arbor.proc_allocation(1, None)
comm    = mpi.COMM_WORLD
#print(f"rank={comm.rank} size={comm.size}")

xs = [comm.rank]*(comm.rank + 1)
gxs = comm.allgather(xs)
print(gxs)

context = arbor.context(alloc, comm)
print(context)

if comm.rank == 0:
    gs = [[0,1]]
elif comm.rank == 1:
    gs = [[2,3,4]]
#elif comm.rank == 3:
#    gs = [[3]]


groups = [arbor.group_description(arbor.cell_kind.cable, g, arbor.backend.multicore) for g in gs]
decomp = arbor.partition_by_group(recipe, context, groups)

sim = arbor.simulation(recipe, decomp, context)

dt = 0.025

sim.set_binning_policy(arbor.binning.regular, dt)

# Set spike generators to record
sim.record(arbor.spike_recording.all)

# Sampler
#handles = [sim.sample((gid, 0), arbor.regular_schedule(0.1)) for gid in range(ncells)]

# Run simulation for 100 ms
sim.run(10, dt=dt)
print('Simulation finished')

# Print spike times
#print('spikes:')
#for sp in sim.spikes():
#    print(' ', sp)

# Plot the results
#print("Plotting results ...")
#df_list = []
#for gid in range(ncells):
#    samples, meta = sim.samples(handles[gid])[0]
#    df_list.append(pandas.DataFrame({'t/ms': samples[:, 0], 'U/mV': samples[:, 1], 'Cell': f"cell {gid}"}))

#df = pandas.concat(df_list,ignore_index=True)
#seaborn.relplot(data=df, kind="line", x="t/ms", y="U/mV",hue="Cell",ci=None)
#plt.show()
