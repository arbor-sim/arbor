#!/usr/bin/env python3

import arbor
import seaborn
import pandas
from math import sqrt

# Make a ball and stick cell model

tree = arbor.segment_tree()

# Construct a cell with the following morphology.
# The soma (at the root of the tree) is marked 's', and
# the end of each branch i is marked 'bi'.
#
#               b4
#              /
#             /
#            b1---b3
#           /
#          /
# s-------b0
#          \
#           \
#            b2

# Start with a spherical soma with radius 6 μm,
# approximated with a cylinder of: length = diameter = 12 μm.

s = tree.append(arbor.mnpos, arbor.mpoint(-12, 0, 0, 6), arbor.mpoint(0, 0, 0, 6), tag=1)

# Add the dendrite cables, labelling those closest to the soma "dendn",
# and those furthest with "dendx" because we will set different electrical
# properties for the two regions.

labels = arbor.label_dict()
labels['soma'] = '(tag 1)'
labels['dendn'] = '(tag 5)'
labels['dendx'] = '(tag 6)'

b0 = tree.append(s, arbor.mpoint(0, 0, 0, 2), arbor.mpoint(100, 0, 0, 2), tag=5)

# Radius tapers from 2 to 0.5 over the length of the branch.

b1 = tree.append(b0, arbor.mpoint(100, 0, 0, 2), arbor.mpoint(100+50/sqrt(2), 50/sqrt(2), 0, 0.5), tag=5)
b2 = tree.append(b0, arbor.mpoint(100, 0, 0, 1), arbor.mpoint(100+50/sqrt(2), -50/sqrt(2), 0, 1), tag=5)
b3 = tree.append(b1, arbor.mpoint(100+50/sqrt(2), 50/sqrt(2), 0, 1), arbor.mpoint(100+50/sqrt(2)+50, 50/sqrt(2), 0, 1), tag=6)
b4 = tree.append(b1, arbor.mpoint(100+50/sqrt(2), 50/sqrt(2), 0, 1), arbor.mpoint(100+2*50/sqrt(2), 2*50/sqrt(2), 0, 1), tag=6)

# Combine the "dendn" and "dendx" regions into a single "dend" region.
# The dendrites were labelled as such so that we can set different
# properties on each sub-region, and then combined so that we can
# set other properties on the whole dendrites.
labels['dend'] = '(join (region "dendn") (region "dendx"))'
# Location of stimuli, in the middle of branch 2.
labels['stim_site'] = '(location 1 0.5)'
# The root of the tree (equivalent to '(location 0 0)')
labels['root'] = '(root)'
# The tips of the dendrites (3 locations at b4, b3, b2).
labels['dtips'] = '(terminal)'

# Extract the cable cell from the builder.
# cell = b.build()
cell = arbor.cable_cell(tree, labels)

# Set initial membrane potential everywhere on the cell to -40 mV.
cell.set_properties(Vm=-40)
# Put hh dynamics on soma, and passive properties on the dendrites.
cell.paint('"soma"', 'hh')
cell.paint('"dend"', 'pas')
# Set axial resistivity in dendrite regions (Ohm.cm)
cell.paint('"dendn"', rL=500)
cell.paint('"dendx"', rL=10000)
# Attach stimuli with duration of 2 ms and current of 0.8 nA.
# There are three stimuli, which activate at 10 ms, 50 ms and 80 ms.
cell.place('"stim_site"', arbor.iclamp( 10, 2, 0.8))
cell.place('"stim_site"', arbor.iclamp( 50, 2, 0.8))
cell.place('"stim_site"', arbor.iclamp( 80, 2, 0.8))
# Add a spike detector with threshold of -10 mV.
cell.place('"root"', arbor.spike_detector(-10))

# Discretization: the default discretization in Arbor is 1 compartment per branch.
# Let's be a bit more precise and make that every 2 micron:
cell.compartments_length(2)

# Make single cell model.
m = arbor.single_cell_model(cell)

# Attach voltage probes, sampling at 10 kHz.
m.probe('voltage', '(location 0 0)', 10000) # at the soma.
m.probe('voltage', '"dtips"',  10000) # at the tips of the dendrites.

# Run simulation for 100 ms of simulated activity.
tfinal=100
m.run(tfinal)
print("Simulation done.")

# Print spike times.
if len(m.spikes)>0:
    print('{} spikes:'.format(len(m.spikes)))
    for s in m.spikes:
        print('  {:7.4f}'.format(s))
else:
    print('no spikes')

# Plot the recorded voltages over time.
print("Plotting results...")
df = pandas.DataFrame()
for t in m.traces:
    df=df.append(pandas.DataFrame({'t/ms': t.time, 'U/mV': t.value, 'Location': str(t.location), "Variable": t.variable}) )

seaborn.relplot(data=df, kind="line", x="t/ms", y="U/mV",hue="Location",col="Variable",ci=None).savefig('single_cell_multi_branch_result.svg')
