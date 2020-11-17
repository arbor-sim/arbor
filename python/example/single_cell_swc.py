#!/usr/bin/env python3

# NOTE: deprecating spherical roots changes the behavior of this model.
# There is no soma, because only the root sample has tag 1, which will be
# ignored as it is always the proximal end of any cable segment.
# The fix is to:
#   - Write an swc interpreter that inserts a cylinder with the
#     appropriate properties.
#   - Extend the cable-only descriptions to handle detached cables, to
#     preserve surface area and correct starting locations of cables
#     attached to the soma.

import arbor
from arbor import mechanism as mech
from arbor import location as loc
import pandas, seaborn
import sys

# Load a cell morphology from an swc file.
# Example present here: ../../test/unit/swc/pyramidal.swc
if len(sys.argv) < 2:
    print("No SWC file passed to the program")
    sys.exit(0)

filename = sys.argv[1]
morpho = arbor.load_swc(filename)

# Define the regions and locsets in the model.
defs = {'soma': '(tag 1)',  # soma has tag 1 in swc files.
        'axon': '(tag 2)',  # axon has tag 2 in swc files.
        'dend': '(tag 3)',  # dendrites have tag 3 in swc files.
        'root': '(root)',   # the start of the soma in this morphology is at the root of the cell.
        'stim_site': '(location 0 0.5)', # site for the stimulus, in the middle of branch 1.
        'axon_end': '(restrict (terminal) (region "axon"))'} # end of the axon.
labels = arbor.label_dict(defs)

# Combine morphology with region and locset definitions to make a cable cell.
cell = arbor.cable_cell(morpho, labels)

print(cell.locations('"axon_end"'))

# Set initial membrane potential to -55 mV
cell.set_properties(Vm=-55)
# Use Nernst to calculate reversal potential for calcium.
cell.set_ion('ca', method=mech('nernst/x=ca'))
# hh mechanism on the soma and axon.
cell.paint('"soma"', 'hh')
cell.paint('"axon"', 'hh')
# pas mechanism the dendrites.
cell.paint('"dend"', 'pas')
# Increase resistivity on dendrites.
cell.paint('"dend"', rL=500)
# Attach stimuli that inject 0.8 nA currents for 1 ms, starting at 3 and 8 ms.
cell.place('"stim_site"', arbor.iclamp(3, 1, current=2))
cell.place('"stim_site"', arbor.iclamp(8, 1, current=4))
# Detect spikes at the soma with a voltage threshold of -10 mV.
cell.place('"axon_end"', arbor.spike_detector(-10))

# Have one compartment between each sample point.
cell.compartments_on_segments()

# Make single cell model.
m = arbor.single_cell_model(cell)

# Attach voltage probes that sample at 50 kHz.
m.probe('voltage', where='"root"',  frequency=50000)
m.probe('voltage', where=loc(2,1),  frequency=50000)
m.probe('voltage', where='"stim_site"',  frequency=50000)
m.probe('voltage', where='"axon_end"', frequency=50000)

# Simulate the cell for 15 ms.
tfinal=15
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
print("Plotting results ...")
df_list = []
for t in m.traces:
    df_list.append(pandas.DataFrame({'t/ms': t.time, 'U/mV': t.value, 'Location': t.location, "Variable": t.variable}))

df = pandas.concat(df_list)

seaborn.relplot(data=df, kind="line", x="t/ms", y="U/mV",hue="Location",col="Variable",ci=None).savefig('single_cell_swc.svg')
