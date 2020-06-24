import arbor
from arbor import mechanism as mech
from arbor import location as loc
import matplotlib.pyplot as plt

# Load a cell morphology from an swc file.
# The model has 31 branches, including soma, dendrites and axon.
tree = arbor.load_swc('../../test/unit/swc/example.swc')

# Define the regions and locsets in the model.
defs = {'soma': '(tag 1)',  # soma has tag 1 in swc files.
        'axon': '(tag 2)',  # axon has tag 2 in swc files.
        'dend': '(tag 3)',  # dendrites have tag 3 in swc files.
        'root': '(root)',   # the start of the soma in this morphology is at the root of the cell.
        'stim_site': '(location 1 0.5)'} # site for the stimulus, in the middle of branch 1.
labels = arbor.label_dict(defs)

# Combine morphology with region and locset definitions to make a cable cell.
cell = arbor.cable_cell(tree, labels)

# Set initial membrane potential to -55 mV
cell.set_properties(Vm=-55)
# Use Nernst to calculate reversal potential for calcium.
cell.set_ion('ca', method=mech('nernst/x=ca'))
# hh mechanism on the soma and axon.
cell.paint('soma', 'hh')
cell.paint('axon', 'hh')
# pas mechanism the dendrites.
cell.paint('dend', 'pas')
# Increase resistivity on dendrites.
cell.paint('dend', rL=500)
# Attach stimuli that inject 0.8 nA currents for 1 ms, starting at 3 and 8 ms.
cell.place('stim_site', arbor.iclamp(3, 1, current=0.8))
cell.place('stim_site', arbor.iclamp(8, 1, 0.8))
# Detect spikes at the soma with a voltage threshold of -10 mV.
cell.place('root', arbor.spike_detector(-10))

# Have one compartment between each sample point.
cell.compartments_on_samples()

# Make single cell model.
m = arbor.single_cell_model(cell)

# Attach voltage probes that sample at 50 kHz.
m.probe('voltage', where='root',  frequency=50000)
m.probe('voltage', where=loc(2,1),  frequency=50000)
m.probe('voltage', where=loc(4,1),  frequency=50000)
m.probe('voltage', where=loc(30,1), frequency=50000)

# Simulate the cell for 15 ms.
tfinal=15
m.run(tfinal)

# Print spike times.
if len(m.spikes)>0:
    print('{} spikes:'.format(len(m.spikes)))
    for s in m.spikes:
        print('  {:7.4f}'.format(s))
else:
    print('no spikes')

# Plot the recorded voltages over time.
fig, ax = plt.subplots()
for t in m.traces:
    ax.plot(t.time, t.value)

legend_labels = ['{}: {}'.format(s.variable, s.location) for s in m.traces]
ax.legend(legend_labels)
ax.set(xlabel='time (ms)', ylabel='voltage (mV)', title='swc morphology demo')
plt.xlim(0,tfinal)
plt.ylim(-80,50)
ax.grid()

plot_to_file=False
if plot_to_file:
    fig.savefig("voltages.png", dpi=300)
else:
    plt.show()
