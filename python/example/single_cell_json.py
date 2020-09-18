import arbor
import sys
import matplotlib.pyplot as plt

defaults_file = sys.argv[1]
cells_file    = sys.argv[2]
swc_file      = sys.argv[3]

tree         = arbor.load_swc(swc_file)
defaults     = arbor.load_default_parameters(defaults_file)
globals      = arbor.load_cell_parameters(cells_file)
locals       = arbor.load_region_parameters(cells_file)
region_mechs = arbor.load_region_mechanisms(cells_file)

# Define the regions and locsets in the model.
defs = {'soma': '(tag 1)',  # soma has tag 1 in swc files.
        'axon': '(tag 2)',  # axon has tag 2 in swc files.
        'dend': '(tag 3)',  # dendrites have tag 3 in swc files.
        'apic': '(tag 4)',  # dendrites have tag 3 in swc files.
        'all' : '(all)',    # all the cell
        'root': '(root)',   # the start of the soma in this morphology is at the root of the cell.
        'mid_soma': '(location 0 0.5)'
        } # end of the axon.
labels = arbor.label_dict(defs)

# Combine morphology with region and locset definitions to make a cable cell.
cell = arbor.cable_cell(tree, labels)

# Set the default cell parameters.
cell.set_default_properties(globals)

# Overwrite the default local cell parameters.
cell.set_local_properties(locals)

# Paint density mechanisms on the regions of the cell.
cell.paint_dynamics(region_mechs)

# Place current clamp and spike detector.
cell.place('mid_soma', arbor.iclamp(0, 3, current=3.5))
cell.place('root', arbor.spike_detector(-10))

# Select a fine discritization for apt comparison with neuron.
cell.compartments_length(0.5)

# Make single cell model.
m = arbor.single_cell_model(cell)

# Set the model default parameters
m.set_default_properties(defaults)

# arbor.output_cell_description(cell, "cell_out.json")

# Extend the default catalogue
m.properties.catalogue.extend(arbor.bbp_catalogue(), "")

# Attach voltage probes that sample at 50 kHz.
m.probe('voltage', where='root',  frequency=50000)
m.probe('voltage', where='mid_soma', frequency=50000)

# Simulate the cell for 20 ms.
tfinal=20
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
ax.grid()

plot_to_file=False
if plot_to_file:
    fig.savefig("voltages.png", dpi=300)
else:
    plt.show()
