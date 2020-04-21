import arbor

# Create a sample tree with a single sample of radius 3 μm
tree = arbor.sample_tree()
tree.append(arbor.msample(x=0, y=0, z=0, radius=3, tag=2))

labels = arbor.label_dict({'soma': '(tag 2)', 'center': '(location 0 0.5)'})

cell = arbor.cable_cell(tree, labels)

# Set initial membrane potential everywhere on the cell to -40 mV.
cell.set_properties(Vm=-40)
# Put hh dynamics on soma, and passive properties on the dendrites.
cell.paint('soma', 'hh')
# Attach stimuli with duration of 2 ms and current of 0.8 nA.
cell.place('center', arbor.iclamp( 10, 2, 0.8))
# Add a spike detector with threshold of -10 mV.
cell.place('center', arbor.spike_detector(-10))

# Make single cell model.
m = arbor.single_cell_model(cell)

# Attach voltage probes, sampling at 10 kHz.
m.probe('voltage', 'center',  10000)

# Run simulation for 100 ms of simulated activity.
tfinal=30
m.run(tfinal)

# Print spike times.
if len(m.spikes)>0:
    print('{} spikes:'.format(len(m.spikes)))
    for s in m.spikes:
        print('  {:7.4f}'.format(s))
else:
    print('no spikes')

# Plot the recorded voltages over time.
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
for t in m.traces:
    ax.plot(t.time, t.value)

legend_labels = ['{}: {}'.format(s.variable, s.location) for s in m.traces]
ax.legend(legend_labels)
ax.set(xlabel='time (ms)', ylabel='voltage (mV)', title='cell builder demo')
plt.xlim(0,tfinal)
plt.ylim(-80,50)
ax.grid()

# Set to True to save the image to file instead of opening a plot window.
plot_to_file=False
if plot_to_file:
    fig.savefig("voltages.png", dpi=300)
else:
    plt.show()
