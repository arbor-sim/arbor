import arbor
from arbor import mechanism as mech
from arbor import location as loc
import matplotlib.pyplot as plt

# Make a ball and stick cell model
b = arbor.flat_cell_builder()

# Construct a cell with the following morphology.
# The soma (at the root of the tree) is marked 's', and
# the end of each branch i is marked 'bi'.
#
#               b5
#              /
#             /
#            b2---b4
#           /
#          /
# s-------b1
#          \
#           \
#            b3

# Add a spherical soma with radius 6 um
s  = b.add_sphere(6, "soma")

# Add the dendrite cables, labelling those closest to the soma "dendn",
# and those furthest with "dendx" because we will set different electrical
# properties for the two regions.
b1 = b.add_cable(parent=s,  length=100, radius=2, name="dendn", ncomp=100)
# Radius tapers from 2 to 0.5 over the length of the branch.
b2 = b.add_cable(parent=b1, length= 50, radius=(2,0.5), name="dendn", ncomp=50)
b3 = b.add_cable(parent=b1, length= 50, radius=1, name="dendn", ncomp=50)
b4 = b.add_cable(parent=b2, length= 50, radius=1, name="dendx", ncomp=50)
b5 = b.add_cable(parent=b2, length= 50, radius=1, name="dendx", ncomp=50)

# Combine the "dendn" and "dendx" regions into a single "dend" region.
# The dendrites were labelled as such so that we can set different
# properties on each sub-region, and then combined so that we can
# set other properties on the whole dendrites.
b.add_label('dend', '(join (region "dendn") (region "dendx"))')
# Location of stimuli, in the middle of branch 2.
b.add_label('stim_site', '(location 2 0.5)')
# The root of the tree (equivalent to '(location 0 0)')
b.add_label('root', '(root)')
# The tips of the dendrites (3 locations at b4, b3, b5).
b.add_label('dtips', '(terminal)')

# Extract the cable cell from the builder.
cell = b.build()

# Set initial membrane potential everywhere on the cell to -40 mV.
cell.set_properties(Vm=-40)
# Put hh dynamics on soma, and passive properties on the dendrites.
cell.paint('soma', 'hh')
cell.paint('dend', 'pas')
# Set axial resistivity in dendrite regions (Ohm.cm)
cell.paint('dendn', rL=500)
cell.paint('dendx', rL=10000)
# Attach stimuli with duration of 2 ms and current of 0.8 nA.
# There are three stimuli, which activate at 10 ms, 50 ms and 80 ms.
cell.place('stim_site', arbor.iclamp( 10, 2, 0.8))
cell.place('stim_site', arbor.iclamp( 50, 2, 0.8))
cell.place('stim_site', arbor.iclamp( 80, 2, 0.8))
# Add a spike detector with threshold of -10 mV.
cell.place('root', arbor.spike_detector(-10))

# Make single cell model.
m = arbor.single_cell_model(cell)

# Attach voltage probes, sampling at 10 kHz.
m.probe('voltage', loc(0,0), 10000) # at the soma.
m.probe('voltage', 'dtips',   10000) # at the tips of the dendrites.

# Run simulation for 100 ms of simulated activity.
tfinal=100
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
