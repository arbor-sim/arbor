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

# Add a spherical soma with radius 6 μm
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

# Combine the "dendn" and "dendx" components into a single "dend" region.
b.add_label('dend', '(join (region "dendn") (region "dendx"))')
b.add_label('stim_site', '(location 2 0.5)')
b.add_label('root', '(root)')

print(b.labels)

# Construct the cable cell.
cell = b.build()

# Set initial membrane potential everywhere on the cell to -40 mV.
cell.set_properties(Vm=-40)
cell.paint('soma', 'hh')
cell.paint('dend', 'pas')
# Set axial resistivity in dendrite regions (Ω·cm)
cell.paint('dendn', rL=500)
cell.paint('dendx', rL=10000)
# Attach stimuli at the root of the cell
cell.place('stim_site', arbor.iclamp( 10, 2, 0.8))
cell.place('stim_site', arbor.iclamp( 50, 2, 0.8))
cell.place('stim_site', arbor.iclamp( 80, 2, 0.8))
cell.place('root', arbor.spike_detector(-10))

# make single cell model
m = arbor.single_cell_model(cell)

# Attach voltage probes, sampling at 10 kHz.
m.probe('voltage', loc(0,0), 10000) # at the soma
m.probe('voltage', loc(3,1), 10000) # at the end of branch 3
m.probe('voltage', loc(4,1), 10000) # at the end of branch 4

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

plot_to_file=False
if plot_to_file:
    fig.savefig("voltages.png", dpi=300)
else:
    plt.show()
