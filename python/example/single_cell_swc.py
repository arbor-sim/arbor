import arbor
from arbor import mechanism as mech
from arbor import location as loc
import matplotlib.pyplot as plt

# make a ball and stick cell model

tree = arbor.load_swc('../../test/unit/swc/example.swc')
morph = arbor.morphology(tree, True)

defs = {'soma': '(tag 1)',
        'axon': '(tag 2)',
        'dend': '(tag 3)',
        'midpoints': '(join (location 1 0.5) (location 1 0.2))',
        'root': '(root)',
        'stim_site': '(location 1 0.5)'}
labels = arbor.label_dict(defs)

cell = arbor.cable_cell(morph, labels, True)

cell.set_properties(Vm=-40)
cell.set_ion('ca', method=mech('nernst/x=ca'))
cell.paint('soma', 'hh')
cell.paint('dend', 'pas')
cell.paint('axon', 'hh')
cell.paint('dend', rL=500)
cell.place('stim_site', arbor.iclamp( 10, 1, 0.8))
cell.place('stim_site', arbor.iclamp( 20, 1, 0.8))
cell.place('stim_site', arbor.iclamp( 25, 1, 0.8))
cell.place('root', arbor.spike_detector(-10))

# make single cell model

m = arbor.single_cell_model(cell)

m.probe('voltage', loc(0,0),  50000)
m.probe('voltage', loc(2,1),  50000)
m.probe('voltage', loc(4,1),  50000)
m.probe('voltage', loc(30,1), 50000)

tfinal=50
m.run(tfinal)

# print spike times.
print('spikes:')
for s in m.spikes:
    print('  ', s)

# plot the recorded voltages over time.
fig, ax = plt.subplots()
for t in m.traces:
    ax.plot(t.time, t.value)

legend_labels = ['{}'.format(s.location) for s in m.traces]
ax.set(xlabel='time (ms)', ylabel='voltage (mV)', title='swc morphology demo')
ax.legend(legend_labels)
plt.xlim(  0,tfinal)
plt.xlim( 19,24)
plt.ylim(-80,50)
ax.grid()
#fig.savefig("voltages.png", dpi=300)
plt.show()
