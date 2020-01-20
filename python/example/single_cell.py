import arbor as arb
import matplotlib.pyplot as plt

# make a ball and stick cell model

b = arb.flat_cell_builder()

s  = b.add_sphere(6, "soma")
b1 = b.add_cable(parent=s, length=100, radius=2, name="dendn", ncomp=100)
b2 = b.add_cable(parent=b1, length=50, radius=(2,0.5), name="dendn", ncomp=50)
b3 = b.add_cable(parent=b1, length=50, radius=1, name="dendn", ncomp=50)
b4 = b.add_cable(parent=b2, length=50, radius=1, name="dendx", ncomp=50)
b5 = b.add_cable(parent=b2, length=50, radius=1, name="dendx", ncomp=50)

b.add_label('dend', '(join (region "dendn") (region "dendx"))')

b.add_label('stim_site', '(location 0 0)')
b.add_label('root', '(root)')

cell = b.build()

cell.paint('soma', 'hh')
cell.paint('dend', 'pas')
cell.paint('dendn', arb.axial_resistivity(50))
cell.paint('dendx', arb.axial_resistivity(1000))
cell.place('root', arb.iclamp(  0, 10, 0.8))
cell.place('root', arb.iclamp( 50, 10, 0.8))
cell.place('root', arb.iclamp(100, 10, 0.8))
cell.place('root', arb.spike_detector(-10))

# make single cell model

m = arb.single_cell_model(cell)

m.probe('voltage', arb.location(0,0), 10000)
m.probe('voltage', arb.location(2,1), 10000)
m.probe('voltage', arb.location(4,1), 10000)

tfinal=150
m.run(tfinal)

# print spike times.
print('spikes:')
for s in m.spikes:
    print('  ', s)

# plot the recorded voltages over time.
fig, ax = plt.subplots()
for t in m.traces:
    ax.plot(t.time, t.value)

ax.set(xlabel='time (ms)', ylabel='voltage (mV)', title='ring demo')
ax.legend(['voltage'])
plt.xlim(  0,tfinal)
plt.ylim(-80,40)
ax.grid()
#fig.savefig("voltages.png", dpi=300)
plt.show()
