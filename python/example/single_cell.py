import arbor as arb

# make a ball and stick cell model

b = arb.flat_cell_builder()

s  = b.add_sphere(6, "soma")
b1 = b.add_cable(parent=s, length=100, radius=2, name="dend", ncomp=1)
b2 = b.add_cable(parent=b1, length=50, radius=(2,0.5), name="dend", ncomp=1)
b3 = b.add_cable(parent=b1, length=50, radius=1, name="dend", ncomp=1)

b.add_label('stim_site', '(location 0 0)')
b.add_label('root', '(root)')

cell = b.build()

cell.paint('soma', 'hh')
cell.paint('dend', 'pas')
cell.place('stim_site', arb.iclamp(0, 10, 0.5))
cell.place('root', arb.spike_detector(-10))

# make single cell model

m = arb.single_cell_model(cell)

m.run(100)

print('spikes:')
for s in m.spikes:
    print('  ', s)
