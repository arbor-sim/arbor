import pyarb as arb

loc = arb.segment_location(0, 0.5)
print(loc)

cell = arb.make_soma_cell()
print(cell)
help(arb.make_soma_cell)

cell.add_stimulus(loc, 10, 100, 0.1)
cell.add_synapse(loc)
cell.add_synapse(loc)
cell.add_synapse(loc)
cell.add_synapse(loc)
cell.add_detector(loc, -40)
print(cell)
