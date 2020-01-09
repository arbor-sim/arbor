import arbor as arb

# make a "ball and stick"
b = arb.flat_cell_builder()

s  = b.add_sphere(6, "soma")
b1 = b.add_cable(parent=s, length=100, radius=2, name="bdend", ncomp=1)
b2 = b.add_cable(parent=b1, length=50, radius=(2,0.5), name="edend", ncomp=1)
b3 = b.add_cable(parent=b1, length=50, radius=1, name="edend", ncomp=1)

b.add_label('dend', '(join (region "edend") (region "bdend"))')
b.add_label('terms', '(terminal)')

# make the cell
cell = b.build()

cell.paint('dend', 'pas')
cell.paint('soma', 'hh')
cell.place('terms', 'expsyn')

print('----------------------------')

print(b.samples)
print(b.labels)

print(cell)

print('----------------------------')

class single_recipe (arb.recipe):

    def __init__(self, cell):
        # The base C++ class constructor must be called first, to ensure that
        # all memory in the C++ class is initialized correctly.
        arb.recipe.__init__(self)
        self.cell = cell
        #self.params = arb.cell_parameters()

    # The num_cells method that returns the total number of cells in the model
    # must be implemented.
    def num_cells(self):
        return 1

    # The cell_description method returns a cell
    def cell_description(self, gid):
        return cell

    def num_targets(self, gid):
        return 1

    def num_sources(self, gid):
        return 1

    # The kind method returns the type of cell with gid.
    # Note: this must agree with the type returned by cell_description.
    def cell_kind(self, gid):
        return arb.cell_kind.cable

    # Make a ring network
    def connections_on(self, gid):
        return []

    # Attach a generator to the first cell in the ring.
    def event_generators(self, gid):
        #if gid==0:
        #    sched = arb.explicit_schedule([1])
        #    return [arb.event_generator(arb.cell_member(0,0), 0.1, sched)]
        return []

    # Define one probe (for measuring voltage at the soma) on each cell.
    def num_probes(self, gid):
        # TODO: we need some probes
        return 1

    def get_probe(self, id):
        # TODO: we need some probes
        #loc = arb.location(0, 0) # at the soma
        #return arb.cable_probe('voltage', id, loc)
        raise 43
        return arb.cable_probe('voltage', id, loc)

rec = single_recipe(cell)

# Make a cell with no spherical root, two branches at the root, and two branches
# hanging off of it.
#b = arb.flat_cell_builder()

#r = b.add_cable(arb.mnpos, length=100, radius=2, name="dend", ncomp=1)
#p = b.add_cable(r,         length=50,  radius=1, name="dend", ncomp=1)
#p = b.add_cable(r,         length=50,  radius=1, name="dend", ncomp=1)
#y = b.add_cable(arb.mnpos, length=100, radius=2, name="axon", ncomp=1)

# User can query properties
#print(b.morphology)
#print(b.samples)
#print(b.labels)


# this blows up right now, because complex soma.
#cell = b.build()

#print('\n----------------------------\n')
#
#b = arb.flat_cell_builder()
#
#r = b.add_cable(arb.mnpos, length=100, radius=2, name="dend", ncomp=1)
#b1 = b.add_cable(r,        length=100, radius=1, name="dend", ncomp=1)
#b2 = b.add_cable(r,        length=50,  radius=2, name="dend", ncomp=1)
#
#cell = b.build()
#
#print(b.samples)
#print()
#print(b.morphology)


