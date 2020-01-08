import arbor as arb

# make a "ball and stick"
b = arb.flat_cell_builder()

s  = b.add_sphere(6, "soma")
b1 = b.add_cable(parent=s, length=100, radius=2, name="bdend", ncomp=1)
b2 = b.add_cable(parent=b1, length=50, radius=(2,0.5), name="edend", ncomp=1)
b3 = b.add_cable(parent=b1, length=50, radius=1, name="edend", ncomp=1)

#labels = {'soma': '(tag 1)',
          #'axon': '(tag 2)',
          #'dend': '(join (tag 3) (tag 4))',
          #'all' : '(all)',
          #'axso': '(join "soma" "axon")'}

b.add_label("dend", "(join (tag 2) (tag 3))")
b.add_label("all", "(all)")

print(b.samples)
print(b.labels)

# make the cell
cell = b.build()

print(cell)

#print('----------------------------')

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


