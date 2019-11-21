import arbor

p = arbor.mpoint(1,2,3,5)
#print(p)

s = arbor.msample(p, 1)
#print(s)

tree = arbor.sample_tree()
#print(tree.size)
#print(tree.empty)
#print(tree.parents)
tree.reserve(42)

tree.append(s)

#print(tree.size)
#print(tree.empty)
#print(tree.parents)

#print(s)
#print(p)
tree.append(0, s)

#print(tree)

tree = arbor.load_swc('../../test/unit/swc/ball_and_stick.swc')
#tree = arbor.load_swc('../../test/unit/swc/example.swc')

print('\n----------------------------- sample tree load from swc -----------------------------\n')
print('sample tree has ', tree.size, ' samples')

m = arbor.morphology(tree, True)
print('morphology has', m.num_branches, ' branches')
print('branch info:')
for i in range(m.num_branches):
    print('  branch', i, ': parent =', m.branch_parent(i), ", children = ", m.branch_children(i), " indexes = ", m.branch_indexes(i))

#print(m.samples)
#print(m.sample_parents)

print('\n----------------------------- make label dictionary -----------------------------\n')
labels = arbor.label_dict()
labels.set('soma', arbor.reg_tag(1))
labels.set('axon', arbor.reg_tag(2))
labels.set('dend', arbor.reg_tag(3))
labels.set('all', arbor.reg_all())
labels.set('terms', arbor.ls_terminal())
print(labels.regions())
print(labels.locsets())

print('\n----------------------------- make cable_cell -----------------------------\n')
cell = arbor.cable_cell(m, labels, True)
print(cell)
print('cell has ', cell.num_branches())

print('\n-------------------------------------------------------------\n')

labels.set('soma', '(tag 1)')
labels.set('axon', '(tag 2)')
labels.set('dend', '(tag 3)')
labels.set('cat', '(join (tag 1) (tag 2))')
