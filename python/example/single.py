import arbor

p = arbor.mpoint(1,2,3,5)
s = arbor.msample(p, 1)
tree = arbor.sample_tree()
print(tree.size, tree.empty, tree.parents)

tree.reserve(10)
tree.append(s)
print(tree.size, tree.empty, tree.parents)
tree.append(0, s)

print(tree)

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
# TODO: this should throw an error because `ax&on` is not a valid name for a region/locset
defs = {'soma': '(tag 1)', 'axon ': '(tag 2)', 'dend': '(tag 3)', 'cat': '(join (tag 1) (tag 2))'}
labels = arbor.label_dict(defs)
print(labels)
#print(labels.regions())
#print(labels.locsets())

print('\n----------------------------- make cable_cell -----------------------------\n')
cell = arbor.cable_cell(m, labels, True)
print(cell)
print('cell has', cell.num_branches, 'branches')

print('\n-------------------------------------------------------------\n')

labels['sdnd'] = '(join (tag 1) (tag 2))'

print(labels['soma'])
print(labels['sdnd'])
print()
print('len(labels)', len(labels))
for name in labels:
    print('  ', name, ':', labels[name])

labels['sdnd'] = '(join (tag 1) (tag 2))'
