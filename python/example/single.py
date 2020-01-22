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

#tree = arbor.load_swc('../../test/unit/swc/ball_and_stick.swc')
tree = arbor.load_swc('../../test/unit/swc/example.swc')

print('\n----------------------------- sample tree load from swc -----------------------------\n')
print('sample tree has ', tree.size, ' samples')

m = arbor.morphology(tree, True)
print('morphology has', m.num_branches, ' branches')
print('branch info:')
for i in range(m.num_branches):
    print('  branch', i, ': parent =', m.branch_parent(i), ", children = ", m.branch_children(i), " indexes = ", m.branch_indexes(i))

print('\n----------------------------- make label dictionary -----------------------------\n')
defs = {'soma': '(tag 1)', 'axon': '(tag 2)', 'dend': '(tag 3)', 'cat': '(join (tag 1) (tag 2))', 'midpoints': '(join (location 1 0.5) (location 1 0.2))'}
labels = arbor.label_dict(defs)
print(labels)
print('regions:',labels.regions)
print('locsets:',labels.locsets)

print('\n----------------------------- make cable_cell -----------------------------\n')
cell = arbor.cable_cell(m, labels, True)
print(cell)
print('cell has', cell.num_branches, 'branches')

print('\n-------------------------------------------------------------\n')

labels['sdnd'] = '(join (tag 1) (tag 2))'
labels['x'] = '(root)'
labels['a'] = '(terminal)'
labels['z'] = '(sum (root) (terminal))'

print('len(labels)', len(labels))
for name in labels:
    print('  ', name, ':', labels[name])

labels['sdnd'] = '(join (tag 1) (tag 2))'

print(labels)
print('regions:',labels.regions)
print('locsets:',labels.locsets)

print(labels.locsets)
labels.locsets[2] = 'hello'
print(labels.locsets)

dmech = arbor.mechanism('expsyn', {'gbar':0.2, 'E':-40})
print(dmech)
print(dmech.values)

cell.place('midpoints', arbor.mechanism('expsyn', {'gbar':12, 'E':-10}))
cell.place('midpoints', arbor.mechanism('expsyn', {'gbar':12, 'E':-10}))
cell.place('midpoints', arbor.gap_junction())
cell.place('midpoints', arbor.iclamp(delay=1, duration=12, amplitude=2))
cell.place('midpoints', arbor.spike_detector(threshold=-10))
print(arbor.iclamp(delay=1, duration=12, amplitude=2))
print(arbor.spike_detector(threshold=-10))
print(cell.num_branches, "branches in this bad boy")
