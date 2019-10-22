import arbor

p = arbor.mpoint(1,2,3,5)
print(p)

s = arbor.msample(p, 1)
print(s)

tree = arbor.sample_tree()
print(tree.size)
print(tree.empty)
print(tree.parents)
tree.reserve(42)

tree.append(s)

print(tree.size)
print(tree.empty)
print(tree.parents)

print(s)
print(p)
tree.append(0, s)

print(tree)

tree = arbor.load_swc('../../test/unit/swc/ball_and_stick.swc')
#tree = arbor.load_swc('../../test/unit/swc/example.swc')

print('sample tree has ', tree.size, ' samples')

m = arbor.morphology(tree, True)
print(m.num_branches)
for i in range(m.num_branches):
    print('  branch', i, ': parent =', m.branch_parent(i), ", children = ", m.branch_children(i))

print()
for i in range(m.num_branches):
    print('  branch', i, ': ', m.branch_indexes(i))

print(m.samples)
print(m.sample_parents)

print()
print()

r = arbor.region()
myreg = arbor.join(arbor.reg_tag(2), arbor.join(r, arbor.reg_tag(1)))
print(myreg)

labels = arbor.label_dict()
labels.set('soma', arbor.reg_tag(1))
labels.set('axon', arbor.reg_tag(2))
labels.set('dend', arbor.reg_tag(3))
labels.set('cat', arbor.join(arbor.reg_tag(3), arbor.reg_tag(1)))
print(labels.regions())
print(labels.locsets())

cell = arbor.cable_cell(m, labels, True)
print(cell)

#labels.set('soma', '(tag 1)')
#labels.set('axon', '(tag 2)')
#labels.set('dend', '(tag 3)')
#labels.set('cat', '(join (tag 1) (tag 2))')
