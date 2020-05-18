import arbor

tree = arbor.sample_tree()
tree.append(           x= 0.0, y= 0.0, z= 0.0, radius= 2.0, tag= 1)
tree.append(parent= 0, x= 4.0, y= 0.0, z= 0.0, radius= 2.0, tag= 1)
tree.append(parent= 1, x= 4.0, y= 0.0, z= 0.0, radius= 0.8, tag= 3)
tree.append(parent= 2, x= 8.0, y= 0.0, z= 0.0, radius= 0.8, tag= 3)
tree.append(parent= 3, x=12.0, y=-0.5, z= 0.0, radius= 0.8, tag= 3)
tree.append(parent= 4, x=20.0, y= 4.0, z= 0.0, radius= 0.4, tag= 3)
tree.append(parent= 5, x=26.0, y= 6.0, z= 0.0, radius= 0.2, tag= 3)
tree.append(parent= 4, x=12.0, y=-0.5, z= 0.0, radius= 0.5, tag= 3)
tree.append(parent= 7, x=19.0, y=-3.0, z= 0.0, radius= 0.5, tag= 3)
tree.append(parent= 8, x=24.0, y=-7.0, z= 0.0, radius= 0.2, tag= 3)
tree.append(parent= 8, x=23.0, y=-1.0, z= 0.0, radius= 0.2, tag= 3)
tree.append(parent=10, x=26.0, y=-2.0, z= 0.0, radius= 0.2, tag= 3)
tree.append(parent= 0, x=-7.0, y= 0.0, z= 0.0, radius= 0.4, tag= 2)
tree.append(parent=12, x=-10.0, y= 0.0, z= 0.0, radius= 0.4, tag= 2)

m = arbor.morphology(tree, spherical_root=False)

regions  = {
            'empty': '(nil)',
            'all': '(all)',
            'tag1': '(tag 1)',
            'tag2': '(tag 2)',
            'tag3': '(tag 3)',
            'tag4': '(tag 4)',
            'soma': '(region "tag1")',
            'axon': '(region "tag2")',
            'dend': '(join (region "tag3") (region "tag4"))',
            'radlt5': '(radius_lt (all) 0.5)',
            'radle5': '(radius_le (all) 0.5)',
            'radgt5': '(radius_gt (all) 0.5)',
            'radge5': '(radius_ge (all) 0.5)',
            'rad36':  '(intersect (radius_gt (all) 0.3) (radius_lt (all) 0.6))',
            'branch0': '(branch 0)',
            'branch3': '(branch 3)',
            'cable_1_01': '(cable 1 0 1)',
            'cable_1_31': '(cable 1 0.3 1)',
            'cable_1_37': '(cable 1 0.3 0.7)',
            'proxint':     '(proximal_interval (locset "proxint_in") 5)',
            'proxintinf':  '(proximal_interval (locset "proxint_in"))',
            'distint':     '(distal_interval   (locset "distint_in") 5)',
            'distintinf':  '(distal_interval   (locset "distint_in"))',
            'lhs' : '(join (cable 0 0.5 1) (cable 1 0 0.5))',
            'rhs' : '(branch 1)',
            'and': '(intersect (region "lhs") (region "rhs"))',
            'or':  '(join      (region "lhs") (region "rhs"))',
          }
locsets = {
            'root': '(root)',
            'term': '(terminal)',
            'rand_dend': '(uniform (region "dend") 0 50 0)',
            'loc15': '(location 1 0.5)',
            'uniform0': '(uniform (tag 3) 0 9 0)',
            'uniform1': '(uniform (tag 3) 0 9 1)',
            'branchmid': '(on_branches 0.5)',
            'sample1': '(sample 1)',
            'distal':  '(distal   (region "rad36"))',
            'proximal':'(proximal (region "rad36"))',
            'distint_in': '(sum (location 1 0.5) (location 2 0.7) (location 5 0.1))',
            'proxint_in': '(sum (location 1 0.8) (location 2 0.3))',
            'loctest' : '(distal (super (join (branch 1) (branch 0))))',
            'restrict': '(restrict  (terminal) (tag 3))',
          }

labels = {**regions, **locsets}

d = arbor.label_dict(labels)

cell = arbor.cable_cell(m, d)

f = open('regloc_input.py', 'w')
f.write('\n############# locsets\n')
for label in locsets:
    locs = [(l.branch, l.pos) for l in cell.locations(label)]
    f.write('ls_{}  = {{\'type\': \'locset\', \'value\': {}}}\n'.format(label, locs))

f.write('\n############# regions\n')
for label in regions:
    comps = [(c.branch, c.prox, c.dist) for c in cell.cables(label)]
    f.write('reg_{} = {{\'type\': \'region\', \'value\': {}}}\n'.format(label, comps))

f.close()
