import arbor as arb
from arbor import mechanism_desc as mech

tree = arb.load_swc('../../test/unit/swc/ball_and_stick.swc')
morph = arb.morphology(tree, spherical_root=True)

labels = arb.label_dict()
labels.set('soma', arb.reg_tag(1))
labels.set('axon', arb.reg_tag(2))
labels.set('dend', arb.join(arb.reg_tag(3), arb.reg_tag(4)))
labels.set('all', arb.reg_all())

cell = arb.cable_cell(morph, labels)

cell.paint('soma', mech('special/ca=fancy_calcium', {'foo': 23, 'bar': 12.1}))
cell.paint('dend', 'pas')

dendprop = arb.local_parameter_set();
dendprop.temperature_K = 285
dendprop.axial_resistivity = 12
dendprop.set_ion('ca', arb.ion_data(12, 24, -50))

cell.paint('dend', dendprop)

cell.paint('soma', arb.local_parameter_set(
    ))

labels = {'soma': '(tag 1)',
          'axon': '(tag 2)',
          'dend': '(join (tag 3) (tag 4))',
          'all' : '(all)',
          'axso': '(join "soma" "axon")'}
print(labels)

# I would like the labels
#labels.set('soma', '(tag 1)')
#labels.set('axon', '(tag 2)')
#labels.set('dend', '(join (tag 3) (tag 4))')
#labels.set('all',  '(all)')
#labels.set('axso', '(join "soma" "axon")')


# oder...
#labels = arb.label_dict(
   #{'soma': '(tag 1)',
    #'axon': '(tag 2)',
    #'dend': '(join (tag 3) (tag 4))',
    #'all' : '(all)',
    #'axso': '(join "soma" "axon")'});


