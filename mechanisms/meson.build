#pas hh expsyn exp2syn test_kin1 test_kinlva test_ca nax kdrmt kamt
modcc_flags = ['-t', 'cpu']

mech_gen = generator(modcc,
                     output : ['@BASENAME@.hpp', '@BASENAME@_cpu.cpp'],
                     arguments : ['@INPUT@', '-t', 'cpu', '-o', '@BUILD_DIR@/@BASENAME@'])

arb_mechs = mech_gen.process('mod/pas.mod',
                             'mod/hh.mod',
                             'mod/expsyn.mod',
                             'mod/exp2syn.mod',
                             'mod/test_kin1.mod',
                             'mod/test_kinlva.mod',
                             'mod/test_ca.mod',
                             'mod/nax.mod',
                             'mod/kdrmt.mod',
                             'mod/kamt.mod')
