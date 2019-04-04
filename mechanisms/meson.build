mechs = ['pas', 'hh', 'expsyn', 'exp2syn', 'test_kin1', 'test_kinlva', 'test_ca', 'nax', 'kdrmt', 'kamt']
modcc_flags = ['-t', 'cpu']

mech_gen = generator(modcc,
                     output : ['@BASENAME@.hpp', '@BASENAME@_cpu.cpp'],
                     arguments : ['@INPUT@', '-o', '@BUILD_DIR@/@BASENAME@']+modcc_flags)

# Make a list of the mod files packaged with Arbor
modfiles = []
foreach m: mechs
  modfiles += 'mod/'+m+'.mod'
endforeach

# Generate the header and source files for mechanisms
arb_mechs = mech_gen.process(modfiles)

# Make the catalog
mech_dir = meson.current_build_dir() + '/generated'
default_catalogue_options = ['-A', 'arbor', '-I', mech_dir, '-B', 'multicore']

gen_cat = find_program('generate_default_catalogue')

mech_catalogue = custom_target('default_catalogue.cpp',
                               output: 'default_catalogue.cpp',
                               command : [gen_cat, default_catalogue_options, '-o', '@OUTPUT@'])

