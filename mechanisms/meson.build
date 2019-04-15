mechs = ['pas', 'hh', 'expsyn', 'exp2syn', 'test_kin1', 'test_kinlva', 'test_ca', 'nax', 'kdrmt', 'kamt']
modcc_flags = ['-t', 'cpu']

mech_srcs = []
foreach m: mechs
  modfile = meson.current_source_dir()+'/mod/'+m+'.mod'
  mech_srcs += custom_target(m+'.mod',
                             output : [m+'.hpp', m+'_cpu.cpp'],
                             command : [modcc, modfile, '-o', meson.current_build_dir()+'/'+m, modcc_flags])
endforeach

# Make the catalog
# ... todo: we hack the "-I ''" flag, and rely on meson including the path used for the output of mechgen.
#     should ensure that all of these go in the mechanisms/generated path
#mech_dir = meson.current_build_dir() + '/generated'
#default_catalogue_options = ['-A', 'arbor', '-I', mech_dir, '-B', 'multicore']
default_catalogue_options = ['-A', 'arbor', '-I', '', '-B', 'multicore']

gen_cat = find_program('generate_default_catalogue')

mech_catalogue = custom_target('default_catalogue.cpp',
                               output: 'default_catalogue.cpp',
                               command : [gen_cat, default_catalogue_options, mechs, '-o', '@OUTPUT@'])

