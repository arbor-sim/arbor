mechs = ['exp2syn', 'expsyn', 'expsyn_stdp', 'gj', 'hh', 'kamt', 'kdrmt', 'nax', 'nernst', 'pas']
modcc_flags = ['-t', 'cpu']

mech_srcs = []
foreach m: mechs
  modfile = meson.current_source_dir()+'/default/'+m+'.mod'
  mech_srcs += custom_target(m+'.mod',
                             output : [m+'.hpp', m+'_cpu.cpp'],
                             command : [modcc, modfile, '-N', 'arb::default_catalogue', '-o', meson.current_build_dir()+'/'+m, modcc_flags])
endforeach

# Make the catalog
# ... todo: we hack the "-I ''" flag, and rely on meson including the path used for the output of mechgen.
#     should ensure that all of these go in the mechanisms/generated path
#mech_dir = meson.current_build_dir() + '/generated'
#default_catalogue_options = ['-A', 'arbor', '-I', mech_dir, '-B', 'multicore']
default_catalogue_options = ['-A', 'arbor', '-I', '', '-B', 'multicore', '-C', 'default', '-N', 'arb::default_catalogue']

gen_cat = find_program('generate_catalogue')

mech_catalogue = custom_target('default_catalogue.cpp',
                               output: 'default_catalogue.cpp',
                               command : [gen_cat, default_catalogue_options, mechs, '-o', '@OUTPUT@'])

mech_srcs += mech_catalogue
