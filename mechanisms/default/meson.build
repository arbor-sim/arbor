name = 'default'
mechs = ['exp2syn', 'expsyn', 'expsyn_stdp', 'gj', 'hh', 'kamt', 'kdrmt', 'nax', 'nernst', 'pas']

mech_srcs = []
foreach m: mechs
  modfile = meson.current_source_dir()+f'/@m@.mod'
  mech_srcs += custom_target(f'mechanism:@name@:@m@',
                             output : [f'@m@.hpp', f'@m@_cpu.cpp'],
                             command : [modcc, modfile, '-N', f'arb::@name@_catalogue', '-o', meson.current_build_dir()+'/'+m, modcc_flags])
endforeach

catalogue_source = f'@name@_catalogue.cpp'
mech_catalogue = custom_target(catalogue_source,
                               output: catalogue_source,
                               command : [gen_cat, catalogue_options, '-C', name, '-N', f'arb::@name@_catalogue', mechs, '-o', '@OUTPUT@'])

catalogue_sources += mech_srcs + mech_catalogue
