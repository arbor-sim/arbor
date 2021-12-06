name = 'bbp'
mechs = ['CaDynamics_E2', 'Ca_HVA', 'Ca_LVAst', 'Ih', 'Im', 'K_Pst', 'K_Tst', 'Nap_Et2', 'NaTa_t', 'NaTs2_t', 'SK_E2', 'SKv3_1']

mech_srcs = []
foreach m: mechs
  modfile = meson.current_source_dir()+f'/@m@.mod'
  mech_srcs += custom_target(f'mechanism:@name@:@m@',
                             input : modfile,
                             output : [f'@m@.hpp', f'@m@_cpu.cpp'],
                             command : [modcc, modfile, '-N', f'arb::@name@_catalogue', '-o', meson.current_build_dir()+'/'+m, arb_modcc_flags])
endforeach

catalogue_source = f'@name@_catalogue.cpp'
mech_catalogue = custom_target(catalogue_source,
                               output: catalogue_source,
                               command : [gen_cat, catalogue_options, '-C', name, '-N', f'arb::@name@_catalogue', mechs, '-o', '@OUTPUT@'])

catalogue_sources += mech_srcs + mech_catalogue
