unit_mechs = [
  'ca_linear',
  'celsius_test',
  'diam_test',
  'fixed_ica_current',
  'gj0',
  'gj1',
  'linear_ca_conc',
  'non_linear',
  'param_as_state',
  'point_ica_current',
  'post_events_syn',
  'read_cai_init',
  'read_eX',
  'test0_kin_compartment',
  'test0_kin_conserve',
  'test0_kin_diff',
  'test0_kin_steadystate',
  'test1_kin_compartment',
  'test1_kin_conserve',
  'test1_kin_diff',
  'test1_kin_steadystate',
  'test2_kin_diff',
  'test3_kin_diff',
  'test4_kin_compartment',
  'test5_nonlinear_diff',
  'test6_nonlinear_diff',
  'test_ca',
  'test_ca_read_valence',
  'test_cl_valence',
  'test_kin1',
  'test_kinlva',
  'test_linear_init',
  'test_linear_init_shuffle',
  'test_linear_state',
  'write_cai_breakpoint',
  'write_eX',
  'write_multiple_eX',
  'write_Xi_Xo'
]

name = 'unit'

unit_mech_sources = []
foreach m: unit_mechs
  modfile = meson.current_source_dir()+f'/@m@.mod'
  generated_files = [f'@m@.hpp', f'@m@_cpu.cpp']
  if arb_with_gpu
    generated_files += [f'@m@_gpu.cu', f'@m@_gpu.cpp']
  endif
  unit_mech_sources += custom_target(f'mechanism:@name@:@m@',
                             input : modfile,
                             output : generated_files,
                             command : [modcc, modfile, '-N', 'testing', '-o', meson.current_build_dir()+'/'+m, arb_modcc_flags])
endforeach

