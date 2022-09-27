name = 'testing'
mechs = [
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

cat_srcs = [f'@name@_catalogue.cpp']
mod_srcs = []
foreach m: mechs
  mod_srcs += meson.current_source_dir()+f'/@m@.mod'
  cat_srcs = [f'@m@.hpp', f'@m@_cpu.cpp']
  if arb_with_gpu
    cat_srcs += [f'@m@_gpu.cu', f'@m@_gpu.cpp']
  endif
endforeach

unit_testing_cat = custom_target(f'@name@_catalogue',
                                 output: cat_srcs,
                                 command: [modcc, '-N', f'arb', '-c', f'@name@', '-o', meson.current_build_dir(), arb_modcc_flags, mod_srcs])
