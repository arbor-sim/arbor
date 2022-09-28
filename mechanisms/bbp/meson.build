name = 'bbp'
mechs = ['CaDynamics_E2', 'Ca_HVA', 'Ca_LVAst', 'Ih', 'Im', 'K_Pst', 'K_Tst', 'Nap_Et2', 'NaTa_t', 'NaTs2_t', 'SK_E2', 'SKv3_1']

cat_srcs = [f'@name@_catalogue.cpp']
mod_srcs = []
foreach m: mechs
  mod_srcs += meson.current_source_dir()+f'/@m@.mod'
  cat_srcs += [f'@m@.hpp', f'@m@_cpu.cpp']
  if arb_with_gpu
    cat_srcs += [f'@m@_gpu.cu', f'@m@_gpu.cpp']
  endif
endforeach

bbp_cat = custom_target(f'@name@_catalogue',
                    output: cat_srcs,
                    command : [modcc, '-N', f'arb', '-c', f'@name@', '-o', meson.current_build_dir(), arb_modcc_flags, mod_srcs])
