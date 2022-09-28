name = 'allen'
mechs = ['CaDynamics', 'Ca_HVA', 'Ca_LVA', 'Ih', 'Im', 'Im_v2', 'Kd', 'K_P', 'K_T', 'Kv2like', 'Kv3_1', 'Nap', 'NaTa', 'NaTs', 'NaV', 'SK']

cat_srcs = [f'@name@_catalogue.cpp']
mod_srcs = []
foreach m: mechs
  mod_srcs += meson.current_source_dir()+f'/@m@.mod'
  cat_srcs += [f'@m@.hpp', f'@m@_cpu.cpp']
  if arb_with_gpu
    cat_srcs += [f'@m@_gpu.cu', f'@m@_gpu.cpp']
  endif
endforeach

allen_cat = custom_target(f'@name@_catalogue',
              output: cat_srcs,
              command : [modcc, '-N', f'arb', '-c', f'@name@', '-o', meson.current_build_dir(), arb_modcc_flags, mod_srcs])
