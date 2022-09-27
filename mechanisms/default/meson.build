name = 'default'
mechs = ['exp2syn', 'expsyn', 'expsyn_stdp', 'gj', 'hh', 'kamt', 'kdrmt', 'nax', 'nernst', 'pas']

cat_srcs = [f'@name@_catalogue.cpp']
mod_srcs = []
foreach m: mechs
  mod_srcs += meson.current_source_dir()+f'/@m@.mod'
  cat_srcs = [f'@m@.hpp', f'@m@_cpu.cpp']
  if arb_with_gpu
    cat_srcs += [f'@m@_gpu.cu', f'@m@_gpu.cpp']
  endif
endforeach

cat = custom_target(f'@name@_catalogue',
              output: cat_srcs,
              command : [modcc, '-N', f'arb', '-c', f'@name@', '-o', meson.current_build_dir(), arb_modcc_flags, mod_srcs])

catalogue_sources += cat
