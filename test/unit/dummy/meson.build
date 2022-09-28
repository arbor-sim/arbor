name = 'dummy'
mechs = ['dummy']

cat_srcs = [f'@name@_catalogue.cpp']
mod_srcs = []
foreach m: mechs
  mod_srcs += meson.current_source_dir()+f'/@m@.mod'
  cat_srcs += [f'@m@.hpp', f'@m@_cpu.cpp']
  if arb_with_gpu
    cat_srcs += [f'@m@_gpu.cu', f'@m@_gpu.cpp']
  endif
endforeach

dummy_cat = custom_target(f'@name@_catalogue',
                          output: cat_srcs,
                          command : [modcc, '-N', f'arb', '-c', f'@name@', '-o', meson.current_build_dir(), arb_modcc_flags, mod_srcs])

dummy_catalogue = shared_library('dummy-catalogue',
                                 name_prefix: '',
                                 name_suffix: 'so',
                                 sources: [dummy_cat],
                                 dependencies: [dep_arb],
                                 cpp_args : ['-DSTANDALONE'] + arbor_private_args,
                                 cuda_args: arbor_cuda_private_args,
                                 install: false,
                                 build_by_default: false)
