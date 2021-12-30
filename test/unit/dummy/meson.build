name = 'dummy'
dummy_mechs = ['dummy']
namespace = f'arb::@name@_catalogue'

dummy_mech_sources = []
foreach m: dummy_mechs
  modfile = meson.current_source_dir()+f'/@m@.mod'
  generated_files = [f'@m@.hpp', f'@m@_cpu.cpp']
  if arb_with_gpu
    generated_files += [f'@m@_gpu.cu', f'@m@_gpu.cpp']
  endif
  dummy_mech_sources += custom_target(f'mechanism:@name@:@m@',
                             input : modfile,
                             output : generated_files,
                             command : [modcc, modfile, '-N', namespace, '-o', meson.current_build_dir()+'/'+m, arb_modcc_flags])
endforeach

dummy_catalogue_source = custom_target('dummy-catalogue-source',
                               output: f'@name@_catalogue.cpp',
                               command : [gen_cat, catalogue_options, '-C', name, '-N', namespace, dummy_mechs, '-o', '@OUTPUT@'])

dummy_catalogue = shared_library('dummy-catalogue',
                            name_prefix: '',
                            sources: [dummy_mech_sources, dummy_catalogue_source],
                            dependencies: [dep_arb],
                            cpp_args : ['-DSTANDALONE'] + arbor_private_args,
                            cuda_args: arbor_cuda_private_args,
                            install: false,
                            build_by_default: false)

