arborenv_src = [
  'affinity.cpp',
  'arbenvexcept.cpp',
  'concurrency.cpp',
  'default_env.cpp',
  'read_envvar.cpp']

if arb_with_gpu
  arborenv_src += ['gpu_uuid.cpp', 'private_gpu.cpp']
endif

lib_arborenv = library('arborenv',
                 sources: arborenv_src,
                 include_directories: 'include',
                 dependencies: [dep_arb],
                 install: true)

install_subdir('include/arborenv', install_dir: 'include')

dep_arborenv = declare_dependency(
              include_directories: ['include'],
              link_with: lib_arborenv,
              dependencies: [])
