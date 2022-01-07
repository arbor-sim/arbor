project('arbor',
  ['cpp'],
  version: files('VERSION'),
  meson_version: '>=0.60',
  default_options : ['buildtype=release', 'cpp_std=c++17', 'default_library=static'],
  license: 'BSD')

arb_with_assertions   = get_option('arb_with_assertions')
arb_with_bundled_libs = get_option('arb_with_bundled_libs')
arb_with_mpi          = get_option('arb_with_mpi')
arb_with_gpu          = get_option('arb_with_gpu')
arb_with_neuroml      = get_option('arb_with_neuroml')
arb_with_profiling    = get_option('arb_with_profiling')
arb_with_vectorize    = get_option('arb_with_vectorize')
arb_with_python       = get_option('arb_with_python')
arb_arch              = get_option('arb_arch')
arb_target_flags      = get_option('arb_target_flags')

#
# Extract version information
#

arb_version = meson.project_version()
arb_version_array = arb_version.split('-')
if arb_version_array.length()==2
  arb_version_dev = 'dev'
else
  arb_version_dev = ''
endif
arb_version_array = arb_version_array[0].split('.')
arb_version_major = arb_version_array[0].to_int()
arb_version_minor = arb_version_array[1].to_int()
arb_version_patch = arb_version_array[2].to_int()

#
# Set all compiler flags and definitions that are used by one or more target and / or exported
#

arb_modcc_flags = []
if arb_with_profiling
  arb_modcc_flags += '--profile'
endif
if arb_with_vectorize
  arb_modcc_flags += '--simd'
endif

# set configure-time defines for arbor, arborenv and arborio
# of the form ARB_HAVE_XXX (private) and ARB_XXX (public)
arb_public_defs = []
arb_private_defs = []
arb_configurations = []
if arb_with_mpi
  arb_private_defs += '-DARB_HAVE_MPI'
  arb_configurations += 'MPI'
endif
# TODO: configuration of hip-clang and cuda-clang
if arb_with_gpu
  arb_private_defs += '-DARB_HAVE_GPU'
  arb_public_defs += '-DARB_CUDA'
  arb_configurations += 'GPU'
endif
if arb_with_profiling
  arb_private_defs += '-DARB_HAVE_PROFIILING'
  arb_configurations += 'PROFILE'
endif
if arb_with_assertions
  arb_private_defs += '-DARB_HAVE_ASSERTIONS'
  arb_configurations += 'ASSERT'
endif
if arb_with_neuroml
  arb_configurations += 'NEUROML'
endif
if arb_with_vectorize
  arb_configurations += 'VECTORIZE'
endif

# Set optimization/target-specific architecture flags
arb_target_args = []
if not (arb_arch=='none')
  # The choice of -mcpu=<arch> or -march=<arch> depends on the architecture + OS.
  cpu_id=host_machine.cpu_family()
  os_id=host_machine.system()
  if (cpu_id=='aarch64') and (os_id=='darwin')
    arb_target_args += f'-mcpu=@arb_arch@'
  elif cpu_id=='aarch64' or cpu_id=='x86_64' or cpu_id=='x86'
    arb_target_args += f'-march=@arb_arch@'
  else
    arb_target_args += f'-mcpu=@arb_arch@'
  endif
else
  arb_target_args = arb_target_flags.strip().split()
endif

# Append the target arguments with '-Xcompiler=' for passing through the nvcc compiler wrapper
cuda_target_args = []
foreach arg: arb_target_args
  cuda_target_args += '-Xcompiler=' + arg
endforeach

arbor_private_args = ['-fvisibility=hidden'] + arb_target_args + arb_private_defs + arb_public_defs
arbor_cuda_private_args = ['-Xcompiler=-fvisibility=hidden'] + cuda_target_args + arb_private_defs + arb_public_defs

message('arbor target flags   :', arb_target_args)
message('arbor public defines :', arb_public_defs)
message('arbor private defines:', arb_private_defs)
message('arbor moddcc flags   :', arb_modcc_flags)
message('arbor full flags     :', arbor_private_args)
message('arbor cuda flags     :', arbor_cuda_private_args)

#
# Set up all dependencies
#

if arb_with_gpu
  add_languages('cuda', native: false, required: true)
  dep_cuda = dependency('cuda', version : '>=10', modules : [])
endif

# find dependencies
if arb_with_bundled_libs
  dep_libjson   = declare_dependency(include_directories: 'ext/json/single_include')
  dep_libfmt    = declare_dependency(include_directories: 'ext/fmt/include')
else
  dep_libjson   = dependency('nlohmann_json', required : true)
  dep_libfmt    = dependency('fmt', required : true)
endif
if arb_with_neuroml
  dep_libxml2 = dependency('libxml2', required: true)
endif

dep_random123 = declare_dependency(include_directories: 'ext/random123/include')
dep_tinyopt   = declare_dependency(include_directories: 'ext/tinyopt/include')

inc_arb_private = include_directories('arbor')
inc_arb_public  = include_directories('arbor/include')
inc_arborio_private = include_directories('arborio') # used in the unit tests

dep_threads = dependency('threads')
dep_dl = meson.get_compiler('cpp').find_library('dl', required : true)

if arb_with_mpi
  dep_mpi = dependency('mpi', language: 'cpp')
endif

#
# Visit all of the paths containing targets: no more configuration of flags and dependencies from here on down.
#

subdir('modcc')
subdir('mechanisms')
subdir('arbor')
subdir('arborio')
subdir('arborenv')
subdir('sup')
subdir('test')
subdir('example')
subdir('lmorpho')
if arb_with_python
  subdir('python')
endif

#
# set up pkgconfig
#

# TODO: output all flags etc
pkg_config_variables = ['arbconfiguration='+','.join(arb_configurations)]
pkg_config_unescaped_variables = []
if arb_with_gpu
  pkg_config_unescaped_variables += 'CudaCflags='+' '.join(cuda_target_args+arbor_cuda_private_args)
endif

pkg = import('pkgconfig')
pkg.generate(lib_arb,
  description: 'A high-performance library for computational neuroscience simulations',
  url: 'https://arbor-sim.org/',
  # we can append additional flags for the compilers here (like our optimization flags)
  extra_cflags: arb_target_args + arb_public_defs,
  # The variables and unescaped_variables can be used to pass additional flags and configuration for downstream consumers.
  unescaped_variables: pkg_config_unescaped_variables,
  variables: pkg_config_variables
)
pkg.generate(lib_arborio)
