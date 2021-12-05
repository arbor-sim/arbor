project('arbor', 'cpp',
  version: '0.5.3-dev',
  default_options : ['buildtype=release'],
  license: 'BSD')

# Initialize up configurable options
arb_with_assertions   = get_option('arb_with_assertions')
arb_with_bundled_libs = get_option('arb_with_bundled_libs')
arb_with_mpi          = get_option('arb_with_mpi')
arb_with_gpu          = get_option('arb_with_gpu')
arb_with_neuroml      = get_option('arb_with_neuroml')
arb_with_profiling    = get_option('arb_with_profiling')
arb_with_vectorize    = get_option('arb_with_vectorize')
arb_arch              = get_option('arb_arch')

if arb_with_mpi
  error('unable to configure mpi')
endif
if arb_with_gpu
  error('unable to configure gpu')
endif
if arb_with_neuroml
  error('unable to configure neuroml')
endif
if arb_with_vectorize
  error('unable to configure vectorization')
endif
if arb_with_profiling
  error('unable to configure profiling')
endif

# Extract version information
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

message('version', arb_version_major, arb_version_minor, arb_version_patch)

# find dependencies
if arb_with_bundled_libs
  dep_libjson   = declare_dependency(include_directories: 'ext/json/single_include')
  dep_libfmt    = declare_dependency(include_directories: 'ext/fmt/include')
else
  dep_libjson   = dependency('nlohmann_json', required : true)
  dep_libfmt    = dependency('fmt', required : true)
endif

dep_random123 = declare_dependency(include_directories: 'ext/random123/include')
dep_tinyopt   = declare_dependency(include_directories: 'ext/tinyopt/include')

inc_arb_private = include_directories('arbor')
inc_arb_public  = include_directories('arbor/include')

dep_threads = dependency('threads')
dep_dl = meson.get_compiler('cpp').find_library('dl', required : true)

subdir('modcc')
subdir('arbor')
#subdir('mechanisms')
#subdir('arborenv')
#subdir('arborio)
#subdir('sup')

#subdir('example')
