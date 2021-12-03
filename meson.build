project('arbor', 'cpp',
  version: '0.5.3-dev',
  license: 'BSD')

arb_with_assertions = get_option('arb_assertions')
arb_with_bundled_libs = get_option('arb_use_bundled_libs')

#
# find dependencies
#
if arb_with_bundled_libs
  libjson   = declare_dependency(include_directories: 'ext/json/single_include')
  libfmt    = declare_dependency(include_directories: 'ext/fmt/include')
else
  libjson   = dependency('nlohmann_json', required : true)
  libfmt    = dependency('fmt', required : true)
endif

random123 = declare_dependency(include_directories: 'ext/random123/include')
tinyopt   = declare_dependency(include_directories: 'ext/tinyopt/include')

inc_arbprivate = include_directories('arbor')
inc_arbpublic  = include_directories('arbor/include')
inc_arbenv     = include_directories('arborenv/include')
inc_arbsup     = include_directories('sup/include')

dep_threads = dependency('threads')

subdir('modcc')
subdir('mechanisms')
#subdir('arbor')
#subdir('arborenv')
#subdir('sup')

#subdir('example')
