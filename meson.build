project('arbor', 'cpp',
  version: '0.2.1dev',
  license: 'BSD')

with_assertions = get_option('arb_assertions')

inc_tclap = include_directories('ext/tclap/include')
inc_json  = include_directories('ext/json/single_include')

inc_arbprivate = include_directories('arbor')
inc_arbpublic  = include_directories('arbor/include')
inc_arbenv     = include_directories('arborenv/include')
inc_arbsup     = include_directories('sup/include')

dep_threads = dependency('threads')

subdir('modcc')
subdir('mechanisms')
subdir('arbor')
subdir('arborenv')
subdir('sup')

subdir('example')
#subdir('example/ring')
#subdir('example/single')
