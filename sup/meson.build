arborsup_src = ['ioutil.cpp', 'json_meter.cpp', 'path.cpp']

lib_arborsup = library('arbsup',
                 sources: arborsup_src,
                 include_directories: ['include'],
                 dependencies: [dep_arb, dep_libjson])

dep_arborsup = declare_dependency(
              include_directories: ['include'],
              link_with: lib_arborsup,
              dependencies: [dep_libjson])

