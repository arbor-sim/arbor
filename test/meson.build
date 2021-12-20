lib_gtest = static_library('gtest',
              sources: ['gtest-all.cpp'],
              include_directories: [],
              dependencies: dep_threads,
              install: false)

dep_gtest = declare_dependency(
              include_directories: ['.'],
              link_with: lib_gtest,
              dependencies: dep_threads)

subdir('unit')
subdir('unit-modcc')
subdir('unit-distributed')
