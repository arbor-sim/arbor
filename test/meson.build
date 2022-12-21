#lib_gtest = static_library('gtest',
#              sources: ['gtest-all.cpp'],
#              include_directories: [],
#              dependencies: dep_threads,
#              install: false)

#dep_gtest = declare_dependency(
#              include_directories: ['.'],
#              link_with: lib_gtest,
#              dependencies: dep_threads)

#gtest = subproject('gtest')
#dep_gtest = gtest.get_variable('gtest_dep')

dep_gtest = subproject('gtest').get_variable('gtest_dep')

#libsimple_proj = subproject('libsimple')
#libsimple_dep = libsimple_proj.get_variable('libsimple_dep')

subdir('unit')
subdir('unit-modcc')
subdir('unit-distributed')
