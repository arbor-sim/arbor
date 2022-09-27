find_package(Threads REQUIRED)

# Using shared gtest library seems to cause problems at test teardown (double free) on certain
# systems, potentially related to this issue https://github.com/google/googletest/issues/930. Always
# building a static library works around this problem.
add_library(gtest STATIC EXCLUDE_FROM_ALL gtest-all.cpp)
set_target_properties(gtest PROPERTIES CXX_VISIBILITY_PRESET hidden)
target_include_directories(gtest PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(gtest PUBLIC Threads::Threads)

# Convenience target that builds all tests.
# Test executable targets should be added to the 'tests' target as dependencies.
add_custom_target(tests)

# Unit tests.
# Builds: unit.
add_subdirectory(unit)

# Test MPI wrappers and distribution operations.
# Builds: unit-local and unit-mpi (if MPI enabled).
add_subdirectory(unit-distributed)

# Test modcc internals.
# Builds: unit-modcc.
add_subdirectory(unit-modcc)

# Microbenchmarks.
# Builds: one executable per microbenchmark under ubench/.
add_subdirectory(ubench)
