find_package(Threads REQUIRED)
add_library(gtest STATIC gtest-all.cpp)
target_include_directories(gtest PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(gtest PUBLIC Threads::Threads)

## Unit tests.
add_subdirectory(unit)
#
## Test validating models, possebly needing other software installed.
#add_subdirectory(validation)
#
## Test MPI wrappers and distribution operations.
#add_subdirectory(unit-mpi)
#
# Test modcc internals.
add_subdirectory(modcc)

## Microbenchmarks.
## Attempt to update git submodule if required.
#check_git_submodule(google_bench "${CMAKE_CURRENT_SOURCE_DIR}/ubench/google-benchmark")
#if (google_bench_avail)
#    add_subdirectory(ubench)
#else()
#    add_error_target(ubenches
#        "Building micro benchmarks"
#        "The git submodule for google benchmark is not available")
#endif()
