# google test framework
add_library(gtest gtest-all.cpp)

# Unit tests
add_subdirectory(unit)

# Test validating models, possebly needing other software installed
add_subdirectory(validation)

# Test for the internode communication (eg. mpi)
add_subdirectory(global_communication)

# Tests for performance: This could include stand alone tests. These do not necessarily be run automatically
add_subdirectory(performance)

# Microbenchmarks.
# Attempt to update git submodule if required.
check_git_submodule(google_bench "${CMAKE_CURRENT_SOURCE_DIR}/ubench/google-benchmark")
if (google_bench_avail)
    add_subdirectory(ubench)
else()
    add_error_target(ubenches
        "Building micro benchmarks"
        "Git submodule for google benchmark is not available")
endif()

# regression / delta tests
# Employing the full simulator. validated using deltas on output data

# modcc tests
if(NOT use_external_modcc)
    add_subdirectory(modcc)
endif()


# Proposed additional test types:

# Test to check integration between components

# Numbered tests based on bugs in the tracker
