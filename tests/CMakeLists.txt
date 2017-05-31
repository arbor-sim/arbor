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
add_subdirectory(ubench)

# regression / delta tests
# Employing the full simulator. validated using deltas on output data

# modcc tests
if(NOT use_external_modcc)
    add_subdirectory(modcc)
endif()


# Proposed additional test types:

# Test to check integration between components

# Numbered tests based on bugs in the tracker
