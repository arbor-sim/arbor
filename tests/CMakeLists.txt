
# google test framework
add_library(gtest gtest-all.cpp)
# tests look for gtest.h here
include_directories(${CMAKE_CURRENT_SOURCE_DIR})


# Unit tests
add_subdirectory(unit)

# Test validating models, possebly needing other software installed
add_subdirectory(validation)

# Test for the internode communication (eg. mpi)
add_subdirectory(global_communication)

# Tests for performance: This could include stand alone tests. These do not necessarily be run automatically
add_subdirectory(performance)


# Proposed additional test types:

# Large test, employing the full simulator. validated using deltas on output data

# Test to check integration between components 


# Numbered tests based on bugs in the tracker

