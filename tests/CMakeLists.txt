# Large test, employing the full simulator. 
# validated using deltas on output data
#add_subdirectory(delta)

# Test for the internode communication (eg. mpi)
#add_subdirectory(global_communication)

# Test to check integration between components of the site_name
#add_subdirectory(integration)

# Check for performance: This could include stand alone tests
# These do not necessarily be run automatically
#add_subdirectory(performance)

# Numbered tests based on bugs in the tracker
#add_subdirectory(regression)

# google test framework
# This fails with lots of  undefined reference to `testing::UnitTest::Run()'
#add_library(GTEST gtest-all.cpp gtest.h)

# Unit tests
add_subdirectory(unit)

# Test validating models, possebly needing other software installed
add_subdirectory(validation)