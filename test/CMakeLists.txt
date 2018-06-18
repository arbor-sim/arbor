find_package(Threads REQUIRED)
add_library(gtest STATIC gtest-all.cpp)
target_include_directories(gtest PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(gtest PUBLIC Threads::Threads)

# Unit tests.
add_subdirectory(unit)

# Test validating models, possebly needing other software installed.
add_subdirectory(validation)

# Test MPI wrappers and distribution operations.
add_subdirectory(unit-distributed)

# Test modcc internals.
add_subdirectory(unit-modcc)

## Microbenchmarks.
add_subdirectory(ubench)
