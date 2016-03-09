cmake_minimum_required (VERSION 2.8)

# project info
project (cell_algorithms)
enable_language(CXX)

# compilation flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++11 -pthread -Wall")

# this generates a .json file with full compilation command for each file
set(CMAKE_EXPORT_COMPILE_COMMANDS "YES")

# generated .a and .so go into /lib
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_SOURCE_DIR})

add_subdirectory(src)
add_subdirectory(tests)
