cmake_minimum_required (VERSION 2.8)

# project info
project (modparser)
enable_language(CXX)

# compilation flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread -Wall -Wno-unused-function -Wno-sign-compare")

# this generates a .json file with full compilation command for each file
set(CMAKE_EXPORT_COMPILE_COMMANDS "YES")

set( MODCC_LOGGING "OFF" CACHE BOOL "enable debug logging in compiler" )
if( "${MODCC_LOGGING}" STREQUAL "ON" )
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DLOGGING")
endif()

# generated .a and .so go into /lib
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

include_directories(${CMAKE_SOURCE_DIR}/external)
include_directories(${CMAKE_SOURCE_DIR}/tests/gtest)

add_subdirectory(src)
add_subdirectory(tests)

