cmake_minimum_required (VERSION 2.8)

# project info
project (cell_algorithms)
enable_language(CXX)

# save incoming CXX flags for forwarding to modparser external project
set(SAVED_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

# compilation flags
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include("CompilerOptions")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CXXOPT_DEBUG} ${CXXOPT_CXX11} ${CXXOPT_PTHREAD} ${CXXOPT_WALL}")

# this generates a .json file with full compilation command for each file
set(CMAKE_EXPORT_COMPILE_COMMANDS "YES")

# generated .a and .so go into /lib
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# TBB support
set(WITH_TBB "ON" CACHE BOOL "use TBB for on-node threading")
if("${WITH_TBB}" STREQUAL "ON")
    find_package(TBB REQUIRED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DWITH_TBB ${TBB_DEFINITIONS}")
endif()

# MPI support
set(WITH_MPI "ON" CACHE BOOL "use MPI for distributed processing")
if("${WITH_MPI}" STREQUAL "ON")
    find_package(MPI REQUIRED)
    include_directories(SYSTEM ${MPI_C_INCLUDE_PATH})
    add_definitions(-DWITH_MPI)
    # unfortunate workaround for C++ detection in system mpi.h
    add_definitions(-DMPICH_SKIP_MPICXX=1 -DOMPI_SKIP_MPICXX=1)
    set_property(DIRECTORY APPEND_STRING PROPERTY COMPILE_OPTIONS "${MPI_C_COMPILE_FLAGS}")
endif()

# targets for extermal dependencies
include(ExternalProject)
externalproject_add(modparser
    PREFIX ${CMAKE_BINARY_DIR}/external
    CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/external"
               "-DCMAKE_CXX_FLAGS=${SAVED_CXX_FLAGS}"
               "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
    BINARY_DIR "${CMAKE_BINARY_DIR}/external/modparser"
    STAMP_DIR  "${CMAKE_BINARY_DIR}/external/"
    TMP_DIR    "${CMAKE_BINARY_DIR}/external/tmp"
    SOURCE_DIR "${CMAKE_SOURCE_DIR}/external/modparser"
    )


include_directories(${CMAKE_SOURCE_DIR}/external)
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_SOURCE_DIR}/miniapp)
include_directories(${CMAKE_SOURCE_DIR})

if( "${WITH_TBB}" STREQUAL "ON" )
    include_directories(${TBB_INCLUDE_DIRS})
endif()

add_subdirectory(mechanisms)
add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(miniapp)

