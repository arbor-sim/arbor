cmake_minimum_required(VERSION 2.8)

# project info
project(cell_algorithms)
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

# enable assertions?
set(WITH_ASSERTIONS OFF CACHE BOOL "enable EXPECTS() assertions in code")
if(WITH_ASSERTIONS)
    add_definitions("-DWITH_ASSERTIONS")
endif()

# enable traces?
set(WITH_TRACE OFF CACHE BOOL "enable TRACE() macros in code")
if(WITH_TRACE)
    add_definitions("-DWITH_TRACE")
endif()

# TBB support
set(WITH_TBB OFF CACHE BOOL "use TBB for on-node threading" )
if(WITH_TBB)
    find_package(TBB REQUIRED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TBB_DEFINITIONS}")
    add_definitions(-DWITH_TBB)
endif()

# MPI support
set(WITH_MPI OFF CACHE BOOL "use MPI for distrubuted parallelism")
if(WITH_MPI)
    find_package(MPI REQUIRED)
    include_directories(SYSTEM ${MPI_C_INCLUDE_PATH})
    add_definitions(-DWITH_MPI)
    # unfortunate workaround for C++ detection in system mpi.h
    add_definitions(-DMPICH_SKIP_MPICXX=1 -DOMPI_SKIP_MPICXX=1)
    set_property(DIRECTORY APPEND_STRING PROPERTY COMPILE_OPTIONS "${MPI_C_COMPILE_FLAGS}")
endif()

# Profiler support
set(WITH_PROFILING OFF CACHE BOOL "use built-in profiling of miniapp" )
if(WITH_PROFILING)
    add_definitions(-DWITH_PROFILING)
endif()

# Cray systems
set(SYSTEM_CRAY OFF CACHE BOOL "add flags for compilation on Cray systems")
if(SYSTEM_CRAY)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -dynamic")
endif()

# vectorization target
set(VECTORIZE_TARGET "none" CACHE STRING "CPU target for vectorization {KNL,AVX,AVX2}")

if(VECTORIZE_TARGET STREQUAL "KNL")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CXXOPT_KNL}")
elseif(VECTORIZE_TARGET STREQUAL "AVX")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CXXOPT_AVX}")
elseif(VECTORIZE_TARGET STREQUAL "AVX2")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CXXOPT_AVX2}")
endif()

# whether to generate optimized kernels from NMODL
set(USE_OPTIMIZED_KERNELS OFF CACHE BOOL "generate optimized code that vectorizes with the Intel compiler")

include_directories(${CMAKE_SOURCE_DIR}/external)
include_directories(${CMAKE_SOURCE_DIR}/external/tclap/include)
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

