set(CUDA_PROPOGATE_HOST_FLAGS OFF)

cmake_minimum_required (VERSION 2.8)
project (vector)
set(CMAKE_EXPORT_COMPILE_COMMANDS "YES")
include_directories(${CMAKE_SOURCE_DIR}/include)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread")

set(CMAKE_SHARED_LIBRARY_LINK_C_FLAGS)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# KNL
set( KNL_BACKEND "OFF" CACHE BOOL "Support KNL HBM memory" )
if( KNL_BACKEND )
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DWITH_KNL")
endif()

# verbose build
set( VERBOSE "OFF" CACHE BOOL "Verbose tracing for debuggin" )
if( VERBOSE )
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVERBOSE")
endif()

set( COLOR_PRINTING "OFF" CACHE BOOL "Use color text output" )
if( COLOR_PRINTING )
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCOLOR_PRINTING")
endif()

# so that std=c++11 won't cause strange warnings when passed via -Xcompiler

# CUDA
set( CUDA_BACKEND "OFF" CACHE BOOL "Compile CUDA version" )
if( CUDA_BACKEND )
    # so that std=c++11 won't cause strange warnings when passed via -Xcompiler
    set(CUDA_PROPOGATE_HOST_FLAGS OFF)

    # need CUDA 6.5 or greater for C++11 support
    find_package( CUDA 6.5 )

    # set the flags to pass to nvcc
    set(CUDA_NVCC_FLAGS -arch=sm_35; -std=c++11)
    add_definitions(-DWITH_CUDA)
    include_directories(${CUDA_INCLUDE_DIRS})

    set(CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} $ENV{CRAY_CUDATOOLKIT_POST_LINK_OPTS} -lcudart")
endif()

add_subdirectory(./src)
add_subdirectory(./tests)

