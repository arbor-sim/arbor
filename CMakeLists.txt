cmake_minimum_required(VERSION 2.8)

# project info
project(cell_algorithms)
enable_language(CXX)

# save incoming CXX flags for forwarding to modcc external project
set(SAVED_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

# compilation flags
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
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

#----------------------------------------------------------
# Option to disable auto running of modcc compiler
#----------------------------------------------------------
option(NMC_AUTO_RUN_MODCC_ON_CHANGES 
  "Rerun modcc compiler whenever *.mod file or modcc compiler change" ON)

# list of libraries to be linked against targets
set(EXTERNAL_LIBRARIES "")

#threading model selection
set(THREADING_MODEL "serial" CACHE STRING "set the threading model, one of serial/tbb/omp")
if(THREADING_MODEL MATCHES "tbb")
    # TBB support
    find_package(TBB REQUIRED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TBB_DEFINITIONS}")
    add_definitions(-DWITH_TBB)
    list(APPEND EXTERNAL_LIBRARIES ${TBB_LIBRARIES})

elseif(THREADING_MODEL MATCHES "omp")
    # OpenMP support
    find_package(OpenMP REQUIRED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    add_definitions(-DWITH_OMP)

elseif(THREADING_MODEL MATCHES "serial")
    #setup previously done

else()
    message( FATAL_ERROR "-- Threading model '${THREADING_MODEL}' not supported, use one of serial/tbb/omp")

endif()

# libunwind for pretty printing stack traces
find_package(Unwind)
if(UNWIND_FOUND)
    add_definitions(-DWITH_UNWIND)
    include_directories(${UNWIND_INCLUDE_DIR})
    list(APPEND EXTERNAL_LIBRARIES ${UNWIND_LIBRARIES})
endif()

# CUDA support
set(WITH_CUDA OFF CACHE BOOL "use CUDA for GPU offload" )
if(WITH_CUDA)
    find_package(CUDA REQUIRED)

    # Turn off annoying and incorrect warnings generated in the JSON file.
    # We also work around the same issue with the intel compiler.
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-Xcudafe \"--diag_suppress=not_used_in_template_function_params\";-Xcudafe \"--diag_suppress=cast_to_qualified_type\")

    # set the CUDA target specfic flags
    # code regions protected by WITH_CUDA should only be available to the CUDA
    # compiler, which regions protected by WITH_GPU are visible to both host
    # and device compiler when targetting GPU.
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-DWITH_CUDA)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-DWITH_GPU)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch=sm_35)
    #set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch=sm_60)

    add_definitions(-DWITH_GPU)
    include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
    list(APPEND EXTERNAL_LIBRARIES ${CUDA_LIBRARIES})
endif()

# BGQ systems
set(SYSTEM_BGQ OFF CACHE BOOL "Flags for BlueGene Q")

# MPI support
set(WITH_MPI OFF CACHE BOOL "use MPI for distrubuted parallelism")
if(WITH_MPI)
    find_package(MPI REQUIRED)
    include_directories(SYSTEM ${MPI_C_INCLUDE_PATH})
    add_definitions(-DWITH_MPI)
    # unfortunate workaround for C++ detection in system mpi.h
    add_definitions(-DMPICH_SKIP_MPICXX=1 -DOMPI_SKIP_MPICXX=1)
    set_property(DIRECTORY APPEND_STRING PROPERTY COMPILE_OPTIONS "${MPI_C_COMPILE_FLAGS}")

   if(SYSTEM_BGQ)
     add_definitions(-DMPICH2_CONST=const)
   endif()
endif()

# Internal profiler support
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

# Only build modcc if it has not already been installed.
# This is useful if cross compiling for KNL, when it is not desirable to compile
# modcc with the same flags that are used for the KNL target.
set(use_external_modcc OFF)
find_program(MODCC_BIN modcc)
if(MODCC_BIN STREQUAL "MODCC_BIN-NOTFOUND")
    set(modcc "${CMAKE_BINARY_DIR}/modcc/modcc")
else()
    set(use_external_modcc ON)
    set(modcc "${MODCC_BIN}")
endif()

# Validation data generation

# destination directory for generated data
set(VALIDATION_DATA_DIR "${PROJECT_SOURCE_DIR}/validation/data" CACHE PATH "location of generated validation data")

# Whether to build validation data at all
set(BUILD_VALIDATION_DATA ON CACHE BOOL "generate validation data")

# Whether to attempt to use julia to build validation data
find_program(JULIA_BIN julia)
if(JULIA_BIN STREQUAL "JULIA_BIN-NOTFOUND")
    message(STATUS "julia not found; will not automatically build validation data sets from julia scripts")
    set(BUILD_JULIA_VALIDATION_DATA FALSE)
else()
    set(BUILD_JULIA_VALIDATION_DATA TRUE)
endif()

# Whether to attempt to use nrniv to build validation data
# (if we find nrniv, do)
find_program(NRNIV_BIN nrniv)
if(NRNIV_BIN STREQUAL "NRNIV_BIN-NOTFOUND")
    message(STATUS "nrniv not found; will not automatically build NEURON validation data sets")
    set(BUILD_NRN_VALIDATION_DATA FALSE)
else()
    set(BUILD_NRN_VALIDATION_DATA TRUE)
endif()

include_directories(${PROJECT_SOURCE_DIR}/tclap)
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/miniapp)
include_directories(${PROJECT_SOURCE_DIR}/modcc)
include_directories(${PROJECT_SOURCE_DIR})
if( "${WITH_TBB}" STREQUAL "ON" )
    include_directories(${TBB_INCLUDE_DIRS})
endif()

# only include validation data if flag is set
if(BUILD_VALIDATION_DATA)
    add_subdirectory(validation)
endif()

# only compile modcc if it is not provided externally
if(NOT use_external_modcc)
    add_subdirectory(modcc)
endif()

add_subdirectory(mechanisms)
add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(miniapp)

