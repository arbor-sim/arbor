cmake_minimum_required(VERSION 3.0)

# project info
project(cell_algorithms)
enable_language(CXX)

# Hide warnings about mixing old and new signatures for target_link_libraries.
# These can't be avoided, because the FindCUDA packed provided by CMake before
# version 3.9 uses the old signature, while other packages use the new signature.
if ("${CMAKE_VERSION}" MATCHES "^3.[0-9].")
    cmake_policy(SET CMP0023 OLD)
endif()

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

#----------------------------------------------------------
# Option to enable assertions
#----------------------------------------------------------
option(ARB_WITH_ASSERTIONS "enable EXPECTS() assertions in code" OFF)
if(ARB_WITH_ASSERTIONS)
    add_definitions("-DARB_HAVE_ASSERTIONS")
endif()

#----------------------------------------------------------
# Option to enable traces
#----------------------------------------------------------
option(ARB_WITH_TRACE "enable TRACE() macros in code" OFF)
if(ARB_WITH_TRACE)
    add_definitions("-DARB_HAVE_TRACE")
endif()

#----------------------------------------------------------
# Option to disable auto running of modcc compiler
#----------------------------------------------------------
option(ARB_AUTO_RUN_MODCC_ON_CHANGES
  "Rerun modcc compiler whenever *.mod file or modcc compiler change" ON)

#----------------------------------------------------------
# prepare list of libraries/includes needed by external libs
#----------------------------------------------------------
set(EXTERNAL_LIBRARIES "")
set(EXTERNAL_INCLUDES "")

#----------------------------------------------------------
# list of libraries built inside Arbor
#----------------------------------------------------------
set(ARB_LIBRARIES "")

#----------------------------------------------------------
# Threading model selection
#----------------------------------------------------------
set(ARB_THREADING_MODEL "cthread" CACHE STRING "set the threading model, one of cthread/tbb/serial")
set_property(CACHE ARB_THREADING_MODEL PROPERTY STRINGS cthread tbb serial )

if(ARB_THREADING_MODEL MATCHES "tbb")
    # download and build TBB
    include(${CMAKE_SOURCE_DIR}/cmake/tbb/TBBGet.cmake)
    include(${CMAKE_SOURCE_DIR}/cmake/tbb/TBBBuild.cmake)
    tbb_get(TBB_ROOT tbb_root SOURCE_CODE)
    tbb_build(TBB_ROOT ${tbb_root} CONFIG_DIR TBB_DIR)

    # configure TBB
    find_package(TBB REQUIRED tbb tbbmalloc)

    add_definitions(-DARB_HAVE_TBB)
    set(ARB_WITH_TBB TRUE)

    # The TBB CMake package creates the TBB::tbb and TBB:tbbmalloc targets.
    # The following should be sufficient, however on some systems (e.g. Cray PE),
    # they don't work (because CMake is fragile):
    #   list(APPEND EXTERNAL_LIBRARIES TBB::tbb)
    #   list(APPEND EXTERNAL_LIBRARIES TBB::tbbmalloc)
    # So, instead, manually add the library links:
    set(TBB_LIB_PATH ${CMAKE_BINARY_DIR}/tbb_cmake_build/tbb_cmake_build_subdir_release)
    list(APPEND EXTERNAL_LIBRARIES ${TBB_LIB_PATH}/libtbb.so)
    list(APPEND EXTERNAL_LIBRARIES ${TBB_LIB_PATH}/libtbbmalloc.so)

    # The TBB headers are in the downloaded TBB source
    list(APPEND EXTERNAL_INCLUDES ${tbb_root}/include)

elseif(ARB_THREADING_MODEL MATCHES "cthread")

    find_package(Threads REQUIRED)
    add_definitions(-DARB_HAVE_CTHREAD)
    set(ARB_WITH_CTHREAD TRUE)
    list(APPEND EXTERNAL_LIBRARIES ${CMAKE_THREAD_LIBS_INIT})

    if(CMAKE_USE_PTHREADS_INIT)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
    endif()
endif()

#----------------------------------------------------------
# libunwind for pretty printing stack traces
#----------------------------------------------------------
find_package(Unwind)
if(UNWIND_FOUND)
    add_definitions(-DWITH_UNWIND)
    include_directories(${UNWIND_INCLUDE_DIR})
    list(APPEND EXTERNAL_LIBRARIES ${UNWIND_LIBRARIES})
endif()

#----------------------------------------------------------
# CUDA support
#----------------------------------------------------------
option(ARB_WITH_CUDA "use CUDA for GPU offload" OFF)
if(ARB_WITH_CUDA)
    find_package(CUDA REQUIRED)

    # Turn off annoying and incorrect warnings generated in the JSON file.
    # We also work around the same issue with the intel compiler.
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-Xcudafe \"--diag_suppress=not_used_in_template_function_params\";-Xcudafe \"--diag_suppress=cast_to_qualified_type\")

    # set the CUDA target specfic flags
    # code regions protected by ARB_HAVE_CUDA should only be available to the CUDA
    # compiler, which regions protected by ARB_HAVE_GPU are visible to both host
    # and device compiler when targetting GPU.
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-DARB_HAVE_CUDA)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-DARB_HAVE_GPU)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch=sm_60)

    add_definitions(-DARB_HAVE_GPU)
    include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
    list(APPEND EXTERNAL_LIBRARIES ${CUDA_LIBRARIES})
endif()

#----------------------------------------------------------
# Cray/BGQ/Generic Linux/other flag?
#----------------------------------------------------------
set(ARB_SYSTEM_TYPE "Generic" CACHE STRING 
    "Choose a system type to customize flags")
set_property(CACHE ARB_SYSTEM_TYPE PROPERTY STRINGS Generic Cray BGQ )

# Cray specific flags
if(${ARB_SYSTEM_TYPE} MATCHES "Cray")
    # we no longer set the -dynamic flag for the compilere here, instead dynamic
    # linking should be enabled by the user by setting the environment variable:
    #   CRAYPE_LINK_TYPE=dynamic
    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -dynamic")
    add_definitions(-DARB_HAVE_CRAY)
endif()

#----------------------------------------------------------
# MPI support
#----------------------------------------------------------
set(ARB_DISTRIBUTED_MODEL "serial" CACHE STRING "set the global communication model, one of serial/mpi/dryrun")
set_property(CACHE ARB_DISTRIBUTED_MODEL PROPERTY STRINGS serial mpi dryrun)

if(ARB_DISTRIBUTED_MODEL MATCHES "mpi")
   # BGQ specific flags
   if(${ARB_SYSTEM_TYPE} MATCHES "BGQ" )
      # On BGQ, set CXX to the mpi wrapper, and pass it a static
      add_definitions(-DMPICH2_CONST=const)
      set(MPI_FOUND TRUE)
    endif()

    if (NOT MPI_FOUND)
      find_package(MPI REQUIRED)
    endif()
    include_directories(SYSTEM ${MPI_C_INCLUDE_PATH})
    add_definitions(-DARB_HAVE_MPI)

    # unfortunate workaround for C++ detection in system mpi.h
    add_definitions(-DMPICH_SKIP_MPICXX=1 -DOMPI_SKIP_MPICXX=1)
    set_property(DIRECTORY APPEND_STRING PROPERTY COMPILE_OPTIONS "${MPI_C_COMPILE_FLAGS}")

    set(ARB_WITH_MPI TRUE)

elseif(ARB_DISTRIBUTED_MODEL MATCHES "dryrun")
    add_definitions(-DARB_HAVE_DRYRUN)
    set(ARB_WITH_DRYRUN TRUE)

elseif(ARB_DISTRIBUTED_MODEL MATCHES "serial")
    # no additional set up needed

else()
    message( FATAL_ERROR "-- Distributed communication model '${ARB_DISTRIBUTED_MODEL}' not supported, use one of serial/mpi/dryrun")
endif()

#----------------------------------------------------------
# Built-in profiler
#----------------------------------------------------------
option(ARB_WITH_PROFILING "use built-in profiling" OFF)
if(ARB_WITH_PROFILING)
    add_definitions(-DARB_HAVE_PROFILING)
endif()

#----------------------------------------------------------
# vectorization target
#----------------------------------------------------------
set(ARB_VECTORIZE_TARGET "none" CACHE STRING "CPU target for vectorization {KNL,AVX2,AVX512}")
set_property(CACHE ARB_VECTORIZE_TARGET PROPERTY STRINGS none KNL AVX2 AVX512)

if(ARB_VECTORIZE_TARGET STREQUAL "KNL")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CXXOPT_KNL} -DSIMD_KNL")
elseif(ARB_VECTORIZE_TARGET STREQUAL "AVX2")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CXXOPT_AVX2} -DSIMD_AVX2")
elseif(ARB_VECTORIZE_TARGET STREQUAL "AVX512")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CXXOPT_AVX512} -DSIMD_AVX512")
endif()

#----------------------------------------------------------
# Only build modcc if it has not already been installed.
# This is useful if cross compiling for KNL, when it is not desirable to compile
# modcc with the same flags that are used for the KNL target.
#----------------------------------------------------------
set(use_external_modcc OFF)
find_program(MODCC_BIN modcc)
if(MODCC_BIN)
    set(use_external_modcc ON)
    set(modcc "${MODCC_BIN}")
else()
    set(modcc $<TARGET_FILE:modcc>)
endif()

#----------------------------------------------------------
# Validation data generation
#----------------------------------------------------------
# destination directory for generated data
set(ARB_VALIDATION_DATA_DIR "${PROJECT_SOURCE_DIR}/validation/data" CACHE PATH
  "location of generated validation data")

#----------------------------------------------------------
# Whether to build validation data
#----------------------------------------------------------
# turn off by default
option(ARB_BUILD_VALIDATION_DATA "generate validation data" OFF)
if (ARB_BUILD_VALIDATION_DATA)
    # Whether to attempt to use julia to build validation data
    find_program(JULIA_BIN julia)
    if(JULIA_BIN STREQUAL "JULIA_BIN-NOTFOUND")
        message(STATUS "julia not found; will not automatically build validation data sets from julia scripts")
        set(ARB_BUILD_JULIA_VALIDATION_DATA FALSE)
    else()
        set(ARB_BUILD_JULIA_VALIDATION_DATA TRUE)
    endif()

    # Whether to attempt to use nrniv to build validation data
    # (if we find nrniv, do)
    find_program(NRNIV_BIN nrniv)
    if(NRNIV_BIN STREQUAL "NRNIV_BIN-NOTFOUND")
        message(STATUS "nrniv not found; will not automatically build NEURON validation data sets")
        set(ARB_BUILD_NRN_VALIDATION_DATA FALSE)
    else()
        set(ARB_BUILD_NRN_VALIDATION_DATA TRUE)
    endif()
endif()

#----------------------------------------------------------
# Setup include dirs
#----------------------------------------------------------
include_directories(
    "${PROJECT_SOURCE_DIR}/tclap"
    "${PROJECT_SOURCE_DIR}/include"
    "${PROJECT_SOURCE_DIR}/src"
    "${PROJECT_SOURCE_DIR}/modcc"
    "${PROJECT_SOURCE_DIR}")
if(EXTERNAL_INCLUDES)
  include_directories("${EXTERNAL_INCLUDES}")
endif()

#----------------------------------------------------------
# Setup subdirs
#----------------------------------------------------------
# only include validation data if flag is set
if(ARB_BUILD_VALIDATION_DATA)
    add_subdirectory(validation)
endif()

# only compile modcc if it is not provided externally
if(NOT use_external_modcc)
    add_subdirectory(modcc)
endif()

add_subdirectory(src)
add_subdirectory(mechanisms) # after src path so that gpu_mechanism library is last on link line
add_subdirectory(tests)
add_subdirectory(example)
add_subdirectory(lmorpho)
add_subdirectory(doc)
