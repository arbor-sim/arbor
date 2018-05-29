cmake_minimum_required(VERSION 3.9)

# project info
project(arbor)
enable_language(CXX)

#----------------------------------------------------------
# Global CMake configuration
#----------------------------------------------------------

# Set release as the default build type.
# Otherwise, CMake will default to debug.
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE release CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "debug" "release")
endif()

# When we find threads, prefer -pthread option.
set(THREADS_PREFER_PTHREAD_FLAG ON)

# Include own CMake modules in search path.
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

# Build paths.
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Installation paths. TODO
# ...

# Generate a .json file with full compilation command for each file.
set(CMAKE_EXPORT_COMPILE_COMMANDS "YES")


# compilation flags
#TODO: move these to target-specific bits
include("CompilerOptions")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CXXOPT_DEBUG} ${CXXOPT_CXX11} ${CXXOPT_PTHREAD} ${CXXOPT_WALL}")

# Used to marshall definitions, include paths and linkage based on configuration state:
add_library(libarbor_deps INTERFACE)

#----------------------------------------------------------
# Configure-time build options for Arbor:
#----------------------------------------------------------

# Enable CUDA support with ARB_GPU_MODEL

set(ARB_GPU_MODEL "none" CACHE STRING "The target GPU architecture: one of {none,K20,K80,P100}")
set_property(CACHE ARB_GPU_MODEL PROPERTY STRINGS none K20 K80 P100 )

# Use in-tree TBB?

option(ARB_PRIVATE_TBBLIB "build and link against in-tree TBB build" OFF)
mark_as_advanced(ARB_PRIVATE_TBBLIB)

# Use externally built modcc?

set(ARB_MODCC "" CACHE STRING "path to external modcc NMODL compiler")

# System-specific configuration for BG/Q, Cray.

set(ARB_SYSTEM_TYPE "Generic" CACHE STRING "Choose a system type to customize flags")
set_property(CACHE ARB_SYSTEM_TYPE PROPERTY STRINGS Generic Cray BGQ )

# # Note: Disabling auto-rebuild of modules is only relevant in
# # some specific development contexts, and should normally be
# # left enabled.
# 
# option(ARB_AUTO_RUN_MODCC_ON_CHANGES
#   "Rerun modcc compiler whenever *.mod file or modcc compiler change" ON)
# mark_as_advanced(ARB_AUTO_RUN_MODCC_ON_CHANGES)


#----------------------------------------------------------
# Configure-time features for Arbor:
#----------------------------------------------------------

set(ARB_THREADING_MODEL "cthread" CACHE STRING "set the threading model, one of cthread/tbb/serial")
set_property(CACHE ARB_THREADING_MODEL PROPERTY STRINGS cthread tbb serial )

set(ARB_DISTRIBUTED_MODEL "serial" CACHE STRING "set the global communication model, one of serial/mpi/dryrun")
set_property(CACHE ARB_DISTRIBUTED_MODEL PROPERTY STRINGS serial mpi dryrun)

option(ARB_WITH_PROFILING "use built-in profiling" OFF)

option(ARB_WITH_ASSERTIONS "enable EXPECTS() assertions in code" OFF)

# Note: TRACE() is used only in development, and should never
# be present in any released code; it is safe to leave TRACE macros enabled.

option(ARB_WITH_TRACE "enable TRACE() macros in code" ON)
mark_as_advanced(ARB_WITH_TRACE)

#----------------------------------------------------------
# Set up flags and dependencies for library target:
#----------------------------------------------------------

# System-specific defines
#-------------------------

if(${ARB_SYSTEM_TYPE} MATCHES "Cray")
    # User should also enable dynamic linking by setting the environment variable:
    #   CRAYPE_LINK_TYPE=dynamic
    add_definitions(-DARB_HAVE_CRAY)
endif()

# Profiling and test features
#-----------------------------

if(ARB_WITH_PROFILING)
    target_compile_definitions(libarbor_deps ARB_HAVE_PROFILING)
endif()
if(ARB_WITH_ASSERTIONS)
    target_compile_definitions(libarbor_deps ARB_HAVE_ASSERTIONS)
endif()
if(ARB_WITH_TRACE)
    target_compile_definitions(libarbor_deps ARB_HAVE_TRACE)
endif()

# Threading model
#-----------------

if(ARB_THREADING_MODEL MATCHES "tbb")
    if(ARB_PRIVATE_TBBLIB)
        # Build and link against a local copy of TBB:

        include(GitSubmodule) # required for check_git_submodule
        check_git_submodule(tbb "${CMAKE_SOURCE_DIR}/tbb")
        if(NOT tbb_avail)
            message(FATAL_ERROR "TBB git submodule required.")
        endif()

        # Turn off proxy malloc library and test compilation.
        option(TBB_BUILD_TBBMALLOC_PROXY OFF)
        option(TBB_BUILD_TESTS           OFF)

        add_subdirectory(tbb)

        target_link_libraries(libarbor_deps INTERFACE tbb_static tbbmalloc_static)
        target_include_directories(libarbor_deps SYSTEM INTERFACE ${CMAKE_SOURCE_DIR}/tbb/include)
    else()
        find_package(TBB REQUIRED)
        target_link_libraries(libarbor_deps INTERFACE TBB::tbb)
    endif()
    target_compile_definitions(libarbor_deps INTERFACE ARB_HAVE_TBB)

elseif(ARB_THREADING_MODEL MATCHES "cthread")
    set(ARB_WITH_CTHREAD TRUE)

    find_package(Threads REQUIRED)
    target_compile_definitions(libarbor_deps ARB_HAVE_CTHREAD)
    target_link_libraries(libarbor_deps INTERFACE Threads::Threads)

endif()

# Distributed model
#-------------------

if(ARB_DISTRIBUTED_MODEL MATCHES "mpi")
    # BGQ specific flags
    if(${ARB_SYSTEM_TYPE} MATCHES "BGQ" )
        # On BGQ, set CXX to the mpi wrapper, and pass it a static
        target_compile_definitions(libarbor_deps INTERFACE MPICH2_CONST=const)
        set(MPI_FOUND TRUE)
    else()
        find_package(MPI REQUIRED CXX)
        target_compile_definitions(MPI::MPI_CXX INTERFACE MPICH_SKIP_MPICXX=1 OMPI_SKIP_MPICXX=1)
    endif()

    target_compile_definitions(libarbor_deps INTERFACE ARB_HAVE_MPI)
    set(ARB_WITH_MPI TRUE)

elseif(ARB_DISTRIBUTED_MODEL MATCHES "dryrun")
    target_compile_definitions(libarbor_deps INTERFACE ARB_HAVE_DRYRUN)
    set(ARB_WITH_DRYRUN TRUE)

elseif(ARB_DISTRIBUTED_MODEL MATCHES "serial")
    # no additional set up needed

else()
    message( FATAL_ERROR "-- Distributed communication model '${ARB_DISTRIBUTED_MODEL}' not supported, use one of serial/mpi/dryrun")
endif()

# CUDA support
#--------------

if(NOT ARB_GPU_MODEL MATCHES "none")
    include(CheckLanguage)
    check_language(CUDA)

    if (CMAKE_CUDA_COMPILER)
        message(FATAL_ERROR "Missing CUDA support.")
    endif()
    enable_language(CUDA)
    set(ARB_WITH_CUDA TRUE)

    target_compile_options(libarbor_deps INTERFACE
        $<$<COMPILE_LANGUAGE:CUDA>:--diag_suppress=integer_sign_change
        $<$<COMPILE_LANGUAGE:CUDA>:--diag_suppress=unsigned_compare_with_zero)
    target_compile_definitions(libarbor_deps ARB_HAVE_GPU)

    if(ARB_GPU_MODEL MATCHES "K20")
        set(cuda_arch sm_35)
        set(arb_cuda_arch 350)
    elseif(ARB_GPU_MODEL MATCHES "K80")
        set(cuda_arch sm_37)
        set(arb_cuda_arch 370)
    elseif(ARB_GPU_MODEL MATCHES "P100")
        set(cuda_arch sm_60)
        set(arb_cuda_arch 600)
    else()
        message(FATAL_ERROR "-- GPU architecture '${ARB_GPU_MODEL}' not supported. Use one of {none, K20, K80, P100}")
    endif()

    target_compile_options(libarbor_deps INTERFACE
        $<$<COMPILE_LANGUAGE:CUDA>:-arch=${cuda_arch})
    target_compile_definitions(libarbor_deps ARB_CUDA_ARCH=${arb_cuda_arch})
endif()

# Use libunwind if available for pretty printing stack traces
#-------------------------------------------------------------

find_package(Unwind)
if(Unwind_FOUND)
    target_link_libraries(libarbor_deps Unwind::unwind)
    target_compile_definitions(libarbor_deps ARB_WITH_UNWIND)
endif()

# Build and use modcc unless explicit path given
#------------------------------------------------

if(ARB_MODCC)
    find_program(modcc NAMES ${ARB_MODCC} NO_CMAKE_PATH NO_CMAKE_ENVIRONMENT_PATH NO_CMAKE_SYSTEM_PATH)
    if(NOT modcc)
        message(FATAL_ERROR "Unable to find modcc executable.")
    endif()
    set(ARB_WITH_EXTERNAL_MODCC TRUE)
else()
    set(modcc $<TARGET_FILE:modcc>)
endif()

#----------------------------------------------------------
# prepare list of libraries/includes needed by external libs
#----------------------------------------------------------

# TODO: Replace with public include directories for libarbor:
# i.e. $<INSTALL_INTERFACE:include> $<BUILD_INTERFACE:.../include>
set(EXTERNAL_LIBRARIES "")
# TODO: Replace with interface libraries on libarbor target:
set(EXTERNAL_INCLUDES "")


#----------------------------------------------------------
# Modcc vectorization target
#----------------------------------------------------------
option(ARB_VECTORIZE "use explicit SIMD code in generated mechanisms" OFF)

#----------------------------------------------------------
# Target microarchitecture for building arbor libraries
#----------------------------------------------------------
set(ARB_ARCH "" CACHE STRING "Target architecture for arbor libraries")
if(ARB_ARCH)
    # Sets CXXOPT_ARCH variable accordingly:
    set_arch_target("${ARB_ARCH}")
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

#----------------------------------------------------------
# set up for targets that require git submodules.
#----------------------------------------------------------
include(GitSubmodule) # required for check_git_submodule
include(ErrorTarget)  # reguired for add_error_target
check_git_submodule(rtdtheme "${PROJECT_SOURCE_DIR}/doc/rtd_theme")

add_subdirectory(src)
add_subdirectory(mechanisms) # after src path so that gpu_mechanism library is last on link line
add_subdirectory(tests)
add_subdirectory(example)
add_subdirectory(lmorpho)
if (rtdtheme_avail)
    add_subdirectory(doc)
else()
    add_error_target( docs
        "Generating Sphinx documentation"
        "The git submodule for read the docs is not available")
endif()

