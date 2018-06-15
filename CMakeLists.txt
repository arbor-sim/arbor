cmake_minimum_required(VERSION 3.9)

project(arbor VERSION 0.1)
enable_language(CXX)

#----------------------------------------------------------
# Global CMake configuration
#----------------------------------------------------------

# Use GNU standard installation path conventions.
include(GNUInstallDirs)

# Include own CMake modules in search path, load common modules.

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
include(GitSubmodule) # required for check_git_submodule
include(ErrorTarget)  # reguired for add_error_target

# Set release as the default build type (CMake default is debug.)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE release CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "debug" "release")
endif()

# When we find threads, prefer -pthread option.
set(THREADS_PREFER_PTHREAD_FLAG ON)

# Build paths.

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Generate a .json file with full compilation command for each file.

set(CMAKE_EXPORT_COMPILE_COMMANDS "YES")

# Compiler options common to library, examples, tests, etc.

include("CompilerOptions")
add_compile_options(${CXXOPT_DEBUG} ${CXXOPT_WALL})
set(CMAKE_CXX_STANDARD 11)

#----------------------------------------------------------
# Configure-time build options for Arbor:
#----------------------------------------------------------

# Enable CUDA support with ARB_GPU_MODEL.

set(ARB_GPU_MODEL "none" CACHE STRING "The target GPU architecture: one of {none,K20,K80,P100}")
set_property(CACHE ARB_GPU_MODEL PROPERTY STRINGS none K20 K80 P100 )

# Specify target archiecture.

set(ARB_ARCH "" CACHE STRING "Target architecture for arbor libraries")

# Perform explicit vectorization?

option(ARB_VECTORIZE "use explicit SIMD code in generated mechanisms" OFF)

# Use in-tree TBB?

option(ARB_PRIVATE_TBBLIB "build and link against in-tree TBB build" OFF)

# Use externally built modcc?

set(ARB_MODCC "" CACHE STRING "path to external modcc NMODL compiler")

# Generate validation data for validation tests?

option(ARB_BUILD_VALIDATION_DATA "generate validation data" OFF)

# Where to generate and find validation data?

set(ARB_VALIDATION_DATA_DIR "${PROJECT_SOURCE_DIR}/validation/data" CACHE PATH
  "location of generated validation data")

#----------------------------------------------------------
# Configure-time features for Arbor:
#----------------------------------------------------------

set(ARB_THREADING_MODEL "cthread" CACHE STRING "set the threading model, one of cthread/tbb/serial")
set_property(CACHE ARB_THREADING_MODEL PROPERTY STRINGS cthread tbb serial )

if(ARB_THREADING_MODEL STREQUAL "tbb")
set(ARB_WITH_TBB TRUE)
endif()

option(ARB_WITH_MPI "build with MPI support" OFF)

option(ARB_WITH_PROFILING "use built-in profiling" OFF)

option(ARB_WITH_ASSERTIONS "enable EXPECTS() assertions in code" OFF)

# Note: TRACE() is used only in development, and should never
# be present in any released code; it is safe to leave TRACE macros enabled.

option(ARB_WITH_TRACE "enable TRACE() macros in code" ON)
mark_as_advanced(ARB_WITH_TRACE)

#----------------------------------------------------------
# Set up flags and dependencies:
#----------------------------------------------------------

# Interface library `arbor-private` collects dependencies, options etc.
# for the arbor library and arbor unit tests.

add_library(arbor-private INTERFACE)

# External libraries in `ext` sub-directory: json, tclap and
# (if ARB_REQUIRE_TBB is true) tbb.
# Creates interface libraries `ext-json`, `ext-tclap` and `ext-tbb`.

add_subdirectory(ext)

# Auxilary/helper utiliies in `aux` are common across test executables
# and examples. Creates interface library `aux`.

add_subdirectory(aux)

# Target microarchitecture for building arbor libraries, tests and examples
#---------------------------------------------------------------------------
if(ARB_ARCH)
    set_arch_target(ARB_CXXOPT_ARCH "${ARB_ARCH}")
    target_compile_options(arbor-private INTERFACE ${ARB_CXXOPT_ARCH})
endif()

# Profiling and test features
#-----------------------------

if(ARB_WITH_PROFILING)
    target_compile_definitions(arbor-private INTERFACE ARB_HAVE_PROFILING)
endif()
if(ARB_WITH_ASSERTIONS)
    target_compile_definitions(arbor-private INTERFACE ARB_HAVE_ASSERTIONS)
endif()

# Threading model
#-----------------

if(ARB_THREADING_MODEL MATCHES "tbb")
    target_link_libraries(arbor-private INTERFACE ext-tbb)
elseif(ARB_THREADING_MODEL MATCHES "cthread")
    set(ARB_WITH_CTHREAD TRUE)

    find_package(Threads REQUIRED)
    target_compile_definitions(arbor-private INTERFACE ARB_HAVE_CTHREAD)
    target_link_libraries(arbor-private INTERFACE Threads::Threads)

endif()

# MPI support
#-------------------

if(ARB_WITH_MPI)
    find_package(MPI REQUIRED CXX)
    target_compile_definitions(MPI::MPI_CXX INTERFACE MPICH_SKIP_MPICXX=1 OMPI_SKIP_MPICXX=1)
    target_compile_definitions(arbor-private INTERFACE ARB_HAVE_MPI)
    target_link_libraries(arbor-private INTERFACE MPI::MPI_CXX)
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

    target_compile_options(arbor-private INTERFACE
        $<$<COMPILE_LANGUAGE:CUDA>:--diag_suppress=integer_sign_change
        $<$<COMPILE_LANGUAGE:CUDA>:--diag_suppress=unsigned_compare_with_zero)
    target_compile_definitions(arbor-private ARB_HAVE_GPU)

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

    target_compile_options(arbor-private INTERFACE
        $<$<COMPILE_LANGUAGE:CUDA>:-arch=${cuda_arch})
    target_compile_definitions(arbor-private ARB_CUDA_ARCH=${arb_cuda_arch})
endif()

# Use libunwind if available for pretty printing stack traces
#-------------------------------------------------------------

find_package(Unwind)
if(Unwind_FOUND)
    target_link_libraries(arbor-private INTERFACE Unwind::unwind)
    target_compile_definitions(arbor-private ARB_WITH_UNWIND)
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
    set(ARB_WITH_EXTERNAL_MODCC FALSE)
endif()


#----------------------------------------------------------
# Configure targets in sub-directories.
#----------------------------------------------------------


# arbor-public-headers:
add_subdirectory(include)

# modcc, libmodcc:
add_subdirectory(modcc)

# arbor, arbor-private-headers:
add_subdirectory(arbor)

# unit, unit-mpi, unit-local, unit-modcc, validate
add_subdirectory(test)

# add_subdirectory(example)

# lmorpho:
add_subdirectory(lmorpho)

# add_subdirectory(doc)

# validation-data:
if(ARB_BUILD_VALIDATION_DATA)
    add_subdirectory(validation) # validation-data
endif()

