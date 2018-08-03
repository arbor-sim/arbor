# 3.8 requirement for CUDA language support.
cmake_minimum_required(VERSION 3.8)

project(arbor VERSION 0.1)
enable_language(CXX)

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

option(ARB_WITH_MPI "build with MPI support" OFF)

option(ARB_WITH_PROFILING "use built-in profiling" OFF)

option(ARB_WITH_ASSERTIONS "enable arb_assert() assertions in code" OFF)

#----------------------------------------------------------
# Global CMake configuration
#----------------------------------------------------------

# Use GNU standard installation path conventions.
include(GNUInstallDirs)

# Include own CMake modules in search path, load common modules.

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
include(GitSubmodule) # required for check_git_submodule
include(ErrorTarget)  # reguired for add_error_target
include(FindThreadsCudaFix) # bug work around

# Set release as the default build type (CMake default is debug.)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE release CACHE STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "debug" "release")
endif()

# When we find threads, prefer -pthread option.
set(THREADS_PREFER_PTHREAD_FLAG ON)

# Add CUDA as a language if GPU support requested.
# (This has to be set early so as to enable CUDA tests in generator
# expressions.)
if(NOT ARB_GPU_MODEL MATCHES "none")
    enable_language(CUDA)
endif()

# Build paths.

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Generate a .json file with full compilation command for each file.

set(CMAKE_EXPORT_COMPILE_COMMANDS "YES")

# Detect and deprecate xlC.

include("CheckCompilerXLC")

# Compiler options common to library, examples, tests, etc.

include("CompilerOptions")
add_compile_options(
    "$<$<COMPILE_LANGUAGE:CXX>:${CXXOPT_DEBUG}>"
    "$<$<COMPILE_LANGUAGE:CXX>:${CXXOPT_WALL}>")
set(CMAKE_CXX_STANDARD 14)

#----------------------------------------------------------
# Set up flags and dependencies:
#----------------------------------------------------------

# Interface library `arbor-private-deps` collects dependencies, options etc.
# for the arbor library.

add_library(arbor-private-deps INTERFACE)

# Interface library `arbor-public-deps` collects requirements for the
# users of the arbor library (e.g. mpi) that will become part
# of arbor's PUBLIC interface.

add_library(arbor-public-deps INTERFACE)

# External libraries in `ext` sub-directory: json and tclap.
# Creates interface libraries `ext-json` and `ext-tclap`.

add_subdirectory(ext)

# Auxilary/helper utiliies in `aux` are common across test executables
# and examples. Creates interface library `arbor-aux`.

add_subdirectory(aux)

# Target microarchitecture for building arbor libraries, tests and examples
#---------------------------------------------------------------------------
if(ARB_ARCH)
    set_arch_target(ARB_CXXOPT_ARCH "${ARB_ARCH}")
    target_compile_options(arbor-private-deps INTERFACE ${ARB_CXXOPT_ARCH})
endif()

# Profiling and test features
#-----------------------------

if(ARB_WITH_PROFILING)
    target_compile_definitions(arbor-private-deps INTERFACE ARB_HAVE_PROFILING)
endif()
if(ARB_WITH_ASSERTIONS)
    target_compile_definitions(arbor-private-deps INTERFACE ARB_HAVE_ASSERTIONS)
endif()

# Threading model
#-----------------

find_package(Threads REQUIRED)
find_threads_cuda_fix()
target_link_libraries(arbor-private-deps INTERFACE Threads::Threads)

# MPI support
#-------------------

if(ARB_WITH_MPI)
    find_package(MPI REQUIRED CXX)
    target_compile_definitions(arbor-private-deps INTERFACE ARB_HAVE_MPI)

    # target_compile_definitions(MPI::MPI_CXX INTERFACE MPICH_SKIP_MPICXX=1 OMPI_SKIP_MPICXX=1)
    # target_link_libraries(arbor-public-deps INTERFACE MPI::MPI_CXX)

    # CMake 3.9 does not allow us to add definitions to an import target.
    # so wrap MPI::MPI_CXX in an interface library 'mpi-wrap' instead.

    add_library(mpi-wrap INTERFACE)
    target_link_libraries(mpi-wrap INTERFACE MPI::MPI_CXX)
    target_compile_definitions(mpi-wrap INTERFACE MPICH_SKIP_MPICXX=1 OMPI_SKIP_MPICXX=1)
    target_link_libraries(arbor-public-deps INTERFACE mpi-wrap)
endif()

# CUDA support
#--------------

if(NOT ARB_GPU_MODEL MATCHES "none")
    set(ARB_WITH_CUDA TRUE)

    add_compile_options(
        "$<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe=--diag_suppress=integer_sign_change>"
        "$<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe=--diag_suppress=unsigned_compare_with_zero>")
    target_compile_definitions(arbor-private-deps INTERFACE ARB_HAVE_GPU)

    if(ARB_GPU_MODEL MATCHES "K20")
        set(cuda_arch sm_35)
        set(arb_cuda_arch 350)
    elseif(ARB_GPU_MODEL MATCHES "K80")
        set(cuda_arch sm_37)
        set(arb_cuda_arch 370)
    elseif(ARB_GPU_MODEL MATCHES "P100")
        set(cuda_code_0 sm_35)
        set(cuda_arch_0 compute_30)
        set(cuda_code_1 sm_37)
        set(cuda_arch_1 compute_30)
        set(cuda_code_2 sm_60)
        set(cuda_arch_2 compute_60)
        set(arb_cuda_arch 600)
    else()
        message(FATAL_ERROR "-- GPU architecture '${ARB_GPU_MODEL}' not supported. Use one of {none, K20, K80, P100}")
    endif()

    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=${cuda_arch_0},code=${cuda_code_0}>)
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=${cuda_arch_1},code=${cuda_code_1}>)
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=${cuda_arch_2},code=${cuda_code_2}>)
    target_compile_definitions(arbor-private-deps INTERFACE ARB_CUDA_ARCH=${arb_cuda_arch})
endif()

# Use libunwind if available for pretty printing stack traces
#-------------------------------------------------------------

find_package(Unwind)
if(Unwind_FOUND)
    target_link_libraries(arbor-private-deps INTERFACE Unwind::unwind)
    target_compile_definitions(arbor-private-deps ARB_WITH_UNWIND)
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

# miniapp, brunel-minapp, event-gen 
add_subdirectory(example)

# lmorpho:
add_subdirectory(lmorpho)

# html:
add_subdirectory(doc)

# validation-data:
if(ARB_BUILD_VALIDATION_DATA)
    add_subdirectory(validation) # validation-data
endif()

