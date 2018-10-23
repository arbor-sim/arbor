# 3.8 requirement for CUDA language support.
cmake_minimum_required(VERSION 3.8)

file(READ VERSION FULL_VERSION_STRING)
string(STRIP "${FULL_VERSION_STRING}" FULL_VERSION_STRING)
string(REGEX MATCH "^[0-9]+(\\.[0-9]+)?(\\.[0-9]+)?(\\.[0-9]+)?" numeric_version "${FULL_VERSION_STRING}")

project(arbor VERSION ${numeric_version})
enable_language(CXX)


#----------------------------------------------------------
# Configure-time build options for Arbor:
#----------------------------------------------------------

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

option(ARB_WITH_GPU "build with GPU support" OFF)

option(ARB_WITH_MPI "build with MPI support" OFF)

option(ARB_WITH_PROFILING "use built-in profiling" OFF)

option(ARB_WITH_ASSERTIONS "enable arb_assert() assertions in code" OFF)

#----------------------------------------------------------
# Global CMake configuration
#----------------------------------------------------------

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
if(ARB_WITH_GPU)
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

# Note: any target dependency of arbor needs to be explicitly added
# to the 'export set', even the private ones, and this must be done
# in the same CMakeLists.txt in which the target is defined.

# Interface library `arbor-private-deps` collects dependencies, options etc.
# for the arbor library.

add_library(arbor-private-deps INTERFACE)
install(TARGETS arbor-private-deps EXPORT arbor-targets)

# Interface library `arbor-public-deps` collects requirements for the
# users of the arbor library (e.g. mpi) that will become part
# of arbor's PUBLIC interface.

add_library(arbor-public-deps INTERFACE)
install(TARGETS arbor-public-deps EXPORT arbor-targets)

# External libraries in `ext` sub-directory: json and tclap.
# Creates interface libraries `ext-json` and `ext-tclap`.

add_subdirectory(ext)

# Support utiliies in `sup` are common across test executables
# and examples. Creates interface library `arbor-sup`.

add_subdirectory(sup)

# Keep track of packages we need to add to the generated CMake config
# file for arbor.

set(arbor_export_dependencies)

# Keep track of which 'components' of arbor are included (this is
# currently just captures MPI support)

set(arbor_supported_components)

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

list(APPEND arbor_export_dependencies "Threads")

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
    install(TARGETS mpi-wrap EXPORT arbor-targets)

    list(APPEND arbor_export_dependencies "MPI\;COMPONENTS\;CXX")
    list(APPEND arbor_supported_components "MPI")
endif()

# CUDA support
#--------------

if(ARB_WITH_GPU)
    set(ARB_WITH_CUDA TRUE)

    add_compile_options(
        "$<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe=--diag_suppress=integer_sign_change>"
        "$<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe=--diag_suppress=unsigned_compare_with_zero>")
    target_compile_definitions(arbor-private-deps INTERFACE ARB_HAVE_GPU)

    target_compile_options(arbor-private-deps INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_35,code=sm_35>)
    target_compile_options(arbor-private-deps INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_37,code=sm_37>)
    target_compile_options(arbor-private-deps INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_60,code=sm_60>)
    target_compile_options(arbor-private-deps INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_70,code=sm_70>)

endif()

# Use libunwind if available for pretty printing stack traces
#-------------------------------------------------------------

find_package(Unwind)
if(Unwind_FOUND)
    target_link_libraries(arbor-private-deps INTERFACE Unwind::unwind)
    target_compile_definitions(arbor-private-deps ARB_WITH_UNWIND)

    list(APPEND arbor_export_dependencies "Unwind")
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

set(ARB_MODCC_FLAGS)
if(ARB_VECTORIZE)
    list(APPEND ARB_MODCC_FLAGS "--simd")
endif()
if(ARB_WITH_PROFILING)
    list(APPEND ARB_MODCC_FLAGS "--profile")
endif()

#----------------------------------------------------------
# Set up install paths, permissions.
#----------------------------------------------------------

# Set up install paths according to GNU conventions.
#
# GNUInstallDirs picks (e.g.) `lib64` for the library install path on some
# systems where this is definitely not correct (e.g. Arch Linux). If there
# are cases where `lib` is inappropriate, we will have to incorporate special
# case behaviour here.

if(NOT CMAKE_INSTALL_LIBDIR)
    set(CMAKE_INSTALL_LIBDIR lib)
endif()
include(GNUInstallDirs)

# Implicitly created directories require permissions to be set explicitly
# via this CMake variable.
#
# Note that this has no effect until CMake version 3.11.

set(CMAKE_INSTALL_DEFAULT_DIRECTORY_PERMISSIONS
    OWNER_READ
    OWNER_WRITE
    OWNER_EXECUTE
    GROUP_READ
    GROUP_EXECUTE
    WORLD_READ
    WORLD_EXECUTE)

# CMake versions 3.11 and 3.12 ignore this variable for directories
# implicitly created by install(DIRECTORY ...), which for us corresponds
# to our doc and include directories. Work-around by trying to install
# a non-existant file to these locations.

foreach(directory "${CMAKE_INSTALL_DOCDIR}" "${CMAKE_INSTALL_INCLUDEDIR}")
    install(FILES _no_such_file_ OPTIONAL DESTINATION "${directory}")
endforeach()

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

#----------------------------------------------------------
# Generate CMake config/version files for install.
#----------------------------------------------------------

# Note: each dependency for the arbor library target, private or otherwise,
# needs to add itself to the arbor-exports EXPORT target in the subdirectory
# in which they are defined, or none of this will work.

set(cmake_config_dir "${CMAKE_INSTALL_LIBDIR}/cmake/arbor")
install(EXPORT arbor-targets NAMESPACE arbor:: DESTINATION "${cmake_config_dir}")

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/arbor-config-version.cmake"
    COMPATIBILITY SameMajorVersion)

# Template file will use contents of arbor_export_dependencies to include the
# required `find_dependency` statements, and arbor_supported_components will
# be used to check feature support.

configure_file(
    cmake/arbor-config.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/arbor-config.cmake"
    @ONLY)

install(
    FILES
        "${CMAKE_CURRENT_BINARY_DIR}/arbor-config.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/arbor-config-version.cmake"
    DESTINATION "${cmake_config_dir}")

