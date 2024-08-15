cmake_minimum_required(VERSION 3.19)
include(CMakeDependentOption)
include(CheckIPOSupported)

# Make CUDA support throw errors if architectures remain unclear
cmake_policy(SET CMP0104 NEW)

file(READ VERSION FULL_VERSION_STRING)
string(STRIP "${FULL_VERSION_STRING}" FULL_VERSION_STRING)
string(REGEX MATCH "^[0-9]+(\\.[0-9]+)?(\\.[0-9]+)?(\\.[0-9]+)?" numeric_version "${FULL_VERSION_STRING}")

project(arbor VERSION ${numeric_version})
enable_language(CXX)

include(GNUInstallDirs)
include(CheckCXXCompilerFlag)

# Effectively adds '-fpic' flag to CXX_FLAGS. Needed for dynamic catalogues.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Have LTO where possible, ie add -flto
check_ipo_supported(RESULT HAVE_LTO OUTPUT ERR_LTO)
if(NOT DEFINED CMAKE_INTERPROCEDURAL_OPTIMIZATION)
  if(HAVE_LTO)
    message (STATUS "LTO support found, enabling")
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
  else()
    message(STATUS "No LTO: ${ERR_LTO}")
  endif()
endif()

# Turn on this option to force the compilers to produce color output when output is
# redirected from the terminal (e.g. when using ninja or a pager).

option(ARBDEV_COLOR "Always produce ANSI-colored output (GNU/Clang only)." OFF)

#----------------------------------------------------------
# Configure-time build options for Arbor:
#----------------------------------------------------------

# Specify target architecture.
check_cxx_compiler_flag("-march=native" CXX_HAS_NATIVE)
if(CXX_HAS_NATIVE)
    set(ARB_DEFAULT_ARCH "native")
else()
    set(ARB_DEFAULT_ARCH "none")
endif()
set(ARB_ARCH ${ARB_DEFAULT_ARCH} CACHE STRING "Target architecture for arbor libraries")

# Perform explicit vectorization?

option(ARB_VECTORIZE "use explicit SIMD code in generated mechanisms" OFF)

# Support for Thread pinning

option(ARB_USE_HWLOC "request support for thread pinning via HWLOC" OFF)
mark_as_advanced(ARB_USE_HWLOC)

# Use externally built modcc?

set(ARB_MODCC "" CACHE STRING "path to external modcc NMODL compiler")

# Use libunwind to generate stack traces on errors?

option(ARB_BACKTRACE "Enable stacktraces on assertion and exceptions (requires Boost)." OFF)

# Specify GPU build type

set(ARB_GPU "none" CACHE STRING "GPU backend and compiler configuration")
set_property(CACHE PROPERTY STRINGS "none" "cuda" "cuda-clang" "hip")
if(NOT ARB_GPU STREQUAL "none")
    set(ARB_USE_GPU_DEP ON)
endif()
cmake_dependent_option(ARB_USE_GPU_RNG
    "Use GPU generated random numbers (only cuda, not bitwise equal to CPU version)" OFF
    "ARB_USE_GPU_DEP" OFF)

# Use bundled 3rd party libraries
option(ARB_USE_BUNDLED_LIBS "Use bundled 3rd party libraries" OFF)

# Use pybind11-stubgen to make type stubs.
option(ARB_BUILD_PYTHON_STUBS "Use pybind11-stubgen to build type stubs." ON)

# Optional additional CXX Flags used for all code that will run on the target
# CPU architecture. Recorded in installed target, for downstream dependencies
# to use.
# Useful, for example, when a user wants to compile with target-specific
# optimization flags.
set(ARB_CXX_FLAGS_TARGET "" CACHE STRING "Optional additional flags for compilation")

#----------------------------------------------------------
# Debug support
#----------------------------------------------------------

# Print builtin catalogue configuration while building
option(ARB_CAT_VERBOSE "Print catalogue build information" OFF)
mark_as_advanced(ARB_CAT_VERBOSE)

#----------------------------------------------------------
# Configure-time features for Arbor:
#----------------------------------------------------------

option(ARB_WITH_MPI "build with MPI support" OFF)

option(ARB_WITH_PROFILING "use built-in profiling" OFF)

option(ARB_WITH_ASSERTIONS "enable arb_assert() assertions in code" OFF)

#----------------------------------------------------------
# Python front end for Arbor:
#----------------------------------------------------------

option(ARB_WITH_PYTHON "enable Python front end" OFF)

#----------------------------------------------------------
# Global CMake configuration
#----------------------------------------------------------

# Include own CMake modules in search path, load common modules.

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
include(GitSubmodule) # required for check_git_submodule
include(ErrorTarget)  # reguired for add_error_target

# Set release as the default build type (CMake default is debug.)

if (NOT CMAKE_BUILD_TYPE AND NOT BUILD_TESTING)
    set(CMAKE_BUILD_TYPE release CACHE STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "debug" "release")
endif()

# Add CUDA as a language if GPU support requested. (This has to be set early so
# as to enable CUDA tests in generator expressions.)
if(ARB_GPU STREQUAL "cuda")
    include(FindCUDAToolkit)
    set(ARB_WITH_NVCC TRUE)
    # CMake 3.18 and later set the default CUDA architecture for
    # each target according to CMAKE_CUDA_ARCHITECTURES. 

    # This fixes nvcc picking up a wrong host compiler for linking, causing
    # issues with outdated libraries, eg libstdc++ and std::filesystem. Must
    # happen before all calls to enable_language(CUDA)
    set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
    enable_language(CUDA)
    find_package(CUDAToolkit)
    if(${CUDAToolkit_VERSION_MAJOR} GREATER_EQUAL 12)
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            # Pascal, Volta, Ampere, Hopper
            set(CMAKE_CUDA_ARCHITECTURES 60 70 80 90)
        endif()
    else()
        message(FATAL_ERROR "Need at least CUDA 12, got ${CUDAToolkit_VERSION_MAJOR}")
    endif()

    # We _still_ need this otherwise CUDA symbols will not be exported
    # from libarbor.a leading to linker errors when link external clients.
    # Unit tests are NOT external enough. Re-review this somewhere in the
    # future.
    find_package(CUDA ${CUDAToolkit_VERSION_MAJOR} REQUIRED)
elseif(ARB_GPU STREQUAL "cuda-clang")
    include(FindCUDAToolkit)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 60 70 80 90)
    endif()
    set(ARB_WITH_CUDA_CLANG TRUE)
    enable_language(CUDA)
elseif(ARB_GPU STREQUAL "hip")
    set(ARB_WITH_HIP_CLANG TRUE)
    # Specify AMD architecture using a (user provided) list.
    # Note: CMake native HIP architectures are introduced with version 3.21.
    set(ARB_HIP_ARCHITECTURES gfx906 gfx900 CACHE STRING "AMD offload architectures (semicolon separated)")
endif()

if(ARB_WITH_NVCC OR ARB_WITH_CUDA_CLANG OR ARB_WITH_HIP_CLANG)
    set(ARB_WITH_GPU TRUE)
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
add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:${CXXOPT_WALL}>")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

#----------------------------------------------------------
# Set up flags and dependencies:
#----------------------------------------------------------

# Note: any target dependency of arbor needs to be explicitly added
# to the 'export set', even the private ones, and this must be done
# in the same CMakeLists.txt in which the target is defined.

# Data and internal scripts go here
set(ARB_INSTALL_DATADIR ${CMAKE_INSTALL_DATAROOTDIR}/arbor)

# Interface library `arbor-config-defs` collects configure-time defines
# for arbor, arborenv, arborio, of the form ARB_HAVE_XXX. These
# defines should _not_ be used in any installed public headers.

add_library(arbor-config-defs INTERFACE)
install(TARGETS arbor-config-defs EXPORT arbor-targets)

# Interface library `arbor-private-deps` collects dependencies, options etc.
# for the arbor library.
add_library(arbor-private-deps INTERFACE)
target_link_libraries(arbor-private-deps INTERFACE arbor-config-defs ext-random123 ${CMAKE_DL_LIBS})
install(TARGETS arbor-private-deps EXPORT arbor-targets)

# Interface library `arborenv-private-deps` collects dependencies, options etc.
# for the arborenv library.

add_library(arborenv-private-deps INTERFACE)
target_link_libraries(arborenv-private-deps INTERFACE arbor-config-defs)
install(TARGETS arborenv-private-deps EXPORT arbor-targets)

# Interface library `arborio-private-deps` collects dependencies, options etc.
# for the arborio library.

add_library(arborio-private-deps INTERFACE)
target_link_libraries(arborio-private-deps INTERFACE arbor-config-defs)
install(TARGETS arborio-private-deps EXPORT arbor-targets)

# Interface library `arbor-public-deps` collects requirements for the
# users of the arbor library (e.g. mpi) that will become part
# of arbor's PUBLIC interface.

add_library(arbor-public-deps INTERFACE)
install(TARGETS arbor-public-deps EXPORT arbor-targets)

# Interface library `arborio-public-deps` collects requirements for the
# users of the arborio library (e.g. xml libs) that will become part
# of arborio's PUBLIC interface.

add_library(arborio-public-deps INTERFACE)
install(TARGETS arborio-public-deps EXPORT arborio-targets)

# Add scripts and supporting CMake for setting up external catalogues

install(PROGRAMS scripts/arbor-build-catalogue DESTINATION ${CMAKE_INSTALL_BINDIR})
install(FILES mechanisms/BuildModules.cmake DESTINATION ${ARB_INSTALL_DATADIR})

# External libraries in `ext` sub-directory: json, tinyopt and random123.
# Creates interface libraries `ext-tinyopt` and `ext-random123`

cmake_dependent_option(ARB_USE_BUNDLED_FMT "Use bundled FMT lib." ON "ARB_USE_BUNDLED_LIBS" OFF)
cmake_dependent_option(ARB_USE_BUNDLED_PUGIXML "Use bundled XML lib." ON "ARB_USE_BUNDLED_LIBS" OFF)
cmake_dependent_option(ARB_USE_BUNDLED_GTEST "Use bundled GoogleTest." ON "ARB_USE_BUNDLED_LIBS" OFF)
# TODO When we get a units spack package...
#cmake_dependent_option(ARB_USE_BUNDLED_UNITS "Use bundled LLNL units." ON "ARB_USE_BUNDLED_LIBS" OFF)
set(ARB_USE_BUNDLED_UNITS ON CACHE STRING "Use bundled LLNL units.")

cmake_dependent_option(ARB_USE_BUNDLED_JSON "Use bundled Niels Lohmann's json library." ON "ARB_USE_BUNDLED_LIBS" OFF)
if(NOT ARB_USE_BUNDLED_JSON)
  find_package(nlohmann_json 3.11.2 CONFIG REQUIRED)
  message(STATUS "Using external JSON = ${nlohmann_json_VERSION}")
endif()

cmake_dependent_option(ARB_USE_BUNDLED_RANDOM123 "Use bundled Random123 lib." ON "ARB_USE_BUNDLED_LIBS" OFF)
add_library(ext-random123 INTERFACE)
if(NOT ARB_USE_BUNDLED_RANDOM123)
    find_package(Random123 REQUIRED)
    target_include_directories(ext-random123 INTERFACE ${RANDOM123_INCLUDE_DIR})
endif()

# in the event we can find hwloc, just add it
find_package(hwloc QUIET)
add_library(ext-hwloc INTERFACE)
if(hwloc_FOUND)
    # We'd like to use the package syntax, here, yet if we do, we'd need to
    # provide the find script to the system.
    target_link_directories(ext-hwloc INTERFACE ${hwloc_LIBRARY_DIRS})
    target_link_libraries(ext-hwloc INTERFACE ${hwloc_LIBRARY})
    target_include_directories(ext-hwloc INTERFACE ${hwloc_INCLUDE_DIR})
    target_compile_definitions(ext-hwloc INTERFACE ARB_HAVE_HWLOC)
    target_link_libraries(arbor-private-deps INTERFACE ext-hwloc)
else()
    if(ARB_USE_HWLOC)
        message(SEND_ERROR "Requested support for hwloc, but CMake couldn't find it.")
    endif()
endif()

add_library(ext-units INTERFACE)
if(ARB_USE_BUNDLED_UNITS)
    target_link_libraries(ext-units INTERFACE units::units)
else()
    message(FATAL, "TODO: At the time of Arbor 0.10.0 there is no Spack package")
endif()


add_subdirectory(ext)
install(TARGETS ext-hwloc EXPORT arbor-targets)
install(TARGETS ext-random123 EXPORT arbor-targets)
target_link_libraries(arbor-public-deps INTERFACE ext-units)
install(TARGETS ext-units EXPORT arbor-targets)
install(TARGETS units compile_flags_target EXPORT arbor-targets)

# Keep track of packages we need to add to the generated CMake config
# file for arbor.

set(arbor_export_dependencies)

# Keep track of which 'components' of arbor are included (this is
# currently just 'MPI' support and 'neuroml' for NeuroML support in
# libarborio.)

set(arbor_supported_components)

# Target microarchitecture for building arbor libraries, tests and examples
#---------------------------------------------------------------------------

# Set the full set of target flags in ARB_CXX_FLAGS_TARGET_FULL, which
# will include target-specific -march flags if ARB_ARCH is not "none".
if(ARB_ARCH STREQUAL "none")
    set(ARB_CXX_FLAGS_TARGET_FULL ${ARB_CXX_FLAGS_TARGET})
    set(ARB_CXX_FLAGS_TARGET_FULL_CPU ${ARB_CXX_FLAGS_TARGET})
else()
    set_arch_target(ARB_CXXOPT_ARCH_CPU ARB_CXXOPT_ARCH ${ARB_ARCH})
    set(ARB_CXX_FLAGS_TARGET_FULL ${ARB_CXX_FLAGS_TARGET} ${ARB_CXXOPT_ARCH})
    set(ARB_CXX_FLAGS_TARGET_FULL_CPU ${ARB_CXX_FLAGS_TARGET} ${ARB_CXXOPT_ARCH_CPU})
endif()

# Add SVE compiler flags if detected/desired
set(ARB_SVE_WIDTH "auto" CACHE STRING "Default SVE vector length in bits. Default: auto (detection during configure time).")
mark_as_advanced(ARB_SVE_WIDTH)
if (ARB_VECTORIZE)
    if (ARB_SVE_WIDTH STREQUAL "auto")
        get_sve_length(ARB_HAS_SVE ARB_SVE_BITS)
        if (ARB_HAS_SVE)
            message(STATUS "SVE detected with vector size = ${ARB_SVE_BITS} bits")
            set(ARB_CXX_SVE_FLAGS " -msve-vector-bits=${ARB_SVE_BITS}")
        else()
            message(STATUS "NO SVE detected")
            set(ARB_CXX_SVE_FLAGS "")
        endif()
    else()
        set(ARB_SVE_BITS ${ARB_SVE_WIDTH})
        set(ARB_CXX_SVE_FLAGS " -msve-vector-bits=${ARB_SVE_BITS}")
    endif()
    list(APPEND ARB_CXX_FLAGS_TARGET_FULL
        "$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CXX>>:${ARB_CXX_SVE_FLAGS}>")
endif()

# Compile with `-fvisibility=hidden` to ensure that the symbols of the generated
# arbor static libraries are hidden from the dynamic symbol tables of any shared
# libraries that link against them.
list(APPEND ARB_CXX_FLAGS_TARGET_FULL
            "$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CXX>>:-fvisibility=hidden>"
            "$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=-fvisibility=hidden>")
separate_arguments(ARB_CXX_FLAGS_TARGET_FULL)

target_compile_options(arbor-private-deps INTERFACE ${ARB_CXX_FLAGS_TARGET_FULL})
target_compile_options(arborenv-private-deps INTERFACE ${ARB_CXX_FLAGS_TARGET_FULL})
target_compile_options(arborio-private-deps INTERFACE ${ARB_CXX_FLAGS_TARGET_FULL})

# Profiling and test features
#-----------------------------

if(ARB_WITH_PROFILING)
    target_compile_definitions(arbor-config-defs INTERFACE ARB_HAVE_PROFILING)
endif()
if(ARB_WITH_ASSERTIONS)
    target_compile_definitions(arbor-config-defs INTERFACE ARB_HAVE_ASSERTIONS)
endif()

# Python bindings
#----------------------------------------------------------

# The minimum version of Python supported by Arbor.
set(arb_py_version 3.9.0)

if(DEFINED PYTHON_EXECUTABLE)
    set(Python3_EXECUTABLE ${PYTHON_EXECUTABLE})
endif()

if(ARB_WITH_PYTHON)
    cmake_dependent_option(ARB_USE_BUNDLED_PYBIND11 "Use bundled pybind11" ON "ARB_WITH_PYTHON;ARB_USE_BUNDLED_LIBS" OFF)

    if(DEFINED ENV{CIBUILDWHEEL} AND (UNIX AND NOT APPLE))
        find_package(Python3 ${arb_py_version} COMPONENTS Interpreter Development.Module REQUIRED)
    else()
        find_package(Python3 ${arb_py_version} COMPONENTS Interpreter Development REQUIRED)
    endif()

else()
    # If not building the Python module, the interpreter is still required
    # to build some targets, e.g. when building the documentation.
    find_package(Python3 ${arb_py_version} COMPONENTS Interpreter)
endif()

if(${Python3_FOUND})
    set(PYTHON_EXECUTABLE "${Python3_EXECUTABLE}")
    message(STATUS "PYTHON_EXECUTABLE: ${PYTHON_EXECUTABLE}")
endif()

# Threading model
#-----------------

find_package(Threads REQUIRED)
target_link_libraries(arbor-private-deps INTERFACE Threads::Threads)

list(APPEND arbor_export_dependencies "Threads")

# MPI support
#-------------------

if(ARB_WITH_MPI)
    find_package(MPI REQUIRED CXX)
    target_compile_definitions(arbor-config-defs INTERFACE ARB_HAVE_MPI)

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
    if(ARB_WITH_NVCC OR ARB_WITH_CUDA_CLANG)
        target_include_directories(arborenv-private-deps INTERFACE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
        add_compile_options(
                "$<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe=--diag_suppress=integer_sign_change>"
                "$<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe=--diag_suppress=unsigned_compare_with_zero>")
    endif()

    if(ARB_WITH_NVCC)
        target_compile_definitions(arbor-private-deps INTERFACE ARB_CUDA)
        target_compile_definitions(arborenv-private-deps INTERFACE ARB_CUDA)
    elseif(ARB_WITH_CUDA_CLANG)
        # Transform cuda archtitecture list into clang cuda flags
        list(TRANSFORM CMAKE_CUDA_ARCHITECTURES PREPEND "--cuda-gpu-arch=sm_" OUTPUT_VARIABLE TMP)
        string(REPLACE ";" " " CUDA_ARCH_STR "${TMP}")

        set(clang_options_ -DARB_CUDA -xcuda ${CUDA_ARCH_STR} --cuda-path=${CUDA_TOOLKIT_ROOT_DIR})
        target_compile_options(arbor-private-deps INTERFACE $<$<COMPILE_LANGUAGE:CXX>:${clang_options_}>)
        target_compile_options(arborenv-private-deps INTERFACE $<$<COMPILE_LANGUAGE:CXX>:${clang_options_}>)
    elseif(ARB_WITH_HIP_CLANG)
        # Transform hip archtitecture list into clang hip flags
        list(TRANSFORM ARB_HIP_ARCHITECTURES PREPEND "--offload-arch=" OUTPUT_VARIABLE TMP)
        string(REPLACE ";" " " HIP_ARCH_STR "${TMP}")

        set(clang_options_ -DARB_HIP -xhip ${HIP_ARCH_STR})
        target_compile_options(arbor-private-deps INTERFACE $<$<COMPILE_LANGUAGE:CXX>:${clang_options_}>)
        target_compile_options(arborenv-private-deps INTERFACE $<$<COMPILE_LANGUAGE:CXX>:${clang_options_}>)
    endif()
endif()

# Use boost::stacktrace if requested for pretty printing stack traces
#--------------------------------------------------------------------

if (ARB_BACKTRACE)
    find_package(Boost REQUIRED
                 COMPONENTS stacktrace_basic
                            stacktrace_addr2line)
    target_link_libraries(arbor-private-deps INTERFACE Boost::stacktrace_basic Boost::stacktrace_addr2line ${CMAKE_DL_LIBS})
    target_compile_definitions(arbor-private-deps INTERFACE WITH_BACKTRACE)
endif()

# Build modcc flags
#------------------------------------------------

if(ARB_MODCC)
    find_program(modcc NAMES ${ARB_MODCC} NO_CMAKE_PATH NO_CMAKE_ENVIRONMENT_PATH NO_CMAKE_SYSTEM_PATH REQUIRED)
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

# Random number creation
# -----------------------------------------------

if(ARB_USE_GPU_RNG AND (ARB_WITH_NVCC OR ARB_WITH_CUDA_CLANG))
    set(ARB_USE_GPU_RNG_IMPL TRUE)
else()
    set(ARB_USE_GPU_RNG_IMPL FALSE)
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
add_subdirectory(arbor/include)

# arbor-sup:
add_subdirectory(sup)

# modcc, libmodcc:
add_subdirectory(modcc)

# arbor, arbor-private-headers:
add_subdirectory(arbor)

# arborenv, arborenv-public-headers:
add_subdirectory(arborenv)

# arborio, arborio-public-headers:
add_subdirectory(arborio)

# unit, unit-mpi, unit-local, unit-modcc
add_subdirectory(test)

# self contained examples:
add_subdirectory(example)

# html:
add_subdirectory(doc)

# python interface:
if(ARB_WITH_PYTHON)
    add_subdirectory(python)
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
#
# To avoid CMake users of the installed arbor library conditionally requiring
# that they add CUDA to their project language, explicitly munge the import
# language and library dependencies on the installed target if ARB_WITH_GPU
# is set, via the variables arbor_override_import_lang and arbor_add_import_libs.
# arbor_build_config records our build type in a way compatible with the
# generated export cmake files.

set(arbor_build_config NOCONFIG)
if(CMAKE_BUILD_TYPE)
    string(TOUPPER "${CMAKE_BUILD_TYPE}" arbor_build_config)
endif()

set(arbor_override_import_lang)
set(arbor_add_import_libs)
set(arborenv_add_import_libs)
set(arborio_add_import_libs)

if(ARB_WITH_GPU)
    set(arbor_override_import_lang CXX)
    set(arbor_add_import_libs ${CUDA_LIBRARIES})
    set(arborenv_add_import_libs ${CUDA_LIBRARIES})
endif()

# (We remove old generated one so that the generation happens every time we run cmake.)
file(REMOVE "${CMAKE_CURRENT_BINARY_DIR}/arbor-config.cmake")
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/arbor-config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/arbor-config.cmake"
    @ONLY)

install(
    FILES
        "${CMAKE_CURRENT_BINARY_DIR}/arbor-config.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/arbor-config-version.cmake"
    DESTINATION "${cmake_config_dir}")

add_subdirectory(lmorpho)
