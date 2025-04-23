cmake_minimum_required(VERSION 3.19)
include(CMakeDependentOption)
include(CheckIPOSupported)

# Usually we don't want to hear those
set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS ON CACHE INTERNAL "" FORCE)

# Make CUDA support throw errors if architectures remain unclear
cmake_policy(SET CMP0104 NEW)
# Ensure CMake is aware of the policies for modern RPATH behavior
cmake_policy(SET CMP0072 NEW)

# Set release as the default build type (CMake default is debug.)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE release CACHE STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "debug" "release")
endif()

set(CPM_USE_LOCAL_PACKAGES ON)
include(cmake/CPM.cmake)

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
    message (VERBOSE "LTO support found, enabling")
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
  else()
    message(STATUS "No LTO: ${ERR_LTO}")
  endif()
endif()

# Use pybind11-stubgen to make type stubs.
cmake_dependent_option(ARB_BUILD_PYTHON_STUBS "Use pybind11-stubgen to build type stubs." ON "ARB_WITH_PYTHON" OFF)

# Turn on this option to force the compilers to produce color output when output is
# redirected from the terminal (e.g. when using ninja or a pager).

option(ARBDEV_COLOR "Always produce ANSI-colored output (GNU/Clang only)." OFF)
mark_as_advanced(FORCE ARBDEV_COLOR)

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

option(BUILD_SHARED_LIBS "Build using shared libraries" OFF)

# Perform explicit vectorization?
option(ARB_VECTORIZE "use explicit SIMD code in generated mechanisms" OFF)

# Support for Thread pinning

option(ARB_USE_HWLOC "request support for thread pinning via HWLOC" OFF)
mark_as_advanced(ARB_USE_HWLOC)

# Build tests and benchmarks, docs

option(BUILD_TESTING "build tests and benchmarks" ON)
option(BUILD_DOCUMENTATION "build documentation" ON)

# Use externally built modcc?

set(ARB_MODCC "" CACHE STRING "path to external modcc NMODL compiler")
mark_as_advanced(FORCE ARB_MODCC)

# Use libunwind to generate stack traces on errors?

option(ARB_BACKTRACE "Enable stacktraces on assertion and exceptions (requires Boost)." OFF)
mark_as_advanced(FORCE ARB_BACKTRACE)

# Specify GPU build type

set(ARB_GPU "none" CACHE STRING "GPU backend and compiler configuration")
set_property(CACHE PROPERTY STRINGS "none" "cuda" "cuda-clang" "hip")
if(NOT ARB_GPU STREQUAL "none")
    set(ARB_USE_GPU_DEP ON)
endif()
cmake_dependent_option(ARB_USE_GPU_RNG
    "Use GPU generated random numbers (only cuda, not bitwise equal to CPU version)" OFF
    "ARB_USE_GPU_DEP" OFF)

# Optional additional CXX Flags used for all code that will run on the target
# CPU architecture. Recorded in installed target, for downstream dependencies
# to use.
# Useful, for example, when a user wants to compile with target-specific
# optimization flag.spr
set(ARB_CXX_FLAGS_TARGET "" CACHE STRING "Optional additional flags for compilation")
mark_as_advanced(FORCE ARB_CXX_FLAGS_TARGET)

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

option(ARB_WITH_PROFILING "enable Tracy profiling" OFF)
cmake_dependent_option(ARB_WITH_STACK_PROFILING "enable stack collection in profiling" OFF "ARB_WITH_PROFILING" OFF)
cmake_dependent_option(ARB_WITH_MEMORY_PROFILING "enable memory in profiling" OFF "ARB_WITH_PROFILING" OFF)
mark_as_advanced(FORCE ARB_WITH_STACK_PROFILING ARB_WITH_MEMORY_PROFILING)

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

# Compiler options common to library, examples, tests, etc.

include("CompilerOptions")
check_supported_cxx()
add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:${CXXOPT_WALL}>")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

mark_as_advanced(FORCE CMAKE_OSX_ARCHITECTURES CMAKE_OSX_DEPLOYMENT_TARGET CMAKE_OSX_SYSROOT)

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

# Add all dependencies.

# Keep track of packages we need to add to the generated CMake config
# file for arbor.
set(arbor_export_dependencies)


# First make ourselves less chatty
set(_saved_CMAKE_MESSAGE_LOG_LEVEL ${CMAKE_MESSAGE_LOG_LEVEL})
set(CMAKE_MESSAGE_LOG_LEVEL STATUS)

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
install(TARGETS ext-hwloc EXPORT arbor-targets)

CPMFindPackage(NAME json
              GITHUB_REPOSITORY nlohmann/json
              VERSION 3.12.0
              OPTIONS "CMAKE_SUPPRESS_DEVELOPER_WARNINGS ON")
install(TARGETS nlohmann_json EXPORT arbor-targets)

add_library(ext-random123 INTERFACE)
CPMFindPackage(NAME random123
              DOWNLOAD_ONLY YES
              GITHUB_REPOSITORY DEShawResearch/random123
              VERSION 1.14.0)
if(random123_ADDED)
    target_include_directories(ext-random123 INTERFACE $<BUILD_INTERFACE:${random123_SOURCE_DIR}/include>)
else()
    target_include_directories(ext-random123 INTERFACE ${RANDOM123_INCLUDE_DIR})
endif()
install(TARGETS ext-random123 EXPORT arbor-targets)

if (ARB_WITH_PYTHON)
    CPMFindPackage(NAME pybind11
                  GITHUB_REPOSITORY pybind/pybind11
                  VERSION 2.13.6
                  OPTIONS "PYBIND11_CPP_STANDARD -std=c++20")
    # required for find_python_module
    include(FindPythonModule)
endif()

CPMFindPackage(NAME pugixml
              GITHUB_REPOSITORY zeux/pugixml
              VERSION 1.13
              DOWNLOAD_ONLY YES)
add_library(ext-pugixml INTERFACE)
if(pugixml_ADDED)
    target_compile_definitions(ext-pugixml INTERFACE PUGIXML_HEADER_ONLY)
    target_include_directories(ext-pugixml INTERFACE $<BUILD_INTERFACE:${pugixml_SOURCE_DIR}/src>)
else()
    list(APPEND arbor_export_dependencies pugixml)
    target_link_libraries(ext-pugixml INTERFACE pugixml::pugixml)
endif()
install(TARGETS ext-pugixml EXPORT arbor-targets)

CPMFindPackage(NAME fmt
              GITHUB_REPOSITORY fmtlib/fmt
              VERSION 10.0.0
              GIT_TAG 10.0.0)

add_library(ext-gtest INTERFACE)
add_library(ext-bench INTERFACE)
if (BUILD_TESTING)
    CPMFindPackage(NAME benchmark
                  GITHUB_REPOSITORY google/benchmark
                  VERSION 1.8.3
                  OPTIONS "BENCHMARK_ENABLE_TESTING OFF" "CMAKE_BUILD_TYPE release" "BUILD_SHARED_LIBS OFF")
    CPMFindPackage(NAME googletest
                  GITHUB_REPOSITORY google/googletest
                  GIT_TAG release-1.12.1
                  VERSION 1.12.1
                  OPTIONS "INSTALL_GTEST OFF" "BUILD_GMOCK OFF")
    if(benchmark_ADDED)
        target_link_libraries(ext-bench INTERFACE benchmark)
    else()
        target_link_libraries(ext-bench INTERFACE benchmark::benchmark)
    endif()
    if(googletest_ADDED)
        target_link_libraries(ext-gtest INTERFACE )
    else()
        target_link_libraries(ext-gtest INTERFACE gtest gtest_main)
    endif()
endif()

CPMFindPackage(NAME units
               GITHUB_REPOSITORY llnl/units
               VERSION 0.13.1
               OPTIONS "UNITS_PROJECT_NAME units"
                       "SKBUILD OFF"
                       "UNITS_INSTALL ON"
                       "BUILD_SHARED_LIBS OFF"
                       "UNITS_BUILD_STATIC_LIBRARY ON"
                       "UNITS_BUILD_SHARED_LIBRARY OFF"
                       "UNITS_BUILD_OBJECT_LIBRARY OFF"
                       "UNITS_ENABLE_TESTS OFF"
                       "UNITS_BUILD_CONVERTER_APP OFF"
                       "UNITS_BUILD_WEBSERVER OFF")
target_link_libraries(arbor-public-deps INTERFACE units::units)
if(units_ADDED)
    install(TARGETS units compile_flags_target EXPORT arbor-targets)
endif()
list(APPEND arbor_export_dependencies units)

CPMFindPackage(NAME tinyopt
              GITHUB_REPOSITORY halfflat/tinyopt
              GIT_TAG 7e6d707d49c6cb4be27ebd253856be65293288df
              DOWNLOAD_ONLY YES)

add_library(ext-tinyopt INTERFACE)
if(tinyopt_ADDED)
  target_include_directories(ext-tinyopt INTERFACE $<BUILD_INTERFACE:${tinyopt_SOURCE_DIR}/include>)
else()
    message(FATAL_ERROR "Could not obtain tinyopt.")
endif()

# hide all internal vars
mark_as_advanced(FORCE benchmark_DIR BENCHMARK_BUILD_32_BITS BENCHMARK_DOWNLOAD_DEPENDENCIES BENCHMARK_ENABLE_ASSEMBLY_TESTS BENCHMARK_ENABLE_DOXYGEN BENCHMARK_ENABLE_EXCEPTIONS BENCHMARK_ENABLE_GTEST_TESTS BENCHMARK_ENABLE_INSTALL BENCHMARK_ENABLE_LIBPFM BENCHMARK_ENABLE_LTO BENCHMARK_ENABLE_WERROR BENCHMARK_FORCE_WERROR BENCHMARK_INSTALL_DOCS BENCHMARK_USE_BUNDLED_GTEST BENCHMARK_USE_LIBCXX)
mark_as_advanced(FORCE googletest_DIR BUILD_GMOCK)
mark_as_advanced(FORCE json_DIR JSON_CI JSON_BuildTests JSON_Diagnostics JSON_DisableEnumSerialization JSON_GlobalUDLs JSON_ImplicitConversions JSON_Install JSON_LegacyDiscardedValueComparison JSON_MultipleHeaders JSON_SystemInclude)
mark_as_advanced(FORCE RANDOM123_INCLUDE_DIR)
mark_as_advanced(FORCE pybind11_DIR PYBIND11_PYTHONLIBS_OVERWRITE PYBIND11_PYTHON_VERSION PYBIND11_FINDPYTHON PYBIND11_INSTALL PYBIND11_INTERNALS_VERSION PYBIND11_NOPYTHON PYBIND11_SIMPLE_GIL_MANAGEMENT PYBIND11_TEST)
mark_as_advanced(FORCE pugixml_DIR)
mark_as_advanced(FORCE fmt_DIR)
mark_as_advanced(FORCE units_DIR UNITS_BUILD_OBJECT_LIBRARY UNITS_BUILD_SHARED_LIBRARY UNITS_HEADER_ONLY UNITS_NAMESPACE UNITS_BUILD_FUZZ_TARGETS UNITS_ENABLE_TESTS)
mark_as_advanced(FORCE tinyopt_DIR)
mark_as_advanced(FORCE CXXFEATURECHECK_DEBUG)
mark_as_advanced(FORCE CPM_DONT_CREATE_PACKAGE_LOCK CPM_DONT_UPDATE_MODULE_PATH CPM_DOWNLOAD_ALL CPM_INCLUDE_ALL_IN_PACKAGE_LOCK CPM_LOCAL_PACKAGES_ONLY CPM_SOURCE_CACHE CPM_USE_NAMED_CACHE_DIRECTORIES)
mark_as_advanced(FORCE FETCHCONTENT_BASE_DIR FETCHCONTENT_FULLY_DISCONNECTED FETCHCONTENT_QUIET FETCHCONTENT_SOURCE_DIR_BENCHMARK FETCHCONTENT_SOURCE_DIR_GOOGLETEST FETCHCONTENT_SOURCE_DIR_JSON FETCHCONTENT_SOURCE_DIR_PYBIND11 FETCHCONTENT_SOURCE_DIR_RANDOM123 FETCHCONTENT_SOURCE_DIR_TINYOPT FETCHCONTENT_SOURCE_DIR_UNITS FETCHCONTENT_UPDATES_DISCONNECTED FETCHCONTENT_UPDATES_DISCONNECTED_BENCHMARK FETCHCONTENT_UPDATES_DISCONNECTED_GOOGLETEST FETCHCONTENT_UPDATES_DISCONNECTED_JSON FETCHCONTENT_UPDATES_DISCONNECTED_PYBIND11 FETCHCONTENT_UPDATES_DISCONNECTED_RANDOM123 FETCHCONTENT_UPDATES_DISCONNECTED_TINYOPT FETCHCONTENT_UPDATES_DISCONNECTED_UNITS)


# Restore chattyness
set(CMAKE_MESSAGE_LOG_LEVEL ${_saved_CMAKE_MESSAGE_LOG_LEVEL})

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
set(arb_py_version 3.10.0)

if(DEFINED PYTHON_EXECUTABLE)
    set(Python3_EXECUTABLE ${PYTHON_EXECUTABLE})
endif()

if(ARB_WITH_PYTHON)
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
    message(VERBOSE "PYTHON_EXECUTABLE: ${PYTHON_EXECUTABLE}")
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

    # CMake 3.9 does not allow us to add definitions to an import target. so
    # wrap MPI::MPI_CXX in an interface library 'mpi-wrap' instead.
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
if (BUILD_TESTING)
    add_subdirectory(test)
endif()

# self contained examples:
add_subdirectory(example)

# html:
if (BUILD_DOCUMENTATION)
    add_subdirectory(doc)
endif()

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
