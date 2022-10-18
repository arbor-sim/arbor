# Niels Lohmann's json library (single-header version).

if(ARB_USE_BUNDLED_JSON)
  add_library(ext-json INTERFACE)
  target_include_directories(ext-json INTERFACE json/single_include)
  set(json_library_name ext-json PARENT_SCOPE)
endif()

# Random123 (DE Shaw Research) counter-based random number generators (header-only)

if(ARB_USE_BUNDLED_RANDOM123)
    check_git_submodule(random123 random123)
    if(NOT random123_avail)
        message(FATAL_ERROR "Required Random123 submodule is not available.")
    endif()
    target_include_directories(ext-random123 INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>/random123/include)
    message(STATUS "Using Random123 submodule: ${CMAKE_CURRENT_SOURCE_DIR}>/random123")
endif()

# tinyopt command line parsing libary (header-only).

add_library(ext-tinyopt INTERFACE)
target_include_directories(ext-tinyopt INTERFACE tinyopt/include)


# functionality for adding external projects
include(ExternalProject)
macro(add_external_cmake_project)
    # handle named arguments
    set(options OPTIONAL)
    set(oneValueArgs NAME SOURCE_DIR INSTALL_DIR LIB)
    set(multiValueArgs CMAKE_ARGS)
    cmake_parse_arguments(EP "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    # populate cmake arguments
    set(all_cmake_args
        "-DCMAKE_INSTALL_PREFIX=${EP_INSTALL_DIR}"
        "-DCMAKE_INSTALL_LIBDIR=${EP_INSTALL_DIR}/lib"
        "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
        "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}")
    list(APPEND all_cmake_args ${EP_CMAKE_ARGS})

    # add external project
    set(EP_BUILD "${EP_NAME}-build")
    ExternalProject_Add(${EP_BUILD}
        # Add dummy DOWNLOAD_COMMAND to stop ExternalProject_Add terminating CMake if the
        # git submodule had not been udpated.
        DOWNLOAD_COMMAND "${CMAKE_COMMAND}" -E echo "Warning: ${EP_SOURCE_DIR} empty or missing."
        SOURCE_DIR "${EP_SOURCE_DIR}"
        CMAKE_ARGS "${all_cmake_args}"
        INSTALL_DIR "${EP_INSTALL_DIR}"
    )
    set_target_properties(${name} PROPERTIES EXCLUDE_FROM_ALL TRUE)

    # make top level interface library which links to external project
    add_library(${EP_NAME} INTERFACE)
    add_dependencies(${EP_NAME} ${EP_BUILD})
    target_include_directories(${EP_NAME} INTERFACE "${EP_INSTALL_DIR}/include")
    target_link_libraries(${EP_NAME} INTERFACE "${EP_INSTALL_DIR}/lib/${EP_LIB}")
endmacro()


# Google benchmark for microbenchmarks:

check_git_submodule(gbench google-benchmark)
if(gbench_avail)
    # Set up google benchmark as an external project.
    add_external_cmake_project(
        NAME ext-benchmark
        SOURCE_DIR ${gbench_src_dir}
        INSTALL_DIR ${gbench_install_dir}
        SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/google-benchmark"
        INSTALL_DIR "${CMAKE_CURRENT_BINARY_DIR}/google-benchmark"
        LIB libbenchmark.a
        CMAKE_ARGS
            "-DCMAKE_BUILD_TYPE=release")
else()
    add_error_target(ext-benchmark-build
        "Building Google benchmark library"
        "The git submodule for google benchmark is not available")
endif()


# Google Test framework:

check_git_submodule(gtest googletest)
if(gtest_avail)
    # Set up google test as an external project.
    add_external_cmake_project(
        NAME ext-gtest
        SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/googletest"
        INSTALL_DIR "${CMAKE_CURRENT_BINARY_DIR}/googletest"
        LIB libgtest.a
        CMAKE_ARGS
            "-DCMAKE_BUILD_TYPE=release"
            "-DBUILD_SHARED_LIBS=OFF"
            "-DBUILD_GMOCK=OFF")

    # on some systems we need link explicitly against threads
    find_package (Threads)
    target_link_libraries(ext-gtest INTERFACE Threads::Threads)
else()
    add_error_target(ext-gtest-build
        "Building Google Test library"
        "The git submodule for google test is not available")
endif()
