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
# Arguments:
# NAME:           project folder
# INTERFACE_NAME: generated cmake target to link against
# LIB:            library name
# CMAKE_ARGS:     additional cmake arguments
function(add_external_cmake_project)
    # handle named arguments: fills variables EP_NAME, EP_INTERFACE_NAME, EP_LIB, and EP_CMAKE_ARGS
    set(options OPTIONAL)
    set(oneValueArgs NAME INTERFACE_NAME LIB)
    set(multiValueArgs CMAKE_ARGS)
    cmake_parse_arguments(EP "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(EP_BUILD       "${EP_NAME}-build")
    set(EP_SOURCE_DIR  "${CMAKE_CURRENT_SOURCE_DIR}/${EP_NAME}")
    set(EP_INSTALL_DIR "${CMAKE_CURRENT_BINARY_DIR}/${EP_NAME}")

    set(EP_INTERFACE_INCLUDE_DIR "${EP_INSTALL_DIR}/include")
    set(EP_INTERFACE_LIB_NAME    "${EP_INSTALL_DIR}/lib/${EP_LIB}")

    check_git_submodule(${EP_NAME}_sub ${EP_NAME})
    if(${EP_NAME}_sub_avail)
        # populate cmake arguments
        set(EP_ALL_CMAKE_ARGS
            "-DCMAKE_INSTALL_PREFIX=${EP_INSTALL_DIR}"
            "-DCMAKE_INSTALL_LIBDIR=${EP_INSTALL_DIR}/lib"
            "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
            "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}")
        list(APPEND EP_ALL_CMAKE_ARGS ${EP_CMAKE_ARGS})

        # add external project
        ExternalProject_Add(${EP_BUILD}
            # Add dummy DOWNLOAD_COMMAND to stop ExternalProject_Add terminating CMake if the
            # git submodule had not been udpated.
            DOWNLOAD_COMMAND "${CMAKE_COMMAND}" -E echo "Warning: ${EP_SOURCE_DIR} empty or missing."
            BUILD_BYPRODUCTS "${EP_INTERFACE_LIB_NAME}"
            SOURCE_DIR "${EP_SOURCE_DIR}"
            CMAKE_ARGS "${EP_ALL_CMAKE_ARGS}"
            INSTALL_DIR "${EP_INSTALL_DIR}"
        )
        set_target_properties(${EP_BUILD} PROPERTIES EXCLUDE_FROM_ALL TRUE)

        # make top level interface library which links to external project
        add_library(${EP_INTERFACE_NAME} INTERFACE)
        add_dependencies(${EP_INTERFACE_NAME} ${EP_BUILD})
        target_include_directories(${EP_INTERFACE_NAME} INTERFACE ${EP_INTERFACE_INCLUDE_DIR})
        target_link_libraries(${EP_INTERFACE_NAME} INTERFACE ${EP_INTERFACE_LIB_NAME})
    else()
        add_error_target(${EP_BUILD}
            "Building ${EP_NAME} library"
            "The git submodule for ${EP_NAME} is not available")
    endif()
endfunction()

# Google benchmark for microbenchmarks:

# Set up google benchmark as an external project.
add_external_cmake_project(
    NAME google-benchmark
    INTERFACE_NAME ext-benchmark
    LIB libbenchmark.a
    CMAKE_ARGS
        "-DCMAKE_BUILD_TYPE=release"
        "-DBUILD_SHARED_LIBS=OFF")

# Google Test framework:

# Set up google test as an external project.
add_external_cmake_project(
    NAME googletest
    INTERFACE_NAME ext-gtest
    LIB libgtest.a
    CMAKE_ARGS
        "-DCMAKE_BUILD_TYPE=release"
        "-DBUILD_SHARED_LIBS=OFF"
        "-DBUILD_GMOCK=OFF")

# on some systems we need link explicitly against threads
if (TARGET ext-gtest)
    find_package (Threads)
    target_link_libraries(ext-gtest INTERFACE Threads::Threads)
endif()
