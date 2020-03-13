# Niels Lohmann's json library (single-header version).

add_library(ext-json INTERFACE)
target_include_directories(ext-json INTERFACE json/single_include)

# tinyopt command line parsing libary (header-only).

add_library(ext-tinyopt INTERFACE)
target_include_directories(ext-tinyopt INTERFACE tinyopt/include)

# Random123 (DE Shaw Research) counter-based random number generators (header-only)

add_library(ext-random123 INTERFACE)
target_include_directories(ext-random123 INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>/random123/include)
install(TARGETS ext-random123 EXPORT arbor-targets)

# Google benchmark for microbenchmarks:

check_git_submodule(gbench google-benchmark)
if(gbench_avail)
    # Set up google benchmark as an external project.

    include(ExternalProject)
    set(gbench_src_dir "${CMAKE_CURRENT_SOURCE_DIR}/google-benchmark")
    set(gbench_install_dir "${CMAKE_CURRENT_BINARY_DIR}/google-benchmark")
    set(gbench_cmake_args
        "-DCMAKE_BUILD_TYPE=release"
        "-DCMAKE_INSTALL_PREFIX=${gbench_install_dir}"
        "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
        "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}")

    ExternalProject_Add(gbench-build
        # Add dummy DOWNLOAD_COMMAND to stop ExternalProject_Add terminating CMake if the
        # git submodule had not been udpated.
        DOWNLOAD_COMMAND "${CMAKE_COMMAND}" -E echo "Warning: ${gbench_src_dir} empty or missing."
        SOURCE_DIR "${gbench_src_dir}"
        CMAKE_ARGS "${gbench_cmake_args}"
        INSTALL_DIR "${gbench_install_dir}"
    )
    set_target_properties(gbench-build PROPERTIES EXCLUDE_FROM_ALL TRUE)

else()
    add_error_target(gbench-build
        "Building Google benchmark library"
        "The git submodule for google benchmark is not available")
endif()

add_library(ext-benchmark INTERFACE)
add_dependencies(ext-benchmark gbench-build)
target_include_directories(ext-benchmark INTERFACE "${gbench_install_dir}/include")
target_link_libraries(ext-benchmark INTERFACE "${gbench_install_dir}/lib/libbenchmark.a")

# Sphinx RTD theme

check_git_submodule(rtdtheme sphinx_rtd_theme)
add_target_if(rtdtheme_avail
    check-sphinx_rtd_theme
    "Checking for Read the Docs Sphinx theme"
    "The git submodule for read the docs is not available")

add_library(ext-sphinx_rtd_theme INTERFACE)
add_dependencies(ext-sphinx_rtd_theme check-sphinx_rtd_theme)
target_include_directories(ext-sphinx_rtd_theme INTERFACE sphinx_rtd_theme)
