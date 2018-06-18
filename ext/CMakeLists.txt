# Niels Lohmann's json library (single-header version).

add_library(ext-json INTERFACE)
target_include_directories(ext-json INTERFACE json/single_include)

# TCLAP command line parsing libary (header-only).

add_library(ext-tclap INTERFACE)
target_include_directories(ext-tclap INTERFACE tclap/include)

# Intel TBB:
# Alias system TBB or build locally and export that, according
# to ARB_PRIVATE_TBBLIB setting.

if(ARB_REQUIRE_TBB)
    if(ARB_PRIVATE_TBBLIB)
        check_git_submodule(tbb tbb)
        if(NOT tbb_avail)
            message(FATAL_ERROR "TBB git submodule required.")
        endif()

        # Turn off proxy malloc library and test compilation.
        option(TBB_BUILD_TBBMALLOC_PROXY OFF)
        option(TBB_BUILD_TESTS           OFF)

        add_subdirectory(tbb)
        add_library(ext-tbb INTERFACE)
        target_link_libraries(ext-tbb INTERFACE tbb_static tbbmalloc_static)
        target_include_directories(ext-tbb SYSTEM INTERFACE tbb/include)
    else()
        find_package(TBB REQUIRED)
        add_library(ext-tbb ALIAS TBB::tbb)
    endif()
endif()

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
