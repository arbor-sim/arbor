include(ExternalProject)

# List of micro benchmarks to build.

set(bench_sources
    accumulate_functor_values.cpp)

# Set up google benchmark as an external project.

set(gbench_src_dir "${CMAKE_CURRENT_SOURCE_DIR}/google-benchmark")
set(gbench_install_dir "${PROJECT_BINARY_DIR}/gbench")

set(gbench_cmake_args
    "-DCMAKE_BUILD_TYPE=release"
    "-DCMAKE_INSTALL_PREFIX=${gbench_install_dir}"
    "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
    "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}")


# Attempt to update git submodule if required.
find_package(Git)
if(NOT EXISTS "${gbench_src_dir}/.git")
    if(GIT_FOUND)
        exec_program("${GIT_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}"
            ARGS submodule update --init google-benchmark)
    else()
        message(WARNING "google-benchmark submodule apparently not updated, but cannot find git")
    endif()
endif()

ExternalProject_Add(gbench
    SOURCE_DIR "${gbench_src_dir}"
    CMAKE_ARGS "${gbench_cmake_args}"
    INSTALL_DIR "${gbench_install_dir}"
)

# Build benches.

foreach(bench_src ${bench_sources})
    string(REGEX REPLACE "\\.[^.]*$" "" bench_exe "${bench_src}")
    add_executable("${bench_exe}" "${bench_src}")
    add_dependencies("${bench_exe}" gbench)
    target_include_directories("${bench_exe}" PRIVATE "${gbench_install_dir}/include")
    target_link_libraries("${bench_exe}" "${gbench_install_dir}/lib/libbenchmark.a")

    list(APPEND bench_exe_list ${bench_exe})
endforeach()

add_custom_target(ubenches DEPENDS ${bench_exe_list})

