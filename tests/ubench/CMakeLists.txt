include(ExternalProject)

set(gbench_install_dir "${PROJECT_BINARY_DIR}/gbench")

set(gbench_cmake_args
    -DCMAKE_BUILD_TYPE=release
    -DCMAKE_INSTALL_PREFIX=${gbench_install_dir}
    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER})

ExternalProject_Add(gbench
    SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/google-benchmark"
    CMAKE_ARGS "${gbench_cmake_args}"
#    INSTALL_DIR "${gbench_install_dir}"
)

message(STATUS "gbench_install_dir:  ${gbench_install_dir}")

set(bench_sources
    accumulate_functor_values.cpp)

foreach(bench_src ${bench_sources})
    string(REGEX REPLACE "\\.[^.]*$" "" bench_exe "${bench_src}")
    add_executable("${bench_exe}" "${bench_src}")
    add_dependencies("${bench_exe}" gbench)
    target_include_directories("${bench_exe}" PRIVATE "${gbench_install_dir}/include")
    target_link_libraries("${bench_exe}" "${gbench_install_dir}/lib/libbenchmark.a")

    list(APPEND bench_exe_list ${bench_exe})
endforeach()

add_custom_target(ubenches DEPENDS ${bench_exe_list})

