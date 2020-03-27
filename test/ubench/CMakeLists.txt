include(ExternalProject)

# List of micro benchmarks to build.

set(bench_sources
    accumulate_functor_values.cpp
    default_construct.cpp
    event_setup.cpp
    event_binning.cpp
    fvm_discretize.cpp
    mech_vec.cpp
    task_system.cpp
)

if(ARB_WITH_CUDA)
list(APPEND bench_sources
    cuda_compare_and_reduce.cu
    cuda_reduce_by_key.cu
)
endif()

# Build benches.

foreach(bench_src ${bench_sources})
    string(REGEX REPLACE "\\.[^.]*$" "" bench_exe ${bench_src})
    add_executable(${bench_exe} EXCLUDE_FROM_ALL "${bench_src}")

    target_link_libraries(${bench_exe} arbor arbor-private-headers ext-benchmark)
    target_compile_options(${bench_exe} PRIVATE ${ARB_CXXOPT_ARCH})
    target_compile_definitions(${bench_exe} PRIVATE "-DDATADIR=\"${CMAKE_CURRENT_SOURCE_DIR}/swc\"")
    list(APPEND bench_exe_list ${bench_exe})
endforeach()

add_custom_target(ubenches DEPENDS ${bench_exe_list})
