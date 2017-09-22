include(ExternalProject)

# List of micro benchmarks to build.

set(bench_sources
    accumulate_functor_values.cpp
    event_setup.cpp
)

set(bench_sources_cuda
    cuda_compare_and_reduce.cu
    cuda_reduce_by_key.cu
)

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
    set(git_failed)
    if(GIT_FOUND)
        message(STATUS "Updating the google-benchmark submodule ${gbench_src_dir}")
        execute_process(
            COMMAND "${GIT_EXECUTABLE}" submodule update --init "${gbench_src_dir}"
            WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
            ERROR_VARIABLE git_error
            RESULT_VARIABLE git_result)
        if(NOT git_result EQUAL 0)
            set(git_failed "${git_error}")
        endif()
    else()
        set(git_failed "git not found")
    endif()

    if(git_failed)
        message(WARNING "Unable to update the google-benchmark submodule: ${git_failed}")
    endif()
endif()

ExternalProject_Add(gbench
    # Add dummy DOWNLOAD_COMMAND to stop ExternalProject_Add terminating CMake if the
    # git submodule had not been udpated.
    DOWNLOAD_COMMAND "${CMAKE_COMMAND}" -E echo "Warning: ${gbench_src_dir} empty or missing."
    SOURCE_DIR "${gbench_src_dir}"
    CMAKE_ARGS "${gbench_cmake_args}"
    INSTALL_DIR "${gbench_install_dir}"
)
set_target_properties(gbench PROPERTIES EXCLUDE_FROM_ALL TRUE)

# Build benches.

foreach(bench_src ${bench_sources})
    string(REGEX REPLACE "\\.[^.]*$" "" bench_exe "${bench_src}")
    add_executable("${bench_exe}" EXCLUDE_FROM_ALL "${bench_src}")
    add_dependencies("${bench_exe}" gbench)
    target_include_directories("${bench_exe}" PRIVATE "${gbench_install_dir}/include")
    target_link_libraries("${bench_exe}" LINK_PUBLIC "${gbench_install_dir}/lib/libbenchmark.a")
    target_link_libraries("${bench_exe}" LINK_PUBLIC ${NESTMC_LIBRARIES})

    list(APPEND bench_exe_list ${bench_exe})
endforeach()


if(NMC_WITH_CUDA)
    cuda_include_directories("${gbench_install_dir}/include")
    foreach(bench_src ${bench_sources_cuda})
        string(REGEX REPLACE "\\.[^.]*$" "" bench_exe "${bench_src}")
        cuda_add_executable("${bench_exe}" EXCLUDE_FROM_ALL "${bench_src}")
        add_dependencies("${bench_exe}" gbench)
        target_link_libraries("${bench_exe}" LINK_PUBLIC "${gbench_install_dir}/lib/libbenchmark.a")
        target_link_libraries("${bench_exe}" LINK_PUBLIC ${NESTMC_LIBRARIES})

        list(APPEND bench_exe_list ${bench_exe})
    endforeach()
endif()

add_custom_target(ubenches DEPENDS ${bench_exe_list})

