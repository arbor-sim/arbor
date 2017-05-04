include(${PROJECT_SOURCE_DIR}/mechanisms/BuildModules.cmake)

# Build prototype mechanisms for testing in test_mechanisms.
set(proto_mechanisms pas hh expsyn exp2syn test_kin1 test_kinlva)
set(mech_proto_dir "${CMAKE_CURRENT_BINARY_DIR}/mech_proto")
file(MAKE_DIRECTORY "${mech_proto_dir}")

build_modules(
    ${proto_mechanisms}
    SOURCE_DIR "${PROJECT_SOURCE_DIR}/mechanisms/mod"
    DEST_DIR "${mech_proto_dir}"
    MECH_SUFFIX _proto
    MODCC_FLAGS -t cpu
    TARGET build_test_mods
)

# Unit test sources

set(TEST_CUDA_SOURCES
    test_atomics.cu
    test_mc_cell_group.cu
    test_gpu_stack.cu
    test_matrix.cu
    test_spikes.cu
    test_vector.cu

    # unit test driver
    test.cpp
)

set(TEST_SOURCES
    # unit tests
    test_algorithms.cpp
    test_backend.cpp
    test_double_buffer.cpp
    test_cell.cpp
    test_compartments.cpp
    test_counter.cpp
    test_cycle.cpp
    test_domain_decomposition.cpp
    test_either.cpp
    test_event_queue.cpp
    test_event_binner.cpp
    test_filter.cpp
    test_fvm_multi.cpp
    test_fs_cell.cpp
    test_mc_cell_group.cpp
    test_lexcmp.cpp
    test_mask_stream.cpp
    test_math.cpp
    test_matrix.cpp
    test_mechanisms.cpp
    test_nop.cpp
    test_optional.cpp
    test_parameters.cpp
    test_partition.cpp
    test_path.cpp
    test_point.cpp
    test_probe.cpp
    test_segment.cpp
    test_range.cpp
    test_span.cpp
    test_spikes.cpp
    test_spike_store.cpp
    test_stimulus.cpp
    test_strprintf.cpp
    test_swcio.cpp
    test_synapses.cpp
    test_tree.cpp
    test_transform.cpp
    test_uninitialized.cpp
    test_vector.cpp

    # unit test driver
    test.cpp
)


set(TARGETS test.exe)

add_executable(test.exe ${TEST_SOURCES} ${HEADERS})
target_compile_definitions(test.exe PUBLIC "-DDATADIR=\"${PROJECT_SOURCE_DIR}/data\"")

if (NMC_AUTO_RUN_MODCC_ON_CHANGES)
  add_dependencies(test.exe build_test_mods)
endif()

target_include_directories(test.exe PRIVATE "${mech_proto_dir}/..")

if(NMC_WITH_CUDA)
    # Omit -DDATADIR for cuda target because of CMake 3.7/3.8 FindCUDA quoting bug.

    set(TARGETS ${TARGETS} test_cuda.exe)
    cuda_add_executable(test_cuda.exe ${TEST_CUDA_SOURCES} ${HEADERS})
    target_link_libraries(test_cuda.exe LINK_PUBLIC gpu)
endif()

foreach(target ${TARGETS})
    target_link_libraries(${target} LINK_PUBLIC gtest nestmc)
    target_link_libraries(${target} LINK_PUBLIC ${EXTERNAL_LIBRARIES})

    if(NMC_WITH_MPI)
        target_link_libraries(${target} LINK_PUBLIC ${MPI_C_LIBRARIES})
        set_property(TARGET ${target} APPEND_STRING PROPERTY LINK_FLAGS "${MPI_C_LINK_FLAGS}")
    endif()

    set_target_properties(${target}
       PROPERTIES
       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
    )
endforeach()
