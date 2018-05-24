include(${PROJECT_SOURCE_DIR}/mechanisms/BuildModules.cmake)

# Build prototype mechanisms for testing in test_mechanisms.
set(proto_mechanisms pas hh expsyn exp2syn test_kin1 test_kinlva test_ca)
set(mech_proto_dir "${CMAKE_CURRENT_BINARY_DIR}/mech_proto")
file(MAKE_DIRECTORY "${mech_proto_dir}")

build_modules(
    ${proto_mechanisms}
    SOURCE_DIR "${PROJECT_SOURCE_DIR}/mechanisms/mod"
    DEST_DIR "${mech_proto_dir}"
    MECH_SUFFIX _proto
    MODCC_FLAGS -t cpu
    GENERATES _cpu.hpp
    TARGET build_test_mods
)

# Unit test sources

set(test_cuda_sources
    test_intrin.cu
    test_gpu_stack.cu
    test_matrix.cu
    test_matrix_cpuvsgpu.cpp
    test_reduce_by_key.cu
    test_vector.cu

    test_mc_cell_group_gpu.cpp
    test_multi_event_stream_gpu.cpp
    test_multi_event_stream_gpu.cu
    test_spikes_gpu.cpp

    # unit test driver
    test.cpp
)

set(test_sources
    # unit tests
    test_algorithms.cpp
    test_any.cpp
    test_backend.cpp
    test_double_buffer.cpp
    test_cell.cpp
    test_compartments.cpp
    test_counter.cpp
    test_cycle.cpp
    test_domain_decomposition.cpp
    test_dss_cell_group.cpp
    test_either.cpp
    test_event_binner.cpp
    test_event_generators.cpp
    test_event_queue.cpp
    test_filter.cpp
    test_fvm_layout.cpp
    test_fvm_lowered.cpp
    test_mc_cell_group.cpp
    test_lexcmp.cpp
    test_lif_cell_group.cpp
    test_maputil.cpp
    test_mask_stream.cpp
    test_math.cpp
    test_matrix.cpp
    test_mechanisms.cpp
    test_mechcat.cpp
    test_merge_events.cpp
    test_multi_event_stream.cpp
    test_nop.cpp
    test_optional.cpp
    test_mechinfo.cpp
    test_padded.cpp
    test_partition.cpp
    test_partition_by_constraint.cpp
    test_path.cpp
    test_point.cpp
    test_probe.cpp
    test_range.cpp
    test_segment.cpp
    test_schedule.cpp
    test_rss_cell.cpp
    test_simd.cpp
    test_span.cpp
    test_spikes.cpp
    test_spike_store.cpp
    test_stats.cpp
    test_strprintf.cpp
    test_swcio.cpp
    test_synapses.cpp
    test_tree.cpp
    test_transform.cpp
    test_uninitialized.cpp
    test_unique_any.cpp
    test_vector.cpp

    # unit test driver
    test.cpp

    # common routines
    stats.cpp
)

set(targets test.exe)

add_executable(test.exe ${test_sources})
target_compile_options(test.exe PRIVATE ${CXXOPT_ARCH})
target_compile_definitions(test.exe PUBLIC "-DDATADIR=\"${PROJECT_SOURCE_DIR}/data\"")

if (ARB_AUTO_RUN_MODCC_ON_CHANGES)
  add_dependencies(test.exe build_test_mods)
endif()

target_include_directories(test.exe PRIVATE "${mech_proto_dir}/..")

if(ARB_WITH_CUDA)
    list(APPEND targets test_cuda.exe)
    cuda_add_executable(test_cuda.exe ${test_cuda_sources})
endif()

foreach(target ${targets})
    target_link_libraries(${target} LINK_PUBLIC gtest)
    target_link_libraries(${target} LINK_PUBLIC ${ARB_LIBRARIES})
    target_link_libraries(${target} LINK_PUBLIC ${EXTERNAL_LIBRARIES})

    if(ARB_WITH_MPI)
        target_link_libraries(${target} LINK_PUBLIC ${MPI_C_LIBRARIES})
        set_property(TARGET ${target} APPEND_STRING PROPERTY LINK_FLAGS "${MPI_C_LINK_FLAGS}")
    endif()

    set_target_properties(${target}
       PROPERTIES
       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
    )
endforeach()
