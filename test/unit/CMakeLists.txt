# TODO: test_mechanism and mechanism prototype comparisons must
# be re-jigged.

# Build prototype mechanisms for testing in test_mechanisms.
#
#include(${PROJECT_SOURCE_DIR}/mechanisms/BuildModules.cmake)
#
# set(proto_mechanisms pas hh expsyn exp2syn test_kin1 test_kinlva test_ca)
# set(mech_proto_dir "${CMAKE_CURRENT_BINARY_DIR}/mech_proto")
# file(MAKE_DIRECTORY "${mech_proto_dir}")
# 
# build_modules(
#     ${proto_mechanisms}
#     SOURCE_DIR "${PROJECT_SOURCE_DIR}/mechanisms/mod"
#     DEST_DIR "${mech_proto_dir}"
#     MECH_SUFFIX _proto
#     MODCC_FLAGS -t cpu
#     GENERATES _cpu.hpp
#     TARGET build_test_mods
# )

# Unit test sources

set(unit_sources
    test_algorithms.cpp
    test_any.cpp
    test_backend.cpp
    test_double_buffer.cpp
    test_compartments.cpp
    test_counter.cpp
    test_cycle.cpp
    test_domain_decomposition.cpp
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
    test_mc_cell.cpp
    test_mechanisms.cpp
    test_mechcat.cpp
    test_merge_events.cpp
    test_multi_event_stream.cpp
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
    test_spike_source.cpp
    test_local_context.cpp
    test_simd.cpp
    test_span.cpp
    test_spikes.cpp
    test_spike_store.cpp
    test_spike_emitter.cpp
    test_stats.cpp
    test_strprintf.cpp
    test_swcio.cpp
    test_synapses.cpp
    test_thread.cpp
    test_tree.cpp
    test_transform.cpp
    test_uninitialized.cpp
    test_unique_any.cpp
    test_vector.cpp
    test_version.cpp

    # unit test driver
    test.cpp

    # common routines
    stats.cpp
)

if(ARB_WITH_CUDA)
    list(APPEND unit_sources

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
    )
endif()

add_executable(unit ${unit_sources})
target_compile_options(unit PRIVATE ${CXXOPT_ARCH})
target_compile_definitions(unit PRIVATE "-DDATADIR=\"${CMAKE_CURRENT_SOURCE_DIR}/swc\"")
target_link_libraries(unit PRIVATE gtest arbor arbor-private-headers arbor-aux)

