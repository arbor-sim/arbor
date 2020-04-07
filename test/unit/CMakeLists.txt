# Build mechanisms used solely in unit tests.

set(test_mechanisms
    celsius_test
    diam_test
    test_linear_state
    test_linear_init
    test_linear_init_shuffle
    test0_kin_diff
    test0_kin_conserve
    test0_kin_compartment
    test0_kin_steadystate
    test1_kin_diff
    test1_kin_conserve
    test1_kin_compartment
    test2_kin_diff
    test3_kin_diff
    test4_kin_compartment
    test1_kin_steadystate
    fixed_ica_current
    point_ica_current
    linear_ca_conc
    test_cl_valence
    test_ca_read_valence
    read_eX
    write_multiple_eX
    write_eX
    read_cai_init
    write_cai_breakpoint
)

include(${PROJECT_SOURCE_DIR}/mechanisms/BuildModules.cmake)

set(external_modcc)
if(ARB_WITH_EXTERNAL_MODCC)
    set(external_modcc MODCC ${modcc})
endif()
set(test_mech_dir ${CMAKE_CURRENT_BINARY_DIR}/mechanisms)

build_modules(
    ${test_mechanisms}
    SOURCE_DIR mod
    DEST_DIR "${test_mech_dir}"
    ${external_modcc}
    MODCC_FLAGS -t cpu -t gpu ${ARB_MODCC_FLAGS} -N testing
    GENERATES .hpp _cpu.cpp _gpu.cpp _gpu.cu
    TARGET build_test_mods
)

set(test_mech_sources)
foreach(mech ${test_mechanisms})
    list(APPEND test_mech_sources ${test_mech_dir}/${mech}_cpu.cpp)
    if(ARB_WITH_CUDA)
        list(APPEND test_mech_sources ${test_mech_dir}/${mech}_gpu.cpp)
        list(APPEND test_mech_sources ${test_mech_dir}/${mech}_gpu.cu)
    endif()
endforeach()


# TODO: test_mechanism and mechanism prototype comparisons must
# be re-jigged.

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
    test_counter.cpp
    test_cv_geom.cpp
    test_cv_layout.cpp
    test_cv_policy.cpp
    test_cycle.cpp
    test_domain_decomposition.cpp
    test_dry_run_context.cpp
    test_double_buffer.cpp
    test_either.cpp
    test_event_binner.cpp
    test_event_delivery.cpp
    test_event_generators.cpp
    test_event_queue.cpp
    test_filter.cpp
    test_fvm_layout.cpp
    test_fvm_lowered.cpp
    test_glob_basic.cpp
    test_kinetic_linear.cpp
    test_lexcmp.cpp
    test_lif_cell_group.cpp
    test_local_context.cpp
    test_maputil.cpp
    test_mask_stream.cpp
    test_math.cpp
    test_matrix.cpp
    test_mcable_map.cpp
    test_mc_cell_group.cpp
    test_mechanisms.cpp
    test_mech_temp_diam.cpp
    test_mechcat.cpp
    test_mechinfo.cpp
    test_merge_events.cpp
    test_merge_view.cpp
    test_morphology.cpp
    test_morph_embedding.cpp
    test_morph_expr.cpp
    test_morph_primitives.cpp
    test_multi_event_stream.cpp
    test_optional.cpp
    test_padded.cpp
    test_partition.cpp
    test_partition_by_constraint.cpp
    test_path.cpp
    test_piecewise.cpp
    test_point.cpp
    test_probe.cpp
    test_range.cpp
    test_ratelem.cpp
    test_schedule.cpp
    test_spike_source.cpp
    test_scope_exit.cpp
    test_simd.cpp
    test_span.cpp
    test_spikes.cpp
    test_spike_store.cpp
    test_stats.cpp
    test_strprintf.cpp
    test_swcio.cpp
    test_synapses.cpp
    test_thread.cpp
    test_threading_exceptions.cpp
    test_tree.cpp
    test_transform.cpp
    test_uninitialized.cpp
    test_unique.cpp
    test_unique_any.cpp
    test_variant.cpp
    test_vector.cpp
    test_version.cpp

    # unit test driver
    test.cpp

    # common routines
    mech_private_field_access.cpp
    stats.cpp
    unit_test_catalogue.cpp
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

add_executable(unit EXCLUDE_FROM_ALL ${unit_sources} ${test_mech_sources})
add_dependencies(unit build_test_mods)
add_dependencies(tests unit)

target_compile_options(unit PRIVATE ${ARB_CXXOPT_ARCH})
target_compile_definitions(unit PRIVATE "-DDATADIR=\"${CMAKE_CURRENT_SOURCE_DIR}/swc\"")
target_include_directories(unit PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
target_link_libraries(unit PRIVATE gtest arbor arborenv arbor-private-headers arbor-sup)
