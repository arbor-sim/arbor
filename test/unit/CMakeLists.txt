# Build mechanisms used solely in unit tests.

set(test_mechanisms
    ca_linear
    celsius_test
    diam_test
    fixed_ica_current
    gj0
    gj1
    linear_ca_conc
    non_linear
    param_as_state
    point_ica_current
    post_events_syn
    read_cai_init
    read_eX
    test0_kin_diff
    test0_kin_conserve
    test0_kin_compartment
    test0_kin_steadystate
    test1_kin_diff
    test1_kin_conserve
    test1_kin_compartment
    test1_kin_steadystate
    test2_kin_diff
    test3_kin_diff
    test4_kin_compartment
    test5_nonlinear_diff
    test6_nonlinear_diff
    test_ca
    test_ca_read_valence
    test_cl_valence
    test_linear_state
    test_linear_init
    test_linear_init_shuffle
    test_kin1
    test_kinlva
    write_cai_breakpoint
    write_eX
    write_multiple_eX
    write_Xi_Xo
    mean_reverting_stochastic_process
    mean_reverting_stochastic_process2
    mean_reverting_stochastic_density_process
    mean_reverting_stochastic_density_process2
    stochastic_volatility
)

include(${PROJECT_SOURCE_DIR}/mechanisms/BuildModules.cmake)

make_catalogue_lib(
    NAME    testing
    MOD     ${test_mechanisms}
    VERBOSE ${ARB_CAT_VERBOSE})

# Unit test sources

set(unit_sources
    ../common_cells.cpp
    test_abi.cpp
    test_asc.cpp
    test_any_cast.cpp
    test_any_ptr.cpp
    test_any_visitor.cpp
    test_backend.cpp
    test_cable_cell.cpp
    test_counter.cpp
    test_cv_geom.cpp
    test_cv_layout.cpp
    test_cv_policy.cpp
    test_domain_decomposition.cpp
    test_dry_run_context.cpp
    test_event_delivery.cpp
    test_event_generators.cpp
    test_event_stream.cpp
    test_expected.cpp
    test_filter.cpp
    test_forest.cpp
    test_fvm_layout.cpp
    test_fvm_lowered.cpp
    test_hash.cpp
    test_gathered_vector.cpp
    test_diffusion.cpp
    test_iexpr.cpp
    test_index.cpp
    test_kinetic_linear.cpp
    test_label_resolution.cpp
    test_lif_cell_group.cpp
    test_local_context.cpp
    test_maputil.cpp
    test_mask_stream.cpp
    test_math.cpp
    test_matrix.cpp
    test_mcable_map.cpp
    test_cable_cell_group.cpp
    test_mech_temp_diam.cpp
    test_mechcat.cpp
    test_mechinfo.cpp
    test_merge_events.cpp
    test_merge_view.cpp
    test_morphology.cpp
    test_morph_components.cpp
    test_morph_embedding.cpp
    test_morph_expr.cpp
    test_morph_place.cpp
    test_morph_primitives.cpp
    test_morph_stitch.cpp
    test_network.cpp
    test_ordered_forest.cpp
    test_padded.cpp
    test_partition.cpp
    test_partition_by_constraint.cpp
    test_piecewise.cpp
    test_pp_util.cpp
    test_probe.cpp
    test_rand.cpp
    test_range.cpp
    test_recipe.cpp
    test_ratelem.cpp
    test_serdes.cpp
    test_schedule.cpp
    test_scope_exit.cpp
    test_sde.cpp
    test_segment_tree.cpp
    test_simd.cpp
    test_simulation.cpp
    test_span.cpp
    test_spatial_tree.cpp
    test_spike_source.cpp
    test_spikes.cpp
    test_spike_store.cpp
    test_stats.cpp
    test_strprintf.cpp
    test_swcio.cpp
    test_synapses.cpp
    test_s_expr.cpp
    test_thread.cpp
    test_threading_exceptions.cpp
    test_timestep_range.cpp
    test_tree.cpp
    test_transform.cpp
    test_uninitialized.cpp
    test_unique.cpp
    test_unique_any.cpp
    test_vector.cpp
    test_version.cpp
    test_v_clamp.cpp
    test_debug.cpp

    # unit test driver
    test.cpp

    # common routines
    mech_private_field_access.cpp
    stats.cpp
    unit_test_catalogue.cpp

    # neuroml
    test_nml_morphology.cpp
)

if(ARB_WITH_GPU)
    list(APPEND unit_sources
        test_event_stream_gpu.cpp
        test_intrin.cu
        test_gpu_stack.cu
        test_reduce_by_key.cu
        test_cable_cell_group_gpu.cpp
        test_spikes_gpu.cpp
        test_vector_gpu.cpp
    )
endif()

if(ARB_WITH_CUDA_CLANG OR ARB_WITH_HIP_CLANG)
    set_source_files_properties(${unit_sources} PROPERTIES LANGUAGE CXX)
endif()

add_executable(unit EXCLUDE_FROM_ALL ${unit_sources})
target_link_libraries(unit PRIVATE catalogue-testing)
add_dependencies(tests unit)

make_catalogue_standalone(
    NAME dummy
    SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/dummy"
    MOD dummy
    CXX
    CXX_FLAGS_TARGET ${ARB_CXX_FLAGS_TARGET_FULL}
    VERBOSE ${ARB_CAT_VERBOSE})

target_link_libraries(dummy-catalogue PRIVATE arbor-private-deps)
add_dependencies(unit dummy-catalogue)

target_link_libraries(unit PRIVATE arbor-private-deps ext-gtest)
target_compile_definitions(unit PRIVATE "-DDATADIR=\"${CMAKE_CURRENT_SOURCE_DIR}/../swc\"")
target_compile_definitions(unit PRIVATE "-DLIBDIR=\"${PROJECT_BINARY_DIR}/lib\"")
target_include_directories(unit PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
target_include_directories(unit PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated/testing")
target_link_libraries(unit PRIVATE gtest gtest_main ext-random123 arbor arborenv arborio arborio-private-headers arbor-private-headers arbor-sup)
target_include_directories(unit PRIVATE $<BUILD_INTERFACE:${unordered_dense_SOURCE_DIR}/include>)
