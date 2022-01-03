subdir('mechanisms')
subdir('dummy')

unit_sources = files(
  'test_abi.cpp',
  'test_asc.cpp',
  'test_any_cast.cpp',
  'test_any_ptr.cpp',
  'test_any_visitor.cpp',
  'test_backend.cpp',
  'test_cable_cell.cpp',
  'test_counter.cpp',
  'test_cv_geom.cpp',
  'test_cv_layout.cpp',
  'test_cv_policy.cpp',
  'test_cycle.cpp',
  'test_domain_decomposition.cpp',
  'test_dry_run_context.cpp',
  'test_event_binner.cpp',
  'test_event_delivery.cpp',
  'test_event_generators.cpp',
  'test_event_queue.cpp',
  'test_expected.cpp',
  'test_filter.cpp',
  'test_forest.cpp',
  'test_fvm_layout.cpp',
  'test_fvm_lowered.cpp',
  'test_index.cpp',
  'test_kinetic_linear.cpp',
  'test_lexcmp.cpp',
  'test_label_resolution.cpp',
  'test_lif_cell_group.cpp',
  'test_local_context.cpp',
  'test_maputil.cpp',
  'test_mask_stream.cpp',
  'test_math.cpp',
  'test_matrix.cpp',
  'test_mcable_map.cpp',
  'test_mc_cell_group.cpp',
  'test_mechanisms.cpp',
  'test_mech_temp_diam.cpp',
  'test_mechcat.cpp',
  'test_mechinfo.cpp',
  'test_merge_events.cpp',
  'test_merge_view.cpp',
  'test_morphology.cpp',
  'test_morph_components.cpp',
  'test_morph_embedding.cpp',
  'test_morph_expr.cpp',
  'test_morph_place.cpp',
  'test_morph_primitives.cpp',
  'test_morph_stitch.cpp',
  'test_multi_event_stream.cpp',
  'test_ordered_forest.cpp',
  'test_padded.cpp',
  'test_partition.cpp',
  'test_partition_by_constraint.cpp',
  'test_path.cpp',
  'test_piecewise.cpp',
  'test_pp_util.cpp',
  'test_probe.cpp',
  'test_range.cpp',
  'test_recipe.cpp',
  'test_ratelem.cpp',
  'test_schedule.cpp',
  'test_scope_exit.cpp',
  'test_segment_tree.cpp',
  'test_simd.cpp',
  'test_simulation.cpp',
  'test_span.cpp',
  'test_spike_source.cpp',
  'test_spikes.cpp',
  'test_spike_store.cpp',
  'test_stats.cpp',
  'test_strprintf.cpp',
  'test_swcio.cpp',
  'test_synapses.cpp',
  'test_s_expr.cpp',
  'test_thread.cpp',
  'test_threading_exceptions.cpp',
  'test_tree.cpp',
  'test_transform.cpp',
  'test_uninitialized.cpp',
  'test_unique.cpp',
  'test_unique_any.cpp',
  'test_vector.cpp',
  'test_version.cpp',

  # unit test driver
  'test.cpp',

  # common routines
  '../common_cells.cpp',
  'mech_private_field_access.cpp',
  'stats.cpp',
  'unit_test_catalogue.cpp'
)

if arb_with_neuroml
    unit_sources += files('test_nml_morphology.cpp')
endif

if arb_with_gpu
    unit_sources += files(
      'test_intrin.cu',
      'test_gpu_stack.cu',
      'test_multi_event_stream_gpu.cu',
      'test_reduce_by_key.cu',
      'test_matrix_cpuvsgpu.cpp',
      'test_matrix_gpu.cpp',
      'test_mc_cell_group_gpu.cpp',
      'test_multi_event_stream_gpu.cpp',
      'test_spikes_gpu.cpp',
      'test_vector_gpu.cpp'
    )
endif

unit_data_dir = meson.current_source_dir() + '/../swc'
unit_dummy_catalogue_dir = meson.current_build_dir() + '/dummy'
unit_build_dir = meson.current_build_dir()

arbor_unit_defines = [
  f'-DDATADIR="@unit_data_dir@"',
  f'-DBUILDDIR="@unit_build_dir@"',
  f'-DCATALOGUEDIR="@unit_dummy_catalogue_dir@"',
  '-DUSE_DYNAMIC_CATALOGUES',
]

unit = executable('unit',
                  sources: [unit_sources, unit_mech_sources],
                  include_directories: [inc_arb_public, inc_arb_private, inc_arborio_private],
                  cpp_args: arbor_private_args + arbor_unit_defines,
                  cuda_args: arbor_cuda_private_args,
                  dependencies: [dep_arb, dep_arborio, dep_arborsup, dep_arborenv, dep_gtest, dep_arb_internal],
                  link_with: [dummy_catalogue],
                  build_by_default: false,
                  install: false)
