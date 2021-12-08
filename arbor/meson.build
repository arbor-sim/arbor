arb_src = [
    'arbexcept.cpp',
    'assert.cpp',
    'backends/multicore/shared_state.cpp',
    'communication/communicator.cpp',
    'communication/dry_run_context.cpp',
    'benchmark_cell_group.cpp',
    'cable_cell.cpp',
    'cable_cell_param.cpp',
    'cell_group_factory.cpp',
    'common_types_io.cpp',
    'cv_policy.cpp',
    'execution_context.cpp',
    'gpu_context.cpp',
    'event_binner.cpp',
    'fvm_layout.cpp',
    'fvm_lowered_cell_impl.cpp',
    'hardware/memory.cpp',
    'hardware/power.cpp',
    'io/locked_ostream.cpp',
    'io/serialize_hex.cpp',
    'label_resolution.cpp',
    'lif_cell_group.cpp',
    'mc_cell_group.cpp',
    'mechcat.cpp',
    'mechinfo.cpp',
    'memory/gpu_wrappers.cpp',
    'memory/util.cpp',
    'morph/embed_pwlin.cpp',
    'morph/label_dict.cpp',
    'morph/locset.cpp',
    'morph/morphexcept.cpp',
    'morph/morphology.cpp',
    'morph/mprovider.cpp',
    'morph/place_pwlin.cpp',
    'morph/primitives.cpp',
    'morph/region.cpp',
    'morph/segment_tree.cpp',
    'morph/stitch.cpp',
    'merge_events.cpp',
    'simulation.cpp',
    'partition_load_balance.cpp',
    'profile/clock.cpp',
    'profile/memory_meter.cpp',
    'profile/meter_manager.cpp',
    'profile/power_meter.cpp',
    'profile/profiler.cpp',
    'schedule.cpp',
    'spike_event_io.cpp',
    'spike_source_cell_group.cpp',
    's_expr.cpp',
    'symmetric_recipe.cpp',
    'threading/threading.cpp',
    'thread_private_spike_store.cpp',
    'tree.cpp',
    'util/dylib.cpp',
    'util/hostname.cpp',
    'util/unwind.cpp',
    'version.cpp',
    ]

# todo: mpi files
# todo: cuda files

subdir('include/arbor')

dep_arb_internal = [dep_threads, dep_random123, dep_dl]

lib_arb = library('arbor',
              sources: [arb_src, catalogue_sources],
              include_directories: [inc_arb_public, inc_arb_private],
              dependencies: dep_arb_internal,
              install: true)

install_subdir(
  'include/arbor',
  install_dir: 'include',
  exclude_files: [
    'arb_types.inc',
    'assert_macro.hpp.disabled',
    'assert_macro.hpp.enabled',
    'meson.build',
    'version.hpp.in'])


dep_arb = declare_dependency(
              include_directories: [inc_arb_public],
              link_with: lib_arb,
              dependencies: dep_arb_internal)
