arb_src = [
  'arbexcept.cpp',
  'assert.cpp',
  'backends/multicore/shared_state.cpp',
  'communication/communicator.cpp',
  'communication/dry_run_context.cpp',
  'benchmark_cell_group.cpp',
  'cable_cell.cpp',
  'iexpr.cpp',
  'cable_cell_param.cpp',
  'cell_group_factory.cpp',
  'common_types_io.cpp',
  'cv_policy.cpp',
  'domain_decomposition.cpp',
  'domdecexcept.cpp',
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
  'morph/cv_data.cpp',
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

mpi_src = [
  'communication/mpi.cpp',
  'communication/mpi_error.cpp',
  'communication/mpi_context.cpp',
]

gpu_src = [
  'backends/gpu/shared_state.cpp',
  'backends/gpu/stimulus.cu',
  'backends/gpu/threshold_watcher.cu',
  'backends/gpu/matrix_assemble.cu',
  'backends/gpu/matrix_fine.cu',
  'backends/gpu/matrix_solve.cu',
  'backends/gpu/multi_event_stream.cpp',
  'backends/gpu/multi_event_stream.cu',
  'backends/gpu/shared_state.cu',
  'backends/gpu/forest.cpp',
  'backends/gpu/stimulus.cu',
  'backends/gpu/threshold_watcher.cu',
  'memory/fill.cu',
]

if arb_with_mpi
  arb_src += mpi_src
endif
if arb_with_gpu
  arb_src += gpu_src
endif

# generate version.hpp header
subdir('include/arbor')

# set dependencies for libarbor
dep_arb_internal = [dep_threads, dep_random123, dep_dl]
if arb_with_mpi
  dep_arb_internal += dep_mpi
endif
if arb_with_gpu
  dep_arb_internal += dep_cuda
endif

lib_arb = library('arbor',
                  sources: [arb_src, default_cat, allen_cat, bbp_cat],
                  include_directories: [inc_arb_public, inc_arb_private],
                  cpp_args: arbor_private_args,
                  cuda_args: arbor_cuda_private_args,
                  dependencies: [dep_arb_internal],
                  install: true)

install_subdir('include/arbor',
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
