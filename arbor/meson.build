arb_src = [
    'arbexcept.cpp',
    'assert.cpp',
    'backends/multicore/mechanism.cpp',
    'backends/multicore/shared_state.cpp',
    'backends/multicore/stimulus.cpp',
    'communication/dry_run_context.cpp',
    'communication/communicator.cpp',
    'benchmark_cell_group.cpp',
    'builtin_mechanisms.cpp',
    'cable_cell_param.cpp',
    'cell_group_factory.cpp',
    'common_types_io.cpp',
    'execution_context.cpp',
    'gpu_context.cpp',
    'event_binner.cpp',
    'fvm_layout.cpp',
    'fvm_lowered_cell_impl.cpp',
    'hardware/memory.cpp',
    'hardware/power.cpp',
    'io/locked_ostream.cpp',
    'io/serialize_hex.cpp',
    'lif_cell_group.cpp',
    'cable_cell.cpp',
    'mc_cell_group.cpp',
    'mechcat.cpp',
    'memory/cuda_wrappers.cpp',
    'memory/util.cpp',
    'merge_events.cpp',
    'simulation.cpp',
    'morph/morphology.cpp',
    'morph/sample_tree.cpp',
    'morph/primitives.cpp',
    'partition_load_balance.cpp',
    'profile/clock.cpp',
    'profile/memory_meter.cpp',
    'profile/meter_manager.cpp',
    'profile/power_meter.cpp',
    'profile/profiler.cpp',
    'schedule.cpp',
    'spike_event_io.cpp',
    'spike_source_cell_group.cpp',
    'swcio.cpp',
    'threading/threading.cpp',
    'thread_private_spike_store.cpp',
    'tree.cpp',
    'util/hostname.cpp',
    'util/unwind.cpp',
    'version.cpp']

# todo: mpi files
# todo: cuda files

# FILES FOR MPI:
#   'communication/mpi.cpp'
#   'communication/mpi_context.cpp'
#   'communication/mpi_error.cpp'


subdir('include/arbor')

arb = library('arbor',
              sources: [arb_src, mech_srcs, mech_catalogue],
              include_directories: [inc_arbpublic, inc_arbprivate],
              dependencies : dep_threads)

