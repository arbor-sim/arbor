set(arbor_cxx_sources
    backends/multicore/mechanism.cpp
    backends/multicore/shared_state.cpp
    backends/multicore/stimulus.cpp
    builtin_mechanisms.cpp
    cell_group_factory.cpp
    common_types_io.cpp
    cell.cpp
    event_binner.cpp
    fvm_layout.cpp
    fvm_lowered_cell_impl.cpp
    hardware/affinity.cpp
    hardware/gpu.cpp
    hardware/memory.cpp
    hardware/node_info.cpp
    hardware/power.cpp
    lif_cell_group.cpp
    mc_cell_group.cpp
    mechcat.cpp
    merge_events.cpp
    simulation.cpp
    morphology.cpp
    partition_load_balance.cpp
    profiling/memory_meter.cpp
    profiling/meter_manager.cpp
    profiling/power_meter.cpp
    profiling/profiler.cpp
    schedule.cpp
    spike_source_cell_group.cpp
    swcio.cpp
    threading/threading.cpp
    util/debug.cpp
    util/hostname.cpp
    util/path.cpp
    util/unwind.cpp
)

if(ARB_WITH_CUDA)
    list(APPEND arbor_cxx_sources
        backends/gpu/mechanism.cpp
        backends/gpu/shared_state.cpp
        backends/gpu/stimulus.cpp
        backends/gpu/stimulus.cu
        backends/gpu/threshold_watcher.cu
    )
endif()

if(ARB_WITH_MPI)
    list(APPEND arbor_cxx_sources
        communication/mpi.cpp)
elseif(ARB_WITH_DRYRUN)
    list(APPEND arbor_cxx_sources
        communication/dryrun_global_policy.cpp)
endif()

if(ARB_WITH_CTHREAD)
    list(APPEND arbor_cxx_sources
        threading/cthread.cpp)
endif()

set(arbor_cuda_sources
    memory/fill.cu
    backends/gpu/matrix_assemble.cu
    backends/gpu/matrix_interleave.cu
    backends/gpu/matrix_solve.cu
    backends/gpu/multi_event_stream.cu
    backends/gpu/shared_state.cu
    backends/gpu/stimulus.cu
    backends/gpu/threshold_watcher.cu
)

add_library(arbor ${arbor_cxx_sources})
target_compile_options(arbor PRIVATE ${CXXOPT_ARCH})
list(APPEND ARB_LIBRARIES arbor)

if(ARB_WITH_CUDA)
    cuda_add_library(arborcu ${arbor_cuda_sources})
    list(APPEND ARB_LIBRARIES arborcu)
endif()

if (ARB_AUTO_RUN_MODCC_ON_CHANGES)
    add_dependencies(arbor build_all_mods)
    if (ARB_WITH_CUDA)
        add_dependencies(arborcu build_all_mods)
    endif()
endif()

# Make changes to the ARB_LIBRARIES visible to the parent scope.
set(ARB_LIBRARIES "${ARB_LIBRARIES}" PARENT_SCOPE)
