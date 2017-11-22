set(BASE_SOURCES
    backends/multicore/fvm.cpp
    cell_group_factory.cpp
    common_types_io.cpp
    cell.cpp
    event_binner.cpp
    hardware/affinity.cpp
    hardware/gpu.cpp
    hardware/memory.cpp
    hardware/node_info.cpp
    hardware/power.cpp
    model.cpp
    morphology.cpp
    partition_load_balance.cpp
    profiling/memory_meter.cpp
    profiling/meter_manager.cpp
    profiling/power_meter.cpp
    profiling/profiler.cpp
    schedule.cpp
    swcio.cpp
    threading/threading.cpp
    util/debug.cpp
    util/hostname.cpp
    util/path.cpp
    util/prefixbuf.cpp
    util/unwind.cpp
)
set(CUDA_SOURCES
    backends/gpu/fvm.cpp
    backends/gpu/fill.cu
    backends/gpu/multi_event_stream.cu
    backends/gpu/kernels/assemble_matrix.cu
    backends/gpu/kernels/interleave.cu
    backends/gpu/kernels/ions.cu
    backends/gpu/kernels/solve_matrix.cu
    backends/gpu/kernels/stim_current.cu
    backends/gpu/kernels/take_samples.cu
    backends/gpu/kernels/test_thresholds.cu
    backends/gpu/kernels/time_ops.cu
)

if(ARB_WITH_MPI)
    set(BASE_SOURCES ${BASE_SOURCES} communication/mpi.cpp)
elseif(ARB_WITH_DRYRUN)
    set(BASE_SOURCES ${BASE_SOURCES} communication/dryrun_global_policy.cpp)
endif()

if(ARB_WITH_CTHREAD)
    set(BASE_SOURCES ${BASE_SOURCES} threading/cthread.cpp)
endif()

add_library(arbor ${BASE_SOURCES})
list(APPEND ARB_LIBRARIES arbor)

if(ARB_WITH_CUDA)
    cuda_add_library(arborcu ${CUDA_SOURCES})
    list(APPEND ARB_LIBRARIES arborcu)
endif()

if (ARB_AUTO_RUN_MODCC_ON_CHANGES)
    add_dependencies(arbor build_all_mods)
    if (ARB_WITH_CUDA)
        add_dependencies(arborcu build_all_gpu_mods)
    endif()
endif()

# Make changes to the ARB_LIBRARIES visible to the parent scope.
set(ARB_LIBRARIES "${ARB_LIBRARIES}" PARENT_SCOPE)
