set(BASE_SOURCES
    common_types_io.cpp
    cell.cpp
    morphology.cpp
    parameter_list.cpp
    profiling/memory_meter.cpp
    profiling/meter_manager.cpp
    profiling/power_meter.cpp
    profiling/profiler.cpp
    swcio.cpp
    threading/affinity.cpp
    util/debug.cpp
    util/hostname.cpp
    util/memory.cpp
    util/path.cpp
    util/power.cpp
    util/unwind.cpp
    backends/multicore/fvm.cpp
)
set(CUDA_SOURCES
    backends/gpu/fvm.cu
    memory/fill.cu
)


if(NMC_WITH_MPI)
    set(BASE_SOURCES ${BASE_SOURCES} communication/mpi.cpp)

elseif(NMC_WITH_DRYRUN)
    set(BASE_SOURCES ${BASE_SOURCES} communication/dryrun_global_policy.cpp)

endif()

if(NMC_WITH_CTHREAD)
    set(BASE_SOURCES ${BASE_SOURCES} threading/cthread.cpp)
endif()

add_library(nestmc ${BASE_SOURCES} ${HEADERS})

if (NMC_AUTO_RUN_MODCC_ON_CHANGES)
  add_dependencies(nestmc build_all_mods)
endif()

if(NMC_WITH_CUDA)
    cuda_add_library(gpu ${CUDA_SOURCES})
    set(NESTMC_LIBRARIES ${NESTMC_LIBRARIES} gpu)
    add_dependencies(gpu build_all_gpu_mods)
endif()

