set(BASE_SOURCES
    common_types_io.cpp
    cell.cpp
    morphology.cpp
    parameter_list.cpp
    profiling/profiler.cpp
    profiling/time_meter.cpp
    profiling/meter_manager.cpp
    swcio.cpp
    threading/affinity.cpp
    util/debug.cpp
    util/path.cpp
    util/unwind.cpp
    backends/fvm_multicore.cpp
)
set(CUDA_SOURCES
    backends/fvm_gpu.cu
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

