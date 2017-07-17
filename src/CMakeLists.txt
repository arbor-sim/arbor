set(BASE_SOURCES
    backends/multicore/fvm.cpp
    common_types_io.cpp
    cell.cpp
    cell_group_factory.cpp
    event_binner.cpp
    model.cpp
    morphology.cpp
    parameter_list.cpp
    profiling/memory_meter.cpp
    profiling/meter_manager.cpp
    profiling/power_meter.cpp
    profiling/profiler.cpp
    swcio.cpp
    hardware/affinity.cpp
    hardware/gpu.cpp
    hardware/memory.cpp
    hardware/power.cpp
    threading/threading.cpp
    util/debug.cpp
    util/hostname.cpp
    util/path.cpp
    util/unwind.cpp
)
set(CUDA_SOURCES
    backends/gpu/fvm.cu
    backends/gpu/multi_event_stream.cu
    memory/fill.cu
)

# The cell_group_factory acts like an interface between the
# front end and back end.
if(NMC_WITH_CUDA)
    set(CUDA_SOURCES ${CUDA_SOURCES} cell_group_factory.cu)
else()
    set(BASE_SOURCES ${BASE_SOURCES} cell_group_factory.cpp)
endif()

if(NMC_WITH_MPI)
    set(BASE_SOURCES ${BASE_SOURCES} communication/mpi.cpp)
elseif(NMC_WITH_DRYRUN)
    set(BASE_SOURCES ${BASE_SOURCES} communication/dryrun_global_policy.cpp)
endif()

if(NMC_WITH_CTHREAD)
    set(BASE_SOURCES ${BASE_SOURCES} threading/cthread.cpp)
endif()

add_library(nestmc ${BASE_SOURCES} ${HEADERS})
list(APPEND NESTMC_LIBRARIES nestmc)

if(NMC_WITH_CUDA)
    cuda_add_library(gpu ${CUDA_SOURCES})
    # FIXME
    # The gpu library uses symbols fron nestmc, so we have to
    # add nestmc to the end. This is a temporary hack that will
    # go away when we have properly seperable front and back end
    # compilation.
    list(APPEND NESTMC_LIBRARIES gpu nestmc)
endif()

if (NMC_AUTO_RUN_MODCC_ON_CHANGES)
    add_dependencies(nestmc build_all_mods)
    if (NMC_WITH_CUDA)
        add_dependencies(gpu build_all_gpu_mods)
    endif()
endif()

# Make changes to the NMC_LIBRARIES visible to the parent scope.
set(NESTMC_LIBRARIES "${NESTMC_LIBRARIES}" PARENT_SCOPE)
