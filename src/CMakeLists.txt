set(HEADERS
    swcio.hpp
)
set(BASE_SOURCES
    common_types_io.cpp
    cell.cpp
    parameter_list.cpp
    profiling/profiler.cpp
    swcio.cpp
    util/debug.cpp
    util/unwind.cpp
    backends/fvm_multicore.cpp
)
set(CUDA_SOURCES
    backends/fvm_gpu.cu
    memory/fill.cu
)

if(${WITH_MPI})
    set(BASE_SOURCES ${BASE_SOURCES} communication/mpi.cpp)
endif()

add_library(nestmc ${BASE_SOURCES} ${HEADERS})

if (NMC_AUTO_RUN_MODCC_ON_CHANGES)
  add_dependencies(nestmc build_all_mods)
endif()

if(WITH_CUDA)
    cuda_add_library(gpu ${CUDA_SOURCES})
    set(NESTMC_LIBRARIES ${NESTMC_LIBRARIES} gpu)
    add_dependencies(gpu build_all_gpu_mods)
endif()

