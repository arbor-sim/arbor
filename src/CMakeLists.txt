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
    backends/catalogue_multicore.cpp
)

if(${WITH_MPI})
    set(BASE_SOURCES ${BASE_SOURCES} communication/mpi.cpp)
endif()

add_library(nestmc ${BASE_SOURCES} ${HEADERS})
add_dependencies(nestmc build_all_mods)

if(WITH_CUDA)
    add_dependencies(nestmc build_all_gpu_mods)
endif()

add_subdirectory(memory)

