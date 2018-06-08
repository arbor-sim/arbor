# Sources:

set(arbor_sources
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
    list(APPEND arbor_sources
        backends/gpu/mechanism.cpp
        backends/gpu/shared_state.cpp
        backends/gpu/stimulus.cpp
        backends/gpu/stimulus.cu
        backends/gpu/threshold_watcher.cu
        backends/gpu/matrix_assemble.cu
        backends/gpu/matrix_interleave.cu
        backends/gpu/matrix_solve.cu
        backends/gpu/multi_event_stream.cu
        backends/gpu/shared_state.cu
        backends/gpu/stimulus.cu
        backends/gpu/threshold_watcher.cu
        memory/fill.cu
    )
endif()

if(ARB_WITH_MPI)
    list(APPEND arbor_sources
        communication/mpi.cpp
        communication/mpi_error.cpp
        communication/mpi_context.cpp)
endif()

if(ARB_WITH_CTHREAD)
    list(APPEND arbor_sources
        threading/cthread.cpp)
endif()

# Mechanisms (generated from .mod files):
# (Sets arbor_mechanism_sources variable, build_all_mods target).
add_subdirectory(../mechanisms "${CMAKE_BINARY_DIR}/mechanisms")

# Source file properties are directory-local?!
set_source_files_properties(${arbor_mechanism_sources} PROPERTIES GENERATED TRUE)

add_library(arbor ${arbor_sources} ${arbor_mechanism_sources})
target_link_libraries(arbor PRIVATE arbor_deps)
target_include_directories(arbor PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/external)
target_include_directories(arbor INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}) # TODO: split out into public headers
add_dependencies(arbor build_all_mods)

