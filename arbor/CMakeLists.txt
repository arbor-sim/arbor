# Sources:

set(arbor_sources
    assert.cpp
    backends/multicore/mechanism.cpp
    backends/multicore/shared_state.cpp
    backends/multicore/stimulus.cpp
    benchmark_cell_group.cpp
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
    io/locked_ostream.cpp
    io/serialize_hex.cpp
    lif_cell_group.cpp
    mc_cell_group.cpp
    mechcat.cpp
    memory/cuda_wrappers.cpp
    memory/util.cpp
    merge_events.cpp
    simulation.cpp
    morphology.cpp
    partition_load_balance.cpp
    profile/clock.cpp
    profile/memory_meter.cpp
    profile/meter_manager.cpp
    profile/power_meter.cpp
    profile/profiler.cpp
    schedule.cpp
    spike_source_cell_group.cpp
    swcio.cpp
    threadinfo.cpp
    threading/threading.cpp
    thread_private_spike_store.cpp
    util/hostname.cpp
    util/path.cpp
    util/unwind.cpp
    version.cpp
        )

if(ARB_WITH_CUDA)
    list(APPEND arbor_sources
        backends/gpu/mechanism.cpp
        backends/gpu/shared_state.cpp
        backends/gpu/stimulus.cpp
        backends/gpu/stimulus.cu
        backends/gpu/threshold_watcher.cu
        backends/gpu/managed_ptr.cpp
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

# Add special target for private include directory, for use by arbor target
# and arbor unit tests. Private headers are also used for the other binaries
# until the process of splitting our private and public headers is complete.

add_library(arbor-private-headers INTERFACE)
target_include_directories(arbor-private-headers INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

# Mechanisms, generated from .mod files; sets arbor_mechanism_sources
# variable, build_all_mods target. Note: CMake source file properties are
# directory-local.

add_subdirectory(../mechanisms "${CMAKE_BINARY_DIR}/mechanisms")
set_source_files_properties(${arbor_mechanism_sources} PROPERTIES GENERATED TRUE)

# Library target:

add_library(arbor ${arbor_sources} ${arbor_mechanism_sources})
add_dependencies(arbor build_all_mods)
target_link_libraries(arbor PRIVATE arbor-private-deps arbor-private-headers)
target_link_libraries(arbor PUBLIC arbor-public-deps arbor-public-headers)

install(TARGETS arbor ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

