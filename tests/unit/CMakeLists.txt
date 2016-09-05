set(HEADERS
    ${PROJECT_SOURCE_DIR}/src/cell.hpp
    ${PROJECT_SOURCE_DIR}/src/cell_tree.hpp
    ${PROJECT_SOURCE_DIR}/src/math.hpp
    ${PROJECT_SOURCE_DIR}/src/point.hpp
    ${PROJECT_SOURCE_DIR}/src/segment.hpp
    ${PROJECT_SOURCE_DIR}/src/swcio.hpp
    ${PROJECT_SOURCE_DIR}/src/tree.hpp
)

set(TEST_SOURCES
    # unit tests
    test_algorithms.cpp
    test_double_buffer.cpp
    test_cell.cpp
    test_compartments.cpp
    test_counter.cpp
    test_either.cpp
    test_event_queue.cpp
    test_fvm.cpp
    test_cell_group.cpp
    test_lexcmp.cpp
    test_mask_stream.cpp
    test_matrix.cpp
    test_mechanisms.cpp
    test_nop.cpp
    test_optional.cpp
    test_parameters.cpp
    test_partition.cpp
    test_point.cpp
    test_probe.cpp
    test_segment.cpp
    test_range.cpp
    test_span.cpp
    test_spikes.cpp
    test_spike_store.cpp
    test_stimulus.cpp
    test_swcio.cpp
    test_synapses.cpp
    test_tree.cpp
    test_transform.cpp
    test_uninitialized.cpp

    # unit test driver
    test.cpp
)

add_definitions("-DDATADIR=\"${CMAKE_SOURCE_DIR}/data\"")
add_executable(test.exe ${TEST_SOURCES} ${HEADERS})

set(TARGETS test.exe)

foreach(target ${TARGETS})
    target_link_libraries(${target} LINK_PUBLIC cellalgo gtest)

    if(WITH_TBB)
    target_link_libraries(${target} LINK_PUBLIC ${TBB_LIBRARIES})
    endif()

    if(WITH_MPI)
    target_link_libraries(${target} LINK_PUBLIC ${MPI_C_LIBRARIES})
    set_property(TARGET ${target} APPEND_STRING PROPERTY LINK_FLAGS "${MPI_C_LINK_FLAGS}")
    endif()

    set_target_properties(${target}
       PROPERTIES
       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
    )
endforeach()

