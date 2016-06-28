set(HEADERS
    ../src/cell.hpp
    ../src/cell_tree.hpp
    ../src/math.hpp
    ../src/point.hpp
    ../src/segment.hpp
    ../src/swcio.hpp
    ../src/tree.hpp
)

# google test framework
add_library(gtest gtest-all.cpp gtest.h)

set(TEST_SOURCES
    # unit tests
    test_algorithms.cpp
    test_cell.cpp
    test_compartments.cpp
    test_event_queue.cpp
    test_fvm.cpp
    test_cell_group.cpp
    test_matrix.cpp
    test_mechanisms.cpp
    test_optional.cpp
    test_parameters.cpp
    test_point.cpp
    test_segment.cpp
    test_spikes.cpp
    test_stimulus.cpp
    test_swcio.cpp
    test_synapses.cpp
    test_tree.cpp
    test_uninitialized.cpp

    # unit test driver
    test.cpp
)

set(VALIDATION_SOURCES
    # unit tests
    validate_ball_and_stick.cpp
    validate_soma.cpp
    #validate_synapses.cpp

    # unit test driver
    validate.cpp
)

add_definitions("-DDATADIR=\"${CMAKE_SOURCE_DIR}/data\"")
add_executable(test.exe ${TEST_SOURCES} ${HEADERS})
add_executable(validate.exe ${VALIDATION_SOURCES} ${HEADERS})

target_link_libraries(test.exe LINK_PUBLIC cellalgo gtest)
target_link_libraries(validate.exe LINK_PUBLIC cellalgo gtest)

if(WITH_TBB)
    target_link_libraries(test.exe LINK_PUBLIC ${TBB_LIBRARIES})
    target_link_libraries(validate.exe LINK_PUBLIC ${TBB_LIBRARIES})
endif()

set_target_properties(test.exe
   PROPERTIES
   RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
)

set_target_properties(validate.exe
   PROPERTIES
   RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
)

