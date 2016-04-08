set(HEADERS
    ../src/cell.hpp
    ../src/cell_tree.hpp
    ../src/math.hpp
    ../src/point.hpp
    ../src/segment.hpp
    ../src/swcio.hpp
    ../src/tree.hpp
)

set(TEST_SOURCES
    # google test framework
    gtest-all.cpp

    # unit tests
    test_cell.cpp
    test_compartments.cpp
    test_point.cpp
    test_algorithms.cpp
    test_run.cpp
    test_segment.cpp
    test_stimulus.cpp
    test_swcio.cpp
    test_tree.cpp

    # unit test driver
    main.cpp
)

add_executable(test.exe ${TEST_SOURCES} ${HEADERS})

target_link_libraries(test.exe LINK_PUBLIC cellalgo)

set_target_properties(test.exe
   PROPERTIES
   RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
)
