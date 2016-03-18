set(HEADERS
    ../src/swcio.hpp
    ../src/tree.hpp
    ../src/cell_tree.hpp
    ../src/point.hpp
    ../src/math.hpp
    ../src/segment.hpp
)

set(TEST_SOURCES
    # google test framework
    gtest-all.cpp

    # unit tests
    test_swcio.cpp
    test_tree.cpp
    test_point.cpp
    test_segment.cpp

    # unit test driver
    main.cpp
)

add_executable(test.exe ${TEST_SOURCES} ${HEADERS})

target_link_libraries(test.exe LINK_PUBLIC cellalgo)

set_target_properties(test.exe
   PROPERTIES
   RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
)
