set(TEST_SOURCES
    # unit tests
    test_lexer.cpp
    test_module.cpp
    test_optimization.cpp
    test_parser.cpp
    #test_printers.cpp
    test_visitors.cpp

    # unit test driver
    driver.cpp
)

add_executable(test_compiler ${TEST_SOURCES})

target_link_libraries(test_compiler LINK_PUBLIC compiler gtest)

set_target_properties( test_compiler
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
)

