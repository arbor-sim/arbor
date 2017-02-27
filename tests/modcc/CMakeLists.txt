set(MODCC_TEST_SOURCES
    # unit tests
    test_lexer.cpp
    test_kinetic_rewriter.cpp
    test_module.cpp
    test_optimization.cpp
    test_parser.cpp
    #test_printers.cpp
    test_visitors.cpp

    # unit test driver
    driver.cpp

    # utility
    expr_expand.cpp
)

add_definitions("-DDATADIR=\"${PROJECT_SOURCE_DIR}/data\"")
add_executable(test_modcc ${MODCC_TEST_SOURCES})

target_link_libraries(test_modcc LINK_PUBLIC compiler gtest)

set_target_properties(test_modcc
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
)
