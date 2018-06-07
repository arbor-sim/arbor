set(unit-modcc_sources
    # unit tests
    test_lexer.cpp
    test_kinetic_rewriter.cpp
    test_module.cpp
    test_msparse.cpp
    test_parser.cpp
    test_prefixbuf.cpp
    test_printers.cpp
    test_removelocals.cpp
    test_simd_backend.cpp
    test_symdiff.cpp
    test_symge.cpp
    test_visitors.cpp

    # unit test driver
    driver.cpp

    # utility
    common.cpp
    expr_expand.cpp
)

add_executable(unit-modcc ${unit-modcc_sources})
target_link_libraries(unit-modcc PRIVATE libmodcc gtest)
target_compile_definitions(unit-modcc PRIVATE "DATADIR=\"${CMAKE_CURRENT_SOURCE_DIR}\"")

set_target_properties(unit-modcc
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
    EXCLUDE_FROM_ALL ${ARB_WITH_EXTERNAL_MODCC}
)
