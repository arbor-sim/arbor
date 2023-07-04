set(unit-modcc_sources
    # unit tests
    test_fn_rewriters.cpp
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

add_executable(unit-modcc EXCLUDE_FROM_ALL ${unit-modcc_sources})
add_dependencies(tests unit-modcc)

target_link_libraries(unit-modcc PRIVATE libmodcc ext-gtest)
target_compile_definitions(unit-modcc PRIVATE "DATADIR=\"${CMAKE_CURRENT_SOURCE_DIR}\"")
