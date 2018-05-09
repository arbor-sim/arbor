set(MODCC_SOURCES
    astmanip.cpp
    errorvisitor.cpp
    expression.cpp
    functionexpander.cpp
    functioninliner.cpp
    lexer.cpp
    kineticrewriter.cpp
    module.cpp
    parser.cpp
    solvers.cpp
    symdiff.cpp
    symge.cpp
    token.cpp

    io/prefixbuf.cpp

    printer/cexpr_emit.cpp
    printer/cprinter.cpp
    printer/cudaprinter.cpp
    printer/infoprinter.cpp
    printer/printerutil.cpp
)

include_directories("${PROJECT_SOURCE_DIR}/modcc")

add_library(compiler ${MODCC_SOURCES})

add_executable(modcc modcc.cpp)

target_link_libraries(modcc LINK_PUBLIC compiler)

set_target_properties(modcc
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/modcc"
)
