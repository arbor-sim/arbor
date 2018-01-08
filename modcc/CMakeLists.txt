set(MODCC_SOURCES
    astmanip.cpp
    cexpr_emit.cpp
    cprinter.cpp
    cprinter2.cpp
    cudaprinter.cpp
    errorvisitor.cpp
    expression.cpp
    functionexpander.cpp
    functioninliner.cpp
    infoprinter.cpp
    lexer.cpp
    kineticrewriter.cpp
    module.cpp
    parser.cpp
    printerutil.cpp
    solvers.cpp
    symdiff.cpp
    symge.cpp
    textbuffer.cpp
    token.cpp

    io/prefixbuf.cpp
)

include_directories("${PROJECT_SOURCE_DIR}/modcc")

add_library(compiler ${MODCC_SOURCES})

add_executable(modcc modcc.cpp)

target_link_libraries(modcc LINK_PUBLIC compiler)

set_target_properties(modcc
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/modcc"
)
