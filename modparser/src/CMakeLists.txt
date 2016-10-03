set(BASE_SOURCES
    token.cpp
    lexer.cpp
    expression.cpp
    parser.cpp
    textbuffer.cpp
    cprinter.cpp
    functionexpander.cpp
    functioninliner.cpp
    cudaprinter.cpp
    expressionclassifier.cpp
    constantfolder.cpp
    errorvisitor.cpp
    module.cpp
)

add_library(compiler ${BASE_SOURCES})

add_executable(modcc modcc.cpp)

target_link_libraries(modcc         LINK_PUBLIC compiler)

set_target_properties( modcc
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

install(TARGETS modcc RUNTIME DESTINATION bin)

