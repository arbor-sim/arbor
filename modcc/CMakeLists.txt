# The modcc compiler is split into a staric library and driver;
# unit tests for the driver also use this library.

set(libmodcc_sources

    astmanip.cpp
    blocks.cpp
    errorvisitor.cpp
    expression.cpp
    functionexpander.cpp
    functioninliner.cpp
    lexer.cpp
    kineticrewriter.cpp
    linearrewriter.cpp
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

set(modcc_sources modcc.cpp)

add_library(libmodcc STATIC ${libmodcc_sources})
target_include_directories(libmodcc PUBLIC .)
set_target_properties(libmodcc PROPERTIES OUTPUT_NAME modcc)

add_executable(modcc ${modcc_sources})
target_link_libraries(modcc libmodcc ext-tinyopt)

set_target_properties(modcc libmodcc PROPERTIES EXCLUDE_FROM_ALL ${ARB_WITH_EXTERNAL_MODCC})

if (NOT ARB_WITH_EXTERNAL_MODCC)
    install(TARGETS modcc RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

