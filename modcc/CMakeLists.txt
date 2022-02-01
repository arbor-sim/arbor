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
    printer/marks.cpp
    printer/gpuprinter.cpp
    printer/infoprinter.cpp
    printer/printerutil.cpp
)

set(modcc_sources modcc.cpp)

add_library(libmodcc STATIC ${libmodcc_sources})
#target_include_directories(libmodcc PUBLIC .)
#target_include_directories(libmodcc PRIVATE ../ext/fmt/include)
target_link_libraries(libmodcc PUBLIC arbor-public-headers)
#target_include_directories(libmodcc PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/include)

target_include_directories(libmodcc
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../ext/fmt/include>)


set_target_properties(libmodcc PROPERTIES OUTPUT_NAME modcc)

export_visibility(libmodcc)


add_executable(modcc ${modcc_sources})
target_link_libraries(modcc PRIVATE libmodcc ext-tinyopt)
#message("modcc current bin dir = " ${CMAKE_CURRENT_BINARY_DIR}/include)
#target_include_directories(modcc PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/include)
#target_link_libraries(modcc PRIVATE arbor-public-headers)
if (ARB_USE_BUNDLED_FMT)
    target_include_directories(modcc PRIVATE ../ext/fmt/include)
else ()
    find_package(fmt)
    target_link_libraries(modcc PRIVATE fmt::fmt-header-only)
endif ()
set_target_properties(modcc libmodcc PROPERTIES EXCLUDE_FROM_ALL ${ARB_WITH_EXTERNAL_MODCC})

if (NOT ARB_WITH_EXTERNAL_MODCC)
    install(TARGETS modcc RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

