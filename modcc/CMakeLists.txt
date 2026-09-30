# The modcc compiler is split into a static library and driver;
# unit tests for the driver also use this library.

set(libmodcc_sources
    astmanip.cpp
    blocks.cpp
    errorvisitor.cpp
    expression.cpp
    functionexpander.cpp
    functioninliner.cpp
    procinliner.cpp
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

if(APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  find_package(llvm)
  message("LLVM Patch ${LLVM_LIBRARY_DIR} AND ${LLVM_INCLUDE_DIRECTORY}")
  # Homebrew deploys libc++ along with the llvm package. We must point to it
  # otherwise clang will use the header files of the package and link to
  # system libc++. See llvm/llvm-project#155531 and Homebrew/homebrew-core#235411
  link_directories(${LLVM_LIBRARY_DIR}/c++)
  include_directories(SYSTEM ${LLVM_LIBRARY_DIR}/../include/c++/v1)
endif()


set(modcc_sources modcc.cpp)

add_library(libmodcc STATIC ${libmodcc_sources})
target_link_libraries(libmodcc PUBLIC arbor-public-headers)

target_include_directories(libmodcc
                           PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                                  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>)

target_link_libraries(libmodcc PRIVATE fmt::fmt-header-only)

set_target_properties(libmodcc PROPERTIES OUTPUT_NAME modcc)

if (APPLE AND LLVM_LIBRARY_DIR)
    # Homebrew deploys libc++ along with the llvm package. We must point to it
    # otherwise clang will use the header files of the package and link to
    # system libc++. See llvm/llvm-project#155531 and Homebrew/homebrew-core#235411
    target_link_directories(libmodcc PUBLIC ${LLVM_LIBRARY_DIR}/c++)
endif()


export_visibility(libmodcc)

add_executable(modcc ${modcc_sources})

target_link_libraries(modcc PRIVATE fmt::fmt-header-only)
target_link_libraries(modcc PRIVATE libmodcc ext-tinyopt)
set_target_properties(modcc libmodcc PROPERTIES EXCLUDE_FROM_ALL ${ARB_WITH_EXTERNAL_MODCC})
install(TARGETS modcc RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
