add_executable(lmorpho lmorpho.cpp lsystem.cpp lsys_models.cpp)

target_link_libraries(lmorpho PRIVATE arbor arbor-sup ext-tinyopt arborio)

if(APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  find_package(llvm)
endif()


if (APPLE AND LLVM_LIBRARY_DIR)
    message("LLVM Patch ${LLVM_LIBRARY_DIR}")
    # Homebrew deploys libc++ along with the llvm package. We must point to it
    # otherwise clang will use the header files of the package and link to
    # system libc++. See llvm/llvm-project#155531 and Homebrew/homebrew-core#235411
    target_link_directories(lmorpho PUBLIC ${LLVM_LIBRARY_DIR}/c++)
endif()


# TODO: resolve public headers
target_link_libraries(lmorpho PRIVATE arbor-private-headers)

install(TARGETS lmorpho RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
