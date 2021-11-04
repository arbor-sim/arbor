add_executable(lmorpho lmorpho.cpp lsystem.cpp lsys_models.cpp)

target_link_libraries(lmorpho PRIVATE arbor arbor-sup ext-tinyopt arborio)

# TODO: resolve public headers
target_link_libraries(lmorpho PRIVATE arbor-private-headers)

install(TARGETS lmorpho RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
