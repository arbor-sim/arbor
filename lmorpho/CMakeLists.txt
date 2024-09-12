add_executable(lmorpho lmorpho.cpp lsystem.cpp lsys_models.cpp)
set_source_files_properties(lmorpho.cpp lsystem.cpp lsys_models.cpp
                            PROPERTIES SKIP_LINTING ON)

target_link_libraries(lmorpho PRIVATE arbor arbor-sup ext-tinyopt arborio)

# TODO: resolve public headers
target_link_libraries(lmorpho PRIVATE arbor-private-headers)

install(TARGETS lmorpho RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
