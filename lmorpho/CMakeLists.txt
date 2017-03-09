add_executable(lmorpho lmorpho.cpp lsystem.cpp lsys_models.cpp morphio.cpp)

target_link_libraries(lmorpho LINK_PUBLIC nestmc)
target_link_libraries(lmorpho LINK_PUBLIC ${EXTERNAL_LIBRARIES})

