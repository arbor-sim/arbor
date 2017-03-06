add_executable(lmorpho lmorpho.cpp lsystem.cpp lsys_models.cpp morphology.cpp morphio.cpp)

target_link_libraries(lmorpho LINK_PUBLIC nestmc)
target_link_libraries(lmorpho LINK_PUBLIC ${EXTERNAL_LIBRARIES})

