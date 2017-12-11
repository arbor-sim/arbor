add_executable(lmorpho lmorpho.cpp lsystem.cpp lsys_models.cpp morphio.cpp)

if(ARB_WITH_MPI)
    target_link_libraries(lmorpho LINK_PUBLIC ${MPI_C_LIBRARIES})
    set_property(TARGET lmorpho APPEND_STRING PROPERTY LINK_FLAGS "${MPI_C_LINK_FLAGS}")
endif()

target_link_libraries(lmorpho LINK_PUBLIC arbor)
target_link_libraries(lmorpho LINK_PUBLIC ${EXTERNAL_LIBRARIES})

