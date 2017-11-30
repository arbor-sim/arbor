set(IPSS_SOURCES
    ipss_exe.cpp
)

add_executable(ipss.exe ${IPSS_SOURCES})

target_link_libraries(ipss.exe LINK_PUBLIC ${ARB_LIBRARIES})
target_link_libraries(ipss.exe LINK_PUBLIC ${EXTERNAL_LIBRARIES})

# if(ARB_WITH_MPI)
    # target_link_libraries(miniapp.exe LINK_PUBLIC ${MPI_C_LIBRARIES})
    # set_property(TARGET miniapp.exe APPEND_STRING PROPERTY LINK_FLAGS "${MPI_C_LINK_FLAGS}")
# endif()

set_target_properties(
    ipss.exe
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/miniapps/ipss"
)
