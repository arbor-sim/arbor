add_executable(event_gen.exe event_gen.cpp)

target_link_libraries(event_gen.exe LINK_PUBLIC ${ARB_LIBRARIES})
target_link_libraries(event_gen.exe LINK_PUBLIC ${EXTERNAL_LIBRARIES})

if(ARB_WITH_MPI)
    target_link_libraries(event_gen.exe LINK_PUBLIC ${MPI_C_LIBRARIES})
    set_property(TARGET event_gen.exe APPEND_STRING PROPERTY LINK_FLAGS "${MPI_C_LINK_FLAGS}")
endif()

set_target_properties(
    event_gen.exe
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/example"
)
