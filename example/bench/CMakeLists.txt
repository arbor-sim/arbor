add_executable(bench.exe bench.cpp recipe.cpp parameters.cpp)

target_link_libraries(bench.exe LINK_PUBLIC ${ARB_LIBRARIES})
target_link_libraries(bench.exe LINK_PUBLIC ${EXTERNAL_LIBRARIES})

if(ARB_WITH_MPI)
    target_link_libraries(bench.exe LINK_PUBLIC ${MPI_C_LIBRARIES})
    set_property(TARGET bench.exe APPEND_STRING PROPERTY LINK_FLAGS "${MPI_C_LINK_FLAGS}")
endif()

set_target_properties(
    bench.exe
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/example"
)
