add_executable(threadbench.exe bench.cpp)

target_link_libraries(threadbench.exe LINK_PUBLIC ${ARB_LIBRARIES})
target_link_libraries(threadbench.exe LINK_PUBLIC ${EXTERNAL_LIBRARIES})

set_target_properties(
    threadbench.exe
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/example"
)

