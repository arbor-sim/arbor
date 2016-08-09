set(HEADERS
    ${PROJECT_SOURCE_DIR}/src/swcio.hpp
)
set(COMMUNICATION_SOURCES
    test_exporter_spike_file.cpp
    test_exporter_spike_single_file.cpp
    # unit test driver
    test.cpp
)

add_executable(global_communication.exe ${COMMUNICATION_SOURCES} ${HEADERS})

set(TARGETS global_communication.exe)

foreach(target ${TARGETS})
    target_link_libraries(${target} LINK_PUBLIC cellalgo gtest)
    
    if(WITH_TBB)
    target_link_libraries(${target} LINK_PUBLIC ${TBB_LIBRARIES})
    endif()

    if(WITH_MPI)
    target_link_libraries(${target} LINK_PUBLIC ${MPI_C_LIBRARIES})
    set_property(TARGET ${target} APPEND_STRING PROPERTY LINK_FLAGS "${MPI_C_LINK_FLAGS}")
    endif()

    set_target_properties(${target}
       PROPERTIES
       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
    )
endforeach()

