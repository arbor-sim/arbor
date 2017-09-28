set(HEADERS
    ${PROJECT_SOURCE_DIR}/src/swcio.hpp
)
set(COMMUNICATION_SOURCES
    test_domain_decomposition.cpp
    test_exporter_spike_file.cpp
    test_communicator.cpp
    test_mpi.cpp

    # unit test driver
    test.cpp
)

add_executable(global_communication.exe ${COMMUNICATION_SOURCES} ${HEADERS})

set(TARGETS global_communication.exe)

foreach(target ${TARGETS})
    target_link_libraries(${target} LINK_PUBLIC arbor gtest)
    target_link_libraries(${target} LINK_PUBLIC ${EXTERNAL_LIBRARIES})

    if(ARB_WITH_MPI)
        target_link_libraries(${target} LINK_PUBLIC ${MPI_C_LIBRARIES})
        set_property(TARGET ${target} APPEND_STRING PROPERTY LINK_FLAGS "${MPI_C_LINK_FLAGS}")
    endif()

    set_target_properties(
        ${target}
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
    )
endforeach()

