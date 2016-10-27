set(VALIDATION_SOURCES
    # unit tests
    validate_ball_and_stick.cpp
    validate_soma.cpp
    validate_synapses.cpp

    # support code
    validation_data.cpp
    trace_analysis.cpp

    # unit test driver
    validate.cpp
)

add_definitions("-DDATADIR=\"${CMAKE_SOURCE_DIR}/data\"")

add_executable(validate.exe ${VALIDATION_SOURCES} ${HEADERS})

set(TARGETS validate.exe)

foreach(target ${TARGETS})
    target_link_libraries(${target} LINK_PUBLIC nestmc gtest)

    if(WITH_TBB)
        target_link_libraries(${target} LINK_PUBLIC ${TBB_LIBRARIES})
    endif()

    if(WITH_MPI)
        target_link_libraries(${target} LINK_PUBLIC ${MPI_C_LIBRARIES})
        set_property(TARGET ${target} APPEND_STRING PROPERTY LINK_FLAGS "${MPI_C_LINK_FLAGS}")
    endif()

    set_target_properties(
        ${target}
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
    )

    if (BUILD_VALIDATION_DATA)
	add_dependencies(${target} validation_data)
    endif()
endforeach()

