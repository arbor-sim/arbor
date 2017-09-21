set(VALIDATION_SOURCES
    # unit tests
    validate_ball_and_stick.cpp
    validate_compartment_policy.cpp
    validate_soma.cpp
    validate_kinetic.cpp
    validate_synapses.cpp

    # support code
    validation_data.cpp
    trace_analysis.cpp

    # unit test driver
    validate.cpp
)

if(NMC_VALIDATION_DATA_DIR)
    if ("${CMAKE_VERSION}" MATCHES "^3.[789].")
        message(WARNING "CMake ${CMAKE_VERSION} has broken FindCUDA; omitting NMC_DATADIR define.")
    else()
        add_definitions("-DNMC_DATADIR=\"${NMC_VALIDATION_DATA_DIR}\"")
    endif()
endif()

add_executable(validate.exe ${VALIDATION_SOURCES})

if(NMC_WITH_CUDA)
    target_link_libraries(validate.exe LINK_PUBLIC gpu)
endif()


target_link_libraries(validate.exe LINK_PUBLIC gtest)
target_link_libraries(validate.exe LINK_PUBLIC ${NESTMC_LIBRARIES})
target_link_libraries(validate.exe LINK_PUBLIC ${EXTERNAL_LIBRARIES})

if(NMC_WITH_MPI)
    target_link_libraries(validate.exe LINK_PUBLIC ${MPI_C_LIBRARIES})
    set_property(TARGET validate.exe APPEND_STRING PROPERTY LINK_FLAGS "${MPI_C_LINK_FLAGS}")
endif()

set_target_properties(
    validate.exe
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
)

if(NMC_BUILD_VALIDATION_DATA)
    add_dependencies(validate.exe validation_data)
endif()
