set(validation_sources
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

add_executable(validate ${validation_sources})
target_compile_definitions(validate PRIVATE "ARB_DATADIR=\"${ARB_VALIDATION_DATA_DIR}\"")
target_link_libraries(validate PRIVATE gtest arbor)
target_include_directories(validate PRIVATE "${PROJECT_SOURCE_DIR}/external")

if(ARB_BUILD_VALIDATION_DATA)
    add_dependencies(validate validation_data)
endif()
