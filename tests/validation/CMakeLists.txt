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

set(VALIDATION_CUDA_SOURCES
    # unit tests
    validate_soma.cu
    validate_ball_and_stick.cu
    validate_kinetic.cu
    validate_synapses.cu

    # support code
    validation_data.cpp
    trace_analysis.cpp

    # unit test driver
    validate.cpp
)

if(NMC_VALIDATION_DATA_DIR)
    if ("${CMAKE_VERSION}" MATCHES "^3.[78].")
        #message(WARNING "CMake ${CMAKE_VERSION} has broken FindCUDA; omitting NMC_DATADIR define.")
    else()
        add_definitions("-DNMC_DATADIR=\"${NMC_VALIDATION_DATA_DIR}\"")
    endif()
endif()

add_executable(validate.exe ${VALIDATION_SOURCES})
set(TARGETS validate.exe)

if(NMC_WITH_CUDA)
    cuda_add_executable(validate_cuda.exe ${VALIDATION_CUDA_SOURCES})
    list(APPEND TARGETS validate_cuda.exe)
    target_link_libraries(validate_cuda.exe LINK_PUBLIC gpu)
endif()


foreach(target ${TARGETS})
    target_link_libraries(${target} LINK_PUBLIC gtest)
    target_link_libraries(${target} LINK_PUBLIC ${NESTMC_LIBRARIES})
    target_link_libraries(${target} LINK_PUBLIC ${EXTERNAL_LIBRARIES})

    if(NMC_WITH_MPI)
        target_link_libraries(${target} LINK_PUBLIC ${MPI_C_LIBRARIES})
        set_property(TARGET ${target} APPEND_STRING PROPERTY LINK_FLAGS "${MPI_C_LINK_FLAGS}")
    endif()

    set_target_properties(
        ${target}
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
    )

    if(NMC_BUILD_VALIDATION_DATA)
        add_dependencies(${target} validation_data)
    endif()
endforeach()

