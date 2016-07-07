set(HEADERS
    ${PROJECT_SOURCE_DIR}/src/cell.hpp
    ${PROJECT_SOURCE_DIR}/src/cell_tree.hpp
    ${PROJECT_SOURCE_DIR}/src/math.hpp
    ${PROJECT_SOURCE_DIR}/src/point.hpp
    ${PROJECT_SOURCE_DIR}/src/segment.hpp
    ${PROJECT_SOURCE_DIR}/src/swcio.hpp
    ${PROJECT_SOURCE_DIR}/src/tree.hpp
)

set(TEST_SOURCES
    # unit test driver
    test.cpp
)

set(VALIDATION_SOURCES
    # unit tests
    validate_ball_and_stick.cpp
    validate_soma.cpp
    #validate_synapses.cpp

    # unit test driver
    validate.cpp
)

# google test framework
add_library(gtest_validate gtest-all.cpp gtest.h)

add_definitions("-DDATADIR=\"${CMAKE_SOURCE_DIR}/data\"")

add_executable(validate.exe ${VALIDATION_SOURCES} ${HEADERS})

set(TARGETS validate.exe)

foreach(target ${TARGETS})
	target_link_libraries(${target} LINK_PUBLIC cellalgo gtest_validate)
    
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

