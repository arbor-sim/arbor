set(py_unit_sources
    # Currently empty: the s expression tests were moved, however keep this
    # open for the inevitable tests that will be written in the future.

    # unit test driver
    test.cpp
)

add_executable(py_unit EXCLUDE_FROM_ALL ${py_unit_sources})
add_dependencies(tests py_unit)

add_library(py_unit_lib STATIC $<TARGET_OBJECTS:pyarb_obj>)
target_link_libraries(py_unit_lib PRIVATE arbor pybind11::module)

target_compile_options(py_unit PRIVATE ${ARB_CXX_FLAGS_TARGET_FULL})
target_include_directories(
    py_unit PRIVATE
    "${CMAKE_CURRENT_BINARY_DIR}"
    "${PROJECT_SOURCE_DIR}/python"
)

target_link_libraries(
    py_unit PRIVATE
    ext-gtest
    arbor
    py_unit_lib
    pybind11::module
)
