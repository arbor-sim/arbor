set(py_unit_sources
    s_expr.cpp

    # unit test driver
    test.cpp
)

add_executable(py_unit EXCLUDE_FROM_ALL ${py_unit_sources})
add_dependencies(tests py_unit)

add_library(py_unit_lib $<TARGET_OBJECTS:pyarb_obj>)
target_link_libraries(py_unit_lib PRIVATE arbor pybind11::module)

target_compile_options(py_unit PRIVATE ${ARB_CXXOPT_ARCH})
target_include_directories(
    py_unit PRIVATE
    "${CMAKE_CURRENT_BINARY_DIR}"
    "${PROJECT_SOURCE_DIR}/python"
)

target_link_libraries(
    py_unit PRIVATE
    gtest
    arbor
    py_unit_lib
    pybind11::module
)
