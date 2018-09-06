# Set up pybind11 as an external project.
set(pb11_src_dir "${PROJECT_SOURCE_DIR}/python/pybind11")
check_git_submodule(pybind11 "${pb11_src_dir}")

if(NOT pybind11_avail)
    message(FATAL_ERROR "The git submodule for pybind11 is not available, required for python support")
endif()

# Set up pybind11, which is used to generate python bindings.
# Pybind11 has good cmake support, so just add the pybind11 directory,
# instead of using find_package.
set(PYBIND11_CPP_STANDARD -std=c++14)
add_subdirectory(pybind11)

# The python library. MODULE will make a python-exclusive model.
add_library(pyarb MODULE
    pyarb.cpp
    strings.cpp
    identifiers.cpp
    context.cpp
    recipe.cpp
    event_generator.cpp
    profiler.cpp
    )

target_link_libraries(pyarb PRIVATE arbor pybind11::module)
set_target_properties(pyarb PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}" SUFFIX "${PYTHON_MODULE_EXTENSION}")

# Determine the installation path, according to the python version.
find_package(PythonInterp REQUIRED)
set(ARB_PYEXECDIR "${CMAKE_INSTALL_LIBDIR}/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/site-packages")
install(TARGETS pyarb LIBRARY DESTINATION ${ARB_PYEXECDIR})
