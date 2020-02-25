include(FindPythonModule) # required for find_python_module

# Set up pybind11 as an external project.
set(pb11_src_dir "${PROJECT_SOURCE_DIR}/python/pybind11")
check_git_submodule(pybind11 "${pb11_src_dir}")

if(NOT pybind11_avail)
    message(FATAL_ERROR "The git submodule for pybind11 is not available, required for python support")
endif()

# Set up pybind11, which is used to generate Python bindings.
# Pybind11 has good cmake support, so just add the pybind11 directory,
# instead of using find_package.
set(PYBIND11_CPP_STANDARD -std=c++14)
add_subdirectory(pybind11)

set(pyarb_source
    cells.cpp
    config.cpp
    context.cpp
    domain_decomposition.cpp
    error.cpp
    event_generator.cpp
    flat_cell_builder.cpp
    identifiers.cpp
    morphology.cpp
    morph_parse.cpp
    mpi.cpp
    profiler.cpp
    pyarb.cpp
    recipe.cpp
    sampling.cpp
    schedule.cpp
    simulation.cpp
    single_cell_model.cpp
    spikes.cpp
    s_expr.cpp
)

# compile the pyarb sources into an object library that will be
# use by both the Python wrapper target (pyarb) and for the
# unit tests of the C++ components in the Python wrapper.
add_library(pyarb_obj OBJECT ${pyarb_source})
target_link_libraries(pyarb_obj PRIVATE arbor pybind11::module)

# The Python library. MODULE will make a Python-exclusive model.
add_library(pyarb MODULE $<TARGET_OBJECTS:pyarb_obj>)

# The output name of the pyarb .so file is "_arbor"
set_target_properties(pyarb PROPERTIES OUTPUT_NAME _arbor)
# With this, the full name of the library will be something like:
#   arbor.cpython-36m-x86_64-linux-gnu.so
set_target_properties(pyarb PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}" SUFFIX "${PYTHON_MODULE_EXTENSION}")

# This dependency has to be spelt out again, despite being added to
# pyarb_obj because CMake.
target_link_libraries(pyarb PRIVATE arbor pybind11::module)

# Add support for mpi4py if available.
if (ARB_WITH_MPI)
    find_python_module(mpi4py)
    if (HAVE_MPI4PY)
        target_include_directories(pyarb PRIVATE "${PY_MPI4PY}/include")
        target_compile_definitions(pyarb PRIVATE -DARB_WITH_MPI4PY)
    endif()
endif()

# For unit tests on C++ side of Python wrappers
add_subdirectory(test)

# Create the Python module in the build directory.
# The module contains the dynamic library, __init__.py and VERSION information.
set(python_mod_path "${CMAKE_CURRENT_BINARY_DIR}/arbor")
set_target_properties(pyarb PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${python_mod_path}")
file(COPY "${CMAKE_SOURCE_DIR}/python/__init__.py" DESTINATION "${python_mod_path}")
file(COPY "${CMAKE_SOURCE_DIR}/VERSION" DESTINATION "${python_mod_path}")

# Determine the installation path, according to the Python version.
# The installation for Python 3.8 would be:
#  ${ARB_PYTHON_PREFIX}/python3.8/site-packages
# By default ARB_PYTHON_PREFIX is set to CMAKE_INSTALL_PREFIX, and can be optionally
# used to install the Python module as a user module, or in a virtualenv.
find_package(PythonInterp REQUIRED)
set(arb_pyexecdir "${ARB_PYTHON_PREFIX}/lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/site-packages")
install(DIRECTORY ${python_mod_path} DESTINATION ${arb_pyexecdir})
