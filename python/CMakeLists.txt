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

# The Python library. MODULE will make a Python-exclusive model.
add_library(pyarb MODULE
    cells.cpp
    config.cpp
    context.cpp
    domain_decomposition.cpp
    event_generator.cpp
    identifiers.cpp
    mpi.cpp
    pyarb.cpp
    recipe.cpp
    schedule.cpp
    simulation.cpp
    spikes.cpp
)

target_link_libraries(pyarb PRIVATE arbor pybind11::module)

# The output name of the pyarb .so file is "arbor", to facilitate "import arbor"
set_target_properties(pyarb PROPERTIES OUTPUT_NAME arbor)
# With this, the full name of the library will be something like:
#   arbor.cpython-37m-x86_64-linux-gnu.so
set_target_properties(pyarb PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}" SUFFIX "${PYTHON_MODULE_EXTENSION}")

# Add support for mpi4py if available.
if (ARB_WITH_MPI)
    find_python_module(mpi4py)
    if (HAVE_MPI4PY)
        target_include_directories(pyarb PRIVATE "${PY_MPI4PY}/include")
        target_compile_definitions(pyarb PRIVATE -DARB_WITH_MPI4PY)
    endif()
endif()

# Determine the installation path, according to the Python version.
find_package(PythonInterp REQUIRED)
set(ARB_PYEXECDIR "${CMAKE_INSTALL_LIBDIR}/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/site-packages")
install(TARGETS pyarb LIBRARY DESTINATION ${ARB_PYEXECDIR})
