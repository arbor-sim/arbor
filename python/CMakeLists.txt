include(GNUInstallDirs)

set(PYBIND11_CPP_STANDARD -std=c++17)

if(ARB_USE_BUNDLED_PYBIND11)
  include(FindPythonModule) # required for find_python_module

  # Set up pybind11 as an external project.
  set(pb11_src_dir "${PROJECT_SOURCE_DIR}/ext/pybind11")
  check_git_submodule(pybind11 "${pb11_src_dir}")

  if(NOT pybind11_avail)
    message(FATAL_ERROR "The git submodule for pybind11 is not available, required for python support")
  endif()

  # Set up pybind11, which is used to generate Python bindings.
  # Pybind11 has good cmake support, so just add the pybind11 directory,
  # instead of using find_package.
  add_subdirectory(${pb11_src_dir} pybind11)
else()
  find_package(pybind11 2.8.1 REQUIRED)
endif()

set(pyarb_source
    cable_cell_io.cpp
    probes.cpp
    cells.cpp
    config.cpp
    context.cpp
    domain_decomposition.cpp
    error.cpp
    event_generator.cpp
    identifiers.cpp
    mechanism.cpp
    morphology.cpp
    mpi.cpp
    profiler.cpp
    pyarb.cpp
    recipe.cpp
    schedule.cpp
    simulation.cpp
    single_cell_model.cpp
    env.cpp
)

set_property(SOURCE config.cpp PROPERTY COMPILE_DEFINITIONS ARB_BINARY="${CMAKE_INSTALL_BINDIR}"       APPEND)
set_property(SOURCE config.cpp PROPERTY COMPILE_DEFINITIONS ARB_LIB="${CMAKE_INSTALL_LIBDIR}"          APPEND)
set_property(SOURCE config.cpp PROPERTY COMPILE_DEFINITIONS ARB_DATA="${CMAKE_INSTALL_DATAROOTDIR}"    APPEND)
set_property(SOURCE config.cpp PROPERTY COMPILE_DEFINITIONS ARB_CXX_COMPILER="${CMAKE_CXX_COMPILER}"   APPEND)
set_property(SOURCE config.cpp PROPERTY COMPILE_DEFINITIONS ARB_PREFIX="${CMAKE_INSTALL_PREFIX}"       APPEND)

# compile the pyarb sources into an object library that will be
# use by both the Python wrapper target (pyarb) and for the
# unit tests of the C++ components in the Python wrapper.
add_library(pyarb_obj OBJECT ${pyarb_source})
set_target_properties(pyarb_obj PROPERTIES CXX_VISIBILITY_PRESET hidden)
target_link_libraries(pyarb_obj PRIVATE arbor arborenv arborio pybind11::module)

# The Python library. MODULE will make a Python-exclusive model.
add_library(pyarb MODULE $<TARGET_OBJECTS:pyarb_obj>)

# The output name of the pyarb .so file is "_arbor"
set_target_properties(pyarb PROPERTIES OUTPUT_NAME _arbor)
# With this, the full name of the library will be something like:
#   arbor.cpython-36m-x86_64-linux-gnu.so
set_target_properties(pyarb PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}" SUFFIX "${PYTHON_MODULE_EXTENSION}")

# This dependency has to be spelt out again, despite being added to
# pyarb_obj because CMake.
target_link_libraries(pyarb PRIVATE arbor arborenv arborio pybind11::module)

# Add support for mpi4py if available.
if (ARB_WITH_MPI)
    find_python_module(mpi4py)
    if (HAVE_MPI4PY)
        target_include_directories(pyarb_obj PRIVATE "${PY_MPI4PY}/include")
        target_compile_definitions(pyarb_obj PRIVATE -DARB_WITH_MPI4PY)
    endif()
endif()

# For unit tests on C++ side of Python wrappers
add_subdirectory(test)

# Create the Python module in the build directory.
# The module contains the dynamic library, __init__.py and VERSION information.
set(python_mod_path "${CMAKE_CURRENT_BINARY_DIR}/arbor")
set_target_properties(pyarb PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${python_mod_path}")
file(COPY "${PROJECT_SOURCE_DIR}/python/__init__.py" DESTINATION "${python_mod_path}")
file(COPY "${PROJECT_SOURCE_DIR}/VERSION" DESTINATION "${python_mod_path}")

# Set the installation path

# Ask Python where it keeps its system (platform) packages.
file(WRITE "${CMAKE_BINARY_DIR}/install-prefix" "${CMAKE_INSTALL_PREFIX}")

execute_process(
    COMMAND ${PYTHON_EXECUTABLE} "${PROJECT_SOURCE_DIR}/scripts/where.py"
    INPUT_FILE "${CMAKE_BINARY_DIR}/install-prefix"
    OUTPUT_VARIABLE ARB_PYTHON_LIB_PATH_DEFAULT
    OUTPUT_STRIP_TRAILING_WHITESPACE)

# Default to installing in that path, override with user specified ARB_PYTHON_LIB_PATH
set(ARB_PYTHON_LIB_PATH ${ARB_PYTHON_LIB_PATH_DEFAULT} CACHE PATH "path for installing Python module for Arbor.")
message(STATUS "Python module installation path: ${ARB_PYTHON_LIB_PATH}")

install(DIRECTORY ${python_mod_path} DESTINATION ${ARB_PYTHON_LIB_PATH})
