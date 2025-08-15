include(GNUInstallDirs)

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
    network.cpp
    profiler.cpp
    pyarb.cpp
    label_dict.cpp
    recipe.cpp
    schedule.cpp
    simulation.cpp
    single_cell_model.cpp
    env.cpp
    remote.cpp
    units.cpp
)

set_property(SOURCE config.cpp PROPERTY COMPILE_DEFINITIONS ARB_BINARY="${CMAKE_INSTALL_BINDIR}"         APPEND)
set_property(SOURCE config.cpp PROPERTY COMPILE_DEFINITIONS ARB_LIB="${CMAKE_INSTALL_LIBDIR}"            APPEND)
set_property(SOURCE config.cpp PROPERTY COMPILE_DEFINITIONS ARB_DATA="${CMAKE_INSTALL_DATAROOTDIR}"      APPEND)
set_property(SOURCE config.cpp PROPERTY COMPILE_DEFINITIONS ARB_CXX_COMPILER="${CMAKE_CXX_COMPILER}"     APPEND)
set_property(SOURCE config.cpp PROPERTY COMPILE_DEFINITIONS ARB_PREFIX="${CMAKE_INSTALL_PREFIX}"         APPEND)
set_property(SOURCE config.cpp PROPERTY COMPILE_DEFINITIONS ARB_PYTHON_LIB_PATH="${ARB_PYTHON_LIB_PATH}" APPEND)

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

# Set RPATH for the installation phase
if(APPLE)
    set_target_properties(pyarb PROPERTIES INSTALL_RPATH "@loader_path")  # Relative RPATH for macOS
else()
    set_target_properties(pyarb PROPERTIES INSTALL_RPATH "$ORIGIN")  # Use relative RPATH on Linux
endif()

# This dependency has to be spelt out again, despite being added to
# pyarb_obj because CMake.
target_link_libraries(pyarb PRIVATE arbor arborenv arborio pybind11::module)

# Ensure RPATH is preserved during installation
set_target_properties(pyarb PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)

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

execute_process(COMMAND ${PYTHON_EXECUTABLE} "${PROJECT_SOURCE_DIR}/scripts/where.py"
                INPUT_FILE "${CMAKE_BINARY_DIR}/install-prefix"
                OUTPUT_VARIABLE ARB_PYTHON_LIB_PATH_DEFAULT_REL
                OUTPUT_STRIP_TRAILING_WHITESPACE)

# convert to absolute path if needed (could be a relative path if ccmake was used)
get_filename_component(ARB_PYTHON_LIB_PATH_DEFAULT "${ARB_PYTHON_LIB_PATH_DEFAULT_REL}"
                       REALPATH BASE_DIR "${CMAKE_BINARY_DIR}")

# Default to installing in that path, override with user specified ARB_PYTHON_LIB_PATH
set(ARB_PYTHON_LIB_PATH ${ARB_PYTHON_LIB_PATH_DEFAULT} CACHE PATH "path for installing Python module for Arbor.")
message(VERBOSE "Python module installation path: ${ARB_PYTHON_LIB_PATH}")
mark_as_advanced(FORCE ARB_PYTHON_LIB_PATH)

if(DEFINED SKBUILD_PROJECT_NAME)
  # Building wheel through scikit-build-core
  set(_python_module_install_path .)
else()
  set(_python_module_install_path ${ARB_PYTHON_LIB_PATH}/arbor)
endif()

# generate type stubs and copy them to the expected places
if(ARB_BUILD_PYTHON_STUBS)
  find_program(PB11_STUBGEN NAMES pybind11-stubgen REQUIRED)
  add_custom_command(TARGET pyarb POST_BUILD
                     COMMAND
                       PYTHONPATH=${CMAKE_BINARY_DIR}/python:$ENV{PYTHONPATH} ${PB11_STUBGEN} -o ${CMAKE_BINARY_DIR}/stubs arbor
                     BYPRODUCTS ${CMAKE_BINARY_DIR}/stubs
                     USES_TERMINAL
                     COMMENT "Generating type stubs")
  install(DIRECTORY ${CMAKE_BINARY_DIR}/stubs/arbor/ DESTINATION ${_python_module_install_path})
endif()

install(TARGETS pyarb DESTINATION ${_python_module_install_path})
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py DESTINATION ${_python_module_install_path})
install(FILES ${PROJECT_SOURCE_DIR}/VERSION ${PROJECT_SOURCE_DIR}/README.md ${PROJECT_SOURCE_DIR}/LICENSE DESTINATION ${_python_module_install_path})
