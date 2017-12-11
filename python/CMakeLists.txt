# Defines the CMake commands/policies
cmake_minimum_required( VERSION 3.5.0 )

# Set the project name
project( pyarbor )

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)
find_package(PythonExtensions REQUIRED)
find_package(Cython REQUIRED)

include_directories(${PYTHON_INCLUDE_DIRS})

#enable_testing()
#find_file( NOSETESTS_EXECUTABLE nosetests )
#add_test( nosetests "${NOSETESTS_EXECUTABLE}" -v --with-xunit )

# Process the CMakeLists.txt in the 'src' and 'bin' directory.
add_subdirectory( miniapp )
#add_subdirectory( bin )
