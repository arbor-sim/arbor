
cmake_minimum_required(VERSION 3.9)
project(efields-cat LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

list(APPEND CMAKE_PREFIX_PATH /Users/hater/src/arbor/python/example/reading-external-fields/.venv/lib/python3.12/site-packages/arbor/lib/cmake/arbor)
list(APPEND CMAKE_PREFIX_PATH /Users/hater/src/arbor/python/example/reading-external-fields/.venv/lib/python3.12/site-packages/arbor/lib/cmake/units)
find_package(arbor REQUIRED)

# GPU: Disabled

set(CMAKE_BUILD_TYPE release)
set(CMAKE_CXX_COMPILER  c++)
set(CMAKE_CXX_FLAGS     ${ARB_CXX_FLAGS})

include(BuildModules.cmake)

set(ARB_WITH_EXTERNAL_MODCC true)
find_program(modcc NAMES modcc PATHS /Users/hater/src/arbor/python/example/reading-external-fields/.venv/lib/python3.12/site-packages/arbor/bin)

make_catalogue_standalone(
  NAME efields
  SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/mod"
  MOD template efield
  CXX reader
  CXX_FLAGS_TARGET ${ARB_CXX_FLAGS_TARGET}
  VERBOSE OFF)
