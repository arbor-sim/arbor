
# CMake find_package() Module for Random123 header-only C++ library
# https://github.com/DEShawResearch/random123
#
# Example usage:
#
# find_package(Random123)
#
# If successful the following variables will be defined
# RANDOM123_FOUND
# RANDOM123_INCLUDE_DIR

find_path(RANDOM123_INCLUDE_DIR Random123/threefry.h)

if(RANDOM123_INCLUDE_DIR)
    set(RANDOM123_FOUND TRUE)
    message(STATUS "Found Random123: ${RANDOM123_INCLUDE_DIR}")
endif()

if(RANDOM123_FIND_REQUIRED AND NOT RANDOM123_FOUND)
    message(FATAL_ERROR "Required package Random123 not found")
endif()

