# CMake (at least sometimes) misidentifies XL 13 for Linux as Clang.

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    try_compile(ignore ${CMAKE_BINARY_DIR} ${PROJECT_SOURCE_DIR}/cmake/dummy.cpp COMPILE_DEFINITIONS --version OUTPUT_VARIABLE cc_out)
    string(REPLACE "\n" ";" cc_out "${cc_out}")
    foreach(line ${cc_out})
        if(line MATCHES "^IBM XL C")
            set(CMAKE_CXX_COMPILER_ID "XL")
        endif()
    endforeach(line)
endif()

# If we _do_ find xlC, don't try and build: too many bugs!

if(CMAKE_CXX_COMPILER_ID STREQUAL "XL")
    message(FATAL_ERROR "Arbor does not support being built by the IBM xlC compiler")
endif()
