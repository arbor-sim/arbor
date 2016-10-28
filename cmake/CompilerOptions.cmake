# Compiler-aware compiler options

set(CXXOPT_DEBUG "-g")
set(CXXOPT_PTHREAD "-pthread")
set(CXXOPT_CXX11 "-std=c++11")
set(CXXOPT_WALL "-Wall")

if(${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
    # Disable 'missing-braces' warning: this will inappropriately
    # flag initializations such as
    #     std::array<int,3> a={1,2,3};

    set(CXXOPT_WALL "${CXXOPT_WALL} -Wno-missing-braces")
endif()

if(${CMAKE_CXX_COMPILER_ID} MATCHES "GNU")

    # compiler flags for generating KNL-specific AVX512 instructions
    # supported in gcc 4.9.x and later
    set(CXXOPT_KNL "-march=knl")
    set(CXXOPT_AVX "-mavx")
    set(CXXOPT_AVX2 "-march=core-avx2")
endif()

if(${CMAKE_CXX_COMPILER_ID} MATCHES "Intel")
    # Disable warning for unused template parameter
    # this is raised by a templated function in the json library

    set(CXXOPT_WALL "${CXXOPT_WALL} -wd488")

    # compiler flags for generating KNL-specific AVX512 instructions
    set(CXXOPT_KNL "-xMIC-AVX512")
    set(CXXOPT_AVX "-mavx")
    set(CXXOPT_AVX2 "-march=core-avx2")
endif()

