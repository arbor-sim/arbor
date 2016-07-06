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

if(${CMAKE_CXX_COMPILER_ID} MATCHES "Intel")
    # Disable warning for unused template parameter
    # this is raised by a templated function in the json library

    set(CXXOPT_WALL "${CXXOPT_WALL} -wd488")
endif()

