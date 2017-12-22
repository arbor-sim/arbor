# Compiler-aware compiler options

set(CXXOPT_DEBUG "-g")
set(CXXOPT_PTHREAD "-pthread")
set(CXXOPT_CXX11 "-std=c++11")
set(CXXOPT_WALL "-Wall")

if(${CMAKE_CXX_COMPILER_ID} MATCHES "XL")
    # Disable 'missing-braces' warning: this will inappropriately
    # flag initializations such as
    #     std::array<int,3> a={1,2,3};
    set(CXXOPT_WALL "${CXXOPT_WALL} -Wno-missing-braces")

    # CMake, bless its soul, likes to insert this unsupported flag. Hilarity ensues.
    string(REPLACE "-qhalt=e" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
endif()

if(${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
    # Disable 'missing-braces' warning: this will inappropriately
    # flag initializations such as
    #     std::array<int,3> a={1,2,3};
    set(CXXOPT_WALL "${CXXOPT_WALL} -Wno-missing-braces")

    # Clang is erroneously warning that T is an 'unused type alias' in code like this:
    # struct X {
    #     using T = decltype(expression);
    #     T x;
    # };
    set(CXXOPT_WALL "${CXXOPT_WALL} -Wno-unused-local-typedef")

    # Ignore warning if string passed to snprintf is not a string literal.
    set(CXXOPT_WALL "${CXXOPT_WALL} -Wno-format-security")
endif()

if(${CMAKE_CXX_COMPILER_ID} MATCHES "GNU")
    # Compiler flags for generating KNL-specific AVX512 instructions
    # supported in gcc 4.9.x and later.
    set(CXXOPT_KNL "-march=knl")
    set(CXXOPT_AVX2 "-mavx2")
    set(CXXOPT_AVX512 "-mavx512f -mavx512cd")

    # Disable 'maybe-uninitialized' warning: this will be raised
    # inappropriately in some uses of util::optional<T> when T
    # is a primitive type.
    set(CXXOPT_WALL "${CXXOPT_WALL} -Wno-maybe-uninitialized")
endif()

if(${CMAKE_CXX_COMPILER_ID} MATCHES "Intel")
    # Compiler flags for generating KNL-specific AVX512 instructions.
    set(CXXOPT_KNL "-xMIC-AVX512")
    set(CXXOPT_AVX2 "-xCORE-AVX2")
    set(CXXOPT_AVX512 "-xCORE-AVX512")

    # Disable warning for unused template parameter
    # this is raised by a templated function in the json library.
    set(CXXOPT_WALL "${CXXOPT_WALL} -wd488")
endif()

