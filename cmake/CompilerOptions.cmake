# Compiler-aware compiler options

set(CXXOPT_DEBUG "-g")
set(CXXOPT_PTHREAD "-pthread")
set(CXXOPT_CXX11 "-std=c++11")
set(CXXOPT_WALL "-Wall")

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

if(CMAKE_CXX_COMPILER_ID MATCHES "XL")
    # Disable 'missing-braces' warning: this will inappropriately
    # flag initializations such as
    #     std::array<int,3> a={1,2,3};
    set(CXXOPT_WALL "${CXXOPT_WALL} -Wno-missing-braces")

    # CMake, bless its soul, likes to insert this unsupported flag. Hilarity ensues.
    string(REPLACE "-qhalt=e" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CXXOPT_KNL "-march=knl")
    set(CXXOPT_AVX2 "-mavx2 -mfma")
    set(CXXOPT_AVX512 "-mavx512f -mavx512cd")

    # Disable 'missing-braces' warning: this will inappropriately
    # flag initializations such as
    #     std::array<int,3> a={1,2,3};
    set(CXXOPT_WALL "${CXXOPT_WALL} -Wno-missing-braces")

    # Disable 'potentially-evaluated-expression' warning: this warns
    # on expressions of the form `typeid(expr)` when `expr` has side
    # effects.
    set(CXXOPT_WALL "${CXXOPT_WALL} -Wno-potentially-evaluated-expression")

    # Clang is erroneously warning that T is an 'unused type alias' in code like this:
    # struct X {
    #     using T = decltype(expression);
    #     T x;
    # };
    set(CXXOPT_WALL "${CXXOPT_WALL} -Wno-unused-local-typedef")

    # Ignore warning if string passed to snprintf is not a string literal.
    set(CXXOPT_WALL "${CXXOPT_WALL} -Wno-format-security")
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    # Disable 'maybe-uninitialized' warning: this will be raised
    # inappropriately in some uses of util::optional<T> when T
    # is a primitive type.
    set(CXXOPT_WALL "${CXXOPT_WALL} -Wno-maybe-uninitialized")
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    # Disable warning for unused template parameter
    # this is raised by a templated function in the json library.
    set(CXXOPT_WALL "${CXXOPT_WALL} -wd488")
endif()

# Set CXXOPT_ARCH in parent scope according to requested architecture.
# Architectures are given by the same names that GCC uses for its
# -mcpu or -march options.

function(set_arch_target arch)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        # Correct compiler option unfortunately depends upon the target architecture family.
        # Extract this information from running the configured compiler with --verbose.

        try_compile(ignore ${CMAKE_BINARY_DIR} ${PROJECT_SOURCE_DIR}/cmake/dummy.cpp COMPILE_DEFINITIONS --verbose OUTPUT_VARIABLE cc_out)
        string(REPLACE "\n" ";" cc_out "${cc_out}")
        set(target)
        foreach(line ${cc_out})
            if(line MATCHES "^Target:")
                string(REGEX REPLACE "^Target: " "" target "${line}")
            endif()
        endforeach(line)
        string(REGEX REPLACE "-.*" "" target_model "${target}")

        # Use -mcpu for all supported targets _except_ for x86, where it should be -march.

        if(target_model MATCHES "x86" OR target_model MATCHES "amd64" OR target_model MATCHES "aarch64")
            set(CXXOPT_ARCH "-march=${arch}")
        else()
            set(CXXOPT_ARCH "-mcpu=${arch}")
        endif()

    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
        # Translate target architecture names to Intel-compatible names.
        # icc 17 recognizes the following specific microarchitecture names for -mtune:
        #     broadwell, haswell, ivybridge, knl, sandybridge, skylake

        if(arch MATCHES "sandybridge")
            set(tune "${arch}")
            set(arch "AVX")
        elseif(arch MATCHES "ivybridge")
            set(tune "${arch}")
            set(arch "CORE-AVX-I")
        elseif(arch MATCHES "broadwell|haswell|skylake")
            set(tune "${arch}")
            set(arch "CORE-AVX2")
        elseif(arch MATCHES "knl")
            set(tune "${arch}")
            set(arch "MIC-AVX512")
        elseif(arch MATCHES "nehalem|westmere")
            set(tune "corei7")
            set(arch "SSE4.2")
        elseif(arch MATCHES "core2")
            set(tune "core2")
            set(arch "SSSE3")
        elseif(arch MATCHES "native")
            unset(tune)
            set(arch "Host")
        else()
            set(tune "generic")
            set(arch "SSE2") # default for icc
        endif()

        if(tune)
            set(CXXOPT_ARCH "-x${arch};-mtune=${tune}")
        else()
            set(CXXOPT_ARCH "-x${arch}")
        endif()

    elseif(CMAKE_CXX_COMPILER_ID MATCHES "XL")
        # xlC 13 for Linux uses -mcpu. Not even attempting to get xlC 12 for BG/Q right
        # at this point: use CXXFLAGS as required!
        #
        # xlC, gcc, and clang all recognize power8 and power9 as architecture keywords.

        if(arch MATCHES "native")
            set(CXXOPT_ARCH "-qarch=auto")
        else()
            set(CXXOPT_ARCH "-mcpu=${arch}")
        endif()
    endif()

    set(CXXOPT_ARCH "${CXXOPT_ARCH}" PARENT_SCOPE)
endfunction()
