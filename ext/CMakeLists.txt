# Niels Lohmann's json library (single-header version).

add_library(ext-json INTERFACE)
target_include_directories(ext-json INTERFACE json/single_include)

# TCLAP command line parsing libary (header-only).

add_library(ext-tclap INTERFACE)
target_include_directories(ext-tclap INTERFACE tclap/include)

# Intel TBB:
# Alias system TBB or build locally and export that, according
# to ARB_PRIVATE_TBBLIB setting.

if(ARB_REQUIRE_TBB)
    if(ARB_PRIVATE_TBBLIB)
        check_git_submodule(tbb tbb)
        if(NOT tbb_avail)
            message(FATAL_ERROR "TBB git submodule required.")
        endif()

        # Turn off proxy malloc library and test compilation.
        option(TBB_BUILD_TBBMALLOC_PROXY OFF)
        option(TBB_BUILD_TESTS           OFF)

        add_subdirectory(tbb)
        add_library(ext-tbb INTERFACE)
        target_link_libraries(ext-tbb INTERFACE tbb_static tbbmalloc_static)
        target_include_directories(ext-tbb SYSTEM INTERFACE tbb/include)
    else()
        find_package(TBB REQUIRED)
        add_library(ext-tbb ALIAS TBB::tbb)
    endif()
endif()

