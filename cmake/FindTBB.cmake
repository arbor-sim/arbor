# Find the Intel Thread Building Blocks library
#
# Sets the following variables:
#
#  TBB_FOUND               - True if libtbb and libtbb_malloc found.
#  TBB_LIBRARIES           - Paths to libtbb and libtbbmalloc.
#  TBB_INCLUDE_DIR         - Base directory for tbb/ includes.
# 
# Generates the import library target TBB:tbb if found.
#
# The default search path can be overriden by setting the
# CMake variable TBB_ROOT_DIR or the environment variables
# TBBROOT or TBB_ROOT.

if(NOT TBB_FOUND)
    find_package(Threads REQUIRED)

    set(_tbb_search_path ${TBB_ROOT_DIR} $ENV{TBBROOT} $ENV{TBB_ROOT})
    set(_tbb_lib_suffixes lib/intel64/gcc4.7 lib/intel64/gcc4.4 lib/android lib/mic lib)

    macro(_tbb_findlib libname)
        find_library(_lib${libname} ${libname}
            PATHS ${_tbb_search_path} NO_DEFAULT_PATH
            PATH_SUFFIXES ${_tbb_lib_suffixes})
        find_library(_lib${libname} ${libname}
            PATH_SUFFIXES _tbb_lib_suffixes)
    endmacro()

    _tbb_findlib(tbb)
    _tbb_findlib(tbbmalloc)

    find_path(TBB_INCLUDE_DIR tbb/tbb.h PATHS ${_tbb_search_path} NO_DEFAULT_PATH PATH_SUFFIXES include)
    find_path(TBB_INCLUDE_DIR tbb/tbb.h)

    include(FindPackageHandleStandardArgs)
    find_package_handle_standard_args(TBB DEFAULT_MSG TBB_INCLUDE_DIR _libtbb _libtbbmalloc)

    if(TBB_FOUND)
        set(TBB_INCLUDE_DIRS ${TBB_INCLUDE_DIR})
        set(TBB_LIBRARIES ${_libtbb} ${_libtbbmalloc})
        if(NOT TARGET TBB::tbb)
            add_library(TBB::tbb UNKNOWN IMPORTED)
            set_target_properties(TBB::tbb PROPERTIES
                    IMPORTED_LOCATION "${_libtbb}"
                    INTERFACE_LINK_LIBRARIES "${_libtbbmalloc}"
                    INTERFACE_INCLUDE_DIRECTORIES "${TBB_INCLUDE_DIR}"
            )
            target_link_libraries(TBB:tbb PUBLIC Threads::Threads ${CMAKE_DL_LIBS})
        endif()
    endif()
    mark_as_advanced(TBB_INCLUDE_DIR)

    unset(_tbb_search_path)
    unset(_tbb_lib_suffixes)
    unset(_libtbb)
    unset(_libtbbmalloc)
endif()
