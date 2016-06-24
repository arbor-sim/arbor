set(HEADERS
    swcio.hpp
)
set(BASE_SOURCES
    cell.cpp
    mechanism_interface.cpp
    parameter_list.cpp
    swcio.cpp
)
if(${WITH_MPI})
    set(BASE_SOURCES ${BASE_SOURCES} communication/mpi.cpp)
endif()

add_library(cellalgo ${BASE_SOURCES} ${HEADERS})
add_dependencies(cellalgo build_all_mods)

