set(HEADERS
    swcio.hpp
)
set(BASE_SOURCES
    cell.cpp
    mechanism_interface.cpp
    parameter_list.cpp
    swcio.cpp
)

add_library(cellalgo ${BASE_SOURCES} ${HEADERS})
