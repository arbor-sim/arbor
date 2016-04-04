set(HEADERS
    swcio.hpp
    )
set(BASE_SOURCES
    parameter_list.cpp
    swcio.cpp
    cell.cpp
)

add_library(cellalgo ${BASE_SOURCES} ${HEADERS})
