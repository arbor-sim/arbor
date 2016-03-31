set(HEADERS
    swcio.hpp
    )
set(BASE_SOURCES
    swcio.cpp
    cell.cpp
)

add_library(cellalgo ${BASE_SOURCES} ${HEADERS})
