set(HEADERS
)
set(MINIAPP_SOURCES
    io.cpp
    miniapp.cpp
)

add_executable(miniapp.exe ${MINIAPP_SOURCES} ${HEADERS})

target_link_libraries(miniapp.exe LINK_PUBLIC cellalgo)
target_link_libraries(miniapp.exe LINK_PUBLIC ${TBB_LIBRARIES})

set_target_properties(miniapp.exe
   PROPERTIES
   RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/miniapp"
)

