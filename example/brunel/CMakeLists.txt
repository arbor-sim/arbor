set(HEADERS
    io.hpp
    partitioner.hpp
)
set(MINIAPP_SOURCES
    brunel_miniapp.cpp
    io.cpp
)

add_executable(brunel_miniapp.exe ${MINIAPP_SOURCES} ${HEADERS})

target_link_libraries(brunel_miniapp.exe LINK_PUBLIC ${ARB_LIBRARIES})
target_link_libraries(brunel_miniapp.exe LINK_PUBLIC ${EXTERNAL_LIBRARIES})

if(ARB_WITH_MPI)
    target_link_libraries(brunel_miniapp.exe LINK_PUBLIC ${MPI_C_LIBRARIES})
    set_property(TARGET brunel_miniapp.exe APPEND_STRING PROPERTY LINK_FLAGS "${MPI_C_LINK_FLAGS}")
endif()

set_target_properties(brunel_miniapp.exe
   PROPERTIES
   RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/example/miniapp/brunel"
)
