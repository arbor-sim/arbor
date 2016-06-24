set(HEADERS
)
set(MINIAPP_SOURCES
    mpi.cpp
    io.cpp
    miniapp.cpp
)

add_executable(miniapp.exe ${MINIAPP_SOURCES} ${HEADERS})

target_link_libraries(miniapp.exe LINK_PUBLIC cellalgo)
target_link_libraries(miniapp.exe LINK_PUBLIC ${TBB_LIBRARIES})

if(WITH_MPI)
    target_link_libraries(miniapp.exe LINK_PUBLIC ${MPI_C_LIBRARIES})
    set_property(TARGET miniapp.exe APPEND_STRING PROPERTY LINK_FLAGS "${MPI_C_LINK_FLAGS}")
endif()

set_target_properties(miniapp.exe
   PROPERTIES
   RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/miniapp"
)

