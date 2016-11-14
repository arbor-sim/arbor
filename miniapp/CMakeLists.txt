set(HEADERS
)
set(MINIAPP_SOURCES
    miniapp.cpp
    io.cpp
    miniapp_recipes.cpp
)
set(MINIAPP_SOURCES_CUDA
    miniapp.cu
    io.cpp
    miniapp_recipes.cpp
)

if(WITH_CUDA)
    cuda_add_executable(miniapp.exe ${MINIAPP_SOURCES_CUDA} ${HEADERS})
    target_link_libraries(miniapp.exe LINK_PUBLIC gpu)
else()
    add_executable(miniapp.exe ${MINIAPP_SOURCES} ${HEADERS})
endif()

target_link_libraries(miniapp.exe LINK_PUBLIC nestmc)
target_link_libraries(miniapp.exe LINK_PUBLIC ${EXTERNAL_LIBRARIES})

if(WITH_MPI)
    target_link_libraries(miniapp.exe LINK_PUBLIC ${MPI_C_LIBRARIES})
    set_property(TARGET miniapp.exe APPEND_STRING PROPERTY LINK_FLAGS "${MPI_C_LINK_FLAGS}")
endif()

set_target_properties(miniapp.exe
   PROPERTIES
   RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/miniapp"
)

