set(HEADERS
)

set(DISK_IO_SOURCES
    disk_io.cpp
)

add_executable(disk_io.exe ${DISK_IO_SOURCES} ${HEADERS})

target_link_libraries(disk_io.exe LINK_PUBLIC nestmc)

if(WITH_TBB)
    target_link_libraries(disk_io.exe LINK_PUBLIC ${TBB_LIBRARIES})
endif()

if(WITH_MPI)
    target_link_libraries(disk_io.exe LINK_PUBLIC ${MPI_C_LIBRARIES})
    set_property(TARGET disk_io.exe APPEND_STRING PROPERTY LINK_FLAGS "${MPI_C_LINK_FLAGS}")
endif()

# Copy the python file that drives the performance tests and produces the output
file(COPY disk_io.py DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
