set(IPSS_SOURCES
    ipss_exe.cpp
)

add_executable(ipss.exe ${IPSS_SOURCES})

target_link_libraries(ipss.exe LINK_PUBLIC ${ARB_LIBRARIES})
target_link_libraries(ipss.exe LINK_PUBLIC ${EXTERNAL_LIBRARIES})

set_target_properties(
    ipss.exe
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/example/ipss"
)

install(FILES parse_and_plot.py
        DESTINATION "${CMAKE_BINARY_DIR}/example/ipss")
