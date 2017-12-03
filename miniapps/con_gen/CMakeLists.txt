set(CON_GEN_SOURCES
    con_gen.cpp
)

add_executable(con_gen.exe ${CON_GEN_SOURCES})

target_link_libraries(con_gen.exe LINK_PUBLIC ${ARB_LIBRARIES})
target_link_libraries(con_gen.exe LINK_PUBLIC ${EXTERNAL_LIBRARIES})

set_target_properties(
    con_gen.exe
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/miniapps/con_gen"
)

install(FILES parse_and_plot.py
        DESTINATION "${CMAKE_BINARY_DIR}/miniapps/con_gen")
