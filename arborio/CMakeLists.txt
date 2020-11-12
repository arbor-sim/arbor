set(arborio-sources
    swcio.cpp
)

add_library(arborio ${arborio-sources})

add_library(arborio-public-headers INTERFACE)
target_include_directories(arborio-public-headers INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(arborio PUBLIC arbor arborio-public-headers)
target_link_libraries(arborio PRIVATE arbor-config-defs arborio-private-deps)

install(DIRECTORY include/arborio
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp")

install(TARGETS arborio-public-headers EXPORT arbor-targets)
install(TARGETS arborio EXPORT arbor-targets ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
