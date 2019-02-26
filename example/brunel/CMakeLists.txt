add_executable(brunel-miniapp EXCLUDE_FROM_ALL
    brunel_miniapp.cpp
    io.cpp)
add_dependencies(examples brunel-miniapp)

target_link_libraries(brunel-miniapp PRIVATE arbor arborenv arbor-sup ext-tclap)
