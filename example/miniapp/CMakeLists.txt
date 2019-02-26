add_executable(miniapp EXCLUDE_FROM_ALL
    miniapp.cpp
    io.cpp
    miniapp_recipes.cpp
    morphology_pool.cpp
    trace.cpp
)
add_dependencies(examples miniapp)

target_link_libraries(miniapp PRIVATE arbor arborenv arbor-sup ext-tclap ext-json)
