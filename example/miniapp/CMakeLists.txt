add_executable(miniapp
    miniapp.cpp
    io.cpp
    miniapp_recipes.cpp
    miniapp_symmetric_recipe.cpp
    morphology_pool.cpp
    trace.cpp
)

target_link_libraries(miniapp PRIVATE arbor arbor-aux ext-tclap ext-json)
