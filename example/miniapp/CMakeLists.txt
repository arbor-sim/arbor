add_executable(miniapp
    miniapp.cpp
    io.cpp
    miniapp_recipes.cpp
    morphology_pool.cpp
    trace.cpp
)

target_link_libraries(miniapp PRIVATE arbor arbor-aux ext-tclap ext-json)

# TODO: resolve public headers
target_link_libraries(miniapp PRIVATE arbor-private-headers)
