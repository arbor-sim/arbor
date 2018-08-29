add_executable(dryrun
    dryrun.cpp
    io.cpp
    dryrun_recipes.cpp
)

target_link_libraries(dryrun PRIVATE arbor arbor-aux ext-tclap ext-json)
