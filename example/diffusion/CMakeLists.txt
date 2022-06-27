add_executable(diffusion EXCLUDE_FROM_ALL diffusion.cpp)
add_dependencies(examples diffusion)
target_link_libraries(diffusion PRIVATE arbor arborio arborenv arbor-sup ext-tinyopt)
