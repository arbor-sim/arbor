add_executable(remote EXCLUDE_FROM_ALL remote.cpp)
add_executable(server EXCLUDE_FROM_ALL server.cpp)
add_dependencies(examples remote server)

target_link_libraries(remote PRIVATE arbor arborio arborenv arbor-sup)
target_link_libraries(server PRIVATE arbor arborio arborenv arbor-sup)
