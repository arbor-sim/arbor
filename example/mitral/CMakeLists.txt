add_executable(mitral mitral.cpp)

target_link_libraries(mitral PRIVATE arbor arbor-sup ext-json)
