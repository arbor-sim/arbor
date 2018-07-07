add_executable(brunel-miniapp
    brunel_miniapp.cpp
    io.cpp)

target_link_libraries(brunel-miniapp PRIVATE arbor arbor-aux ext-tclap)

# TODO: resolve public headers
target_link_libraries(brunel-miniapp PRIVATE arbor-private-headers)
