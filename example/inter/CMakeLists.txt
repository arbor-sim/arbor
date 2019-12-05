add_executable(arb_inter EXCLUDE_FROM_ALL arb.cpp)
add_executable(nest_proxy EXCLUDE_FROM_ALL nest.cpp)
add_executable(arbor_proxy EXCLUDE_FROM_ALL arbor_proxy.cpp)

configure_file(arbor_proxy.py  ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)

add_dependencies(examples arb_inter)
add_dependencies(examples nest_proxy)
add_dependencies(examples arbor_proxy)

target_link_libraries(arb_inter PRIVATE arbor arborenv arbor-sup ext-json)
target_link_libraries(nest_proxy PRIVATE arbor arborenv arbor-sup ext-json)
target_link_libraries(arbor_proxy PRIVATE arbor arborenv arbor-sup ext-json)

