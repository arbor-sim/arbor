add_executable(arb_inter arb.cpp)
add_executable(nest_inter nest.cpp)
add_executable(arbor_proxy arbor_proxy.cpp)
add_executable(intercomm intercomm.cpp)

target_link_libraries(arb_inter PRIVATE arbor arborenv arbor-sup ext-json)
target_link_libraries(nest_inter PRIVATE arbor arborenv arbor-sup ext-json)
target_link_libraries(arbor_proxy PRIVATE arbor arborenv arbor-sup ext-json)
target_link_libraries(intercomm PRIVATE arbor)
