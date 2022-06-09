add_executable(lfp EXCLUDE_FROM_ALL lfp.cpp)
add_dependencies(examples lfp)
target_link_libraries(lfp PRIVATE arbor arborio ext-tinyopt)
file(COPY plot-lfp.py DESTINATION "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
