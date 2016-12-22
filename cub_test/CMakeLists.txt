if(WITH_CUDA)
    cuda_add_executable(cub_test test.cu)

    target_link_libraries(cub_test LINK_PUBLIC nestmc)
    target_link_libraries(cub_test LINK_PUBLIC gpu)
    target_link_libraries(cub_test LINK_PUBLIC ${EXTERNAL_LIBRARIES})

    set_target_properties(cub_test
       PROPERTIES
       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    )
endif()
