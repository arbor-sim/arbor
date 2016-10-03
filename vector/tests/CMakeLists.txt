set(DRIVER_SOURCES
    driver.cpp
    host_coordinator_unittest.cpp
    array_unittest.cpp
    array_reference_unittest.cpp
    host_vector_unittest.cpp
    allocator_unittest.cpp
    array_view_unittest.cpp
    split_range_unittest.cpp
    gtest-all.cc
)
set(DRIVER_CUDA_SOURCES
    device_vector_unittest.cpp
    device_coordinator_unittest.cpp
    pinned_vector_unittest.cpp
)
set(DRIVER_KNL_SOURCES
    knl_vector_unittest.cpp
)

if( CUDA_BACKEND )
    set(DRIVER_SOURCES ${DRIVER_SOURCES} ${DRIVER_CUDA_SOURCES})
    add_executable(driver.cuda ${DRIVER_SOURCES})
    target_link_libraries(driver.cuda LINK_PUBLIC gpu)
elseif( KNL_BACKEND )
    set(DRIVER_SOURCES ${DRIVER_SOURCES} ${DRIVER_KNL_SOURCES})
    add_executable(driver.knl ${DRIVER_SOURCES})
else()
    add_executable(driver.exe ${DRIVER_SOURCES})
endif()


