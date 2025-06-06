set(unit-distributed_sources
    distributed_listener.cpp
    test_domain_decomposition.cpp
    test_communicator.cpp
    test_mpi.cpp
    test_distributed_for_each.cpp
    test_network_generation.cpp
    # unit test driver
    test.cpp
)

add_executable(unit-local EXCLUDE_FROM_ALL ${unit-distributed_sources})
add_dependencies(tests unit-local)

target_compile_options(unit-local PRIVATE ${ARB_CXX_FLAGS_TARGET_FULL})
target_compile_definitions(unit-local PRIVATE TEST_LOCAL)
target_link_libraries(unit-local PRIVATE gtest gtest_main arbor arborenv arborio arbor-sup arbor-private-headers ext-tinyopt)
target_include_directories(unit-local PRIVATE $<BUILD_INTERFACE:${unordered_dense_SOURCE_DIR}/include>)

if(ARB_WITH_MPI)
    add_executable(unit-mpi EXCLUDE_FROM_ALL ${unit-distributed_sources})
    add_dependencies(tests unit-mpi)
    target_include_directories(unit-mpi PRIVATE $<BUILD_INTERFACE:${unordered_dense_SOURCE_DIR}/include>)

    target_compile_options(unit-mpi PRIVATE ${ARB_CXX_FLAGS_TARGET_FULL})
    target_compile_definitions(unit-mpi PRIVATE TEST_MPI)
    target_link_libraries(unit-mpi PRIVATE gtest gtest_main arbor arborenv arborio arbor-sup arbor-private-headers ext-tinyopt)
endif()
