set(unit-distributed_sources
    distributed_listener.cpp
    test_domain_decomposition.cpp
    test_communicator.cpp
    test_mpi.cpp

    # unit test driver
    test.cpp
)

add_executable(unit-local EXCLUDE_FROM_ALL ${unit-distributed_sources})
add_dependencies(tests unit-local)

target_compile_options(unit-local PRIVATE ${ARB_CXX_FLAGS_TARGET_FULL})
target_compile_definitions(unit-local PRIVATE TEST_LOCAL)
target_link_libraries(unit-local PRIVATE ext-gtest arbor arborenv arbor-sup arbor-private-headers ext-tinyopt)

if(ARB_WITH_MPI)
    add_executable(unit-mpi EXCLUDE_FROM_ALL ${unit-distributed_sources})
    add_dependencies(tests unit-mpi)

    target_compile_options(unit-mpi PRIVATE ${ARB_CXX_FLAGS_TARGET_FULL})
    target_compile_definitions(unit-mpi PRIVATE TEST_MPI)
    target_link_libraries(unit-mpi PRIVATE ext-gtest arbor arborenv arbor-sup arbor-private-headers ext-tinyopt)
endif()

