set(unit-distributed_sources
    distributed_listener.cpp
    test_domain_decomposition.cpp
    test_communicator.cpp
    test_mpi.cpp

    # unit test driver
    test.cpp
)

add_executable(unit-local ${unit-distributed_sources})
target_compile_options(unit-local PRIVATE ${ARB_CXXOPT_ARCH})
target_compile_definitions(unit-local PRIVATE TEST_LOCAL)
target_link_libraries(unit-local PRIVATE gtest arbor arbor-anc arbor-private-headers)

if(ARB_WITH_MPI)
    add_executable(unit-mpi ${unit-distributed_sources})
    target_compile_options(unit-mpi PRIVATE ${ARB_CXXOPT_ARCH})
    target_compile_definitions(unit-mpi PRIVATE TEST_MPI)
    target_link_libraries(unit-mpi PRIVATE gtest arbor arbor-anc arbor-private-headers)
endif()

