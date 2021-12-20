unit_distributed_sources = [
  'distributed_listener.cpp',
  'test_domain_decomposition.cpp',
  'test_communicator.cpp',
  'test_mpi.cpp',

  # unit test driver
  'test.cpp',
]

# The distributed unit tests are all configured to be compiled with either a
# local (non-distributed) communicator, or with an MPI communicator.
# A version of the tests with the local communicatior is always configured and
# built, and an MPI version is optionally built if MPI support is enabled.


unit_local = executable('unit-local',
                   sources: unit_distributed_sources,
                   dependencies: [dep_arb, dep_arborenv, dep_arborsup, dep_gtest, dep_tinyopt],
                   include_directories: [inc_arb_private, inc_arb_public],
                   cpp_args: ['-DTEST_LOCAL'] + arbor_private_args,
                   install: false,
                   build_by_default: false)

if arb_with_mpi
  unit_mpi = executable('unit-mpi',
                     sources: unit_distributed_sources,
                     dependencies: [dep_arb, dep_arborenv, dep_arborsup, dep_gtest, dep_tinyopt],
                     include_directories: [inc_arb_private, inc_arb_public],
                     cpp_args: ['-DTEST_MPI'] + arbor_private_args,
                     install: false,
                     build_by_default: false)
endif
