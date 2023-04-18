find_package(Threads REQUIRED)

# Convenience target that builds all tests.
# Test executable targets should be added to the 'tests' target as dependencies.
add_custom_target(tests)

# Unit tests.
# Builds: unit.
add_subdirectory(unit)

# Test MPI wrappers and distribution operations.
# Builds: unit-local and unit-mpi (if MPI enabled).
add_subdirectory(unit-distributed)

# Test modcc internals.
# Builds: unit-modcc.
add_subdirectory(unit-modcc)

# Microbenchmarks.
# Builds: one executable per microbenchmark under ubench/.
add_subdirectory(ubench)

# GNU Make expects 'check' target to execute tests:
# https://www.gnu.org/software/make/manual/make.html#Standard-Targets
# Spack's CMakePackage in turn relies on this:
# https://spack.readthedocs.io/en/latest/packaging_guide.html#adding-installation-phase-tests
add_custom_target(check COMMAND unit)
add_dependencies(check tests)
