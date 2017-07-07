#include "validate_kinetic.hpp"

#include "../gtest.h"

const auto backend = nest::mc::backend_policy::multicore;

TEST(kinetic, kin1_numeric_ref) {
    validate_kinetic_kin1(backend);
}

TEST(kinetic, kinlva_numeric_ref) {
    validate_kinetic_kinlva(backend);
}
