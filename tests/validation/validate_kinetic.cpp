#include "validate_kinetic.hpp"

#include "../gtest.h"

using lowered_cell = nest::mc::fvm::fvm_multicell<nest::mc::multicore::backend>;

TEST(kinetic, kin1_numeric_ref) {
    validate_kinetic_kin1<lowered_cell>();
}

TEST(kinetic, kinlva_numeric_ref) {
    validate_kinetic_kinlva<lowered_cell>();
}
