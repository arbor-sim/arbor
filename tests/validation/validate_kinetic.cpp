#include "validate_kinetic.hpp"

#include "../gtest.h"

TEST(kinetic, kin1_numeric_ref) {
    validate_kinetic_kin1(nest::mc::backend_kind::multicore);
    validate_kinetic_kin1(nest::mc::backend_kind::gpu);
}

TEST(kinetic, kinlva_numeric_ref) {
    validate_kinetic_kinlva(nest::mc::backend_kind::multicore);
    validate_kinetic_kinlva(nest::mc::backend_kind::gpu);
}
