#include <fvm_multicell.hpp>
#include <hardware/gpu.hpp>

#include "validate_kinetic.hpp"

#include "../gtest.h"

using namespace nest::mc;

TEST(kinetic, kin1_numeric_ref) {
    validate_kinetic_kin1(backend_kind::multicore);
    if (hw::num_gpus()) {
        validate_kinetic_kin1(nest::mc::backend_kind::gpu);
    }
}

TEST(kinetic, kinlva_numeric_ref) {
    validate_kinetic_kinlva(backend_kind::multicore);
    if (hw::num_gpus()) {
        validate_kinetic_kinlva(nest::mc::backend_kind::gpu);
    }
}
