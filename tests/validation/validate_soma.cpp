#include <fvm_multicell.hpp>
#include <hardware/gpu.hpp>

#include "validate_soma.hpp"

#include "../gtest.h"

using namespace nest::mc;

TEST(soma, numeric_ref) {
    validate_soma(backend_kind::multicore);
    if (hw::num_gpus()) {
        validate_soma(backend_kind::gpu);
    }
}
