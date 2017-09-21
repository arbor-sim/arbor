#include <fvm_multicell.hpp>
#include <hardware/gpu.hpp>

#include "../gtest.h"
#include "validate_ball_and_stick.hpp"

using namespace nest::mc;

TEST(ball_and_stick, neuron_ref) {
    validate_ball_and_stick(backend_kind::multicore);
    if (hw::num_gpus()) {
        validate_ball_and_stick(backend_kind::gpu);
    }
}

TEST(ball_and_taper, neuron_ref) {
    validate_ball_and_taper(backend_kind::multicore);
    if (hw::num_gpus()) {
        validate_ball_and_taper(backend_kind::gpu);
    }
}

TEST(ball_and_3stick, neuron_ref) {
    validate_ball_and_3stick(backend_kind::multicore);
    if (hw::num_gpus()) {
        validate_ball_and_3stick(backend_kind::gpu);
    }
}

TEST(rallpack1, numeric_ref) {
    validate_rallpack1(backend_kind::multicore);
    if (hw::num_gpus()) {
        validate_rallpack1(backend_kind::gpu);
    }
}

TEST(ball_and_squiggle, neuron_ref) {
    validate_ball_and_squiggle(backend_kind::multicore);
    if (hw::num_gpus()) {
        validate_ball_and_squiggle(backend_kind::gpu);
    }
}
