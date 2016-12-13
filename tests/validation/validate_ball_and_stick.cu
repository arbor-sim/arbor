#include "../gtest.h"
#include "validate_ball_and_stick.hpp"

#include <fvm_multicell.hpp>

using lowered_cell = nest::mc::fvm::fvm_multicell<nest::mc::gpu::backend>;

TEST(ball_and_stick, neuron_ref) {
    validate_ball_and_stick<lowered_cell>();
}

TEST(ball_and_taper, neuron_ref) {
    validate_ball_and_taper<lowered_cell>();
}

TEST(ball_and_3stick, neuron_ref) {
    validate_ball_and_3stick<lowered_cell>();
}

TEST(rallpack1, numeric_ref) {
    validate_rallpack1<lowered_cell>();
}

TEST(ball_and_squiggle, neuron_ref) {
    validate_ball_and_squiggle<lowered_cell>();
}
