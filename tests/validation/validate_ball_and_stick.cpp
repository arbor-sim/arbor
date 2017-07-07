#include "../gtest.h"
#include "validate_ball_and_stick.hpp"

#include <fvm_multicell.hpp>

const auto backend = nest::mc::backend_policy::multicore;

TEST(ball_and_stick, neuron_ref) {
    validate_ball_and_stick(backend);
}

TEST(ball_and_taper, neuron_ref) {
    validate_ball_and_taper(backend);
}

TEST(ball_and_3stick, neuron_ref) {
    validate_ball_and_3stick(backend);
}

TEST(rallpack1, numeric_ref) {
    validate_rallpack1(backend);
}

TEST(ball_and_squiggle, neuron_ref) {
    validate_ball_and_squiggle(backend);
}
