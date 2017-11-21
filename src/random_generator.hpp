#pragma once
#include <math.h>
#include <random123/threefry.h>
#include <random123/uniform.hpp>

#define sample_randomly threefry2x32

namespace arb {
namespace random_generator {
    using RNG = r123::Threefry2x32;

    float sample_poisson(double lambda, unsigned counter, unsigned key);
} // namespace random_generator
} // namespace arb 
