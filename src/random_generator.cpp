#include <random_generator.hpp>

namespace arb {
namespace random_generator {

float sample_poisson(double lambda, unsigned counter, unsigned key) {
    RNG::ctr_type c = {{}};
    RNG::key_type k = {{}};

    k[0] = key;
    c.v[0] = counter;

    RNG::ctr_type r = sample_randomly(c, k);
    float unif = r123::u01<float>(r.v[0]);
    float sample = -lambda * log(1 - unif);

    return sample;
    }
} // namespace random_generator
} // namespace arb
