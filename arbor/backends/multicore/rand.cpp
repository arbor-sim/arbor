#include "backends/rand_impl.hpp"

namespace arb {
namespace multicore {

void generate_random_numbers(
    arb_value_type* dst,        // points to random number storage
    std::size_t width,          // number of sites
    std::size_t width_padded,   // padded number of sites
    std::size_t num_rv,         // number of random variables
    cbprng::value_type seed,    // simulation seed value
    cbprng::value_type mech_id, // mechanism id
    cbprng::value_type counter, // step counter
    arb_size_type const * gid,  // global cell ids (size = width)
    arb_size_type const * idx   // per-cell location index (size = width)
    ) {
    for (std::size_t n=0; n<num_rv; ++n) {
        for (std::size_t i=0; i<width; ++i) {
            const auto r = cbprng::generator{}({seed, mech_id, n, counter},
                    {gid[i], idx[i], 0xdeadf00dull, 0xdeadbeefull});
            const auto [a0, a1] = r123::boxmuller(r[0], r[1]);
            const auto [a2, a3] = r123::boxmuller(r[2], r[3]);
            dst[i + width_padded*(0 + cbprng::cache_size()*n)] = a0;
            dst[i + width_padded*(1 + cbprng::cache_size()*n)] = a1;
            dst[i + width_padded*(2 + cbprng::cache_size()*n)] = a2;
            dst[i + width_padded*(3 + cbprng::cache_size()*n)] = a3;
        }
    }
}

} // namespace multicore
} // namespace arb

