#pragma once

#include <util/pimpl_src.hpp>
#include <memory/gpu_wrappers.hpp>
#include <backends/gpu/rand.hpp>

namespace arb {
namespace gpu {

struct random_numbers::impl {
    // pointer to random number device storage
    arb_value_type* random_numbers_;

    // general parameters
    std::size_t value_width_padded_;
    arb_seed_type seed_;

    // auxillary random number device storage
    std::vector<arb_value_type> random_numbers_h_;
    std::vector<arb_size_type> gid_;
    std::vector<arb_size_type> idx_;

    impl(arb_value_type* data, mechanism& m, std::size_t value_width_padded,
        const mechanism_layout& pos_data, arb_seed_type seed) :
        random_numbers_{data},
        value_width_padded_{value_width_padded},
        seed_{seed},
        random_numbers_h_(m.mech_.n_random_variables * cbprng::cache_size() * value_width_padded),
        gid_{pos_data.gid},
        idx_{pos_data.idx} {}

    void update(mechanism& m, cbprng::value_type counter) {
        const auto num_rv = m.mech_.n_random_variables;
        const auto width = m.ppack_.width;
        const cbprng::value_type mech_id = m.mechanism_id();
        // generate random numbers on the host
        arb_value_type* dst = random_numbers_h_.data();
        arb::multicore::generate_random_numbers(dst, width, value_width_padded_, num_rv, seed_,
            mech_id, counter, gid_.data(), idx_.data());
        // transfer values to device
        memory::gpu_memcpy_h2d(random_numbers_, dst,
            (num_rv*cbprng::cache_size()*value_width_padded_)*sizeof(arb_value_type));
    }
};

} // namespace gpu
} // namespace arb
