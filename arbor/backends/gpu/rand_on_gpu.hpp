#pragma once

#include <util/pimpl_src.hpp>
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
    sarray sindices_;
    std::vector<arb_size_type*> prng_indices_;
    memory::device_vector<arb_size_type*> prng_indices_d_;

    impl(arb_value_type* data, mechanism& m, std::size_t value_width_padded,
        const mechanism_layout& pos_data, arb_seed_type seed) :
        random_numbers_{data},
        value_width_padded_{value_width_padded},
        seed_{seed} {

        sindices_ = sarray(2*value_width_padded);
        chunk_writer writer(sindices_.data(), value_width_padded);
        prng_indices_.resize(2);
        prng_indices_[0] = writer.append_with_padding(pos_data.gid, 0);
        prng_indices_[1] = writer.append_with_padding(pos_data.idx, 0);
        prng_indices_d_ = memory::on_gpu(prng_indices_);
    }

    void update(mechanism& m, cbprng::value_type counter) {
        const auto num_rv = m.mech_.n_random_variables;
        const auto width = m.ppack_.width;
        const cbprng::value_type mech_id = m.mechanism_id();
        // generate random numbers on the device
        generate_random_numbers(random_numbers_, width, value_width_padded_, num_rv, seed_,
            mech_id, counter, prng_indices_[0], prng_indices_[1]);
    }
};

} // namespace gpu
} // namespace arb
