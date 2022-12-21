#pragma once

#include <array>
#include <vector>

#include <arbor/mechanism.hpp>

#include <util/pimpl.hpp>
#include <backends/rand_fwd.hpp>
#include <backends/gpu/gpu_store_types.hpp>

namespace arb {
namespace gpu {

class random_numbers {
public:
    void instantiate(mechanism& m, std::size_t value_width_padded, const mechanism_layout& pos_data,
        arb_seed_type seed);

    void update(mechanism& m);

  private:
    // random number device storage
    array data_;
    std::array<std::vector<arb_value_type*>, cbprng::cache_size()>           random_numbers_;
    std::array<memory::device_vector<arb_value_type*>, cbprng::cache_size()> random_numbers_d_;

    // time step counter
    cbprng::counter_type random_number_update_counter_ = 0u;

    // pimpl
    struct impl;
    util::pimpl<impl> impl_;
};

} // namespace gpu
} // namespace arb
