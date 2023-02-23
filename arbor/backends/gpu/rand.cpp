#include <util/span.hpp>
#include <backends/gpu/chunk_writer.hpp>
#include <backends/gpu/rand.hpp>

#ifdef ARB_ARBOR_NO_GPU_RAND
#include <backends/gpu/rand_on_cpu.hpp>
#else
#include <backends/gpu/rand_on_gpu.hpp>
#endif

ARB_INSTANTIATE_PIMPL(arb::gpu::random_numbers::impl)

namespace arb {
namespace gpu {

void random_numbers::instantiate(mechanism& m, std::size_t value_width_padded,
    const mechanism_layout& pos_data, arb_seed_type seed) {
    using util::make_span;

    // bail out if there are no random variables
    if (m.mech_.n_random_variables == 0) return;

    // Allocate view pointers for random nubers
    std::size_t num_random_numbers_per_cv = m.mech_.n_random_variables;
    std::size_t random_number_storage = num_random_numbers_per_cv*cbprng::cache_size();
    for (auto& v : random_numbers_) {
        v.resize(num_random_numbers_per_cv);
    }

    // Allocate bulk storage
    std::size_t count = random_number_storage*value_width_padded;
    data_ = array(count, NAN);

    // Set random numbers
    chunk_writer writer(data_.data(), value_width_padded);
    for (auto idx_v: make_span(num_random_numbers_per_cv)) {
        for (auto idx_c: make_span(cbprng::cache_size())) {
            random_numbers_[idx_c][idx_v] = writer.fill(0);
        }
    }

    // Shift data to GPU
    for (auto idx_c: make_span(cbprng::cache_size()))
        random_numbers_d_[idx_c] = memory::on_gpu(random_numbers_[idx_c]);

    // Instantiate implementation details
    impl_ = util::make_pimpl<impl>(random_numbers_[0][0], m, value_width_padded, pos_data, seed);
}

void random_numbers::update(mechanism& m) {
    // bail out if there are no random variables
    if (!impl_) return;

    // Assign new random numbers by selecting the next cache
    const auto counter = random_number_update_counter_++;
    const auto cache_idx = cbprng::cache_index(counter);
    m.ppack_.random_numbers = random_numbers_d_[cache_idx].data();

    // Generate random numbers every cbprng::cache_size() iterations:
    // For each random variable we will generate cbprng::cache_size() values per site
    // and there are width sites.
    // The RNG will be seeded by a global seed, the mechanism id, the variable index, the
    // current site's global cell, the site index within its cell and a counter representing
    // time.
    if (cache_idx == 0) impl_->update(m, counter);
}

} // namespace gpu
} // namespace arb
