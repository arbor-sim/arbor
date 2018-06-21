#include <arbor/fvm_types.hpp>

#include "cuda_atomic.hpp"
#include "cuda_common.hpp"
#include "stimulus.hpp"

namespace arb {
namespace gpu {

namespace kernel {
    __global__
    void stimulus_current_impl(int n, stimulus_pp pp) {
        auto i = threadIdx.x + blockDim.x*blockIdx.x;
        if (i<n) {
            auto t = pp.vec_t_[pp.vec_ci_[i]];
            if (t>=pp.delay[i] && t<pp.delay[i]+pp.duration[i]) {
                // use subtraction because the electrode currents are specified
                // in terms of current into the compartment
                cuda_atomic_add(pp.vec_i_+pp.node_index_[i], -pp.weight_[i]*pp.amplitude[i]);
            }
        }
    }
} // namespace kernel


void stimulus_current_impl(int n, const stimulus_pp& pp) {
    constexpr unsigned block_dim = 128;
    const unsigned grid_dim = impl::block_count(n, block_dim);

    kernel::stimulus_current_impl<<<grid_dim, block_dim>>>(n, pp);
}

} // namespace gpu
} // namespace arb
