#include <backends/fvm_types.hpp>
#include <backends/gpu/intrinsics.hpp>

namespace arb{
namespace gpu {

namespace kernels {
    template <typename T, typename I>
    __global__
    void stim_current(
        const T* delay, const T* duration, const T* amplitude, const T* weights,
        const I* node_index, int n, const I* cell_index, const T* time, T* current)
    {
        using value_type = T;
        using iarray = I;

        auto i = threadIdx.x + blockDim.x*blockIdx.x;

        if (i<n) {
            auto t = time[cell_index[i]];
            if (t>=delay[i] && t<delay[i]+duration[i]) {
                // use subtraction because the electrode currents are specified
                // in terms of current into the compartment
                cuda_atomic_add(current+node_index[i], -weights[i]*amplitude[i]);
            }
        }
    }
} // namespace kernels


void stim_current(
    const fvm_value_type* delay, const fvm_value_type* duration,
    const fvm_value_type* amplitude, const fvm_value_type* weights,
    const fvm_size_type* node_index, int n,
    const fvm_size_type* cell_index, const fvm_value_type* time,
    fvm_value_type* current)
{
    constexpr unsigned thread_dim = 192;
    dim3 dim_block(thread_dim);
    dim3 dim_grid((n+thread_dim-1)/thread_dim);

    kernels::stim_current<fvm_value_type, fvm_size_type><<<dim_grid, dim_block>>>
        (delay, duration, amplitude, weights, node_index, n, cell_index, time, current);

}

} // namespace gpu
} // namespace arb
