#include <backends/fvm_types.hpp>

#include "detail.hpp"
#include "stack.hpp"

namespace nest {
namespace mc {
namespace gpu {

/// kernel used to test for threshold crossing test code.
/// params:
///     t       : current time (ms)
///     t_prev  : time of last test (ms)
///     size    : number of values to test
///     is_crossed  : crossing state at time t_prev (true or false)
///     prev_values : values at sample points (see index) sampled at t_prev
///     index      : index with locations in values to test for crossing
///     values     : values at t_prev
///     thresholds : threshold values to watch for crossings
__global__
void test_thresholds_kernel(
    const fvm_size_type* cv_to_cell, const fvm_value_type* t_after, const fvm_value_type* t_before,
    int size,
    stack_base<threshold_crossing>& stack,
    fvm_size_type* is_crossed, fvm_value_type* prev_values,
    const fvm_size_type* cv_index, const fvm_value_type* values, const fvm_value_type* thresholds)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;

    bool crossed = false;
    float crossing_time;

    if (i<size) {
        // Test for threshold crossing
        const auto cv     = cv_index[i];
        const auto cell   = cv_to_cell[cv];
        const auto v_prev = prev_values[i];
        const auto v      = values[cv];
        const auto thresh = thresholds[i];

        if (!is_crossed[i]) {
            if (v>=thresh) {
                // The threshold has been passed, so estimate the time using
                // linear interpolation
                auto pos = (thresh - v_prev)/(v - v_prev);
                crossing_time = impl::lerp(t_before[cell], t_after[cell], pos);

                is_crossed[i] = 1;
                crossed = true;
            }
        }
        else if (v<thresh) {
            is_crossed[i]=0;
        }

        prev_values[i] = v;
    }

    if (crossed) {
        push_back(stack, {fvm_size_type(i), crossing_time});
    }
}

void test_thresholds(
    const fvm_size_type* cv_to_cell, const fvm_value_type* t_after, const fvm_value_type* t_before,
    int size,
    stack_base<threshold_crossing>& stack,
    fvm_size_type* is_crossed, fvm_value_type* prev_values,
    const fvm_size_type* cv_index, const fvm_value_type* values, const fvm_value_type* thresholds)
{
    constexpr int block_dim = 128;
    const int grid_dim = impl::block_count(size, block_dim);
    test_thresholds_kernel<<<grid_dim, block_dim>>>(
        cv_to_cell, t_after, t_before, size, stack, is_crossed, prev_values, cv_index, values, thresholds);
}

} // namespace gpu
} // namespace mc
} // namespace nest
