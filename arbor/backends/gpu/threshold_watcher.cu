#include <cmath>

#include <arbor/fvm_types.hpp>
#include <arbor/gpu/math_cu.hpp>

#include "backends/threshold_crossing.hpp"
#include "stack_cu.hpp"

namespace arb {
namespace gpu {

namespace kernel {

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
void test_thresholds_impl(
    int size,
    const arb_index_type* __restrict__ const cv_to_intdom,
    const arb_value_type* __restrict__ const t_after,
    const arb_value_type* __restrict__ const t_before,
    const arb_index_type* __restrict__ const src_to_spike,
    arb_value_type* __restrict__ const time_since_spike,
    stack_storage<threshold_crossing>& stack,
    arb_index_type* __restrict__ const is_crossed,
    arb_value_type* __restrict__ const prev_values,
    const arb_index_type* __restrict__ const cv_index,
    const arb_value_type* __restrict__ const values,
    const arb_value_type* __restrict__ const thresholds,
    bool record_time_since_spike)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;

    bool crossed = false;
    float crossing_time;

    if (i<size) {
        // Test for threshold crossing
        const auto cv     = cv_index[i];
        const auto intdom = cv_to_intdom[cv];
        const auto v_prev = prev_values[cv];
        const auto v      = values[cv];
        const auto thresh = thresholds[i];
        arb_index_type spike_idx = 0;

        // Reset all spike times to -1.0 indicating no spike has been recorded on the detector
        if (record_time_since_spike) {
            spike_idx = src_to_spike[i];
            time_since_spike[spike_idx] = -1.0;
        }
        if (!is_crossed[i]) {
            if (v>=thresh) {
                // The threshold has been passed, so estimate the time using
                // linear interpolation
                auto pos = (thresh - v_prev)/(v - v_prev);
                crossing_time = gpu::lerp(t_before[intdom], t_after[intdom], pos);

                if (record_time_since_spike) {
                    time_since_spike[spike_idx] = t_after[intdom] - crossing_time;
                }

                is_crossed[i] = 1;
                crossed = true;
            }
        }
        else if (v<thresh) {
            is_crossed[i]=0;
        }

        prev_values[cv] = v;
    }

    if (crossed) {
        push_back(stack, {arb_size_type(i), crossing_time});
    }
}

__global__
extern void reset_crossed_impl(
    int size,
    arb_index_type* __restrict__ const is_crossed,
    const arb_index_type* __restrict__ const cv_index,
    const arb_value_type* __restrict__ const values,
    const arb_value_type* __restrict__ const thresholds)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if (i<size) {
        is_crossed[i] = values[cv_index[i]] >= thresholds[i];
    }
}

} // namespace kernel

void test_thresholds_impl(
    int size,
    const arb_index_type* cv_to_intdom, const arb_value_type* t_after, const arb_value_type* t_before,
    const arb_index_type* src_to_spike, arb_value_type* time_since_spike, stack_storage<threshold_crossing>& stack,
    arb_index_type* is_crossed, arb_value_type* prev_values,
    const arb_index_type* cv_index, const arb_value_type* values, const arb_value_type* thresholds,
    bool record_time_since_spike)
{
    if (size>0) {
        constexpr int block_dim = 128;
        const int grid_dim = impl::block_count(size, block_dim);
        kernel::test_thresholds_impl<<<grid_dim, block_dim>>>(
            size, cv_to_intdom, t_after, t_before, src_to_spike, time_since_spike,
            stack, is_crossed, prev_values, cv_index, values, thresholds, record_time_since_spike);
    }
}

void reset_crossed_impl(
    int size, arb_index_type* is_crossed,
    const arb_index_type* cv_index, const arb_value_type* values, const arb_value_type* thresholds)
{
    if (size>0) {
        constexpr int block_dim = 128;
        const int grid_dim = impl::block_count(size, block_dim);
        kernel::reset_crossed_impl<<<grid_dim, block_dim>>>(size, is_crossed, cv_index, values, thresholds);
    }
}

} // namespace gpu
} // namespace arb
