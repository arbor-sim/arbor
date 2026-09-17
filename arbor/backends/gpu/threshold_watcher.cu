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
///     t           : current time (ms)
///     t_prev      : time of last test (ms)
///     size        : number of values to test
///     is_crossed  : crossing state at time t_prev (true or false)
///     prev_values : values at sample points (see index) sampled at t_prev
///     index       : index with locations in values to test for crossing
///     values      : values at t_prev
///     thresholds  : threshold values to watch for crossings
__global__
void test_thresholds_impl(int size,
                          arb_value_type t_after,
                          arb_value_type t_before,
                          stack_storage<threshold_crossing>& stack,
                          arb_index_type*       __restrict__ const is_crossed,
                          arb_value_type*       __restrict__ const prev_values,
                          const arb_index_type* __restrict__ const cv_index,
                          const arb_value_type* __restrict__ const values,
                          const arb_value_type* __restrict__ const thresholds) {
    int tid = threadIdx.x + blockIdx.x*blockDim.x;

    if (tid < size) {
        // Test for threshold crossing
        const auto cv     = cv_index[tid];
        const auto v_prev = prev_values[cv];
        const auto v      = values[cv];
        const auto thresh = thresholds[tid];

        // We check this to detect the rising flank only
        // TODO: Why don't we just rely on v_prev here?
        //       We could save 32B x num_detectors in memory, three memory accesses, and
        //       a potentially diverging branch.
        if (!is_crossed[tid]) {
            if (v >= thresh) {
                // The threshold has been passed, so estimate the time using
                // linear interpolation
                auto pos = (thresh - v_prev)/(v - v_prev);
                auto crossing_time = gpu::lerp(t_before, t_after, pos);

                is_crossed[tid] = 1;
                push_back(stack, {arb_size_type(tid), crossing_time});
            }
        }
        else if (v < thresh) {
            is_crossed[tid] = 0;
        }
        prev_values[cv] = v;
    }
}

/// kernel used to test for threshold crossing test code.
/// params:
///     t           : current time (ms)
///     t_prev      : time of last test (ms)
///     size        : number of values to test
///     is_crossed  : crossing state at time t_prev (true or false)
///     prev_values : values at sample points (see index) sampled at t_prev
///     index       : index with locations in values to test for crossing
///     values      : values at t_prev
///     thresholds  : threshold values to watch for crossings
__global__
void test_thresholds_record_impl(int size,
                                 arb_value_type t_after,
                                 arb_value_type t_before,
                                 const arb_index_type* __restrict__ const src_to_spike,
                                 arb_value_type*       __restrict__ const time_since_spike,
                                 stack_storage<threshold_crossing>& stack,
                                 arb_index_type*       __restrict__ const is_crossed,
                                 arb_value_type*       __restrict__ const prev_values,
                                 const arb_index_type* __restrict__ const cv_index,
                                 const arb_value_type* __restrict__ const values,
                                 const arb_value_type* __restrict__ const thresholds) {
    int tid = threadIdx.x + blockIdx.x*blockDim.x;

    if (tid < size) {
        // Test for threshold crossing
        const auto cv     = cv_index[tid];
        const auto v_prev = prev_values[cv];
        const auto v      = values[cv];
        const auto thresh = thresholds[tid];

        // Reset all spike times to -1.0 indicating no spike has been recorded on the detector
        auto spike_idx = src_to_spike[tid];
        time_since_spike[spike_idx] = -1.0;

        // We check this to detect the rising flank only
        // TODO: Why don't we just rely on v_prev here?
        //       We could save 32B x num_detectors in memory, three memory accesses, and
        //       a potentially diverging branch.
        if (!is_crossed[tid]) {
            if (v >= thresh) {
                // Estimate the crossing time
                auto pos = (thresh - v_prev)/(v - v_prev);
                auto crossing_time = gpu::lerp(t_before, t_after, pos);

                // record the spike time
                time_since_spike[spike_idx] = t_after - crossing_time;

                is_crossed[tid] = 1;
                push_back(stack, {arb_size_type(tid), crossing_time});
            }
        }
        else if (v < thresh) {
            is_crossed[tid] = 0;
        }
        prev_values[cv] = v;
    }
}

__global__
extern void reset_crossed_impl(int size,
                               arb_index_type*       __restrict__ const is_crossed,
                               const arb_index_type* __restrict__ const cv_index,
                               const arb_value_type* __restrict__ const values,
                               const arb_value_type* __restrict__ const thresholds) {
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    if (tid < size) {
        is_crossed[tid] = values[cv_index[tid]] >= thresholds[tid];
    }
}

} // namespace kernel

void test_thresholds_impl(int size,
                          const arb_value_type t_after,
                          const arb_value_type t_before,
                          stack_storage<threshold_crossing>& stack,
                          arb_index_type* is_crossed,
                          arb_value_type* prev_values,
                          const arb_index_type* cv_index,
                          const arb_value_type* values,
                          const arb_value_type* thresholds,
                          bool record_time_since_spike) {
    launch_1d(size, 128, kernel::test_thresholds_impl,
              size, t_after, t_before, stack, is_crossed, prev_values, cv_index, values, thresholds);
}

void test_thresholds_record_impl(int size,
                                 const arb_value_type t_after,
                                 const arb_value_type t_before,
                                 const arb_index_type* src_to_spike,
                                 arb_value_type* time_since_spike,
                                 stack_storage<threshold_crossing>& stack,
                                 arb_index_type* is_crossed,
                                 arb_value_type* prev_values,
                                 const arb_index_type* cv_index,
                                 const arb_value_type* values,
                                 const arb_value_type* thresholds) {
    launch_1d(size, 128, kernel::test_thresholds_record_impl,
              size, t_after, t_before, src_to_spike, time_since_spike, stack, is_crossed, prev_values, cv_index, values, thresholds);
}

    
void reset_crossed_impl(int size,
                        arb_index_type* is_crossed,
                        const arb_index_type* cv_index,
                        const arb_value_type* values,
                        const arb_value_type* thresholds) {
    launch_1d(size, 128, kernel::reset_crossed_impl,
              size, is_crossed, cv_index, values, thresholds);
}

} // namespace gpu
} // namespace arb
