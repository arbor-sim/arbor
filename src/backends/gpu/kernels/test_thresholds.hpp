#pragma once

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
template <typename T, typename I, typename Stack>
__global__
void test_thresholds(
    float t, float t_prev, int size,
    Stack& stack,
    I* is_crossed, T* prev_values,
    const I* index, const T* values, const T* thresholds)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;

    bool crossed = false;
    float crossing_time;

    if (i<size) {
        // Test for threshold crossing
        const auto v_prev = prev_values[i];
        const auto v      = values[index[i]];
        const auto thresh = thresholds[i];

        if (!is_crossed[i]) {
            if (v>=thresh) {
                // The threshold has been passed, so estimate the time using
                // linear interpolation
                auto pos = (thresh - v_prev)/(v - v_prev);
                crossing_time = t_prev + pos*(t - t_prev);

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
        stack.push_back({I(i), crossing_time});
    }
}

} // namespace gpu
} // namespace mc
} // namespace nest
