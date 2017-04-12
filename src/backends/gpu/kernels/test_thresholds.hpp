#pragma once

namespace nest {
namespace mc {
namespace gpu {

/// Cuda lerp by u on [a,b]: (1-u)*a + u*b.
template <typename T>
__host__ __device__
inline T lerp(T a, T b, T u) {
    return std::fma(u, b, std::fma(-u, a, a));
}

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
    const I* cv_to_cell, const T* t_after, const T* t_before,
    int size,
    Stack& stack,
    I* is_crossed, T* prev_values,
    const I* cv_index, const T* values, const T* thresholds)
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
                crossing_time = lerp(t_before[cell], t_after[cell], pos);

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
