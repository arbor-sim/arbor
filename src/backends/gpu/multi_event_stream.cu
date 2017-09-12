#include <common_types.hpp>
#include <backends/event.hpp>
#include <backends/gpu/multi_event_stream.hpp>
#include <memory/array.hpp>
#include <memory/copy.hpp>
#include <util/rangeutil.hpp>

namespace nest {
namespace mc {
namespace gpu {

namespace kernels {
    template <typename T, typename I>
    __global__ void mark_until_after(
        I n,
        I* mark,
        const I* span_end,
        const T* ev_time,
        const T* t_until)
    {
        I i = threadIdx.x+blockIdx.x*blockDim.x;
        if (i<n) {
            auto t = t_until[i];
            auto end = span_end[i];
            auto &m = mark[i];

            while (m!=end && !(ev_time[m]>t)) {
                ++m;
            }
        }
    }

    template <typename T, typename I>
    __global__ void mark_until(
        I n,
        I* mark,
        const I* span_end,
        const T* ev_time,
        const T* t_until)
    {
        I i = threadIdx.x+blockIdx.x*blockDim.x;
        if (i<n) {
            auto t = t_until[i];
            auto end = span_end[i];
            auto &m = mark[i];

            while (m!=end && t>ev_time[m]) {
                ++m;
            }
        }
    }

    template <typename T, typename I>
    __global__ void drop_marked_events(
        I n,
        I* n_nonempty,
        I* span_begin,
        const I* span_end,
        const I* mark)
    {
        I i = threadIdx.x+blockIdx.x*blockDim.x;
        if (i<n) {
            bool emptied = (span_begin[i]<span_end[i] && mark[i]==span_end[i]);
            span_begin[i] = mark[i];
            if (emptied) {
                atomicAdd(n_nonempty, (I)-1);
            }
        }
    }

    template <typename T, typename I>
    __global__ void event_time_if_before(
        I n,
        const I* span_begin,
        const I* span_end,
        const T* ev_time,
        T* t_until)
    {
        I i = threadIdx.x+blockIdx.x*blockDim.x;
        if (i<n) {
            if (span_begin[i]<span_end[i]) {
                auto ev_t = ev_time[span_begin[i]];
                if (t_until[i]>ev_t) {
                    t_until[i] = ev_t;
                }
            }
        }
    }
} // namespace kernels

void multi_event_stream_base::clear() {
    memory::fill(span_begin_, 0u);
    memory::fill(span_end_, 0u);
    memory::fill(mark_, 0u);
    n_nonempty_stream_[0] = 0;
}

// Designate for processing events `ev` at head of each event stream `i`
// until `event_time(ev)` > `t_until[i]`.
void multi_event_stream_base::mark_until_after(const_view t_until) {
    EXPECTS(n_streams()==util::size(t_until));

    constexpr int blockwidth = 128;
    int nblock = 1+(n_stream_-1)/blockwidth;
    kernels::mark_until_after<value_type, size_type><<<nblock, blockwidth>>>(
        n_stream_, mark_.data(), span_end_.data(), ev_time_.data(), t_until.data());
}

// Designate for processing events `ev` at head of each event stream `i`
// while `t_until[i]` >  `event_time(ev)`.
void multi_event_stream_base::mark_until_after(const_view t_until) {
    EXPECTS(n_streams()==util::size(t_until));

    constexpr int blockwidth = 128;
    int nblock = 1+(n_stream_-1)/blockwidth;
    kernels::mark_until<value_type, size_type><<<nblock, blockwidth>>>(
        n_stream_, mark_.data(), span_end_.data(), ev_time_.data(), t_until.data());
}

// Remove marked events from front of each event stream.
void multi_event_stream_base::drop_marked_events() {
    constexpr int blockwidth = 128;
    int nblock = 1+(n_stream_-1)/blockwidth;
    kernels::drop_marked_events<value_type, size_type><<<nblock, blockwidth>>>(
        n_stream_, n_nonempty_stream_.data(), span_begin_.data(), span_end_.data(), mark_.data());
}

// If the head of `i`th event stream exists and has time less than `t_until[i]`, set
// `t_until[i]` to the event time.
void multi_event_stream_base::event_time_if_before(view t_until) {
    constexpr int blockwidth = 128;
    int nblock = 1+(n_stream_-1)/blockwidth;
    kernels::event_time_if_before<value_type, size_type><<<nblock, blockwidth>>>(
        n_stream_, span_begin_.data(), span_end_.data(), ev_time_.data(), t_until.data());
}


} // namespace gpu
} // namespace nest
} // namespace mc
