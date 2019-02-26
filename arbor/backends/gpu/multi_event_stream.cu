#include <arbor/common_types.hpp>

#include "backends/event.hpp"
#include "backends/gpu/multi_event_stream.hpp"
#include "cuda_common.hpp"

namespace arb {
namespace gpu {

namespace kernels {
    template <typename T, typename I>
    __global__ void mark_until_after(
        unsigned n,
        I* mark,
        const I* span_end,
        const T* ev_time,
        const T* t_until)
    {
        unsigned i = threadIdx.x+blockIdx.x*blockDim.x;
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
        unsigned n,
        I* mark,
        const I* span_end,
        const T* ev_time,
        const T* t_until)
    {
        unsigned i = threadIdx.x+blockIdx.x*blockDim.x;
        if (i<n) {
            auto t = t_until[i];
            auto end = span_end[i];
            auto &m = mark[i];

            while (m!=end && t>ev_time[m]) {
                ++m;
            }
        }
    }

    template <typename I>
    __global__ void drop_marked_events(
        unsigned n,
        I* n_nonempty,
        I* span_begin,
        const I* span_end,
        const I* mark)
    {
        unsigned i = threadIdx.x+blockIdx.x*blockDim.x;
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
        unsigned n,
        const I* span_begin,
        const I* span_end,
        const T* ev_time,
        T* t_until)
    {
        unsigned i = threadIdx.x+blockIdx.x*blockDim.x;
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

// Designate for processing events `ev` at head of each event stream `i`
// until `event_time(ev)` > `t_until[i]`.
void multi_event_stream_base::mark_until_after(const_view t_until) {
    arb_assert(n_streams()==t_until.size());

    constexpr int block_dim = 128;

    unsigned n = n_stream_;
    int nblock = impl::block_count(n, block_dim);
    kernels::mark_until_after<<<nblock, block_dim>>>(
        n, mark_.data(), span_end_.data(), ev_time_.data(), t_until.data());
}

// Designate for processing events `ev` at head of each event stream `i`
// while `t_until[i]` > `event_time(ev)`.
void multi_event_stream_base::mark_until(const_view t_until) {
    arb_assert(n_streams()==t_until.size());
    constexpr int block_dim = 128;

    unsigned n = n_stream_;
    int nblock = impl::block_count(n, block_dim);
    kernels::mark_until<<<nblock, block_dim>>>(
        n, mark_.data(), span_end_.data(), ev_time_.data(), t_until.data());
}

// Remove marked events from front of each event stream.
void multi_event_stream_base::drop_marked_events() {
    constexpr int block_dim = 128;

    unsigned n = n_stream_;
    int nblock = impl::block_count(n, block_dim);
    kernels::drop_marked_events<<<nblock, block_dim>>>(
        n, n_nonempty_stream_.data(), span_begin_.data(), span_end_.data(), mark_.data());
}

// If the head of `i`th event stream exists and has time less than `t_until[i]`, set
// `t_until[i]` to the event time.
void multi_event_stream_base::event_time_if_before(view t_until) {
    constexpr int block_dim = 128;
    int nblock = impl::block_count(n_stream_, block_dim);
    kernels::event_time_if_before<<<nblock, block_dim>>>(
        n_stream_, span_begin_.data(), span_end_.data(), ev_time_.data(), t_until.data());
}


} // namespace gpu
} // namespace arb
