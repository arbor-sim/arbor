#include <common_types.hpp>
#include <backends/event.hpp>
#include <backends/gpu/multi_event_stream.hpp>
#include <memory/array.hpp>
#include <memory/copy.hpp>
#include <util/rangeutil.hpp>

namespace nest {
namespace mc {
namespace gpu {

namespace kernel {
    template <typename T, typename I>
    __global__ void mark_until_after(
        I n,
        I* mark,
        const I* span_end,
        const T* ev_time,
        const T* t_until)
    {
        I i = threadIdx.x+blockIdx.x*blockDim.x;
        if (i>=n) return;

        auto t = t_until[i];
        auto end = span_end[i];
        auto &m = mark[i];

        while (m!=end && !(ev_time[m]>t)) { ++m; }
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
        if (i>=n) return;

        bool emptied = (span_begin[i]<span_end[i] && mark[i]==span_end[i]);
        span_begin[i] = mark[i];
        if (emptied) {
            atomicAdd(n_nonempty, (cell_size_type)-1);
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
        if (i>=n) return;

        if (span_begin[i]<span_end[i]) {
            auto ev_t = ev_time[span_begin[i]];
            if (t_until[i]>ev_t) {
                t_until[i] = ev_t;
            }
        }
    }
} // namespace kernel

void multi_event_stream::clear() {
    memory::fill(span_begin_, 0u);
    memory::fill(span_end_, 0u);
    memory::fill(mark_, 0u);
    n_nonempty_stream_[0] = 0;
}

void multi_event_stream::init(const std::vector<deliverable_event>& staged) {
    if (staged.size()>std::numeric_limits<size_type>::max()) {
        throw std::range_error("too many events");
    }

    // Build vectors in host memory here and transfer to the device at end.
    std::vector<deliverable_event> ev(staged);
    std::size_t n_ev = ev.size();

    std::vector<size_type> divisions(n_stream_+1, 0);
    std::vector<size_type> tmp_ev_indices(n_ev);
    std::vector<value_type> tmp_ev_values(n_ev);

    ev_time_ = array(n_ev);
    ev_weight_ = array(n_ev);
    ev_mech_id_ = iarray(n_ev);
    ev_index_ = iarray(n_ev);

    util::stable_sort_by(ev, [](const deliverable_event& e) { return e.handle.cell_index; });

    // Split out event fields and copy to device.
    util::assign_by(tmp_ev_values, ev, [](const deliverable_event& e) { return e.weight; });
    memory::copy(tmp_ev_values, ev_weight_);

    util::assign_by(tmp_ev_values, ev, [](const deliverable_event& e) { return e.time; });
    memory::copy(tmp_ev_values, ev_time_);

    util::assign_by(tmp_ev_indices, ev, [](const deliverable_event& e) { return e.handle.mech_id; });
    memory::copy(tmp_ev_indices, ev_mech_id_);

    util::assign_by(tmp_ev_indices, ev, [](const deliverable_event& e) { return e.handle.index; });
    memory::copy(tmp_ev_indices, ev_index_);

    // Determine divisions by `cell_index` in ev list and copy to device.
    size_type ev_i = 0;
    size_type n_nonempty = 0;
    for (size_type s = 1; s<=n_stream_; ++s) {
        while (ev_i<n_ev && ev[ev_i].handle.cell_index<s) ++ev_i;
        divisions[s] = ev_i;
        n_nonempty += (divisions[s]!=divisions[s-1]);
    }

    memory::copy(memory::make_view(divisions)(0,n_stream_), span_begin_);
    memory::copy(memory::make_view(divisions)(1,n_stream_+1), span_end_);
    memory::copy(span_begin_, mark_);
    n_nonempty_stream_[0] = n_nonempty;
}


// Designate for processing events `ev` at head of each event stream `i`
// until `event_time(ev)` > `t_until[i]`.
void multi_event_stream::mark_until_after(const_view t_until) {
    EXPECTS(n_streams()==util::size(t_until));

    constexpr int blockwidth = 128;
    int nblock = 1+(n_stream_-1)/blockwidth;
    kernel::mark_until_after<value_type, size_type><<<nblock, blockwidth>>>(
        n_stream_, mark_.data(), span_end_.data(), ev_time_.data(), t_until.data());
}

// Remove marked events from front of each event stream.
void multi_event_stream::drop_marked_events() {
    constexpr int blockwidth = 128;
    int nblock = 1+(n_stream_-1)/blockwidth;
    kernel::drop_marked_events<value_type, size_type><<<nblock, blockwidth>>>(
        n_stream_, n_nonempty_stream_.data(), span_begin_.data(), span_end_.data(), mark_.data());
}

// If the head of `i`th event stream exists and has time less than `t_until[i]`, set
// `t_until[i]` to the event time.
void multi_event_stream::event_time_if_before(view t_until) {
    constexpr int blockwidth = 128;
    int nblock = 1+(n_stream_-1)/blockwidth;
    kernel::event_time_if_before<value_type, size_type><<<nblock, blockwidth>>>(
        n_stream_, span_begin_.data(), span_end_.data(), ev_time_.data(), t_until.data());
}


} // namespace gpu
} // namespace nest
} // namespace mc
