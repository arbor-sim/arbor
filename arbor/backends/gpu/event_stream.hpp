#pragma once

// Indexed collection of pop-only event queues --- CUDA back-end implementation.

#include "backends/multicore/event_stream.hpp"
#include "memory/memory.hpp"

namespace arb {
namespace gpu {

template <typename Event>
class event_stream : public ::arb::multicore::event_stream<Event> {
public:
    using base = ::arb::multicore::event_stream<Event>;

    using size_type = typename base::size_type;
    using event_type = typename base::event_type;

    using event_time_type = typename base::event_time_type;
    using event_data_type = typename base::event_data_type;

    using state = typename base::state;

    void init(const std::vector<Event>& staged, const timestep_range& dts) {
        base::init(staged, dts);
        // copy to GPU
        copy(base::ev_data_, device_ev_data_);
    }

    state marked_events() const {
        if (base::empty()) return {nullptr, nullptr};
        return {device_ev_data_.data()+base::offsets_[base::index_-1], device_ev_data_.data()+base::offsets_[base::index_]};
    }

private:
    template<typename H, typename D>
    static void copy(const H& h, D& d) {
        const auto s = h.size();
        // resize if necessary
        if (d.size() < s) {
            d = D(s);
        }
        memory::copy_async(h, memory::make_view(d)(0u, s));
    }

private:
    memory::device_vector<event_data_type> device_ev_data_;
};

} // namespace gpu
} // namespace arb
