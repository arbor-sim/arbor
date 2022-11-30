#pragma once

#include "backends/multi_event_stream.hpp"
#include "memory/memory.hpp"

namespace arb {
namespace gpu {

template <typename Event>
class multi_event_stream : public ::arb::multi_event_stream<Event> {
public:
    using base = ::arb::multi_event_stream<Event>;
    using event_data_type = typename base::event_data_type;
    using state = typename base::state;

    void init(const std::vector<Event>& staged) {
        base::init(staged);

        device_ev_data_ = memory::on_gpu(base::ev_data_);
    }

    state marked_events() const {
        return {base::n_streams(), base::n_marked(), device_ev_data_.data(), base::span_begin_.data(), base::span_end_.data()};
    }

private:
    memory::device_vector<event_data_type> device_ev_data_;
};

} // namespace gpu
} // namespace arb
