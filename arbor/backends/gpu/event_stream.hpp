#pragma once

#include "backends/event_stream.hpp"
#include "memory/memory.hpp"

namespace arb {
namespace gpu {

template <typename Event>
class event_stream : public ::arb::event_stream<Event> {
public:
    using base = ::arb::event_stream<Event>;
    using event_data_type = typename base::event_data_type;
    using state = typename base::state;

    void init(const std::vector<Event>& staged) {
        base::init(staged);

        device_ev_data_ = memory::on_gpu(base::ev_data_);
    }

    state marked_events() const {
        return {device_ev_data_.data(), base::span_begin_vec_.data(), base::span_end_vec_.data(),
            (arb_size_type)base::ev_kind_.size(), base::marked_};
    }

private:
    memory::device_vector<event_data_type> device_ev_data_;
};

} // namespace gpu
} // namespace arb
