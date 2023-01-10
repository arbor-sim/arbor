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
    using size_type = typename base::size_type;

    void init(const std::vector<Event>& staged) {
        base::init(staged);
        device_ev_data_ = memory::on_gpu(base::ev_data_);
        device_ev_time_ = memory::on_gpu(base::ev_time_);
        device_span_begin_ = memory::on_gpu(base::span_begin_);
        std::vector<size_type> tmp(base::offsets_.begin()+1, base::offsets_.end());
        device_span_end_ = memory::on_gpu(tmp);
    }

    state marked_events() /*const*/ {
        return {
            base::n_streams(),
            base::n_marked(),
            device_ev_data_.data(),
            device_span_begin_.data(),
            device_span_end_.data(),
            device_ev_time_.data(),
            base::t_start_,
            base::t_end_
        };
    }

private:
    memory::device_vector<event_data_type> device_ev_data_;
    memory::device_vector<double> device_ev_time_;
    memory::device_vector<size_type> device_span_begin_;
    memory::device_vector<size_type> device_span_end_;
};

} // namespace gpu
} // namespace arb
