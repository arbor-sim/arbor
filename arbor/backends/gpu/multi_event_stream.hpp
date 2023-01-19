#pragma once

#include "backends/multicore/multi_event_stream.hpp"
#include "memory/memory.hpp"

namespace arb {
namespace gpu {

class multi_event_stream : public ::arb::multicore::multi_event_stream {
public:
    using base = ::arb::multicore::multi_event_stream;

    using size_type = typename base::size_type;
    using event_type = typename base::event_type;

    using event_time_type = typename base::event_time_type;
    using event_data_type = typename base::event_data_type;
    using event_index_type = typename base::event_index_type;


    void init(const std::vector<event_type>& staged, const timestep_range& dts) {
        base::init(staged, dts);
        device_ranges_ = memory::make_view(base::ranges_);
        device_ev_data_ = memory::make_view(base::ev_data_);
    }

    arb_deliverable_event_stream marked_events() const {
        if (base::empty()) return {0, nullptr, nullptr};
        return {
            base::num_streams_[base::index_],
            device_ev_data_.data(),
            device_ranges_.data() + base::stream_lookup_[base::index_]
        };
    }

private:
    memory::device_vector<arb_deliverable_event_range> device_ranges_;
    memory::device_vector<event_data_type> device_ev_data_;
};

} // namespace gpu
} // namespace arb

