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

    using device_ranges_array = memory::device_vector<arb_deliverable_event_range>;
    using device_data_array = memory::device_vector<event_data_type>;

    void init(const std::vector<event_type>& staged, const timestep_range& dts) {
        base::init(staged, dts);
        copy(base::ranges_, device_ranges_);
        copy(base::ev_data_, device_ev_data_);
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
    device_ranges_array device_ranges_;
    device_data_array device_ev_data_;
};

} // namespace gpu
} // namespace arb

