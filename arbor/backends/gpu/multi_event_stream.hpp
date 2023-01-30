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

    void init(const mechanism_event_map& staged, const timestep_range& dts) {
        base::init(staged, dts);
        if (!base::ev_data_.size()) return;

        // calculate lookup array (exclusive scan over size(base::ranges))
        stream_lookup_.clear();
        stream_lookup_.reserve(base::num_dt_);
        stream_lookup_.push_back(0u);
        for (size_type i = 1u; i < num_dt_; ++i) {
            const auto n = base::ranges_[i-1].size();
            const auto sum = stream_lookup_.back()+n;
            stream_lookup_.push_back(size_type(sum));
        }
        const size_type total_streams = stream_lookup_.back() + base::ranges_[num_dt_-1].size();

        // transfer the ranges to a contiguous array;
        ranges_.clear();
        ranges_.reserve(total_streams);
        for (size_type i = 0u; i < num_dt_; ++i) {
            ranges_.insert(ranges_.end(), base::ranges_[i].begin(), base::ranges_[i].end());
        }
        arb_assert(ranges_.size() == total_streams);

        // copy to GPU
        copy(ranges_, device_ranges_);
        copy(base::ev_data_, device_ev_data_);
    }

    arb_deliverable_event_stream marked_events() const {
        if (base::empty()) return {0, nullptr, nullptr};
        return {
            size_type(base::ranges_[base::index_-1].size()),
            device_ev_data_.data(),
            device_ranges_.data() + stream_lookup_[base::index_-1]
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
    std::vector<size_type> stream_lookup_;
    std::vector<arb_deliverable_event_range> ranges_;
    device_ranges_array device_ranges_;
    device_data_array device_ev_data_;
};

} // namespace gpu
} // namespace arb

