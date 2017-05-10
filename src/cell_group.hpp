#pragma once

#include <memory>
#include <vector>

#include <cell.hpp>
#include <common_types.hpp>
#include <event_binner.hpp>
#include <event_queue.hpp>
#include <probes.hpp>
#include <sampler_function.hpp>
#include <spike.hpp>
#include <util/optional.hpp>
#include <util/make_unique.hpp>

namespace nest {
namespace mc {

class cell_group {
public:
    virtual ~cell_group() = default;

    virtual cell_kind get_cell_kind() const = 0;

    virtual void reset() = 0;
    virtual void set_binning_policy(binning_kind policy, time_type bin_interval) = 0;
    virtual void advance(time_type tfinal, time_type dt) = 0;
    virtual void enqueue_events(const std::vector<postsynaptic_spike_event>& events) = 0;
    virtual const std::vector<spike>& spikes() const = 0;
    virtual void clear_spikes() = 0;
    virtual void add_sampler(cell_member_type probe_id, sampler_function s, time_type start_time = 0) = 0;
    virtual const std::vector<probe_record>& probes() const = 0;
};

using cell_group_ptr = std::unique_ptr<cell_group>;

template <typename T, typename... Args>
cell_group_ptr make_cell_group(Args&&... args) {
    return cell_group_ptr(new T(std::forward<Args>(args)...));
}

} // namespace mc
} // namespace nest
