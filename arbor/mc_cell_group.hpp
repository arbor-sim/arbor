#pragma once

#include <cstdint>
#include <functional>
#include <iterator>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <arbor/export.hpp>
#include <arbor/common_types.hpp>
#include <arbor/recipe.hpp>
#include <arbor/sampling.hpp>
#include <arbor/spike.hpp>

#include "backends/event.hpp"
#include "cell_group.hpp"
#include "epoch.hpp"
#include "event_binner.hpp"
#include "event_queue.hpp"
#include "fvm_lowered_cell.hpp"
#include "label_resolution.hpp"
#include "sampler_map.hpp"

namespace arb {

class ARB_ARBOR_API mc_cell_group: public cell_group {
public:
    mc_cell_group() = default;

    mc_cell_group(const std::vector<cell_gid_type>& gids,
                  const recipe& rec,
                  cell_label_range& cg_sources,
                  cell_label_range& cg_targets,
                  fvm_lowered_cell_ptr lowered);

    cell_kind get_cell_kind() const override {
        return cell_kind::cable;
    }

    void reset() override;

    void set_binning_policy(binning_kind policy, time_type bin_interval) override;

    void advance(epoch ep, time_type dt, const event_lane_subrange& event_lanes) override;

    const std::vector<spike>& spikes() const override {
        return spikes_;
    }

    void clear_spikes() override {
        spikes_.clear();
    }

    void add_sampler(sampler_association_handle h, cell_member_predicate probeset_ids,
                     schedule sched, sampler_function fn, sampling_policy policy) override;

    void remove_sampler(sampler_association_handle h) override;

    void remove_all_samplers() override;

    std::vector<probe_metadata> get_probe_metadata(cell_member_type probeset_id) const override;

private:
    // List of the gids of the cells in the group.
    std::vector<cell_gid_type> gids_;

    // Map from gid to integration domain id
    std::vector<arb_index_type> cell_to_intdom_;

    // Hash table for converting gid to local index
    std::unordered_map<cell_gid_type, cell_gid_type> gid_index_map_;

    // The lowered cell state (e.g. FVM) of the cell.
    fvm_lowered_cell_ptr lowered_;

    // Spike detectors attached to the cell.
    std::vector<cell_member_type> spike_sources_;

    // Spikes that are generated.
    std::vector<spike> spikes_;

    // Event time binning manager.
    std::vector<event_binner> binners_;

    // List of events to deliver
    std::vector<deliverable_event> staged_events_;

    // Pending samples to be taken.
    event_queue<sample_event> sample_events_;

    // Handles for accessing lowered cell.
    std::vector<target_handle> target_handles_;

    // Maps probe ids to probe handles (from lowered cell) and tags (from probe descriptions).
    probe_association_map probe_map_;

    // Collection of samplers to be run against probes in this group.
    sampler_association_map sampler_map_;

    // Mutex for thread-safe access to sampler associations.
    std::mutex sampler_mex_;

    // Lookup table for target ids -> local target handle indices.
    std::vector<std::size_t> target_handle_divisions_;
};

} // namespace arb
