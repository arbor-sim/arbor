#pragma once

#include <mutex>
#include <vector>

#include <arbor/export.hpp>
#include <arbor/common_types.hpp>
#include <arbor/recipe.hpp>
#include <arbor/sampling.hpp>
#include <arbor/spike.hpp>

#include "cell_group.hpp"
#include "epoch.hpp"
#include "fvm_lowered_cell.hpp"
#include "label_resolution.hpp"
#include "sampler_map.hpp"
#include "timestep_range.hpp"

namespace arb {


struct ARB_ARBOR_API cable_cell_group: public cell_group {
    cable_cell_group() = default;
    cable_cell_group(const std::vector<cell_gid_type>& gids,
                     const recipe& rec,
                     cell_label_range& cg_sources,
                     cell_label_range& cg_targets,
                     fvm_lowered_cell_ptr lowered);

    cell_kind get_cell_kind() const override { return cell_kind::cable; }

    void reset() override;

    void advance(epoch ep, time_type dt, const event_lane_subrange& event_lanes) override;

    const std::vector<spike>& spikes() const override { return spikes_; }

    void clear_spikes() override { spikes_.clear(); }

    void add_sampler(sampler_association_handle h, cell_member_predicate probeset_ids,
                     schedule sched, sampler_function fn) override;

    void remove_sampler(sampler_association_handle h) override;

    void remove_all_samplers() override;

    std::vector<probe_metadata> get_probe_metadata(const cell_address_type&) const override;

    ARB_SERDES_ENABLE(cable_cell_group, gids_, spikes_, lowered_);

    void t_serialize(serializer& ser, const std::string& k) const override;
    void t_deserialize(serializer& ser, const std::string& k) override;
private:
    // List of the gids of the cells in the group.
    std::vector<cell_gid_type> gids_;

    // The lowered cell state (e.g. FVM) of the cell.
    fvm_lowered_cell_ptr lowered_;

    // Spike detectors attached to the cell.
    std::vector<cell_member_type> spike_sources_;

    // Spikes that are generated.
    std::vector<spike> spikes_;

    // Range of timesteps within current epoch
    timestep_range timesteps_;

    // List of samples to be taken
    std::vector<std::vector<sample_event>> sample_events_;

    // Maps probe ids to probe handles (from lowered cell) and tags (from probe descriptions).
    probe_association_map probe_map_;

    // Collection of samplers to be run against probes in this group.
    sampler_association_map sampler_map_;

    // Mutex for thread-safe access to sampler associations.
    std::mutex sampler_mex_;
};

} // namespace arb
