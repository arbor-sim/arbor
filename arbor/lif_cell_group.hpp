#pragma once

#include <vector>
#include <mutex>

#include <arbor/export.hpp>
#include <arbor/common_types.hpp>
#include <arbor/lif_cell.hpp>
#include <arbor/recipe.hpp>
#include <arbor/sampling.hpp>
#include <arbor/spike.hpp>

#include "sampler_map.hpp"
#include "cell_group.hpp"
#include "label_resolution.hpp"

namespace arb {

class ARB_ARBOR_API lif_cell_group: public cell_group {
public:
    using value_type = double;

    lif_cell_group() = default;

    // Constructor containing gid of first cell in a group and a container of all cells.
    lif_cell_group(const std::vector<cell_gid_type>& gids, const recipe& rec, cell_label_range& cg_sources, cell_label_range& cg_targets);

    virtual cell_kind get_cell_kind() const override;
    virtual void reset() override;
    virtual void set_binning_policy(binning_kind policy, time_type bin_interval) override;
    virtual void advance(epoch epoch, time_type dt, const event_lane_subrange& events) override;

    virtual const std::vector<spike>& spikes() const override;
    virtual void clear_spikes() override;

    // Sampler association methods below should be thread-safe, as they might be invoked
    // from a sampler call back called from a different cell group running on a different thread.
    virtual void add_sampler(sampler_association_handle, cell_member_predicate, schedule, sampler_function, sampling_policy) override;
    virtual void remove_sampler(sampler_association_handle) override;
    virtual void remove_all_samplers() override;

    virtual std::vector<probe_metadata> get_probe_metadata(cell_member_type) const override;

private:
    enum class lif_probe_kind { voltage };

    struct lif_probe_info {
        probe_tag tag;
        lif_probe_kind kind;
        lif_probe_metadata metadata;
    };


    // Advances a single cell (lid) with the exact solution (jumps can be arbitrary).
    // Parameter dt is ignored, since we make jumps between two consecutive spikes.
    void advance_cell(time_type tfinal, time_type dt, cell_gid_type lid, const event_lane_subrange& event_lane);

    // List of the gids of the cells in the group.
    std::vector<cell_gid_type> gids_;

    // Cells that belong to this group.
    std::vector<lif_cell> cells_;

    // Spikes that are generated (not necessarily sorted).
    std::vector<spike> spikes_;

    // Time when the cell was last updated.
    std::vector<time_type> last_time_updated_;
    // Time when the cell can _next_ be updated;
    std::vector<time_type> next_time_updatable_;

    // SAFETY: We need to access samplers_ through a mutex since
    // simulation::add_sampler might be called concurrently.
    std::mutex sampler_mex_;
    sampler_association_map samplers_;

    // LIF probe metadata, precalculated to pass to callbacks
    std::unordered_map<cell_member_type, lif_probe_info> probes_;
};

} // namespace arb
