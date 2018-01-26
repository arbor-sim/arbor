#pragma once
#include <algorithm>
#include <threading/timer.hpp>
#include <cell_group.hpp>
#include <event_queue.hpp>
#include <lif_cell_description.hpp>
#include <profiling/profiler.hpp>
#include <recipe.hpp>
#include <util/unique_any.hpp>
#include <vector>

namespace arb {
class lif_cell_group: public cell_group {
public:
    using value_type = double;

    lif_cell_group() = default;

    // Constructor containing gid of first cell in a group and a container of all cells.
    lif_cell_group(std::vector<cell_gid_type> gids, const recipe& rec);

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

private:
    // Advances a single cell (lid) with the exact solution (jumps can be arbitrary).
    // Parameter dt is ignored, since we make jumps between two consecutive spikes.
    void advance_cell(time_type tfinal, time_type dt, cell_gid_type lid, pse_vector& event_lane);

    // List of the gids of the cells in the group.
    std::vector<cell_gid_type> gids_;

    // Cells that belong to this group.
    std::vector<lif_cell_description> cells_;

    // Spikes that are generated (not necessarily sorted).
    std::vector<spike> spikes_;

    // Time when the cell was last updated.
    std::vector<time_type> last_time_updated_;
};
} // namespace arb
