#pragma once

#include <benchmark_cell.hpp>
#include <cell_group.hpp>
#include <recipe.hpp>
#include <time_sequence.hpp>

namespace arb {

class benchmark_cell_group: public cell_group {
public:
    benchmark_cell_group(std::vector<cell_gid_type> gids, const recipe& rec);

    cell_kind get_cell_kind() const override;

    void advance(epoch ep, time_type dt, const event_lane_subrange& event_lanes) override;

    void reset() override;

    void set_binning_policy(binning_kind policy, time_type bin_interval) override;

    const std::vector<spike>& spikes() const override;

    void clear_spikes() override;

    void add_sampler(sampler_association_handle h, cell_member_predicate probe_ids, schedule sched, sampler_function fn, sampling_policy policy) override;

    void remove_sampler(sampler_association_handle h) override;

    void remove_all_samplers() override;

private:
    time_type t_;

    std::vector<benchmark_cell> cells_;
    std::vector<spike> spikes_;
    std::vector<cell_gid_type> gids_;
};

} // namespace arb

