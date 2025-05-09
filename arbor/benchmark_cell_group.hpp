#pragma once

#include <arbor/benchmark_cell.hpp>
#include <arbor/common_types.hpp>
#include <arbor/recipe.hpp>
#include <arbor/sampling.hpp>
#include <arbor/spike.hpp>

#include "cell_group.hpp"
#include "epoch.hpp"
#include "label_resolution.hpp"

namespace arb {

class benchmark_cell_group: public cell_group {
public:
    benchmark_cell_group(const std::vector<cell_gid_type>& gids, const recipe& rec, cell_label_range& cg_sources, cell_label_range& cg_targets);

    cell_kind get_cell_kind() const override;

    void advance(epoch ep, time_type dt, const event_lane_subrange& event_lanes) override;

    void reset() override;

    const std::vector<spike>& spikes() const override;

    void clear_spikes() override;

    void add_sampler(sampler_association_handle h, cell_member_predicate probeset_ids, schedule sched, sampler_function fn) override;

    void remove_sampler(sampler_association_handle h) override {}

    void remove_all_samplers() override {}

    ARB_SERDES_ENABLE(benchmark_cell_group, cells_, spikes_, gids_);

    void t_serialize(serializer& ser, const std::string& k) const override;
    void t_deserialize(serializer& ser, const std::string& k) override;

private:
    std::vector<benchmark_cell> cells_;
    std::vector<spike> spikes_;
    std::vector<cell_gid_type> gids_;
};

} // namespace arb

