#pragma once

#include <utility>
#include <random>


#include <cell_group.hpp>
#include <recipe.hpp>
#include <ipss_cell_description.hpp>
#include <ipss_cell.hpp>
#include <util/unique_any.hpp>


#include <iostream>

namespace arb {

/// Cell group collecting a group of ipss cells.

class ipss_cell_group: public cell_group {
public:
    ipss_cell_group(std::vector<cell_gid_type> gids, const recipe& rec) {
        cells_.reserve(gids.size());
        for (auto gid: gids) {
            cells_.emplace_back(
                util::any_cast<ipss_cell_description>(rec.get_cell_description(gid)), gid);
        }
        reset();
    }

    cell_kind get_cell_kind() const override {
        return cell_kind::inhomogeneous_poisson_spike_source;
    }

    void reset() override {
        clear_spikes();
        for (auto &cell: cells_) {
            cell.reset();
        }
    }

    void set_binning_policy(binning_kind policy, time_type bin_interval) override {}

    void advance(epoch ep, time_type dt, const event_lane_subrange& events) override {
        for (auto& cell : cells_) {
            cell.advance(ep, spikes_);
        }
    }

    const std::vector<spike>& spikes() const override {
        return spikes_;
    }

    void clear_spikes() override {
        spikes_.clear();
    }

    void add_sampler(sampler_association_handle, cell_member_predicate, schedule, sampler_function, sampling_policy) override {
        std::logic_error("ipss_cell does not support sampling");
    }

    void remove_sampler(sampler_association_handle) override {}

    void remove_all_samplers() override {}

private:
    // RSS cell descriptions.
    std::vector<ipss_cell> cells_;

    // Spikes that are generated.
    std::vector<spike> spikes_;

};
} // namespace arb

