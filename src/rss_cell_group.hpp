#pragma once

#include <utility>

#include <cell_group.hpp>
#include <recipe.hpp>
#include <rss_cell.hpp>
#include <util/unique_any.hpp>

namespace arb {

/// Cell group implementing RSS cells.

class rss_cell_group: public cell_group {
public:
    rss_cell_group(std::vector<cell_gid_type> gids, const recipe& rec) {
        cells_.reserve(gids.size());
        for (auto gid: gids) {
            cells_.emplace_back(
                util::any_cast<rss_cell>(rec.get_cell_description(gid)),
                gid);
        }
        reset();
    }

    cell_kind get_cell_kind() const override {
        return cell_kind::regular_spike_source;
    }

    void reset() override {
        clear_spikes();
        for (auto& cell : cells_) {
            cell.step = 0;
        }
    }

    void set_binning_policy(binning_kind policy, time_type bin_interval) override {}

    void advance(epoch ep, time_type dt, const event_lane_subrange& events) override {
        for (auto& cell: cells_) {

            auto t_end = std::min(cell.stop_time, ep.tfinal);
            auto t = cell.start_time + cell.step*cell.period;

            while (t < t_end) {
                spikes_.push_back({{cell.gid, 0}, t});

                // Increasing time with period time step has float issues
                ++cell.step;
                t = cell.start_time + cell.step*cell.period;
            }
        }
    }

    const std::vector<spike>& spikes() const override {
        return spikes_;
    }

    void clear_spikes() override {
        spikes_.clear();
    }

    void add_sampler(sampler_association_handle, cell_member_predicate, schedule, sampler_function, sampling_policy) override {
        std::logic_error("rss_cell does not support sampling");
    }

    void remove_sampler(sampler_association_handle) override {}

    void remove_all_samplers() override {}

private:
    // RSS description plus gid for each RSS cell.
    struct rss_info: public rss_cell {
        rss_info(rss_cell desc, cell_gid_type gid):
            rss_cell(std::move(desc)), gid(gid)
        {}

        cell_gid_type gid;

        // We do not store the time but a count of the number of step since start
        // of cell. This prevents float problems at high number.
        std::size_t step;
    };

    // RSS cell descriptions.
    std::vector<rss_info> cells_;


    // Spikes that are generated.
    std::vector<spike> spikes_;
};

} // namespace arb

