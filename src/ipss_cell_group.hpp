#pragma once

#include <utility>
#include <random>


#include <cell_group.hpp>
#include <recipe.hpp>
#include <ipss_cell.hpp>
#include <util/unique_any.hpp>


#include <iostream>

namespace arb {

/// Cell group implementing RSS cells.



class ipss_cell_group: public cell_group {
public:
    ipss_cell_group(std::vector<cell_gid_type> gids, const recipe& rec) {
        cells_.reserve(gids.size());
        for (auto gid: gids) {
            cells_.emplace_back(
                util::any_cast<ipss_cell>(rec.get_cell_description(gid)), gid);
        }

        distribution = std::uniform_real_distribution<float>(0.f, 1.0f);
        reset();
    }

    cell_kind get_cell_kind() const override {
        return cell_kind::inhomogeneous_poisson_spike_source;
    }

    void reset() override {
        clear_spikes();
        for (auto &cell: cells_) {
            cell.time = 0.0;

            // Reset the random number generator!
            cell.generator = std::mt19937(cell.gid);
        }
    }

    void set_binning_policy(binning_kind policy, time_type bin_interval) override {}

    void advance(epoch ep, time_type dt, const event_lane_subrange& events) override {
        for (auto& cell: cells_)
        {
            // Get begin and start range: cell config and epoch ranges
            auto t = std::max(cell.start_time, cell.time);
            auto t_end = std::min(cell.stop_time, ep.tfinal);

            // if cell is not active skip
            if (t >= t_end) {
                continue;
            }

            // The probability per sample step
            time_type prob_per_dt = (cell.rates_per_time.at(0).second / 1000.0) * cell.sample_delta;

            // Float noise might result in a final step larger then t_end.
            while (t < t_end)
            {
                // roll a dice between 0 and 1, if below prop we have a spike
                if (distribution(cell.generator) < prob_per_dt) {
                    spikes_.push_back({ { cell.gid, 0 }, t });
                }
                t += cell.sample_delta;
            }

            cell.time = t;
        }
    }

    const std::vector<spike>& spikes() const override {
        return spikes_;
    }

    void clear_spikes() override {
        spikes_.clear();
    }

    void add_sampler(sampler_association_handle, cell_member_predicate, schedule, sampler_function, sampling_policy) override {
        std::logic_error("pi_cell does not support sampling");
    }

    void remove_sampler(sampler_association_handle) override {}

    void remove_all_samplers() override {}

private:
    // RSS description plus gid for each RSS cell.
    struct ipss_info: public ipss_cell {
        ipss_info(ipss_cell desc, cell_gid_type gid):
            ipss_cell(std::move(desc)),
            gid(gid),
            time(0.0),
            generator(gid)
        {}

        cell_gid_type gid;
        time_type time;
        std::mt19937 generator;

    };

    // RSS cell descriptions.
    std::vector<ipss_info> cells_;

    // Spikes that are generated.
    std::vector<spike> spikes_;

    // Distribution for Poisson generation
    std::uniform_real_distribution<float> distribution;
};
} // namespace arb

