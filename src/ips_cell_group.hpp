#pragma once

#include <utility>
#include <random>


#include <cell_group.hpp>
#include <recipe.hpp>
#include <ips_cell.hpp>
#include <util/unique_any.hpp>


#include <iostream>

namespace arb {

/// Cell group implementing RSS cells.



class ips_cell_group: public cell_group {
struct ips_info;
using cell_gen_time = std::tuple<std::vector<ips_info>::const_iterator,
                                 std::vector<std::mt19937>::iterator,
                                 std::vector<time_type>::iterator>;

public:
    ips_cell_group(std::vector<cell_gid_type> gids, const recipe& rec) {
        cells_.reserve(gids.size());
        for (auto gid: gids) {
            cells_.emplace_back(
                util::any_cast<ips_cell>(rec.get_cell_description(gid)),
                gid);

            random_generators.push_back(std::mt19937(gid));
            times_.push_back(0.0);
        }



        reset();
    }

    cell_kind get_cell_kind() const override {
        return cell_kind::inhomogeneous_poisson_source;
    }

    void reset() override {
        clear_spikes();
        for (auto &time: times_) {
            time = 0.0;
        }

    }

    void set_binning_policy(binning_kind policy, time_type bin_interval) override {}

    void advance(epoch ep, time_type dt, const event_lane_subrange& events) override {
        // We only know the dt at this stage.
        auto distribution = std::uniform_real_distribution<float>(0.f, 1.0f);
        for (cell_gen_time cell_gen_time_it(cells_.cbegin(), random_generators.begin(), times_.begin());
            std::get<0>(cell_gen_time_it) != cells_.cend();
            ++std::get<0>(cell_gen_time_it), ++std::get<1>(cell_gen_time_it), ++std::get<2>(cell_gen_time_it))

        {
            auto cell = *std::get<0>(cell_gen_time_it);
            auto time = *std::get<2>(cell_gen_time_it);

            // Get begin and start range, depending on the cell config and
            // end the epoch ranges
            auto t = std::max(cell.start_time, time);
            auto t_end = std::min(cell.stop_time, ep.tfinal);

            // if cell does not fire in current advance step continue
            if (t >= t_end) {
                continue;
            }

            // Take a dt step, roll a dice determine if a spike is produced
            // First convert the rate from spikes/sec to ms
            time_type prob_per_dt = (cell.rate / 1000.0) * cell.sample_delta;

            //while (t < t_end) does not work
            // after x floating points this might result in t-small delta where
            // you would expect it to be t_end. This while works as expected
            while (t < t_end)
            {
                // roll a dice between 0 and 1, if below prop we have a spike
                if (distribution(*std::get<1>(cell_gen_time_it)) < prob_per_dt) {
                    spikes_.push_back({ { cell.gid, 0 }, t });
                }
                t += cell.sample_delta;
            }
            // Save the cell specific time
            *std::get<2>(cell_gen_time_it) = t;
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
    struct ips_info: public ips_cell {
        ips_info(ips_cell desc, cell_gid_type gid):
            ips_cell(std::move(desc)), gid(gid)
        {}

        cell_gid_type gid;
    };

    // RSS cell descriptions.
    std::vector<ips_info> cells_;

    // Simulation time for all RSS cells in the group.
    std::vector<time_type> times_;

    // Spikes that are generated.
    std::vector<spike> spikes_;

    // Random number generator
    std::vector<std::mt19937> random_generators;
};

} // namespace arb

