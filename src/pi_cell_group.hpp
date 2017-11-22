#pragma once

#include <utility>
#include <random>


#include <cell_group.hpp>
#include <recipe.hpp>
#include <pi_cell.hpp>
#include <util/unique_any.hpp>


#include <iostream>

namespace arb {

/// Cell group implementing RSS cells.



class pi_cell_group: public cell_group {
struct pi_info;
using cell_gen_pair = std::pair<std::vector<pi_info>::const_iterator, std::vector<std::mt19937>::iterator>;

public:
    pi_cell_group(std::vector<cell_gid_type> gids, const recipe& rec) {
        cells_.reserve(gids.size());
        for (auto gid: gids) {
            cells_.emplace_back(
                util::any_cast<pi_cell>(rec.get_cell_description(gid)),
                gid);

            random_generators.push_back(std::mt19937(gid));
        }



        reset();
    }

    cell_kind get_cell_kind() const override {
        return cell_kind::regular_spike_source;
    }

    void reset() override {
        clear_spikes();
        time_ = 0;
    }

    void set_binning_policy(binning_kind policy, time_type bin_interval) override {}

    void advance(epoch ep, time_type dt, const event_lane_subrange& events) override {
        // We only know the dt at this stage.
        auto distribution = std::uniform_real_distribution<float>(0.f, 1.0f);
        for (cell_gen_pair cell_gen(cells_.cbegin(), random_generators.begin());
            cell_gen.first != cells_.cend(); ++cell_gen.first, ++cell_gen.second){
            auto cell = *cell_gen.first;

            // Get begin and start range, depending on the cell config and
            // end the epoch ranges
            auto t = std::max(cell.start_time, time_);
            auto t_end = std::min(cell.stop_time, ep.tfinal);

            // if cell does not fire in current advance step continue
            if (t >= t_end) {
                continue;
            }

            // Take a dt step, roll a dice determine if a spike is produced
            // First convert the rate from spikes/sec to ms
            time_type prob_per_dt = (cell.rate / 1000.0) * dt;

            //while (t < t_end) does not work
            // after x floating points this might result in t-small delta where
            // you would expect it to be t_end. This while works as expected
            int counter = 0;
            while (t < t_end)
            {

                // roll a dice between 0 and 1, if below prop we have a spike
                if (distribution(*cell_gen.second) < prob_per_dt) {
                    spikes_.push_back({ { cell.gid, 0 }, t });

                    std::cout << counter << "," << t << std::endl;
                }
                counter++;
                t += dt;
            }
        }

        // To assure that the time is segmented correctly we always have to loop
        // over the time steps in this epoch at least once.
        // We will advance the internal clock of the cell to the first dt that is
        // bigger (or equal) as ep.tfinal.
        while (time_ < ep.tfinal) {
            time_ += dt;
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
    struct pi_info: public pi_cell {
        pi_info(pi_cell desc, cell_gid_type gid):
            pi_cell(std::move(desc)), gid(gid)
        {}

        cell_gid_type gid;
    };

    // RSS cell descriptions.
    std::vector<pi_info> cells_;

    // Simulation time for all RSS cells in the group.
    time_type time_;

    // Spikes that are generated.
    std::vector<spike> spikes_;

    // Random number generator
    std::vector<std::mt19937> random_generators;
};

} // namespace arb

