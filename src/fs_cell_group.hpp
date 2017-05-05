#pragma once

#include <cell_group.hpp>
#include <fs_cell.hpp>

namespace nest {
namespace mc {

/// Cell_group to collect cells that spike at a set frequency
class fs_cell_group : public cell_group {
public:
    using source_id_type = cell_member_type;

    fs_cell_group(cell_gid_type first_gid, std::vector<fs_cell> cells):
        gid_base_{ first_gid },
        cells_(cells)
    {
        // Create a list of the global identifiers for the spike sources
        auto source_gid = cell_gid_type{ gid_base_ };

        for (const auto& cell : cells_) {
            spike_sources_.push_back(source_id_type{ source_gid, 0u});
            ++source_gid;
        }
    }

    virtual ~fs_cell_group() = default;

    cell_kind get_cell_kind() const override
    {
        return cell_kind::regular_frequency;
    }

    void reset() override
    {
        for (auto cell : cells_) {
            cell.reset();
        }
    }

    void set_binning_policy(binning_kind policy, time_type bin_interval) override
    {} // Nothing to do  ?

    void advance(time_type tfinal, time_type dt) override
    {
        auto source_gid = cell_gid_type{ gid_base_ };
        for (auto cell : cells_) {
            for (auto spike_time : cell.spikes_until(tfinal)) {
                spikes_.push_back({ spike_sources_[source_gid], spike_time});
            }
            ++source_gid;
        }
    };


    void enqueue_events(const std::vector<postsynaptic_spike_event>& events) override
    {} // TODO: Fail silently or throw?

    const std::vector<spike>& spikes() const override {
        return spikes_;
    }

    void clear_spikes() override {
        spikes_.clear();
    }


    void add_sampler(cell_member_type probe_id, sampler_function s, time_type start_time = 0) override
    {} // TODO: Fail silently or throw?

private:
    // gid of first cell in group.
    cell_gid_type gid_base_;
    //time_type start_time_;
    //time_type period_;
    //time_type stop_time_;

    // Spikes that are generated.
    std::vector<spike> spikes_;

    // Spike generators attached to the cell
    std::vector<source_id_type> spike_sources_;

    // Store a reference to the cell actually implementing the spiking
    std::vector<fs_cell> & cells_;
};

} // namespace mc
} // namespace nest

