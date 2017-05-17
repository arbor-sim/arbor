#pragma once

#include <cell_group.hpp>
#include <fs_cell.hpp>
#include <util/span.hpp>
#include <util/unique_any.hpp>


#include <iostream>
namespace nest {
namespace mc {

/// Cell_group to collect cells that spike at a set frequency
/// Cell are lightweight and are not executed in anybackend implementation
class fs_cell_group : public cell_group {
public:
    using source_id_type = cell_member_type;

    fs_cell_group(cell_gid_type first_gid, const std::vector<util::unique_any>& cells):
        gid_base_{ first_gid }
    {
        using util::make_span;

        auto source_gid = cell_gid_type{ gid_base_ };
        for (auto i : make_span(0, cells.size())) {
            // Copy all the fs_cells
            cells_.push_back(util::any_cast<fs_cell>(cells[i]));

            // create a lid to gid map
            spike_sources_.push_back(source_id_type{ source_gid, 0 });
            ++source_gid;
        }
    }

    virtual ~fs_cell_group() = default;

    cell_kind get_cell_kind() const override
    {
        return cell_kind::fs_neuron;
    }

    void reset() override
    {
        for (auto cell : cells_) {
            cell.reset();
        }
    }

    void set_binning_policy(binning_kind policy, time_type bin_interval) override
    {} // Nothing to do?

    void advance(time_type tfinal, time_type dt) override
    {
        auto source_lid = cell_gid_type{ 0 };
        for (auto &cell  : cells_) {
            for (auto spike_time : cell.spikes_until(tfinal)) {
                spikes_.push_back({ spike_sources_[source_lid], spike_time});
            }

            ++source_lid;
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

    std::vector<probe_record> probes() const override {
        return probes_;
    }

    void add_sampler(cell_member_type probe_id, sampler_function s, time_type start_time = 0) override
    {} // TODO: Fail silently or throw?

private:
    // gid of first cell in group.
    cell_gid_type gid_base_;

    // Spikes that are generated.
    std::vector<spike> spikes_;

    // Spike generators attached to the cell
    std::vector<source_id_type> spike_sources_;

    // Store a reference to the cell actually implementing the spiking
    std::vector<fs_cell> cells_;

    std::vector<probe_record> probes_;
};

} // namespace mc
} // namespace nest

