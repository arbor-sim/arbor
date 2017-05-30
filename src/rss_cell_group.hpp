#pragma once

#include <cell_group.hpp>
#include <rss_cell.hpp>
#include <util/span.hpp>
#include <util/unique_any.hpp>


#include <iostream>
namespace nest {
namespace mc {

/// Cell_group to collect cells that spike at a set frequency
/// Cell are lightweight and are not executed in anybackend implementation
class rss_cell_group : public cell_group {
public:
    using source_id_type = cell_member_type;

    rss_cell_group(cell_gid_type first_gid, const std::vector<util::unique_any>& cell_descriptions):
        gid_base_(first_gid)
    {
        using util::make_span;

        for (cell_gid_type i: make_span(0, cell_descriptions.size())) {
            // Copy all the rss_cells
            cells_.push_back(rss_cell(
                util::any_cast<rss_cell::rss_cell_descr>(cell_descriptions[i])
            ));

            // create a lid to gid map
            spike_sources_.push_back({gid_base_+i, 0});
        }
    }

    virtual ~rss_cell_group() = default;

    cell_kind get_cell_kind() const override {
        return cell_kind::regular_spike_source;
    }

    void reset() override {
        for (auto cell: cells_) {
            cell.reset();
        }
    }

    void set_binning_policy(binning_kind policy, time_type bin_interval) override
    {} // Nothing to do?

    void advance(time_type tfinal, time_type dt) override {
        // TODO: Move source information to rss_cell implementation
        for (auto i: util::make_span(0, cells_.size())) {
            for (auto spike_time: cells_[i].spikes_until(tfinal)) {
                spikes_.push_back({spike_sources_[i], spike_time});
            }
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
    std::vector<rss_cell> cells_;

    std::vector<probe_record> probes_;
};

} // namespace mc
} // namespace nest

