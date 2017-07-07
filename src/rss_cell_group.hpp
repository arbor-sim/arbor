#pragma once

#include <cell_group.hpp>
#include <rss_cell.hpp>
#include <util/optional.hpp>
#include <util/span.hpp>
#include <util/unique_any.hpp>


#include <iostream>
namespace nest {
namespace mc {

/// Cell_group to collect cells that spike at a set frequency
/// Cell are lightweight and are not executed in anybackend implementation
class rss_cell_group: public cell_group {
public:
    using source_id_type = cell_member_type;

    rss_cell_group(std::vector<cell_gid_type> gids,
                   const std::vector<util::unique_any>& cell_descriptions):
        gids_(gids)
    {
        using util::make_span;

        // Build lookup table for gid to local index
        for (auto i: util::make_span(0, gids_.size())) {
            gid2lid_[gids_[i]] = i;
        }

        for (cell_gid_type i: make_span(0, cell_descriptions.size())) {
            cells_.push_back(rss_cell(
                util::any_cast<rss_cell::rss_cell_description>(cell_descriptions[i])
            ));
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
    {}

    void advance(time_type tfinal, time_type dt) override {
        // TODO: Move rss_cell implementation into the rss_cell_group
        for (auto i: util::make_span(0, cells_.size())) {
            for (auto spike_time: cells_[i].spikes_until(tfinal)) {
                spikes_.push_back({{gids_[i], 0}, spike_time});
            }
        }
    };

    void enqueue_events(const std::vector<postsynaptic_spike_event>& events) override {
        std::logic_error("The rss_cells do not support incoming events!");
    }

    const std::vector<spike>& spikes() const override {
        return spikes_;
    }

    void clear_spikes() override {
        spikes_.clear();
    }

    std::vector<probe_record> probes() const override {
        return {};
    }

    void add_sampler(cell_member_type probe_id, sampler_function s, time_type start_time = 0) override {
        std::logic_error("The rss_cells do not support sampling of internal state!");
    }

private:
    // List of the gids of the cells in the group
    std::vector<cell_gid_type> gids_;

    // Hash table for converting gid to local index
    std::unordered_map<cell_gid_type, cell_gid_type> gid2lid_;

    // convenience function for performing conversion
    util::optional<cell_gid_type> gid2lid(cell_gid_type gid) const {
        auto it = gid2lid_.find(gid);
        return it==gid2lid_.end()? util::nothing: util::optional<cell_gid_type>(it->second);
    }

    // Spikes that are generated.
    std::vector<spike> spikes_;

    // Store a reference to the cell actually implementing the spiking
    std::vector<rss_cell> cells_;
};

} // namespace mc
} // namespace nest

