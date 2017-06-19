#pragma once

#include <cstdint>
#include <functional>
#include <iterator>
#include <vector>

#include <algorithms.hpp>
#include <cell_group.hpp>
#include <cell.hpp>
#include <common_types.hpp>
#include <event_binner.hpp>
#include <event_queue.hpp>
#include <recipe.hpp>
#include <sampler_function.hpp>
#include <spike.hpp>
#include <util/debug.hpp>
#include <util/partition.hpp>
#include <util/range.hpp>
#include <util/unique_any.hpp>

#include <profiling/profiler.hpp>

namespace nest {
namespace mc {

template <typename LoweredCell>
class mc_cell_group: public cell_group {
public:
    using lowered_cell_type = LoweredCell;
    using value_type = typename lowered_cell_type::value_type;
    using size_type  = typename lowered_cell_type::value_type;
    using source_id_type = cell_member_type;

    mc_cell_group() = default;

    template <typename Cells>
    mc_cell_group(cell_gid_type first_gid, const Cells& cells):
        gid_base_{first_gid}
    {
        // Create lookup structure for probe and target ids.
        build_handle_partitions(cells);
        std::size_t n_probes = probe_handle_divisions_.back();
        std::size_t n_targets = target_handle_divisions_.back();
        std::size_t n_detectors = algorithms::sum(util::transform_view(
            cells, [](const cell& c) { return c.detectors().size(); }));

        // Allocate space to store handles.
        target_handles_.resize(n_targets);
        probe_handles_.resize(n_probes);

        lowered_.initialize(cells, target_handles_, probe_handles_);

        // Create a list of the global identifiers for the spike sources
        auto source_gid = cell_gid_type{gid_base_};
        for (const auto& cell: cells) {
            for (cell_lid_type lid=0u; lid<cell.detectors().size(); ++lid) {
                spike_sources_.push_back(source_id_type{source_gid, lid});
            }
            ++source_gid;
        }
        EXPECTS(spike_sources_.size()==n_detectors);

        // Create the enumeration of probes attached to cells in this cell group
        probes_.reserve(n_probes);
        for (auto i: util::make_span(0, cells.size())){
            const cell_gid_type probe_gid = gid_base_ + i;
            const auto probes_on_cell = cells[i].probes();
            for (cell_lid_type lid: util::make_span(0, probes_on_cell.size())) {
                // get the unique global identifier of this probe
                cell_member_type id{probe_gid, lid};

                // get the location and kind information of the probe
                const auto p = probes_on_cell[lid];

                // record the combined identifier and probe details
                probes_.push_back(probe_record{id, p.location, p.kind});
            }
        }
    }

    mc_cell_group(cell_gid_type first_gid, const std::vector<util::unique_any>& cells):
        mc_cell_group(
            first_gid,
            util::transform_view(
                cells,
                [](const util::unique_any& c) -> const cell& {return util::any_cast<const cell&>(c);})
        )
    {}

    cell_kind get_cell_kind() const override {
        return cell_kind::cable1d_neuron;
    }

    void reset() override {
        spikes_.clear();
        events_.clear();
        reset_samplers();
        binner_.reset();
        lowered_.reset();
    }

    void set_binning_policy(binning_kind policy, time_type bin_interval) override {
        binner_ = event_binner(policy, bin_interval);
    }

    void advance(time_type tfinal, time_type dt) override {
        EXPECTS(lowered_.state_synchronized());

        // Bin pending events and enqueue on lowered state.
        time_type ev_min_time = lowered_.max_time(); // (but we're synchronized here)
        while (auto ev = events_.pop_if_before(tfinal)) {
            auto handle = get_target_handle(ev->target);
            auto binned_ev_time = binner_.bin(ev->target.gid, ev->time, ev_min_time);
            lowered_.add_event(binned_ev_time, handle, ev->weight);
        }

        lowered_.setup_integration(tfinal, dt);

        util::optional<time_type> first_sample_time = sample_events_.time_if_before(tfinal);
        std::vector<sample_event> requeue_sample_events;
        while (!lowered_.integration_complete()) {
            // Take any pending samples.
            // TODO: Placeholder: this will be replaced by a backend polling implementation.

            if (first_sample_time) {
                PE("sampling");
                time_type cell_max_time = lowered_.max_time();

                requeue_sample_events.clear();
                while (auto m = sample_events_.pop_if_before(cell_max_time)) {
                    auto& s = samplers_[m->sampler_index];
                    EXPECTS((bool)s.sampler);

                    time_type cell_time = lowered_.time(s.cell_gid-gid_base_);
                    if (cell_time<m->time) {
                        // This cell hasn't reached this sample time yet.
                        requeue_sample_events.push_back(*m);
                    }
                    else {
                        auto next = s.sampler(cell_time, lowered_.probe(s.handle));
                        if (next) {
                            m->time = std::max(*next, cell_time);
                            requeue_sample_events.push_back(*m);
                        }
                    }
                }
                for (auto& ev: requeue_sample_events) {
                    sample_events_.push(std::move(ev));
                }
                first_sample_time = sample_events_.time_if_before(tfinal);
                PL();
            }

            // Ask lowered_ cell to integrate 'one step', delivering any
            // events accordingly.
            // TODO: Placeholder: with backend polling for samplers, we will
            // request that the lowered cell perform the integration all the
            // way to tfinal.

            lowered_.step_integration();

            if (util::is_debug_mode() && !lowered_.is_physical_solution()) {
                std::cerr << "warning: solution out of bounds for cell "
                          << gid_base_ << " at (max) t " << lowered_.max_time() << " ms\n";
            }
        }

        // Copy out spike voltage threshold crossings from the back end, then
        // generate spikes with global spike source ids. The threshold crossings
        // record the local spike source index, which must be converted to a
        // global index for spike communication.
        PE("events");
        for (auto c: lowered_.get_spikes()) {
            spikes_.push_back({spike_sources_[c.index], time_type(c.time)});
        }
        // Now that the spikes have been generated, clear the old crossings
        // to get ready to record spikes from the next integration period.
        lowered_.clear_spikes();
        PL();
    }

    void enqueue_events(const std::vector<postsynaptic_spike_event>& events) override {
        for (auto& e: events) {
            events_.push(e);
        }
    }

    const std::vector<spike>& spikes() const override {
        return spikes_;
    }

    void clear_spikes() override {
        spikes_.clear();
    }

    const std::vector<source_id_type>& spike_sources() const {
        return spike_sources_;
    }

    void add_sampler(cell_member_type probe_id, sampler_function s, time_type start_time) override {
        auto handle = get_probe_handle(probe_id);

        using size_type = sample_event::size_type;
        auto sampler_index = size_type(samplers_.size());
        samplers_.push_back({handle, probe_id.gid, s});
        sampler_start_times_.push_back(start_time);
        sample_events_.push({sampler_index, start_time});
    }

    std::vector<probe_record> probes() const override {
        return probes_;
    }

private:
    // gid of first cell in group.
    cell_gid_type gid_base_;

    // The lowered cell state (e.g. FVM) of the cell.
    lowered_cell_type lowered_;

    // Spike detectors attached to the cell.
    std::vector<source_id_type> spike_sources_;

    // Spikes that are generated.
    std::vector<spike> spikes_;

    // Event time binning manager.
    event_binner binner_;

    // Pending events to be delivered.
    event_queue<postsynaptic_spike_event> events_;

    // Pending samples to be taken.
    event_queue<sample_event> sample_events_;
    std::vector<time_type> sampler_start_times_;

    // Handles for accessing lowered cell.
    using target_handle = typename lowered_cell_type::target_handle;
    std::vector<target_handle> target_handles_;

    using probe_handle = typename lowered_cell_type::probe_handle;
    std::vector<probe_handle> probe_handles_;

    struct sampler_entry {
        typename lowered_cell_type::probe_handle handle;
        cell_gid_type cell_gid;
        sampler_function sampler;
    };

    // Collection of samplers to be run against probes in this group.
    std::vector<sampler_entry> samplers_;

    // Lookup table for probe ids -> local probe handle indices.
    std::vector<std::size_t> probe_handle_divisions_;

    // Lookup table for target ids -> local target handle indices.
    std::vector<std::size_t> target_handle_divisions_;

    // Enumeration of the probes that are attached to the cells in the cell group
    std::vector<probe_record> probes_;

    // Build handle index lookup tables.
    template <typename Cells>
    void build_handle_partitions(const Cells& cells) {
        auto probe_counts =
            util::transform_view(cells, [](const cell& c) { return c.probes().size(); });
        auto target_counts =
            util::transform_view(cells, [](const cell& c) { return c.synapses().size(); });

        make_partition(probe_handle_divisions_, probe_counts);
        make_partition(target_handle_divisions_, target_counts);
    }

    // Use handle partition to get index from id.
    template <typename Divisions>
    std::size_t handle_partition_lookup(const Divisions& divisions, cell_member_type id) const {
        // NB: without any assertion checking, this would just be:
        // return divisions[id.gid-gid_base_]+id.index;

        EXPECTS(id.gid>=gid_base_);

        auto handle_partition = util::partition_view(divisions);
        EXPECTS(id.gid-gid_base_<handle_partition.size());

        auto ival = handle_partition[id.gid-gid_base_];
        std::size_t i = ival.first + id.index;
        EXPECTS(i<ival.second);

        return i;
    }

    // Get probe handle from probe id.
    probe_handle get_probe_handle(cell_member_type probe_id) const {
        return probe_handles_[handle_partition_lookup(probe_handle_divisions_, probe_id)];
    }

    // Get target handle from target id.
    target_handle get_target_handle(cell_member_type target_id) const {
        return target_handles_[handle_partition_lookup(target_handle_divisions_, target_id)];
    }

    void reset_samplers() {
        // clear all pending sample events and reset to start at time 0
        sample_events_.clear();
        using size_type = sample_event::size_type;
        for(size_type i=0; i<samplers_.size(); ++i) {
            sample_events_.push({i, sampler_start_times_[i]});
        }
    }
};

} // namespace mc
} // namespace nest
