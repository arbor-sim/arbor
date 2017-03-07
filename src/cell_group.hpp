#pragma once

#include <cstdint>
#include <functional>
#include <iterator>
#include <vector>

#include <algorithms.hpp>
#include <cell.hpp>
#include <common_types.hpp>
#include <event_queue.hpp>
#include <spike.hpp>
#include <util/debug.hpp>
#include <util/partition.hpp>
#include <util/range.hpp>

#include <profiling/profiler.hpp>

namespace nest {
namespace mc {

template <typename LoweredCell>
class cell_group {
public:
    using iarray = cell_gid_type;
    using lowered_cell_type = LoweredCell;
    using value_type = typename lowered_cell_type::value_type;
    using size_type  = typename lowered_cell_type::value_type;
    using source_id_type = cell_member_type;

    using time_type = float;
    using sampler_function = std::function<util::optional<time_type>(time_type, double)>;

    cell_group() = default;

    template <typename Cells>
    cell_group(cell_gid_type first_gid, const Cells& cells):
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

        cell_.initialize(cells, target_handles_, probe_handles_);

        // Create a list of the global identifiers for the spike sources
        auto source_gid = cell_gid_type{gid_base_};
        for (const auto& cell: cells) {
            for (cell_lid_type lid=0u; lid<cell.detectors().size(); ++lid) {
                spike_sources_.push_back(source_id_type{source_gid, lid});
            }
            ++source_gid;
        }
        EXPECTS(spike_sources_.size()==n_detectors);
    }

    void reset() {
        spikes_.clear();
        clear_events();
        reset_samplers();
        cell_.reset();
    }

    time_type min_step(time_type dt) {
        return 0.1*dt;
    }

    void advance(time_type tfinal, time_type dt) {
        while (cell_.time()<tfinal) {
            // take any pending samples
            time_type cell_time = cell_.time();

            PE("sampling");
            while (auto m = sample_events_.pop_if_before(cell_time)) {
                auto& s = samplers_[m->sampler_index];
                EXPECTS((bool)s.sampler);
                auto next = s.sampler(cell_.time(), cell_.probe(s.handle));

                if (next) {
                    m->time = std::max(*next, cell_time);
                    sample_events_.push(*m);
                }
            }
            PL();

            // look for events in the next time step
            time_type tstep = cell_.time()+dt;
            tstep = std::min(tstep, tfinal);
            auto next = events_.pop_if_before(tstep);

            // apply events that are due within the smallest allowed time step.
            while (next && (next->time-cell_.time()) < min_step(dt)) {
                auto handle = get_target_handle(next->target);
                cell_.deliver_event(handle, next->weight);
                next = events_.pop_if_before(tstep);
            }

            // integrate cell state
            time_type tnext = next ? next->time: tstep;
            cell_.advance(tnext - cell_.time());

            if (!cell_.is_physical_solution()) {
                std::cerr << "warning: solution out of bounds for cell "
                          << gid_base_ << " at t " << cell_.time() << " ms\n";
            }

            // apply events
            PE("events");
            if (next) {
                auto handle = get_target_handle(next->target);
                cell_.deliver_event(handle, next->weight);
            }
            PL();
        }

        // Copy out spike voltage threshold crossings from the back end, then
        // generate spikes with global spike source ids. The threshold crossings
        // record the local spike source index, which must be converted to a
        // global index for spike communication.
        PE("events");
        for (auto c: cell_.get_spikes()) {
            spikes_.push_back({spike_sources_[c.index], time_type(c.time)});
        }
        // Now that the spikes have been generated, clear the old crossings
        // to get ready to record spikes from the next integration period.
        cell_.clear_spikes();
        PL();
    }

    template <typename R>
    void enqueue_events(const R& events) {
        for (auto e : events) {
            events_.push(e);
        }
    }

    const std::vector<spike<source_id_type, time_type>>&
    spikes() const {
        return spikes_;
    }

    void clear_spikes() {
        spikes_.clear();
    }

    const std::vector<source_id_type>&
    spike_sources() const {
        return spike_sources_;
    }

    void clear_events() {
        events_.clear();
    }

    void add_sampler(cell_member_type probe_id, sampler_function s, time_type start_time = 0) {
        auto handle = get_probe_handle(probe_id);

        auto sampler_index = uint32_t(samplers_.size());
        samplers_.push_back({handle, s});
        sampler_start_times_.push_back(start_time);
        sample_events_.push({sampler_index, start_time});
    }

    void remove_samplers() {
        sample_events_.clear();
        samplers_.clear();
        sampler_start_times_.clear();
    }

    void reset_samplers() {
        // clear all pending sample events and reset to start at time 0
        sample_events_.clear();
        for(uint32_t i=0u; i<samplers_.size(); ++i) {
            sample_events_.push({i, sampler_start_times_[i]});
        }
    }

    value_type probe(cell_member_type probe_id) const {
        return cell_.probe(get_probe_handle(probe_id));
    }

private:
    /// gid of first cell in group
    cell_gid_type gid_base_;

    /// the lowered cell state (e.g. FVM) of the cell
    lowered_cell_type cell_;

    /// spike detectors attached to the cell
    std::vector<source_id_type> spike_sources_;

    /// spikes that are generated
    std::vector<spike<source_id_type, time_type>> spikes_;

    /// pending events to be delivered
    event_queue<postsynaptic_spike_event<time_type>> events_;

    /// pending samples to be taken
    event_queue<sample_event<time_type>> sample_events_;
    std::vector<time_type> sampler_start_times_;

    /// the global id of the first target (e.g. a synapse) in this group
    iarray first_target_gid_;

    /// handles for accessing lowered cell
    using target_handle = typename lowered_cell_type::target_handle;
    std::vector<target_handle> target_handles_;

    using probe_handle = typename lowered_cell_type::probe_handle;
    std::vector<probe_handle> probe_handles_;

    struct sampler_entry {
        typename lowered_cell_type::probe_handle handle;
        sampler_function sampler;
    };

    /// collection of samplers to be run against probes in this group
    std::vector<sampler_entry> samplers_;

    /// lookup table for probe ids -> local probe handle indices
    std::vector<std::size_t> probe_handle_divisions_;

    /// lookup table for target ids -> local target handle indices
    std::vector<std::size_t> target_handle_divisions_;

    /// build handle index lookup tables
    template <typename Cells>
    void build_handle_partitions(const Cells& cells) {
        auto probe_counts = util::transform_view(cells, [](const cell& c) { return c.probes().size(); });
        auto target_counts = util::transform_view(cells, [](const cell& c) { return c.synapses().size(); });

        make_partition(probe_handle_divisions_, probe_counts);
        make_partition(target_handle_divisions_, target_counts);
    }

    /// use handle partition to get index from id
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

    /// get probe handle from probe id
    probe_handle get_probe_handle(cell_member_type probe_id) const {
        return probe_handles_[handle_partition_lookup(probe_handle_divisions_, probe_id)];
    }

    /// get target handle from target id
    target_handle get_target_handle(cell_member_type target_id) const {
        return target_handles_[handle_partition_lookup(target_handle_divisions_, target_id)];
    }
};

} // namespace mc
} // namespace nest
