#pragma once

#include <cstdint>
#include <vector>

#include <catypes.hpp>
#include <cell.hpp>
#include <event_queue.hpp>
#include <spike.hpp>
#include <spike_source.hpp>

#include <profiling/profiler.hpp>

namespace nest {
namespace mc {

// samplers take a time and sample value, and return an optional time
// for the next desired sample.

struct sampler {
    using time_type = float;
    using value_type = double;

    cell_member_type probe_id;   // samplers are attached to probes
    std::function<util::optional<time_type>(time_type, value_type)> sample;
};

template <typename Cell>
class cell_group {
public:
    using index_type = cell_gid_type;
    using cell_type = Cell;
    using value_type = typename cell_type::value_type;
    using size_type  = typename cell_type::value_type;
    using spike_detector_type = spike_detector<Cell>;
    using source_id_type = cell_member_type;

    struct spike_source_type {
        source_id_type source_id;
        spike_detector_type source;
    };

    cell_group() = default;

    cell_group(cell_gid_type gid, const cell& c) :
        gid_base_{gid}, cell_{c}
    {
        cell_.voltage()(memory::all) = -65.;
        cell_.initialize();

        source_id_type source_id={gid_base_,0};
        for (auto& d : c.detectors()) {
            ++source_id.index;
            spike_sources_.push_back({
                source_id, spike_detector_type(cell_, d.location, d.threshold, 0.f)
            });
        }
    }

    void advance(double tfinal, double dt) {
        while (cell_.time()<tfinal) {
            // take any pending samples
            float cell_time = cell_.time();

            util::profiler_enter("sampling");
            while (auto m = sample_events_.pop_if_before(cell_time)) {
                auto& sampler = samplers_[m->sampler_index];
                EXPECTS((bool)sampler.sample);

                index_type probe_index = sampler.probe_id.index;
                auto next = sampler.sample(cell_.time(), cell_.probe(probe_index));
                if (next) {
                    m->time = std::max(*next, cell_time);
                    sample_events_.push(*m);
                }
            }
            util::profiler_leave();

            // look for events in the next time step
            auto tstep = std::min(tfinal, cell_.time()+dt);
            auto next = events_.pop_if_before(tstep);
            auto tnext = next ? next->time: tstep;

            // integrate cell state
            cell_.advance(tnext - cell_.time());
            if (!cell_.is_physical_solution()) {
                std::cerr << "warning: solution out of bounds\n";
            }

            util::profiler_enter("events");
            // check for new spikes
            for (auto& s : spike_sources_) {
                if (auto spike = s.source.test(cell_, cell_.time())) {
                    spikes_.push_back({s.source_id, spike.get()});
                }
            }

            // apply events
            if (next) {
                cell_.apply_event(next.get());
                // apply events that are due within some epsilon of the current
                // time step. This should be a parameter. e.g. with for variable
                // order time stepping, use the minimum possible time step size.
                while(auto e = events_.pop_if_before(cell_.time()+dt/10.)) {
                    cell_.apply_event(e.get());
                }
            }
            util::profiler_leave();
        }

    }

    template <typename R>
    void enqueue_events(const R& events) {
        for (auto e : events) {
            e.target -= first_target_gid_;
            events_.push(e);
        }
    }

    const std::vector<spike<source_id_type>>&
    spikes() const { return spikes_; }

    cell_type&       cell()       { return cell_; }
    const cell_type& cell() const { return cell_; }

    const std::vector<spike_source_type>&
    spike_sources() const {
        return spike_sources_;
    }

    void clear_spikes() {
        spikes_.clear();
    }

    void add_sampler(const sampler& s, float start_time = 0) {
        auto sampler_index = uint32_t(samplers_.size());
        samplers_.push_back(s);
        sample_events_.push({sampler_index, start_time});
    }

private:
    /// gid of first cell in group
    cell_gid_type gid_base_;

    /// the lowered cell state (e.g. FVM) of the cell
    cell_type cell_;

    /// spike detectors attached to the cell
    std::vector<spike_source_type> spike_sources_;

    //. spikes that are generated
    std::vector<spike<source_id_type>> spikes_;

    /// pending events to be delivered
    event_queue<postsynaptic_spike_event> events_;

    /// pending samples to be taken
    event_queue<sample_event> sample_events_;

    /// the global id of the first target (e.g. a synapse) in this group
    index_type first_target_gid_;
 
    /// the global id of the first probe in this group
    index_type first_probe_gid_;

    /// collection of samplers to be run against probes in this group
    std::vector<sampler> samplers_;
};

} // namespace mc
} // namespace nest
