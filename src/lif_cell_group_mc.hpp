#pragma once
#include <threading/timer.hpp>
#include <cell_group.hpp>
#include <event_queue.hpp>
#include <lif_cell_description.hpp>
#include <profiling/profiler.hpp>
#include <random123/threefry.h>
#include <random123/uniform.hpp>
#include <util/unique_any.hpp>
#include <vector>

namespace nest {
namespace mc {
    class lif_cell_group_mc: public cell_group {
    public:
        using value_type = double;

        lif_cell_group_mc() = default;

        // Constructor containing gid of first cell in a group and a container of all cells.
        lif_cell_group_mc(cell_gid_type first_gid, const std::vector<util::unique_any>& cells);

        virtual cell_kind get_cell_kind() const override;

        virtual void advance(time_type tfinal, time_type dt) override;

        virtual void enqueue_events(const std::vector<postsynaptic_spike_event>& events) override;

        virtual const std::vector<spike>& spikes() const override;

        virtual void clear_spikes() override;

        virtual void add_sampler(cell_member_type probe_id, sampler_function s, time_type start_time = 0) override;

        virtual void set_binning_policy(binning_kind policy, time_type bin_interval) override;

        virtual std::vector<probe_record> probes() const override;

        virtual void reset() override;

    private:
        // Samples next poisson spike.
        void sample_next_poisson(cell_gid_type lid);

        // Returns the time of the next poisson event for given neuron,
        // taking into accout the delay of poisson spikes,
        // without sampling a new Poisson event time.
        time_type next_poisson_event(cell_gid_type lid);

        // Returns the next most recent event that is yet to be processed.
        // It can be either Poisson event or the queue event.
        // Only events that happened before tfinal are considered.
        util::optional<postsynaptic_spike_event> next_event(cell_gid_type lid, time_type tfinal);

        // Advances a single cell (lid) with the exact solution (jumps can be arbitrary).
        // Parameter dt is ignored, since we make jumps between two consecutive spikes.
        void advance_cell(time_type tfinal, time_type dt, cell_gid_type lid);

        // Gid of first cell in group.
        cell_gid_type gid_base_;

        // Cells that belong to this group.
        std::vector<lif_cell_description> cells_;

        // Spikes that are generated (not necessarily sorted).
        std::vector<spike> spikes_;

        // Pending events per cell.
        std::vector<event_queue<postsynaptic_spike_event> > cell_events_;

        // Time when the cell was last updated.
        std::vector<time_type> last_time_updated_;

        // External spike generation.

        // lambda = 1/(n_poiss * rate) for each cell.
        std::vector<double> lambda_;

        // Sampled next Poisson event time for each cell.
        std::vector<time_type> next_poiss_time_;

        // Counts poisson events. 
        // Used as an argument to random123 (since partially describes a state)
        std::vector<unsigned> poiss_event_counter_;
    };
} // namespace mc
} // namespace nest
