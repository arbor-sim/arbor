#pragma once
#include <algorithm>
#include <threading/timer.hpp>
#include <cell_group.hpp>
#include <event_queue.hpp>
#include <lif_cell_description.hpp>
#include <profiling/profiler.hpp>
#include <random_generator.hpp>
#include <recipe.hpp>
#include <util/unique_any.hpp>
#include <vector>

namespace arb {
    class lif_cell_group_mc: public cell_group {
    public:
        using value_type = double;

        lif_cell_group_mc() = default;

        // Constructor containing gid of first cell in a group and a container of all cells.
        lif_cell_group_mc(std::vector<cell_gid_type> gids, const recipe& rec);

        virtual cell_kind get_cell_kind() const override;
        virtual void reset() override;
        virtual void set_binning_policy(binning_kind policy, time_type bin_interval) override;
        virtual void advance(epoch epoch, time_type dt, const event_lane_subrange& events) override;

        virtual const std::vector<spike>& spikes() const override;
        virtual void clear_spikes() override;

      // Sampler association methods below should be thread-safe, as they might be invoked
      // from a sampler call back called from a different cell group running on a different thread.
        virtual void add_sampler(sampler_association_handle, cell_member_predicate, schedule, sampler_function, sampling_policy) override;
        virtual void remove_sampler(sampler_association_handle) override;
        virtual void remove_all_samplers() override;

    private:
        // Samples next poisson spike.
        void sample_next_poisson(cell_gid_type lid);

        // Returns the time of the next poisson event for given neuron,
        // taking into accout the delay of poisson spikes,
        // without sampling a new Poisson event time.
        template <typename Pred>
        util::optional<time_type> next_poisson_event(cell_gid_type lid, time_type tfinal, Pred should_pop);

        // Returns the next most recent event that is yet to be processed.
        // It can be either Poisson event or the queue event.
        // Only events that happened before tfinal are considered.
        template <typename Pred>
        util::optional<postsynaptic_spike_event> next_event(cell_gid_type lid, time_type tfinal, pse_vector& event_lane, Pred should_pop);

        // Advances a single cell (lid) with the exact solution (jumps can be arbitrary).
        // Parameter dt is ignored, since we make jumps between two consecutive spikes.
        void advance_cell(time_type tfinal, time_type dt, cell_gid_type lid, pse_vector& event_lane);

        // List of the gids of the cells in the group.
        std::vector<cell_gid_type> gids_;

        // Hash table for converting gid to local index
        std::unordered_map<cell_gid_type, cell_gid_type> gid_index_map_;

        // Cells that belong to this group.
        std::vector<lif_cell_description> cells_;

        // Spikes that are generated (not necessarily sorted).
        std::vector<spike> spikes_;

        // Pending events per cell.
        // std::vector<event_queue<postsynaptic_spike_event> > cell_events_;
        std::vector<std::vector<postsynaptic_spike_event> > cell_events_;

        // Time when the cell was last updated.
        std::vector<time_type> last_time_updated_;
        std::vector<unsigned> next_queue_event_index;

        // External spike generation.

        // lambda = 1/(n_poiss * rate) for each cell.
        std::vector<double> lambda_;

        // Sampled next Poisson event time for each cell.
        std::vector<time_type> next_poiss_time_;

        // Counts poisson events. 
        // Used as an argument to random123 (since partially describes a state)
        std::vector<unsigned> poiss_event_counter_;

        cell_gid_type gid_to_index(cell_gid_type gid) const {
          auto it = gid_index_map_.find(gid);
          EXPECTS(it!=gid_index_map_.end());
          return it->second;
        }
    };
} // namespace arb
