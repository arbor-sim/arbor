#pragma once
#include <threading/timer.hpp>
#include <cell_group.hpp>
#include <event_queue.hpp>
#include <lif_cell_description.hpp>
#include <profiling/profiler.hpp>
#include <util/unique_any.hpp>
#include <vector>
#include <backends/gpu/stack.hpp>
#include <memory/memory.hpp>
#include <memory/managed_ptr.hpp>
#include <Random123/threefry.h>

namespace nest {
namespace mc {


struct threshold_crossing {
    cell_gid_type index;    // index of variable
    time_type time;    // time of crossing
 };

class lif_cell_group_gpu: public cell_group {
public:
    using value_type = double;

    lif_cell_group_gpu() = default;

    // Constructor containing gid of first cell in a group and a container of all cells.
    lif_cell_group_gpu(cell_gid_type first_gid, const std::vector<util::unique_any>& cells);

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

    template <typename T>
    using managed_vector = std::vector<T, memory::managed_allocator<T> >;

    // Events management.
    managed_vector<postsynaptic_spike_event> event_buffer;
    managed_vector<unsigned> cell_begin;
    managed_vector<unsigned> cell_end;
    std::vector<event_queue<postsynaptic_spike_event> > cell_events_;

    // LIF parameters.
    managed_vector<double> tau_m;
    managed_vector<double> V_th;
    managed_vector<double> C_m;
    managed_vector<double> E_L;
    managed_vector<double> V_m;
    managed_vector<double> V_reset;
    managed_vector<double> t_ref;

    // External Poisson input parameters.
    managed_vector<unsigned> n_poiss;
    managed_vector<float> rate;
    managed_vector<float> w_poiss;
    managed_vector<float> d_poiss;

    memory::device_vector<unsigned> poiss_event_counter;
    // Samples next poisson spike.
    void sample_next_poisson(cell_gid_type lid);

    // Gid of first cell in group.
    cell_gid_type gid_base_;

    std::vector<lif_cell_description> cells_;

    // Spikes that are generated (not necessarily sorted).
    std::vector<spike> spikes_;

    // Time when the cell was last updated.
    managed_vector<time_type> last_time_updated_;

    // External spike generation.
    // lambda = 1/(n_poiss * rate) for each cell.
    managed_vector<double> lambda_;

    // Random number generators.
    // Each cell has a separate generator in order to achieve the independence.
    managed_vector<std::mt19937> generator_;

    // Unit exponential distribution (with mean 1).
    std::exponential_distribution<time_type> exp_dist_ = std::exponential_distribution<time_type>(1.0);

    // Sampled next Poisson event time for each cell.
    managed_vector<time_type> next_poiss_time_;

};
} // namespace mc
} // namespace nest
