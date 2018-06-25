#pragma once

#include <array>
#include <memory>
#include <unordered_map>
#include <vector>

#include <arbor/common_types.hpp>
#include <arbor/distributed_context.hpp>

#include "backends.hpp"
#include "cell_group.hpp"
#include "communication/communicator.hpp"
#include "domain_decomposition.hpp"
#include "epoch.hpp"
#include "recipe.hpp"
#include "sampling.hpp"
#include "util/nop.hpp"
#include "util/handle_set.hpp"

namespace arb {

class spike_double_buffer;

class simulation {
public:
    using spike_export_function = std::function<void(const std::vector<spike>&)>;

    simulation(const recipe& rec, const domain_decomposition& decomp, const distributed_context* ctx);

    void reset();

    time_type run(time_type tfinal, time_type dt);

    // Note: sampler functions may be invoked from a different thread than that
    // which called the `run` method.

    sampler_association_handle add_sampler(cell_member_predicate probe_ids,
        schedule sched, sampler_function f, sampling_policy policy = sampling_policy::lax);

    void remove_sampler(sampler_association_handle);

    void remove_all_samplers();

    std::size_t num_spikes() const;

    // Set event binning policy on all our groups.
    void set_binning_policy(binning_kind policy, time_type bin_interval);

    // Register a callback that will perform a export of the global
    // spike vector.
    void set_global_spike_callback(spike_export_function export_callback);

    // Register a callback that will perform a export of the rank local
    // spike vector.
    void set_local_spike_callback(spike_export_function export_callback);

    // Add events directly to targets.
    // Must be called before calling simulation::run, and must contain events that
    // are to be delivered at or after the current simulation time.
    void inject_events(const pse_vector& events);

    ~simulation();

private:
    // Private helper function that sets up the event lanes for an epoch.
    // See comments on implementation for more information.
    void setup_events(time_type t_from, time_type time_to, std::size_t epoch_id);

    std::vector<pse_vector>& event_lanes(std::size_t epoch_id);

    std::size_t num_groups() const;

    // keep track of information about the current integration interval
    epoch epoch_;

    time_type t_ = 0.;
    time_type min_delay_;
    std::vector<cell_group_ptr> cell_groups_;

    // one set of event_generators for each local cell
    std::vector<std::vector<event_generator>> event_generators_;

    std::unique_ptr<spike_double_buffer> local_spikes_;

    spike_export_function global_export_callback_ = util::nop_function;
    spike_export_function local_export_callback_ = util::nop_function;

    // Hash table for looking up the the local index of a cell with a given gid
    std::unordered_map<cell_gid_type, cell_size_type> gid_to_local_;

    util::optional<cell_size_type> local_cell_index(cell_gid_type);

    communicator communicator_;

    // Pending events to be delivered.
    std::array<std::vector<pse_vector>, 2> event_lanes_;
    std::vector<pse_vector> pending_events_;

    // Sampler associations handles are managed by a helper class.
    util::handle_set<sampler_association_handle> sassoc_handles_;
};

} // namespace arb
