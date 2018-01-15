#pragma once

#include <array>
#include <unordered_map>
#include <vector>

#include <backends.hpp>
#include <cell_group.hpp>
#include <common_types.hpp>
#include <communication/communicator.hpp>
#include <communication/global_policy.hpp>
#include <domain_decomposition.hpp>
#include <epoch.hpp>
#include <recipe.hpp>
#include <sampling.hpp>
#include <thread_private_spike_store.hpp>
#include <util/nop.hpp>
#include <util/handle_set.hpp>
#include <util/unique_any.hpp>

namespace arb {

class model {
public:
    using communicator_type = communication::communicator<communication::global_policy>;
    using spike_export_function = std::function<void(const std::vector<spike>&)>;

    model(const recipe& rec, const domain_decomposition& decomp);

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
    // Must be called before calling model::run, and must contain events that
    // are to be delivered at or after the current model time.
    void inject_events(const pse_vector& events);

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
    std::vector<std::vector<event_generator_ptr>> event_generators_;

    using local_spike_store_type = thread_private_spike_store;
    util::double_buffer<local_spike_store_type> local_spikes_;

    spike_export_function global_export_callback_ = util::nop_function;
    spike_export_function local_export_callback_ = util::nop_function;

    // Hash table for looking up the the local index of a cell with a given gid
    std::unordered_map<cell_gid_type, cell_size_type> gid_to_local_;

    util::optional<cell_size_type> local_cell_index(cell_gid_type);

    communicator_type communicator_;

    // Convenience functions that map the spike buffers onto the appropriate
    // integration interval.
    //
    // To overlap communication and computation, integration intervals of
    // size Delta/2 are used, where Delta is the minimum delay in the global
    // system.
    // From the frame of reference of the current integration period we
    // define three intervals: previous, current and future
    // Then we define the following :
    //      current_spikes : spikes generated in the current interval
    //      previous_spikes: spikes generated in the preceding interval

    local_spike_store_type& current_spikes()  { return local_spikes_.get(); }
    local_spike_store_type& previous_spikes() { return local_spikes_.other(); }

    // Pending events to be delivered.
    std::array<std::vector<pse_vector>, 2> event_lanes_;
    std::vector<pse_vector> pending_events_;

    // Sampler associations handles are managed by a helper class.
    util::handle_set<sampler_association_handle> sassoc_handles_;
};

} // namespace arb
