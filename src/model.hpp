#pragma once

#include <vector>

#include <backends.hpp>
#include <cell_group.hpp>
#include <common_types.hpp>
#include <domain_decomposition.hpp>
#include <communication/communicator.hpp>
#include <communication/global_policy.hpp>
#include <recipe.hpp>
#include <sampler_function.hpp>
#include <thread_private_spike_store.hpp>
#include <util/nop.hpp>
#include <util/rangeutil.hpp>
#include <util/unique_any.hpp>

namespace nest {
namespace mc {

class model {
public:
    using communicator_type = communication::communicator<communication::global_policy>;
    using spike_export_function = std::function<void(const std::vector<spike>&)>;

    model(const recipe& rec, const domain_decomposition& decomp);

    void reset();

    time_type run(time_type tfinal, time_type dt);

    void attach_sampler(cell_member_type probe_id, sampler_function f, time_type tfrom = 0);

    const std::vector<probe_record>& probes() const;

    std::size_t num_spikes() const;

    // Set event binning policy on all our groups.
    void set_binning_policy(binning_kind policy, time_type bin_interval);

    // access cell_group directly
    // TODO: depricate. Currently used in some validation tests to inject
    // events directly into a cell group. This should be done with a spiking
    // neuron.
    cell_group& group(int i);

    // register a callback that will perform a export of the global
    // spike vector
    void set_global_spike_callback(spike_export_function export_callback);

    // register a callback that will perform a export of the rank local
    // spike vector
    void set_local_spike_callback(spike_export_function export_callback);

private:
    std::size_t num_groups() const;

    time_type t_ = 0.;
    std::vector<cell_group_ptr> cell_groups_;
    std::vector<probe_record> probes_;

    using event_queue_type = typename communicator_type::event_queue;
    util::double_buffer<std::vector<event_queue_type>> event_queues_;

    using local_spike_store_type = thread_private_spike_store;
    util::double_buffer<local_spike_store_type> local_spikes_;

    spike_export_function global_export_callback_ = util::nop_function;
    spike_export_function local_export_callback_ = util::nop_function;

    // Hash table for looking up the group index of the cell_group that
    // contains gid
    std::unordered_map<cell_gid_type, cell_gid_type> gid_groups_;

    communicator_type communicator_;

    // Convenience functions that map the spike buffers and event queues onto
    // the appropriate integration interval.
    //
    // To overlap communication and computation, integration intervals of
    // size Delta/2 are used, where Delta is the minimum delay in the global
    // system.
    // From the frame of reference of the current integration period we
    // define three intervals: previous, current and future
    // Then we define the following :
    //      current_spikes : spikes generated in the current interval
    //      previous_spikes: spikes generated in the preceding interval
    //      current_events : events to be delivered at the start of
    //                       the current interval
    //      future_events  : events to be delivered at the start of
    //                       the next interval

    local_spike_store_type& current_spikes()  { return local_spikes_.get(); }
    local_spike_store_type& previous_spikes() { return local_spikes_.other(); }

    std::vector<event_queue_type>& current_events()  { return event_queues_.get(); }
    std::vector<event_queue_type>& future_events()   { return event_queues_.other(); }
};

} // namespace mc
} // namespace nest
