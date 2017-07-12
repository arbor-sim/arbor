#include <model.hpp>

#include <vector>

#include <backends.hpp>
#include <cell_group.hpp>
#include <cell_group_factory.hpp>
#include <domain_decomposition.hpp>
#include <recipe.hpp>
#include <util/span.hpp>
#include <util/unique_any.hpp>
#include <profiling/profiler.hpp>

namespace nest {
namespace mc {

model::model(const recipe& rec, const domain_decomposition& decomp):
    domain_(decomp)
{
    // set up communicator based on partition
    communicator_ = communicator_type(domain_.gid_group_partition());

    // generate the cell groups in parallel, with one task per cell group
    cell_groups_.resize(domain_.num_local_groups());

    // thread safe vector for constructing the list of probes in parallel
    threading::parallel_vector<probe_record> probe_tmp;

    threading::parallel_for::apply(0, cell_groups_.size(),
        [&](cell_gid_type i) {
            PE("setup", "cells");

            auto group = domain_.get_group(i);
            std::vector<util::unique_any> cell_descriptions(group.end-group.begin);

            for (auto gid: util::make_span(group.begin, group.end)) {
                auto i = gid-group.begin;
                cell_descriptions[i] = rec.get_cell_description(gid);
            }

            cell_groups_[i] = cell_group_factory(
                    group.kind, group.begin, cell_descriptions, domain_.backend());
            PL(2);
        });

    // store probes
    for (const auto& c: cell_groups_) {
        util::append(probes_, c->probes());
    }

    // generate the network connections
    for (cell_gid_type i: util::make_span(domain_.cell_begin(), domain_.cell_end())) {
        for (const auto& cc: rec.connections_on(i)) {
            connection conn{cc.source, cc.dest, cc.weight, cc.delay};
            communicator_.add_connection(conn);
        }
    }
    communicator_.construct();

    // Allocate an empty queue buffer for each cell group
    // These must be set initially to ensure that a queue is available for each
    // cell group for the first time step.
    current_events().resize(num_groups());
    future_events().resize(num_groups());
}

void model::reset() {
    t_ = 0.;
    for (auto& group: cell_groups_) {
        group->reset();
    }

    communicator_.reset();

    for(auto& q : current_events()) {
        q.clear();
    }
    for(auto& q : future_events()) {
        q.clear();
    }

    current_spikes().clear();
    previous_spikes().clear();

    util::profilers_restart();
}

time_type model::run(time_type tfinal, time_type dt) {
    // Calculate the size of the largest possible time integration interval
    // before communication of spikes is required.
    // If spike exchange and cell update are serialized, this is the
    // minimum delay of the network, however we use half this period
    // to overlap communication and computation.
    time_type t_interval = communicator_.min_delay()/2;
    time_type tuntil;

    // task that updates cell state in parallel.
    auto update_cells = [&] () {
        threading::parallel_for::apply(
            0u, cell_groups_.size(),
             [&](unsigned i) {
                auto &group = cell_groups_[i];

                PE("stepping","events");
                group->enqueue_events(current_events()[i]);
                PL();

                 PE("cells");
                group->advance(tuntil, dt);
                 PL();
                PE("events");
                current_spikes().insert(group->spikes());
                group->clear_spikes();
                PL(2);
            });
    };

    // task that performs spike exchange with the spikes generated in
    // the previous integration period, generating the postsynaptic
    // events that must be delivered at the start of the next
    // integration period at the latest.
    auto exchange = [&] () {
        PE("stepping", "communication");

        PE("exchange");
        auto local_spikes = previous_spikes().gather();
        auto global_spikes = communicator_.exchange(local_spikes);
        PL();

        PE("spike output");
        local_export_callback_(local_spikes);
        global_export_callback_(global_spikes.values());
        PL();

        PE("events");
        future_events() = communicator_.make_event_queues(global_spikes);
        PL();

        PL(2);
    };

    while (t_<tfinal) {
        tuntil = std::min(t_+t_interval, tfinal);

        event_queues_.exchange();
        local_spikes_.exchange();

        // empty the spike buffers for the current integration period.
        // these buffers will store the new spikes generated in update_cells.
        current_spikes().clear();

        // run the tasks, overlapping if the threading model and number of
        // available threads permits it.
        threading::task_group g;
        g.run(exchange);
        g.run(update_cells);
        g.wait();

        t_ = tuntil;
    }

    // Run the exchange one last time to ensure that all spikes are output
    // to file.
    event_queues_.exchange();
    local_spikes_.exchange();
    exchange();

    return t_;
}

void model::attach_sampler(cell_member_type probe_id, sampler_function f, time_type tfrom) {
    const auto idx = domain_.local_group_from_gid(probe_id.gid);

    // only attach samplers for local cells
    if (idx) {
        cell_groups_[*idx]->add_sampler(probe_id, f, tfrom);
    }
}

const std::vector<probe_record>& model::probes() const {
    return probes_;
}

std::size_t model::num_spikes() const {
    return communicator_.num_spikes();
}

std::size_t model::num_groups() const {
    return cell_groups_.size();
}

std::size_t model::num_cells() const {
    return domain_.num_local_cells();
}

void model::set_binning_policy(binning_kind policy, time_type bin_interval) {
    for (auto& group: cell_groups_) {
        group->set_binning_policy(policy, bin_interval);
    }
}

cell_group& model::group(int i) {
    return *cell_groups_[i];
}

void model::set_global_spike_callback(spike_export_function export_callback) {
    global_export_callback_ = export_callback;
}

void model::set_local_spike_callback(spike_export_function export_callback) {
    local_export_callback_ = export_callback;
}

} // namespace mc
} // namespace nest
