#include <vector>

#include <backends.hpp>
#include <cell_group.hpp>
#include <cell_group_factory.hpp>
#include <domain_decomposition.hpp>
#include <model.hpp>
#include <recipe.hpp>
#include <util/span.hpp>
#include <util/unique_any.hpp>
#include <profiling/profiler.hpp>

#include <cuda_profiler_api.h>

namespace arb {

model::model(const recipe& rec, const domain_decomposition& decomp):
    communicator_(rec, decomp)
{
    for (auto i: util::make_span(0, decomp.groups.size())) {
        for (auto gid: decomp.groups[i].gids) {
            gid_groups_[gid] = i;
        }
    }

    // Generate the cell groups in parallel, with one task per cell group.
    cell_groups_.resize(decomp.groups.size());
    threading::parallel_for::apply(0, cell_groups_.size(),
        [&](cell_gid_type i) {
            PE("setup", "cells");
            cell_groups_[i] = cell_group_factory(rec, decomp.groups[i]);
            PL(2);
        });
}

void model::reset() {
    t_ = 0.;
    for (auto& group: cell_groups_) {
        group->reset();
    }

    communicator_.reset();

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
                PE("stepping");
                auto &group = cell_groups_[i];

                group->advance(tuntil, dt, epoch_);

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
        PE("from-spikes");
        auto events = communicator_.make_event_queues(global_spikes);
        PL();

        PE("enqueue");
        for (auto i: util::make_span(0, cell_groups_.size())) {
            cell_groups_[i]->enqueue_events(
                util::subrange_view(events, communicator_.group_queue_range(i)),
                tuntil, epoch_);
        }
        PL();

        PL(3);
    };

    //time_type ttt = 30.;
    while (t_<tfinal) {
        tuntil = std::min(t_+t_interval, tfinal);

        local_spikes_.exchange();

        // TODO
        //bool pon = (t_>(ttt-1) && t_<(ttt-1+10));
        //if (pon) cudaProfilerStart();

        // empty the spike buffers for the current integration period.
        // these buffers will store the new spikes generated in update_cells.
        current_spikes().clear();

        // run the tasks, overlapping if the threading model and number of
        // available threads permits it.
        threading::task_group g;
        g.run(exchange);
        g.run(update_cells);
        g.wait();

        // TODO
        //if (pon) cudaProfilerStop();

        t_ = tuntil;
        ++epoch_;
    }

    // Run the exchange one last time to ensure that all spikes are output
    // to file.
    local_spikes_.exchange();
    exchange();

    return t_;
}

sampler_association_handle model::add_sampler(cell_member_predicate probe_ids, schedule sched, sampler_function f, sampling_policy policy) {
    sampler_association_handle h = sassoc_handles_.acquire();

    threading::parallel_for::apply(0, cell_groups_.size(),
        [&](std::size_t i) {
            cell_groups_[i]->add_sampler(h, probe_ids, sched, f, policy);
        });

    return h;
}

void model::remove_sampler(sampler_association_handle h) {
    threading::parallel_for::apply(0, cell_groups_.size(),
        [&](std::size_t i) {
            cell_groups_[i]->remove_sampler(h);
        });

    sassoc_handles_.release(h);
}

void model::remove_all_samplers() {
    threading::parallel_for::apply(0, cell_groups_.size(),
        [&](std::size_t i) {
            cell_groups_[i]->remove_all_samplers();
        });

    sassoc_handles_.clear();
}

std::size_t model::num_spikes() const {
    return communicator_.num_spikes();
}

std::size_t model::num_groups() const {
    return cell_groups_.size();
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

} // namespace arb
