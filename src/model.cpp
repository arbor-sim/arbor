#include <set>
#include <vector>

#include <backends.hpp>
#include <cell_group.hpp>
#include <cell_group_factory.hpp>
#include <domain_decomposition.hpp>
#include <merge_events.hpp>
#include <model.hpp>
#include <recipe.hpp>
#include <util/filter.hpp>
#include <util/span.hpp>
#include <util/unique_any.hpp>
#include <profiling/profiler.hpp>

namespace arb {

model::model(const recipe& rec, const domain_decomposition& decomp):
    communicator_(rec, decomp)
{
    event_generators_.resize(communicator_.num_local_cells());
    cell_local_size_type lidx = 0;
    const auto& grps = decomp.groups;
    for (auto i: util::make_span(0, grps.size())) {
        for (auto gid: grps[i].gids) {
            // Store mapping of gid to local cell index.
            gid_to_local_[gid] = lidx;

            // Set up the event generators for cell gid.
            auto rec_gens = rec.event_generators(gid);
            auto& gens = event_generators_[lidx];
            if (rec_gens.size()) {
                // Allocate two empty event generators that will be used to
                // merge events from the communicator and those already queued
                // for delivery in future epochs.
                gens.reserve(2+rec_gens.size());
                gens.resize(2);
                for (auto& g: rec_gens) {
                    gens.push_back(std::move(g));
                }
            }
            lidx++;
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


    // Create event lane buffers.
    // There is one set for each epoch: current (0) and next (1).
    // For each epoch there is one lane for each cell in the cell group.
    event_lanes_[0].resize(communicator_.num_local_cells());
    event_lanes_[1].resize(communicator_.num_local_cells());
  }

  void model::reset() {
    t_ = 0.;

    for (auto& group: cell_groups_) {
      group->reset();
    }

    for (auto& lanes: event_lanes_) {
      for (auto& lane: lanes) {
        lane.clear();
      }
    }

    for (auto& lane: event_generators_) {
        for (auto& gen: lane) {
            if (gen) {
                gen->reset();
            }
        }
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

                                       auto queues = util::subrange_view(
                                                                         event_lanes(epoch_.id),
                                                                         communicator_.group_queue_range(i));
                                       group->advance(epoch_, dt, queues);
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

        PE("events","from-spikes");
        auto events = communicator_.make_event_queues(global_spikes);
        PL();

        PE("enqueue");
        threading::parallel_for::apply(0, communicator_.num_local_cells(),
            [&](cell_size_type i) {
                const auto epid = epoch_.id;
                merge_events(
                    epoch_.tfinal,
                    epoch_.tfinal+std::min(t_+t_interval, tfinal),
                    event_lanes(epid)[i],
                    events[i],
                    event_generators_[i],
                    event_lanes(epid+1)[i]);
            });
        PL(2);

        PL(2);
    };

    tuntil = std::min(t_+t_interval, tfinal);
    epoch_ = epoch(0, tuntil);
    while (t_<tfinal) {
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

      tuntil = std::min(t_+t_interval, tfinal);
      epoch_.advance(tuntil);
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

  std::vector<pse_vector>& model::event_lanes(std::size_t epoch_id) {
    return event_lanes_[epoch_id%2];
  }

  void model::set_binning_policy(binning_kind policy, time_type bin_interval) {
    for (auto& group: cell_groups_) {
      group->set_binning_policy(policy, bin_interval);
    }
  }

  void model::set_global_spike_callback(spike_export_function export_callback) {
    global_export_callback_ = export_callback;
  }

  void model::set_local_spike_callback(spike_export_function export_callback) {
    local_export_callback_ = export_callback;
  }

  util::optional<cell_size_type> model::local_cell_index(cell_gid_type gid) {
    auto it = gid_to_local_.find(gid);
    return it==gid_to_local_.end()?
        util::nullopt:
        util::optional<cell_size_type>(it->second);
}

  void model::inject_events(const pse_vector& events) {
    auto& lanes = event_lanes(epoch_.id);

    // Append all events that are to be delivered to local cells to the
    // appropriate lane. At the same time, keep track of which lanes have been
    // modified, because the lanes will have to be sorted once all events have
    // been added.
    pse_vector local_events;
    std::set<cell_size_type> modified_lanes;
    for (auto& e: events) {
      if (e.time<t_) {
        throw std::runtime_error("model::inject_events(): attempt to inject an event at time " + std::to_string(e.time) + ", when model state is at time " + std::to_string(t_));
      }
      if (auto lidx = local_cell_index(e.target.gid)) {
        lanes[*lidx].push_back(e);
        modified_lanes.insert(*lidx);
      }
    }

    // Sort events in the event lanes that were modified
    for (auto l: modified_lanes) {
      util::sort(lanes[l]);
    }
}
} // namespace arb
