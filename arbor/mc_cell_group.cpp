#include <functional>
#include <unordered_set>
#include <vector>

#include <arbor/assert.hpp>
#include <arbor/common_types.hpp>
#include <arbor/sampling.hpp>
#include <arbor/recipe.hpp>
#include <arbor/spike.hpp>

#include "backends/event.hpp"
#include "cell_group.hpp"
#include "event_binner.hpp"
#include "event_queue.hpp"
#include "fvm_lowered_cell.hpp"
#include "mc_cell_group.hpp"
#include "profile/profiler_macro.hpp"
#include "sampler_map.hpp"
#include "util/filter.hpp"
#include "util/maputil.hpp"
#include "util/partition.hpp"
#include "util/range.hpp"
#include "util/span.hpp"

namespace arb {

mc_cell_group::mc_cell_group(const std::vector<cell_gid_type>& gids, const recipe& rec, fvm_lowered_cell_ptr lowered):
    lowered_(std::move(lowered))
{
    generate_deps_gids(rec, gids);
    // Default to no binning of events
    set_binning_policy(binning_kind::none, 0);

    // Build lookup table for gid to local index.
    for (auto i: util::count_along(gids_)) {
        gid_index_map_[gids_[i]] = i;
    }

    // Create lookup structure for target ids.
    util::make_partition(target_handle_divisions_,
            util::transform_view(gids_, [&rec](cell_gid_type i) { return rec.num_targets(i); }));
    std::size_t n_targets = target_handle_divisions_.back();

    // Pre-allocate space to store handles, probe map.
    auto n_probes = util::sum_by(gids_, [&rec](cell_gid_type i) { return rec.num_probes(i); });
    probe_map_.reserve(n_probes);
    target_handles_.reserve(n_targets);

    // Construct cell implementation, retrieving handles and maps. 
    lowered_->initialize(gids_, deps_, rec, target_handles_, probe_map_);

    // Create a list of the global identifiers for the spike sources
    for (auto source_gid: gids_) {
        for (cell_lid_type lid = 0; lid<rec.num_sources(source_gid); ++lid) {
            spike_sources_.push_back({source_gid, lid});
        }
    }
    spike_sources_.shrink_to_fit();
}

// Fills gids_ and deps_: gids_ are sorted such that members of the same supercell are consecutive
void mc_cell_group::generate_deps_gids(const recipe& rec, std::vector<cell_gid_type> gids) {

    std::unordered_map<cell_gid_type, cell_size_type> gid_to_loc;
    for (auto i: util::count_along(gids)) {
        gid_to_loc[gids[i]] = i;
    }

    deps_.reserve(gids_.size());
    std::unordered_set<cell_gid_type> visited;
    std::queue<cell_gid_type> scq;

    std::sort(gids.begin(), gids.end());

    for (auto gid: gids) {
        if (visited.count(gid)) continue;
        visited.insert(gid);

        cell_size_type sc_size = 0;
        scq.push(gid);
        while (!scq.empty()) {
            auto g = scq.front();
            scq.pop();

            gids_.push_back(g);
            ++sc_size;

            for (auto gj: rec.gap_junctions_on(g)) {
                cell_gid_type peer =
                        gj.local.gid==g? gj.peer.gid:
                        gj.peer.gid==g?  gj.local.gid:
                        throw bad_cell_description(cell_kind::cable1d_neuron, g);

                if (!gid_to_loc.count(peer)) {
                    // actually an error in the domain decomposition...
                    throw bad_cell_description(cell_kind::cable1d_neuron, g);
                }

                if (!visited.count(peer)) {
                    visited.insert(peer);
                    scq.push(peer);
                }
            }
        }

        deps_.push_back(sc_size>1? sc_size: 0);
        deps_.insert(deps_.end(), sc_size-1, 0);
    }

    perm_gids_.reserve(gids_.size());
    for (auto gid: gids_) {
        perm_gids_.push_back(gid_to_loc[gid]);
    }
}

void mc_cell_group::reset() {
    spikes_.clear();

    sample_events_.clear();
    for (auto &assoc: sampler_map_) {
        assoc.sched.reset();
    }

    for (auto& b: binners_) {
        b.reset();
    }

    lowered_->reset();
}

void mc_cell_group::set_binning_policy(binning_kind policy, time_type bin_interval) {
    binners_.clear();
    binners_.resize(gids_.size(), event_binner(policy, bin_interval));
}

void mc_cell_group::advance(epoch ep, time_type dt, const event_lane_subrange& event_lanes) {
    time_type tstart = lowered_->time();

    PE(advance_eventsetup);
    staged_events_.clear();
    // skip event binning if empty lanes are passed
    if (event_lanes.size()) {
        for (auto lid: util::count_along(gids_)) {
            auto& lane = event_lanes[perm_gids_[lid]];
            for (auto e: lane) {
                if (e.time>=ep.tfinal) break;
                e.time = binners_[lid].bin(e.time, tstart);
                auto h = target_handles_[target_handle_divisions_[lid]+e.target.index];
                auto ev = deliverable_event(e.time, h, e.weight);
                staged_events_.push_back(ev);
            }
        }
    }
    PL();

    // Create sample events and delivery information.
    //
    // For each (schedule, sampler, probe set) in the sampler association
    // map that will be triggered in this integration interval, create
    // sample events for the lowered cell, one for each scheduled sample
    // time and probe in the probe set.
    //
    // Each event is associated with an offset into the sample data and
    // time buffers; these are assigned contiguously such that one call to
    // a sampler callback can be represented by a `sampler_call_info`
    // value as defined below, grouping together all the samples of the
    // same probe for this callback in this association.

    struct sampler_call_info {
        sampler_function sampler;
        cell_member_type probe_id;
        probe_tag tag;

        // Offsets are into lowered cell sample time and event arrays.
        sample_size_type begin_offset;
        sample_size_type end_offset;
    };

    PE(advance_samplesetup);
    std::vector<sampler_call_info> call_info;

    std::vector<sample_event> sample_events;
    sample_size_type n_samples = 0;
    sample_size_type max_samples_per_call = 0;

    for (auto& sa: sampler_map_) {
        auto sample_times = util::make_range(sa.sched.events(tstart, ep.tfinal));
        if (sample_times.empty()) {
            continue;
        }

        sample_size_type n_times = sample_times.size();
        max_samples_per_call = std::max(max_samples_per_call, n_times);

        for (cell_member_type pid: sa.probe_ids) {
            auto cell_index = gid_index_map_.at(pid.gid);
            auto p = probe_map_[pid];

            call_info.push_back({sa.sampler, pid, p.tag, n_samples, n_samples+n_times});

            for (auto t: sample_times) {
                sample_event ev{t, cell_index, {p.handle, n_samples++}};
                sample_events.push_back(ev);
            }
        }
    }

    // Sample events must be ordered by time for the lowered cell.
    util::sort_by(sample_events, [](const sample_event& ev) { return event_time(ev); });
    PL();

    // Run integration and collect samples, spikes.
    auto result = lowered_->integrate(ep.tfinal, dt, staged_events_, std::move(sample_events));

    // For each sampler callback registered in `call_info`, construct the
    // vector of sample entries from the lowered cell sample times and values
    // and then call the callback.

    PE(advance_sampledeliver);
    std::vector<sample_record> sample_records;
    sample_records.reserve(max_samples_per_call);

    for (auto& sc: call_info) {
        sample_records.clear();
        for (auto i = sc.begin_offset; i!=sc.end_offset; ++i) {
           sample_records.push_back(sample_record{time_type(result.sample_time[i]), &result.sample_value[i]});
        }

        sc.sampler(sc.probe_id, sc.tag, sc.end_offset-sc.begin_offset, sample_records.data());
    }
    PL();

    // Copy out spike voltage threshold crossings from the back end, then
    // generate spikes with global spike source ids. The threshold crossings
    // record the local spike source index, which must be converted to a
    // global index for spike communication.

    for (auto c: result.crossings) {
        spikes_.push_back({spike_sources_[c.index], time_type(c.time)});
    }
}

void mc_cell_group::add_sampler(sampler_association_handle h, cell_member_predicate probe_ids,
                                schedule sched, sampler_function fn, sampling_policy policy)
{
    std::vector<cell_member_type> probeset =
        util::assign_from(util::filter(util::keys(probe_map_), probe_ids));

    if (!probeset.empty()) {
        sampler_map_.add(h, sampler_association{std::move(sched), std::move(fn), std::move(probeset)});
    }
}

void mc_cell_group::remove_sampler(sampler_association_handle h) {
    sampler_map_.remove(h);
}

void mc_cell_group::remove_all_samplers() {
    sampler_map_.clear();
}

} // namespace arb
