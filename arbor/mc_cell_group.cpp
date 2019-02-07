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

ARB_DEFINE_LEXICOGRAPHIC_ORDERING(arb::target_handle,(a.mech_id,a.mech_index,a.intdom_index),(b.mech_id,b.mech_index,b.intdom_index))
ARB_DEFINE_LEXICOGRAPHIC_ORDERING(arb::deliverable_event,(a.time,a.handle,a.weight),(b.time,b.handle,b.weight))

mc_cell_group::mc_cell_group(const std::vector<cell_gid_type>& gids, const recipe& rec, fvm_lowered_cell_ptr lowered):
    lowered_(std::move(lowered))
{
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
    lowered_->initialize(gids_, rec, sc_ids_, target_handles_, probe_map_);

    // Create a list of the global identifiers for the spike sources
    for (auto source_gid: gids_) {
        for (cell_lid_type lid = 0; lid<rec.num_sources(source_gid); ++lid) {
            spike_sources_.push_back({source_gid, lid});
        }
    }
    spike_sources_.shrink_to_fit();
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

    cell_size_type ev_begin = 0, ev_mid = 0, ev_end = 0;
    // skip event binning if empty lanes are passed
    if (event_lanes.size()) {

        std::vector<cell_size_type> sorted_intdom(sc_ids_.size());
        cell_size_type n = 0;
        std::generate(sorted_intdom.begin(), sorted_intdom.end(), [&]{ return n++; });
        std::sort(sorted_intdom.begin(), sorted_intdom.end(),
            [&](cell_size_type a, cell_size_type b) {
            return sc_ids_[a] < sc_ids_[b];
        });

        for(auto j: sorted_intdom) {
            printf("%d\n", j);
        }
        printf("\n");

        /*for (auto lid: util::count_along(gids_)) {
            auto& lane = event_lanes[perm_gids_[lid]];
            for (auto e: lane) {
                if (e.time>=ep.tfinal) break;
                e.time = binners_[lid].bin(e.time, tstart);
                auto h = target_handles_[target_handle_divisions_[lid]+e.target.index];
                auto ev = deliverable_event(e.time, h, e.weight);
                staged_events_.push_back(ev);
            }
        }*/

       /* cell_size_type cg_id=0;

        auto sc_part = util::partition_view(sc_bounds_);
        for (auto sci: count_along(sc_part)) {

            for(auto dd_id : util::subrange_view(perm_gids_, sc_part[sci])) {
                unsigned count_staged = 0;
                auto& lane = event_lanes[dd_id];
                for (auto e: lane) {
                    if (e.time>=ep.tfinal) break;
                    e.time = binners_[cg_id].bin(e.time, tstart);
                    auto h = target_handles_[target_handle_divisions_[cg_id]+e.target.index];
                    auto ev = deliverable_event(e.time, h, e.weight);
                    staged_events_.push_back(ev);
                    count_staged++;
                }
                cg_id++;

                ev_end += count_staged;
                std::inplace_merge(staged_events_.begin() + ev_begin,
                                   staged_events_.begin() + ev_mid,
                                   staged_events_.begin() + ev_end);

                ev_mid += count_staged;
            }
            ev_begin = ev_end;
        }*/
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
