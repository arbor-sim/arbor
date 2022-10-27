#include <arbor/arbexcept.hpp>

#include "label_resolution.hpp"
#include "lif_cell_group.hpp"
#include "profile/profiler_macro.hpp"
#include "util/rangeutil.hpp"
#include "util/span.hpp"
#include "util/filter.hpp"
#include "util/maputil.hpp"

using namespace arb;

// Constructor containing gid of first cell in a group and a container of all cells.
lif_cell_group::lif_cell_group(const std::vector<cell_gid_type>& gids, const recipe& rec, cell_label_range& cg_sources, cell_label_range& cg_targets):
    gids_(gids)
{
    for (auto gid: gids_) {
        auto probes = rec.get_probes(gid);
        for (const auto lid: util::count_along(probes)) {
            const auto& probe = probes[lid];
            cell_member_type id = {gid, static_cast<cell_lid_type>(lid)};
            probe_tags_[id] = probe.tag;
            probe_meta_[id] = {};
            if (probe.address.type() == typeid(lif_probe_voltage)) {
                probe_kinds_[id] = lif_probe_kind::voltage;
            }
            else {
                throw bad_cell_probe{cell_kind::lif, gid};
            }
        }
    }
    // Default to no binning of events
    lif_cell_group::set_binning_policy(binning_kind::none, 0);

    cells_.reserve(gids_.size());
    last_time_updated_.resize(gids_.size());
    next_time_updatable_.resize(gids_.size());

    for (auto lid: util::make_span(gids_.size())) {
        cells_.push_back(util::any_cast<lif_cell>(rec.get_cell_description(gids_[lid])));
    }

    for (const auto& c: cells_) {
        cg_sources.add_cell();
        cg_targets.add_cell();
        cg_sources.add_label(c.source, {0, 1});
        cg_targets.add_label(c.target, {0, 1});
    }
}

cell_kind lif_cell_group::get_cell_kind() const {
    return cell_kind::lif;
}

void lif_cell_group::advance(epoch ep, time_type dt, const event_lane_subrange& event_lanes) {
    PE(advance:lif);
    for (auto lid: util::make_span(gids_.size())) {
        // Advance each cell independently.
        advance_cell(ep.t1, dt, lid, event_lanes);
    }
    PL();
}

const std::vector<spike>& lif_cell_group::spikes() const {
    return spikes_;
}

void lif_cell_group::clear_spikes() {
    spikes_.clear();
}

// TODO: implement sampler
void lif_cell_group::add_sampler(sampler_association_handle h,
                                 cell_member_predicate probeset_ids,
                                 schedule sched,
                                 sampler_function fn,
                                 sampling_policy policy) {
    std::lock_guard<std::mutex> guard(sampler_mex_);
    std::vector<cell_member_type> probeset =
        util::assign_from(util::filter(util::keys(probe_tags_), probeset_ids));
    auto assoc = arb::sampler_association{std::move(sched),
                                          std::move(fn),
                                          std::move(probeset),
                                          policy};
    auto result = samplers_.insert({h, std::move(assoc)});
    arb_assert(result.second);
}

void lif_cell_group::remove_sampler(sampler_association_handle h) {
    std::lock_guard<std::mutex> guard(sampler_mex_);
    samplers_.erase(h);
}
void lif_cell_group::remove_all_samplers() {
    std::lock_guard<std::mutex> guard(sampler_mex_);
    samplers_.clear();
}

// TODO: implement binner_
void lif_cell_group::set_binning_policy(binning_kind policy, time_type bin_interval) {
}

void lif_cell_group::reset() {
    spikes_.clear();
    util::fill(last_time_updated_, 0.);
    util::fill(next_time_updatable_, 0.);
}

// Advances a single cell (lid) with the exact solution (jumps can be arbitrary).
// Parameter dt is ignored, since we make jumps between two consecutive spikes.
void lif_cell_group::advance_cell(time_type tfinal, time_type dt, cell_gid_type lid, const event_lane_subrange& event_lanes) {
    // our gid
    const auto gid = gids_[lid];
    // time of last update.
    auto t = last_time_updated_[lid];
    auto& cell = cells_[lid];
    // integrate until tfinal using the exact solution of membrane voltage differential equation.
    // spikes to process
    const auto n_events = event_lanes.size() ? event_lanes[lid].size() : 0;
    int e_idx = 0;
    // collected sampling data
    std::unordered_map<sampler_association_handle,
                       std::unordered_map<cell_member_type,
                                          std::vector<sample_record>>> sampled;
    // samples to process
    std::vector<std::pair<time_type, sampler_association_handle>> samples;
    std::size_t count = 0;
    {
        std::lock_guard<std::mutex> guard(sampler_mex_);
        for (auto& [hdl, assoc]: samplers_) {
            // Count up the samplers touching _our_ gid
            std::size_t delta = 0;
            for (const auto& pid: assoc.probeset_ids) delta += pid.gid == gid;
            if (delta == 0) continue;
            // Construct sampling times
            const auto& times = util::make_range(assoc.sched.events(t, tfinal));
            count += delta*times.size();
            // We only ever use exact sampling, so we over-provision for lax and
            // never look at the policy
            for (auto t: times) samples.emplace_back(t, hdl);
        }
    }
    std::sort(samples.begin(), samples.end());
    const auto n_samples = samples.size();
    int s_idx = 0;
    // Now allocate some scratch space for the probed values, if we don't,
    // re-alloc might move our data
    std::vector<value_type> sampled_voltages;
    sampled_voltages.reserve(count);
    for (;;) {
        const auto e_time = e_idx < n_events ? event_lanes[lid][e_idx].time : tfinal;
        const auto s_time = s_idx < n_samples ? samples[s_idx].first : tfinal;
        const auto time = std::min(e_time, s_time);
        const auto next = next_time_updatable_[lid];
        // bail at end of integration interval
        if (time >= tfinal) break;
        // Check what to do, we put events before samples, if they collide we'll
        // see the update in sampling.
        // We need to incorporate an event
        if (time == e_time) {
            const auto& event_lane = event_lanes[lid];
            // process all events at time t
            auto weight = 0;
            for (; e_idx < n_events && event_lane[e_idx].time <= time; ++e_idx) {
                weight += event_lane[e_idx].weight;
            }
            // skip event if a neuron is in refactory period
            if (time >= next) {
                // Let the membrane potential decay.
                cell.V_m *= exp((t - time) / cell.tau_m);
                // Add jump due to spike.
                cell.V_m += weight / cell.C_m;
                t = time;
                // If crossing threshold occurred
                if (cell.V_m >= cell.V_th) {
                    // save spike
                    spikes_.push_back({{gid, 0}, t});
                    // Advance the last_time_updated to account for the refractory period.
                    next_time_updatable_[lid] = t + cell.t_ref;
                    // Reset the voltage to resting potential.
                    cell.V_m = cell.E_L;
                }
            }
        }
        // We need to probe, so figure out what to do.
        if (time == s_time) {
            // Consume all sample events at this time
            for (; s_idx < n_samples && samples[s_idx].first <= time; ++s_idx) {
                const auto& [s_time, hdl] = samples[s_idx];
                for (const auto& key: samplers_[hdl].probeset_ids) {
                    const auto& kind = probe_kinds_[key];
                    // This is the only thing we know how to do: Probing U(t)
                    switch (kind) {
                        case lif_probe_kind::voltage: {
                            // Compute, but do not _set_ V_m
                            auto U = cell.V_m;
                            // Honour the refractory period
                            if (time >= next) U *= exp((t - time) / cell.tau_m);
                            // Store U for later use.
                            sampled_voltages.push_back(U);
                            // Set up reference to sampled value
                            auto data_ptr = sampled_voltages.data() + sampled_voltages.size() - 1;
                            sampled[hdl][key].push_back(sample_record{time, {data_ptr}});
                            break;
                        }
                        default:
                            throw arbor_internal_error{"Invalid LIF probe kind"};
                    }
                }
            }
        }
        if ((time != s_time) && (time != e_time)) {
            throw arbor_internal_error{"LIF cell group: Must select either sample or spike event; got neither."};
        }
    }
    // Now we need to call all sampler callbacks with the data we have collected
    {
        std::lock_guard<std::mutex> guard(sampler_mex_);
        for (const auto& [k, vs]: sampled) {
            const auto& fun = samplers_[k].sampler;
            for (const auto& [id, us]: vs) {
                fun(probe_metadata{id, {}, 0, nullptr}, us.size(), us.data());
            }
        }
    }
    // This is the last time a cell was updated.
    last_time_updated_[lid] = t;
}

std::vector<probe_metadata> lif_cell_group::get_probe_metadata(cell_member_type key) const {
    if (probe_meta_.count(key)) {
        return {probe_metadata{key, {}, 0, {&probe_meta_.at(key)}}};
    } else {
        return {};
    }
}
