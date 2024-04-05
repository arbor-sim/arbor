#include "adex_cell_group.hpp"

#include <arbor/arbexcept.hpp>

#include "arbor/math.hpp"
#include "util/rangeutil.hpp"
#include "util/span.hpp"
#include "label_resolution.hpp"
#include "profile/profiler_macro.hpp"

#include <iostream>

using namespace arb;

// Constructor containing gid of first cell in a group and a container of all cells.
adex_cell_group::adex_cell_group(const std::vector<cell_gid_type>& gids,
                               const recipe& rec,
                               cell_label_range& cg_sources,
                               cell_label_range& cg_targets):
    gids_(gids) {

    for (auto gid: gids_) {
        const auto& cell = util::any_cast<adex_cell>(rec.get_cell_description(gid));
        // set up cell state
        cells_.push_back(cell);
        // tell our caller about this cell's connections
        cg_sources.add_cell();
        cg_targets.add_cell();
        cg_sources.add_label(hash_value(cell.source), {0, 1});
        cg_targets.add_label(hash_value(cell.target), {0, 1});
        // insert probes where needed
        auto probes = rec.get_probes(gid);
        for (const auto& probe: probes) {
            if (probe.address.type() == typeid(adex_probe_voltage)) {
                cell_address_type addr{gid, probe.tag};
                if (probes_.count(addr)) throw dup_cell_probe(cell_kind::adex, gid, probe.tag);
                probes_.insert_or_assign(addr, adex_probe_info{adex_probe_kind::voltage, {}});
            }
            else if (probe.address.type() == typeid(adex_probe_adaption)) {
                cell_address_type addr{gid, probe.tag};
                if (probes_.count(addr)) throw dup_cell_probe(cell_kind::adex, gid, probe.tag);
                probes_.insert_or_assign(addr, adex_probe_info{adex_probe_kind::adaption, {}});
            }
            else {
                throw bad_cell_probe{cell_kind::adex, gid};
            }
        }
        // set up the internal state
        next_update_.push_back(0);
        current_time_.push_back(0);
    }
}

cell_kind adex_cell_group::get_cell_kind() const {
    return cell_kind::adex;
}

void adex_cell_group::advance(epoch ep, time_type dt, const event_lane_subrange& event_lanes) {
    PE(advance:adex);
    for (auto lid: util::make_span(gids_.size())) {
        // Advance each cell independently.
        advance_cell(ep.t1, dt, lid, event_lanes);
    }
    PL();
}

const std::vector<spike>& adex_cell_group::spikes() const {
    return spikes_;
}

void adex_cell_group::clear_spikes() {
    spikes_.clear();
}

void adex_cell_group::add_sampler(sampler_association_handle h,
                                  cell_member_predicate probeset_ids,
                                  schedule sched,
                                  sampler_function fn) {
    std::lock_guard<std::mutex> guard(sampler_mex_);
    std::vector<cell_address_type> probeset;
    for (const auto& [k, v]: probes_) {
        if (probeset_ids(k)) probeset.push_back(k);
    }
    auto assoc = arb::sampler_association{std::move(sched),
                                          std::move(fn),
                                          std::move(probeset)};
    auto result = samplers_.insert({h, std::move(assoc)});
    arb_assert(result.second);
}

void adex_cell_group::remove_sampler(sampler_association_handle h) {
    std::lock_guard<std::mutex> guard(sampler_mex_);
    samplers_.erase(h);
}
void adex_cell_group::remove_all_samplers() {
    std::lock_guard<std::mutex> guard(sampler_mex_);
    samplers_.clear();
}

void adex_cell_group::reset() {
    spikes_.clear();
}

// integrate a single cell's state from current time `cur` to final time `end`.
// Extra parameters
// * the cell cannot be updated until time `nxt`, which might be in the past or future.
//
// We can be in three states:
// 1. nxt <= cur: we can simply update the cell without further consideration
// 2. cur < nxt <= end: we perform two steps:
//    a. cur - nxt: refractory period, just manipulate w
//    b. nxt - end: normal dynamics, add spike
// 3. nxt > end. Skip everything
void integrate_until(adex_lowered_cell& cell, const time_type end, const time_type& nxt, time_type& cur) {
    // perform pre-step to skip refractory period. This _might_ put cell state beyond the epoch end.
    if (nxt > cur) cur = std::min(nxt, end);
    // if we still have time left, perform the integration.
    if (nxt > end) return;
    auto delta = end - cur;
    auto dE = cell.V_m - cell.E_L;
    auto il = cell.g*dE;
    auto is = cell.g*cell.delta*exp((cell.V_m - cell.V_th)/cell.delta);
    auto dV = (is - il - cell.w)/cell.C_m;
    cell.V_m += delta*dV;

    auto dW = (cell.a*dE - cell.w)/cell.tau;
    cell.w += delta*dW;
    cur = end;
}

void check_spike(adex_lowered_cell& cell, const time_type time, time_type& nxt, const cell_gid_type gid, std::vector<spike>& spikes) {
    if (time > nxt && cell.V_m >= cell.V_th) {
        spikes.emplace_back(cell_member_type{gid, 0}, time);
        // reset membrane potential
        cell.V_m = cell.E_R;
        // schedule next update
        nxt = time + cell.t_ref;
        cell.w += cell.b;
    }
}

void adex_cell_group::advance_cell(time_type t_fin,
                                   time_type dt,
                                   cell_gid_type lid,
                                   const event_lane_subrange& event_lanes) {
    auto time = current_time_[lid];
    auto gid = gids_[lid];
    // Flattened sampler map
    std::vector<probe_metadata> sample_metadata;
    std::vector<sampler_association_handle> sample_callbacks;
    std::vector<std::vector<sample_record>> sample_records;

    struct sample_event {
        time_type time;
        adex_probe_kind kind;
        double* data;
    };

    std::vector<sample_event> sample_events;
    std::vector<double> sample_data;

    if (!samplers_.empty()) {
        auto tlast = time;
        std::vector<size_t> sample_sizes;
        std::size_t total_size = 0;
        {
            std::lock_guard<std::mutex> guard(sampler_mex_);
            for (auto& [hdl, assoc]: samplers_) {
                // No need to generate events
                if (assoc.probeset_ids.empty()) continue;
                // Construct sampling times, might give us the last time we sampled, so skip that.
                auto times = util::make_range(assoc.sched.events(tlast, t_fin));
                // while (!times.empty() && times.front() == tlast) times.left++;
                if (times.empty()) continue;
                for (unsigned idx = 0; idx < assoc.probeset_ids.size(); ++idx) {
                    const auto& pid = assoc.probeset_ids[idx];
                    if (pid.gid != gid) continue;
                    const auto& probe = probes_.at(pid);
                    sample_metadata.push_back({pid, idx, util::any_ptr{&probe.metadata}});
                    sample_callbacks.push_back(hdl);
                    sample_records.emplace_back();
                    auto& records = sample_records.back();
                    sample_sizes.push_back(times.size());
                    total_size += times.size();
                    for (auto t: times) {
                        records.push_back(sample_record{t, nullptr});
                        sample_events.push_back(sample_event{t, probe.kind, nullptr});
                    }
                }
            }
        }
        // Flat list of things to sample
        // NOTE: Need to allocate in one go, else reallocation will mess up the pointers!
        sample_data.resize(total_size);
        auto rx = 0;
        for (unsigned ix = 0; ix < sample_sizes.size(); ++ix) {
            auto size = sample_sizes[ix];
            for (int kx = 0; kx < size; ++kx) {
                sample_records[ix][kx].data = const_cast<const double*>(sample_data.data() + rx);
                sample_events[rx].data = sample_data.data() + rx;
                ++rx;
            }
        }
    }
    util::sort_by(sample_events, [](const auto& s) { return s.time; });
    auto n_samples = sample_events.size();

    auto& cell = cells_[lid];
    auto n_events = static_cast<int>(!event_lanes.empty() ? event_lanes[lid].size() : 0);
    auto evt_idx = 0;
    auto spl_idx = 0;
    while (time < t_fin) {
        auto t_end = std::min(t_fin, time + dt);
        // forward progress?
        arb_assert(t_end > time);
        auto V_0 = cell.V_m;
        auto W_0 = cell.w;
        // Process events in [time, time + dt)
        // delivering each at the exact time
        for (;; ++evt_idx) {
            if (evt_idx >= n_events) break;
            if (event_lanes[lid][evt_idx].time >= t_end) break;

            const auto& evt = event_lanes[lid][evt_idx];
            integrate_until(cell, evt.time, next_update_[lid], current_time_[lid]);
            // NOTE we _could check here instead or in addition.
            // check_spike(cell, evt.time, next_update_[lid], gid, spikes_);
            if (next_update_[lid] <= evt.time) cell.V_m += evt.weight/cell.C_m;
            check_spike(cell, evt.time, next_update_[lid], gid, spikes_);
        }
        // if there's time left before t_end, integrate until that
        integrate_until(cell, t_end, next_update_[lid], current_time_[lid]);
        check_spike(cell, t_end, next_update_[lid], gid, spikes_);

        // now process the sampling events
        for (;; ++spl_idx) {
            if (spl_idx >= n_samples) break;
            const auto& evt = sample_events[spl_idx];
            if (evt.time > t_end) break;
            // interpolation paramter
            auto t = (evt.time - time)/dt;
            if (evt.kind == adex_probe_kind::voltage)  *evt.data = math::lerp(V_0, cell.V_m, t);
            if (evt.kind == adex_probe_kind::adaption) *evt.data = math::lerp(W_0, cell.w, t);
        }

        time = t_end;
    }

    arb_assert(time == t_fin);
    arb_assert(evt_idx == n_events);
    arb_assert(spl_idx == n_samples);

    auto n_samplers = sample_callbacks.size();
    {
        std::lock_guard<std::mutex> guard{sampler_mex_};
        for (int s_idx = 0; s_idx < n_samplers; ++s_idx) {
            const auto& sd = sample_records[s_idx];
            auto hdl = sample_callbacks[s_idx];
            const auto& fun = samplers_[hdl].sampler;
            arb_assert(fun);
            fun(sample_metadata[s_idx], sd.size(), sd.data());
        }
    }
}

void adex_cell_group::t_serialize(serializer& ser, const std::string& k) const {
    serialize(ser, k, *this);
}

void adex_cell_group::t_deserialize(serializer& ser, const std::string& k) {
    deserialize(ser, k, *this);
}

std::vector<probe_metadata> adex_cell_group::get_probe_metadata(const cell_address_type& key) const {
    // SAFETY: Probe associations are fixed after construction, so we do not
    //         need to grab the mutex.
    if (probes_.count(key)) {
        return {probe_metadata{key, 0, &probes_.at(key).metadata}};
    } else {
        return {};
    }
}
