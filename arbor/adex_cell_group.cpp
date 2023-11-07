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
        cg_sources.add_label(cell.source, {0, 1});
        cg_targets.add_label(cell.target, {0, 1});
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

void adex_cell_group::advance_cell(time_type tfinal,
                                   time_type dt,
                                   cell_gid_type lid,
                                   const event_lane_subrange& event_lanes) {
    auto gid = gids_[lid];
    auto& cell = cells_[lid];
    auto n_events = static_cast<int>(!event_lanes.empty() ? event_lanes[lid].size() : 0);

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
        auto tlast = time_;
        std::vector<size_t> sample_sizes;
        std::size_t total_size = 0;
        {
            std::lock_guard<std::mutex> guard(sampler_mex_);
            for (auto& [hdl, assoc]: samplers_) {
                // No need to generate events
                if (assoc.probeset_ids.empty()) continue;
                // Construct sampling times, might give us the last time we sampled, so skip that.
                auto times = util::make_range(assoc.sched.events(tlast, tfinal));
                while (!times.empty() && times.front() == tlast) times.left++;
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
        for (int ix = 0; ix < sample_sizes.size(); ++ix) {
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

    // integrate until tfinal using the exact solution of membrane voltage differential equation.
    // prepare event processing
    auto e_idx = 0;
    auto s_idx = 0;
    for (; time_ < tfinal; time_ += dt) {
        auto dE = cell.V_m - cell.E_L;
        auto il = cell.g*dE;
        auto is = cell.g*cell.delta*exp((cell.V_m - cell.V_th)/cell.delta);
        auto dV = (is - il - cell.w)/cell.C_m;
        auto V_m = cell.V_m + dt*dV;

        auto dW = (cell.a*dE - cell.w)/cell.tau;
        auto w = cell.w + dt*dW;

        auto weight = 0.0;

        // Process events in [time, time + dt)
        for (;;) {
            auto e_t = e_idx < n_events  ? event_lanes[lid][e_idx].time : tfinal;
            if (e_t < time_) {
                ++e_idx;
                continue;
            }
            auto s_t = s_idx < n_samples ? sample_events[s_idx].time    : tfinal;
            if (s_t < time_) {
                ++s_idx;
                continue;
            }
            auto t = std::min(e_t, s_t);
            if (t >= time_ + dt || t >= tfinal) break;
            if (t == e_t) {
                weight += event_lanes[lid][e_idx].weight;
                ++e_idx;
            }
            else {
                auto& [time, kind, ptr] = sample_events[s_idx];
                auto t = (time - time_)/dt;
                if (kind == adex_probe_kind::voltage) {
                    if (next_update_[lid] > time_) {
                        *ptr = cell.E_R;
                    } else {
                        *ptr = math::lerp(cell.V_m, V_m + weight/cell.C_m, t);
                    }
                }
                else if (kind == adex_probe_kind::adaption) {
                    *ptr = math::lerp(cell.w, w, t);
                }
                else {
                    // impossible!
                    throw arbor_internal_error{"Unknown ADEX probe."};
                }
                ++s_idx;
            }
        }
        cell.w = w;
        // if we are still in refractory period, bail now, before we alter membrane voltage
        if (next_update_[lid] > time_) continue;
        V_m += weight/cell.C_m;
        // enter refractory period and emit spike
        if (V_m >= cell.V_th) {
            // interpolate the spike time and emit event.
            // NOTE: _Do_ the interpolation
            auto t_spike = time_;
            spikes_.emplace_back(cell_member_type{gid, 0}, t_spike);
            // reset membrane potential
            V_m = cell.E_R;
            // schedule next update
            next_update_[lid] = time_ + cell.t_ref;
            w += cell.b;
        }
        cell.V_m = V_m;
    }

    auto n_samplers = sample_callbacks.size();
    {
        std::lock_guard<std::mutex> guard{sampler_mex_};
        for (int s_idx = 0; s_idx < n_samplers; ++s_idx) {
            const auto& sd = sample_records[s_idx];
            auto hdl = sample_callbacks[s_idx];
            const auto& fun = samplers_[hdl].sampler;
            if (!fun) throw std::runtime_error{"NO sampler"};
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
