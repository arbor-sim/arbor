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
lif_cell_group::lif_cell_group(const std::vector<cell_gid_type>& gids,
                               const recipe& rec,
                               cell_label_range& cg_sources,
                               cell_label_range& cg_targets):
    gids_(gids) {

    for (auto gid: gids_) {
        const auto& cell = util::any_cast<lif_cell>(rec.get_cell_description(gid));
        // set up cell state
        cells_.push_back(cell);
        last_time_updated_.push_back(0.0);
        last_time_sampled_.push_back(-1.0);
        // tell our caller about this cell's connections
        cg_sources.add_cell();
        cg_targets.add_cell();
        cg_sources.add_label(cell.source, {0, 1});
        cg_targets.add_label(cell.target, {0, 1});
        // insert probes where needed
        auto probes = rec.get_probes(gid);
        for (const auto& probe: probes) {
            if (probe.address.type() == typeid(lif_probe_voltage)) {
                cell_address_type id{gid, probe.tag};
                probes_.insert_or_assign(id, lif_probe_info{id, lif_probe_kind::voltage, {}});
            }
            else {
                throw bad_cell_probe{cell_kind::lif, gid};
            }
        }
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

void lif_cell_group::add_sampler(sampler_association_handle h,
                                 cell_member_predicate probeset_ids,
                                 schedule sched,
                                 sampler_function fn) {
    std::lock_guard<std::mutex> guard(sampler_mex_);
    auto probeset =
        util::assign_from(util::filter(util::keys(probes_), probeset_ids));
    auto assoc = arb::sampler_association{std::move(sched),
                                          std::move(fn),
                                          std::move(probeset)};
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

void lif_cell_group::reset() {
    spikes_.clear();
    util::fill(last_time_updated_, 0.);
    util::fill(next_time_updatable_, 0.);
}

// produce voltage V_m at t1, given cell state at t0 and no spikes in [t0, t1)
static double
lif_decay(const lif_cell& cell, double t0, double t1) {
    return (cell.V_m - cell.E_L)*exp((t0 - t1)/cell.tau_m) + cell.E_L;
}

// Advances a single cell (lid) with the exact solution (jumps can be arbitrary).
// Parameter dt is ignored, since we make jumps between two consecutive spikes.
void lif_cell_group::advance_cell(time_type tfinal,
                                  time_type dt,
                                  cell_gid_type lid,
                                  const event_lane_subrange& event_lanes) {
    const auto gid = gids_[lid];
    auto& cell = cells_[lid];
    // time of last update.
    auto t = last_time_updated_[lid];
    // spikes to process
    const auto n_events = static_cast<int>(event_lanes.size() ? event_lanes[lid].size() : 0);
    int event_idx = 0;
    // collected sampling data
    std::unordered_map<sampler_association_handle,
                       std::unordered_map<cell_address_type,
                                          std::vector<sample_record>>> sampled;
    // samples to process
    std::size_t n_values = 0;
    std::vector<std::pair<time_type, sampler_association_handle>> samples;
    if (!samplers_.empty()) {
        auto tlast = last_time_sampled_[lid];
        std::lock_guard<std::mutex> guard(sampler_mex_);
        for (auto& [hdl, assoc]: samplers_) {
             // No need to generate events
            if (assoc.probeset_ids.empty()) continue;
            // Construct sampling times, might give us the last time we sampled, so skip that.
            auto times = util::make_range(assoc.sched.events(tlast, tfinal));
            while (!times.empty() && times.front() == tlast) times.left++;
            if (times.empty()) continue;
            const auto n_times = times.size();
            // Count up the samplers touching _our_ gid
            int delta = 0;
            for (const auto& pid: assoc.probeset_ids) {
                if (pid.gid != gid) continue;
                arb_assert (0 == sampled[hdl].count(pid));
                sampled[hdl][pid].reserve(n_times);
                delta += n_times;
            }
            if (delta == 0) continue;
            n_values += delta;
            for (auto t: times) samples.emplace_back(t, hdl);
        }
    }
    std::sort(samples.begin(), samples.end());
    int n_samples = samples.size();
    int sample_idx = 0;
    // Now allocate some scratch space for the probed values, if we don't,
    // re-alloc might move our data
    std::vector<double> sampled_voltages;
    sampled_voltages.reserve(n_values);
    // integrate until tfinal using the exact solution of membrane voltage differential equation.
    for (;;) {
        const auto event_time = event_idx < n_events ? event_lanes[lid][event_idx].time : tfinal;
        const auto sample_time = sample_idx < n_samples ? samples[sample_idx].first : tfinal;
        const auto time = std::min(event_time, sample_time);
        // bail at end of integration interval
        if (time >= tfinal) break;
        // Check what to do, we might need to process events **and/or** perform
        // sampling.
        // NB. we put events before samples, if they collide we'll see
        // the update in sampling.

        bool do_event  = time == event_time;
        bool do_sample = time == sample_time;

        if (do_event) {
            const auto& event_lane = event_lanes[lid];
            // process all events at time t
            auto weight = 0.0;
            for (; event_idx < n_events && event_lane[event_idx].time <= time; ++event_idx) {
                weight += event_lane[event_idx].weight;
            }
            // skip event if neuron is in refactory period
            if (time >= t) {
                // Let the membrane potential decay towards E_L and add spike contribution(s)
                cell.V_m = lif_decay(cell, t, time) + weight / cell.C_m;
                // Update current time
                t = time;
                // If crossing threshold occurred
                if (cell.V_m >= cell.V_th) {
                    // save spike
                    spikes_.push_back({{gid, 0}, time});
                    // Advance to account for the refractory period.
                    // This means decay will also start at t + t_ref
                    t += cell.t_ref;
                    // Reset the voltage.
                    cell.V_m = cell.E_R;
                }
            }
        }

        if (do_sample) {
            // Consume all sample events at this time
            for (; sample_idx < n_samples && samples[sample_idx].first <= time; ++sample_idx) {
                const auto& [s_time, hdl] = samples[sample_idx];
                for (const auto& key: samplers_[hdl].probeset_ids) {
                    const auto& kind = probes_.at(key).kind;
                    // This is the only thing we know how to do: Probing U(t)
                    switch (kind) {
                        case lif_probe_kind::voltage: {
                            // Compute, but do not _set_ V_m
                            // default value, if _in_ refractory period, this
                            // will be E_R, so no further action needed.
                            auto U = cell.V_m;
                            if (time >= t) {
                                // we are not in the refractory period, apply decay
                                U = lif_decay(cell, t, time);
                            }
                            // Store U for later use.
                            sampled_voltages.push_back(U);
                            // Set up reference to sampled value
                            sampled[hdl][key].push_back(sample_record{time, {&sampled_voltages.back()}});
                            break;
                        }
                        default:
                            throw arbor_internal_error{"Invalid LIF probe kind"};
                    }
                }
            }
            last_time_sampled_[lid] = time;
        }
        if (!(do_sample || do_event)) {
            throw arbor_internal_error{"LIF cell group: Must select either sample or spike event; got neither."};
        }
        last_time_updated_[lid] = t;
    }
    arb_assert (sampled_voltages.size() <= n_values);
    // Now we need to call all sampler callbacks with the data we have collected
    {
        std::lock_guard<std::mutex> guard(sampler_mex_);
        for (auto& [k, vs]: sampled) {
            const auto& fun = samplers_[k].sampler;
            for (auto& [id, us]: vs) {
                auto meta = get_probe_metadata(id)[0];
                fun(meta, us.size(), us.data());
                us.clear();
            }
        }
    }
}

void lif_cell_group::t_serialize(serializer& ser, const std::string& k) const {
    serialize(ser, k, *this);
}

void lif_cell_group::t_deserialize(serializer& ser, const std::string& k) {
    deserialize(ser, k, *this);
}

std::vector<probe_metadata> lif_cell_group::get_probe_metadata(const cell_address_type& key) const {
    std::lock_guard<std::mutex> guard(probe_mex_);
    if (probes_.count(key)) {
        return {probe_metadata{key, 0, &probes_.at(key).metadata}};
    } else {
        return {};
    }
}
