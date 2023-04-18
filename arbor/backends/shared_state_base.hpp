#pragma once

#include <arbor/mechanism_abi.h>
#include <arbor/common_types.hpp>

#include "backends/event.hpp"
#include "backends/common_types.hpp"
#include "fvm_layout.hpp"

#include "util/rangeutil.hpp"

namespace arb {

// Common functionality for CPU/GPU shared state.
template <typename D, typename array, typename ion_state>
struct shared_state_base {

    void update_time_to(const timestep_range::timestep& ts) {
        auto d = static_cast<D*>(this);
        d->time_to = ts.t_end();
        d->dt = ts.dt();
    }

    void next_time_step() {
        auto d = static_cast<D*>(this);
        d->time = d->time_to;
    }

    void begin_epoch(const std::vector<std::vector<std::vector<deliverable_event>>>& staged_events_per_mech_id,
                     const std::vector<std::vector<sample_event>>& samples,
                     const timestep_range& dts) {
        auto d = static_cast<D*>(this);
        // events
        auto& storage = d->storage;
        for (auto& [mech_id, store] : storage) {
            if (mech_id < staged_events_per_mech_id.size() && staged_events_per_mech_id[mech_id].size())
            {
                store.deliverable_events_.init(staged_events_per_mech_id[mech_id]);
            }
        }
        // samples
        auto n_samples = util::sum_by(samples, [] (const auto& s) {return s.size();});
        if (d->sample_time.size() < n_samples) {
            d->sample_time = array(n_samples);
            d->sample_value = array(n_samples);
        }
        d->sample_events.init(samples);
        // thresholds
        d->watcher.clear_crossings();
    }

    void add_ion(const std::string& ion_name,
                 int charge,
                 const fvm_ion_config& ion_info,
                 typename ion_state::solver_ptr ptr=nullptr) {
        auto d = static_cast<D*>(this);
        d->ion_data.emplace(std::piecewise_construct,
                            std::forward_as_tuple(ion_name),
                            std::forward_as_tuple(charge, ion_info, d->alignment, std::move(ptr)));
    }

    arb_value_type* mechanism_state_data(const mechanism& m,
                                         const std::string& key) {
        auto d = static_cast<D*>(this);
        const auto& store = d->storage.at(m.mechanism_id());

        for (arb_size_type i = 0; i<m.mech_.n_state_vars; ++i) {
            if (key==m.mech_.state_vars[i].name) {
                return store.state_vars_[i];
            }
        }
        return nullptr;
    }

    void mark_events() {
        auto d = static_cast<D*>(this);
        auto& storage = d->storage;
        for (auto& s : storage) {
            s.second.deliverable_events_.mark();
        }
    }

    void deliver_events(mechanism& m) {
        auto d = static_cast<D*>(this);
        auto& storage = d->storage;
        if (auto it = storage.find(m.mechanism_id()); it != storage.end()) {
            auto& deliverable_events = it->second.deliverable_events_;
            if (!deliverable_events.empty()) {
                auto state = deliverable_events.marked_events();
                m.deliver_events(state);
            }
        }
    }

    void reset_thresholds() {
        auto d = static_cast<D*>(this);
        d->watcher.reset(d->voltage);
    }

    void test_thresholds() {
        auto d = static_cast<D*>(this);
        d->watcher.test(d->time_since_spike, d->time, d->time_to);
    }

    void configure_stimulus(const fvm_stimulus_config& stims) {
        auto d = static_cast<D*>(this);
        d->stim_data = {stims, d->alignment};
    }

    void add_stimulus_current() {
        auto d = static_cast<D*>(this);
        d->stim_data.add_current(d->time, d->current_density);
    }

    void ions_init_concentration() {
        auto d = static_cast<D*>(this);
        for (auto& i: d->ion_data) {
            i.second.init_concentration();
        }
    }

    void integrate_cable_state() {
        auto d = static_cast<D*>(this);
        d->solver.solve(d->voltage, d->dt, d->current_density, d->conductivity);
        for (auto& [ion, data]: d->ion_data) {
            if (data.solver) {
                data.solver->solve(data.Xd_,
                                   d->dt,
                                   d->voltage,
                                   data.iX_,
                                   data.gX_,
                                   data.charge[0]);
            }
        }
    }

    fvm_integration_result get_integration_result() {
        auto d = static_cast<D*>(this);
        const auto& crossings = d->watcher.crossings();
        d->update_sample_views();

        return { util::range_pointer_view(crossings),
                 util::range_pointer_view(d->sample_time_host),
                 util::range_pointer_view(d->sample_value_host) };
    }
};

} // namespace arb
