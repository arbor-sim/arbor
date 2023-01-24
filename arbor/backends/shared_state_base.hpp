#pragma once

#include <arbor/mechanism_abi.h>
#include <arbor/common_types.hpp>

namespace arb {

// Common functionality for CPU/GPU shared state.
template <typename D, typename array, typename ion_state>
struct shared_state_base {

    void update_time_step(time_type dt_max, time_type tfinal) {
        auto d = static_cast<D*>(this);
        d->deliverable_events.drop_marked_events();
        d->update_time_to(dt_max, tfinal);
        d->deliverable_events.event_time_if_before(d->time_to);
        d->set_dt();
    }

    void begin_epoch(std::vector<deliverable_event> deliverables,
                     std::vector<sample_event> samples) {
        auto d = static_cast<D*>(this);
        // events
        d->deliverable_events.init(std::move(deliverables));
        // samples
        auto n_samples = samples.size();
        if (d->sample_time.size() < n_samples) {
            d->sample_time = array(n_samples);
            d->sample_value = array(n_samples);
        }
        d->sample_events.init(std::move(samples));
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

    arb_deliverable_event_stream mark_deliverable_events() {
        auto d = static_cast<D*>(this);
        d->deliverable_events.mark_until_after(d->time);
        auto state = d->deliverable_events.marked_events();
        arb_deliverable_event_stream result;
        result.n_streams = state.n;
        result.begin     = state.begin_offset;
        result.end       = state.end_offset;
        result.events    = (arb_deliverable_event_data*) state.ev_data; // FIXME(TH): This relies on bit-castability
        return result;
    }


    void next_time_step() {
        auto d = static_cast<D*>(this);
        std::swap(d->time_to, d->time);
    }

    void reset_thresholds() {
        auto d = static_cast<D*>(this);
        d->watcher.reset(d->voltage);
    }

    void test_thresholds() {
        auto d = static_cast<D*>(this);
        d->watcher.test(d->time_since_spike);
    }

    void configure_stimulus(const fvm_stimulus_config& stims) {
        auto d = static_cast<D*>(this);
        d->stim_data = {stims, d->alignment};
    }

    void add_stimulus_current() {
        auto d = static_cast<D*>(this);
        d->stim_data.add_current(d->time, d->cv_to_intdom, d->current_density);
    }

    void ions_init_concentration() {
        auto d = static_cast<D*>(this);
        for (auto& i: d->ion_data) {
            i.second.init_concentration();
        }
    }

    void integrate_cable_state() {
        auto d = static_cast<D*>(this);
        d->solver.solve(d->voltage, d->dt_intdom, d->current_density, d->conductivity);
        for (auto& [ion, data]: d->ion_data) {
            if (data.solver) {
                data.solver->solve(data.Xd_,
                                   d->dt_intdom,
                                   d->voltage,
                                   data.iX_,
                                   data.gX_,
                                   data.charge[0]);
            }
        }
    }
};

}
