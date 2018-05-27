#pragma once

// Implementations for fvm_lowered_cell are parameterized
// on the back-end class.
//
// Classes here are exposed in a header only so that
// implementation details may be tested in the unit tests.
// It should otherwise only be used in `fvm_lowered_cell.cpp`.

#include <cmath>
#include <iterator>
#include <utility>
#include <vector>
#include <stdexcept>

#include <common_types.hpp>
#include <builtin_mechanisms.hpp>
#include <fvm_layout.hpp>
#include <fvm_lowered_cell.hpp>
#include <ion.hpp>
#include <matrix.hpp>
#include <profiling/profiler.hpp>
#include <recipe.hpp>
#include <sampler_map.hpp>
#include <util/meta.hpp>
#include <util/range.hpp>
#include <util/rangeutil.hpp>
#include <util/transform.hpp>

#include <util/debug.hpp>

namespace arb {

template <class Backend>
class fvm_lowered_cell_impl: public fvm_lowered_cell {
public:
    using backend = Backend;
    using value_type = fvm_value_type;
    using index_type = fvm_index_type;
    using size_type = fvm_size_type;

    void reset() override;

    void initialize(
        const std::vector<cell_gid_type>& gids,
        const recipe& rec,
        std::vector<target_handle>& target_handles,
        probe_association_map<probe_handle>& probe_map) override;

    fvm_integration_result integrate(
        value_type tfinal,
        value_type max_dt,
        std::vector<deliverable_event> staged_events,
        std::vector<sample_event> staged_samples,
        bool check_physical = false) override;

    value_type time() const override { return tmin_; }
    
    std::vector<mechanism_ptr>& mechanisms() {
        return mechanisms_;
    }

private:
    // Host or GPU-side back-end dependent storage.
    using array = typename backend::array;
    using shared_state = typename backend::shared_state;
    using sample_event_stream = typename backend::sample_event_stream;
    using threshold_watcher = typename backend::threshold_watcher;

    std::unique_ptr<shared_state> state_; // Cell state shared across mechanisms.

    // TODO: Can we move the backend-dependent data structures below into state_?
    sample_event_stream sample_events_;
    array sample_time_;
    array sample_value_;
    matrix<backend> matrix_;
    threshold_watcher threshold_watcher_;

    value_type tmin_ = 0;
    value_type initial_voltage_ = NAN;
    value_type temperature_ = NAN;
    std::vector<mechanism_ptr> mechanisms_;

    // Host-side views/copies and local state.
    decltype(backend::host_view(sample_time_)) sample_time_host_;
    decltype(backend::host_view(sample_value_)) sample_value_host_;

    void update_ion_state();

    // Throw if absolute value of membrane voltage exceeds bounds.
    void assert_voltage_bounded(fvm_value_type bound);

    // Throw if any cell time not equal to tmin_
    void assert_tmin();

    // Assign tmin_ and call assert_tmin() if assertions on.
    void set_tmin(value_type t) {
        tmin_ = t;
        EXPECTS((assert_tmin(), true));
    }

    static unsigned dt_steps(value_type t0, value_type t1, value_type dt) {
        return t0>=t1? 0: 1+(unsigned)((t1-t0)/dt);
    }
};

template <typename Backend>
void fvm_lowered_cell_impl<Backend>::assert_tmin() {
    auto time_minmax = state_->time_bounds();
    if (time_minmax.first != time_minmax.second) {
        throw std::logic_error("inconsistent times across cells");
    }
    if (time_minmax.first != tmin_) {
        throw std::logic_error("out of synchronziation with cell state time");
    }
}

template <typename Backend>
void fvm_lowered_cell_impl<Backend>::reset() {
    state_->reset(initial_voltage_, temperature_);
    set_tmin(0);

    for (auto& m: mechanisms_) {
        m->nrn_init();
    }

    update_ion_state();

    // NOTE: Threshold watcher reset must come after the voltage values are set,
    // as voltage is implicitly read by watcher to set initial state.
    threshold_watcher_.reset();
}

template <typename Backend>
fvm_integration_result fvm_lowered_cell_impl<Backend>::integrate(
    value_type tfinal,
    value_type dt_max,
    std::vector<deliverable_event> staged_events,
    std::vector<sample_event> staged_samples,
    bool check_physical)
{
    using util::as_const;

    // Integration setup
    PE(advance_integrate_setup);
    threshold_watcher_.clear_crossings();

    auto n_samples = staged_samples.size();
    if (sample_time_.size() < n_samples) {
        sample_time_ = array(n_samples);
        sample_value_ = array(n_samples);
    }

    state_->deliverable_events.init(std::move(staged_events));
    sample_events_.init(std::move(staged_samples));

    EXPECTS((assert_tmin(), true));
    unsigned remaining_steps = dt_steps(tmin_, tfinal, dt_max);
    PL();

    // TODO: Consider devolving more of this to back-end routines (e.g.
    // per-compartment dt probably not a win on GPU), possibly rumbling
    // complete fvm state into shared state object.

    while (remaining_steps) {
        // Deliver events and accumulate mechanism current contributions.

        PE(advance_integrate_events);
        state_->deliverable_events.mark_until_after(state_->time);
        PL();

        PE(advance_integrate_current);
        state_->zero_currents();
        for (auto& m: mechanisms_) {
            m->deliver_events();
            m->nrn_current();
        }
        PL();

        PE(advance_integrate_events);
        state_->deliverable_events.drop_marked_events();

        // Update event list and integration step times.

        state_->update_time_to(dt_max, tfinal);
        state_->deliverable_events.event_time_if_before(state_->time_to);
        state_->set_dt();
        PL();

        // Take samples at cell time if sample time in this step interval.

        PE(advance_integrate_samples);
        sample_events_.mark_until(state_->time_to);
        state_->take_samples(sample_events_.marked_events(), sample_time_, sample_value_);
        sample_events_.drop_marked_events();
        PL();

        // Integrate voltage by matrix solve.

        PE(advance_integrate_matrix_build);
        matrix_.assemble(state_->dt_cell, state_->voltage, state_->current_density);
        PL();
        PE(advance_integrate_matrix_solve);
        matrix_.solve();
        memory::copy(matrix_.solution(), state_->voltage);
        PL();

        // Integrate mechanism state.

        PE(advance_integrate_state);
        for (auto& m: mechanisms_) {
            m->nrn_state();
        }
        PL();

        // Update ion concentrations.

        PE(advance_integrate_ionupdate);
        update_ion_state();
        PL();

        // Update time and test for spike threshold crossings.

        PE(advance_integrate_threshold);
        memory::copy(state_->time_to, state_->time);
        threshold_watcher_.test();
        PL();

        // Check for non-physical solutions:

        if (check_physical) {
            PE(advance_integrate_physicalcheck);
            assert_voltage_bounded(1000.);
            PL();
        }

        // Check for end of integration.

        PE(advance_integrate_stepsupdate);
        if (!--remaining_steps) {
            tmin_ = state_->time_bounds().first;
            remaining_steps = dt_steps(tmin_, tfinal, dt_max);
        }
        PL();
    }

    set_tmin(tfinal);

    const auto& crossings = threshold_watcher_.crossings();
    sample_time_host_ = backend::host_view(sample_time_);
    sample_value_host_ = backend::host_view(sample_value_);

    return fvm_integration_result{
        util::range_pointer_view(crossings),
        util::range_pointer_view(sample_time_host_),
        util::range_pointer_view(sample_value_host_)
    };
}

template <typename B>
void fvm_lowered_cell_impl<B>::update_ion_state() {
    state_->ions_init_concentration();
    for (auto& m: mechanisms_) {
        m->write_ions();
    }
    state_->ions_nernst_reversal_potential(temperature_);
}

template <typename B>
void fvm_lowered_cell_impl<B>::assert_voltage_bounded(fvm_value_type bound) {
    auto v_minmax = state_->voltage_bounds();
    if (v_minmax.first>=-bound && v_minmax.second<=bound) {
        return;
    }

    auto t_minmax = state_->time_bounds();
    throw std::out_of_range("voltage solution out of bounds for t in ["+
        std::to_string(t_minmax.first)+", "+std::to_string(t_minmax.second)+"]");
}

template <typename B>
void fvm_lowered_cell_impl<B>::initialize(
    const std::vector<cell_gid_type>& gids,
    const recipe& rec,
    std::vector<target_handle>& target_handles,
    probe_association_map<probe_handle>& probe_map)
{
    using util::any_cast;
    using util::count_along;
    using util::make_span;
    using util::value_by_key;
    using util::keys;

    std::vector<cell> cells;
    const std::size_t ncell = gids.size();

    cells.reserve(ncell);
    for (auto gid: gids) {
        cells.push_back(any_cast<cell>(rec.get_cell_description(gid)));
    }

    auto rec_props = rec.get_global_properties(cell_kind::cable1d_neuron);
    auto global_props = rec_props.has_value()? any_cast<cell_global_properties>(rec_props): cell_global_properties{};

    const mechanism_catalogue* catalogue = global_props.catalogue;
    initial_voltage_ = global_props.init_membrane_potential_mV;
    temperature_ = global_props.temperature_K;

    // Mechanism instantiator helper.
    auto mech_instance = [&catalogue](const std::string& name) {
        auto cat = builtin_mechanisms().has(name)? &builtin_mechanisms(): catalogue;
        return cat->instance<backend>(name);
    };

    // Discretize cells, build matrix.

    fvm_discretization D = fvm_discretize(cells);
    EXPECTS(D.ncell == ncell);
    matrix_ = matrix<backend>(D.parent_cv, D.cell_cv_bounds, D.cv_capacitance, D.face_conductance, D.cv_area);
    sample_events_ = sample_event_stream(ncell);

    // Discretize mechanism data.

    fvm_mechanism_data mech_data = fvm_build_mechanism_data(*catalogue, cells, D);

    // Create shared cell state.
    // (SIMD padding requires us to check each mechanism for alignment/padding constraints.)

    unsigned data_alignment = util::max_value(
        util::transform_view(keys(mech_data.mechanisms),
            [&](const std::string& name) { return mech_instance(name)->data_alignment(); }));

    state_ = util::make_unique<shared_state>(ncell, D.cv_to_cell, data_alignment? data_alignment: 1u);

    // Instantiate mechanisms and ions.

    for (auto& i: mech_data.ions) {
        ionKind kind = i.first;

        if (auto ion = value_by_key(global_props.ion_default, to_string(kind))) {
            state_->add_ion(ion.value(), i.second.cv, i.second.iconc_norm_area, i.second.econc_norm_area);
        }
    }

    target_handles.resize(mech_data.ntarget);

    for (auto& m: mech_data.mechanisms) {
        auto& name = m.first;
        auto& config = m.second;
        unsigned mech_id = mechanisms_.size();

        mechanism::layout layout;
        layout.cv = config.cv;
        layout.weight.resize(layout.cv.size());

        // Mechanism weights are F·α where α ∈ [0, 1] is the proportional
        // contribution in the CV, and F is the scaling factor required
        // to convert from the mechanism current contribution units to A/m².

        if (config.kind==mechanismKind::point) {
            // Point mechanism contributions are in [nA]; CV area A in [µm^2].
            // F = 1/A * [nA/µm²] / [A/m²] = 1000/A.

            for (auto i: count_along(layout.cv)) {
                auto cv = layout.cv[i];
                layout.weight[i] = 1000/D.cv_area[cv];

                // (builtin stimulus, for example, has no targets)
                if (!config.target.empty()) {
                    target_handles[config.target[i]] = target_handle(mech_id, i, D.cv_to_cell[cv]);
                }
            }
        }
        else {
            // Density Current density contributions from mechanism are in [mA/cm²]
            // (NEURON compatibility). F = [mA/cm²] / [A/m²] = 10.

            for (auto i: count_along(layout.cv)) {
                layout.weight[i] = 10*config.norm_area[i];
            }
        }

        auto mech = mech_instance(name);
        mech->instantiate(mech_id, *state_, layout);

        for (auto& pv: config.param_values) {
            mech->set_parameter(pv.first, pv.second);
        }
        mechanisms_.push_back(mechanism_ptr(mech.release()));
    }

    // Collect detectors, probe handles.

    std::vector<index_type> detector_cv;
    std::vector<value_type> detector_threshold;

    for (auto cell_idx: make_span(ncell)) {
        cell_gid_type gid = gids[cell_idx];

        for (auto detector: cells[cell_idx].detectors()) {
            detector_cv.push_back(D.segment_location_cv(cell_idx, detector.location));
            detector_threshold.push_back(detector.threshold);
        }

        for (cell_lid_type j: make_span(rec.num_probes(gid))) {
            probe_info pi = rec.get_probe({gid, j});
            auto where = any_cast<cell_probe_address>(pi.address);

            auto cv = D.segment_location_cv(cell_idx, where.location);
            probe_handle handle;

            switch (where.kind) {
            case cell_probe_address::membrane_voltage:
                handle = state_->voltage.data()+cv;
                break;
            case cell_probe_address::membrane_current:
                handle = state_->current_density.data()+cv;
                break;
            default:
                throw std::logic_error("unrecognized probeKind");
            }

            probe_map.insert({pi.id, {handle, pi.tag}});
        }
    }

    threshold_watcher_ = threshold_watcher(state_->cv_to_cell.data(), state_->time.data(),
        state_->time_to.data(), state_->voltage.data(), detector_cv, detector_threshold);

    reset();
}

} // namespace arb
