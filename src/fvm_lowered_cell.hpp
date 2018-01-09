#pragma once

// 3. Replace min_time with time; this can optionally assert
//    synchronization, but invariant is that time is
//    synchronized across integration calls.
// 5. Have everyone use same target_handle, probe_handle, deliverable_event classes.

#include <cmath>
#include <vector>
#include <stdexcept>

#include <common_types.hpp>
#include <backends/fvm_types.hpp>
#include <builtin_mechanisms.hpp>
#include <fvm_layout.hpp>
#include <ion.hpp>
#include <matrix.hpp>
#include <profiling/profiler.hpp>
#include <recipe.hpp>
#include <sampler_map.hpp>
#include <util/meta.hpp>
#include <util/range.hpp>

#include <util/debug.hpp>

namespace arb {

struct fvm_integration_result {
    util::range<const threshold_crossing*> crossings;
    util::range<const fvm_value_type*> sample_time;
    util::range<const fvm_value_type*> sample_value;
};

template <class Backend>
class fvm_lowered_cell {
public:
    using backend = Backend;
    using value_type = fvm_value_type;
    using size_type = fvm_size_type;

    void reset();

    void initialize(
        const std::vector<cell_gid_type>& gids,
        const recipe& rec,
        std::vector<target_handle>& target_handles,
        probe_association_map<probe_handle>& probe_map);

    fvm_integration_result integrate(
        value_type tfinal,
        value_type max_dt,
        std::vector<deliverable_event> staged_events,
        std::vector<sample_event> staged_samples,
        bool check_physical = false);

    value_type time() const { return tmin_; }

private:
    // Host or GPU-side back-end dependent storage.
    using array = typename backend::array;
    using host_view = typename backend::host_view;
    using shared_state = typename backend::shared_state;
    using sample_event_stream = typename backend::sample_event_stream;
    using threshold_watcher = typename backend::threshold_watcher;

    shared_state state_; // Cell state shared across mechanisms.

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
    host_view sample_time_host_;
    host_view sample_value_host_;

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
void fvm_lowered_cell<Backend>::assert_tmin() {
    auto time_minmax = backend::minmax_value(state_.time);
    if (time_minmax.first != time_minmax.second) {
        throw std::logic_error("inconsistent times across cells");
    }
    if (time_minmax.first != tmin_) {
        throw std::logic_error("out of synchronziation with cell state time");
    }
}

template <typename Backend>
void fvm_lowered_cell<Backend>::reset() {
    memory::fill(state_.voltage, initial_voltage_);
    memory::fill(state_.current_density, 0);
    memory::fill(state_.time, 0);
    memory::fill(state_.time_to, 0);
    set_tmin(0);

    for (auto& m: mechanisms_) {
        m->nrn_init();
    }

    for (auto& i: state_.ion_data) {
        i.second.reset(temperature_);
    }
    update_ion_state();

    // NOTE: Threshold watcher reset must come after the voltage values are set,
    // as voltage is implicitly read by watcher to set initial state.
    threshold_watcher_.reset();
}

template <typename Backend>
fvm_integration_result fvm_lowered_cell<Backend>::integrate(
    value_type tfinal,
    value_type dt_max,
    std::vector<deliverable_event> staged_events,
    std::vector<sample_event> staged_samples,
    bool check_physical)
{
    using util::as_const;

    // Integration setup
    threshold_watcher_.clear_crossings();

    auto n_samples = staged_samples.size();
    if (sample_time_.size() < n_samples) {
        sample_time_ = array(n_samples);
        sample_value_ = array(n_samples);
    }

    state_.deliverable_events.init(std::move(staged_events));
    sample_events_.init(std::move(staged_samples));

    EXPECTS((assert_tmin(), true));
    unsigned remaining_steps = dt_steps(tmin_, tfinal, dt_max);

    // TODO: Consider devolving more of this to back-end routines (e.g.
    // per-compartment dt probably not a win on GPU), possibly rumbling
    // complete fvm state into shared state object.

    while (remaining_steps) {
        // Deliver events and accumulate mechanism current contributions.

        PE("current");
        state_.deliverable_events.mark_until_after(state_.time);

        memory::fill(state_.current_density, 0.);
        for (auto& i: state_.ion_data) {
            auto& ion = i.second;
            memory::fill(ion.current_density(), 0.);
        }
        for (auto& m: mechanisms_) {
            PE(m->internal_name().c_str());
            m->deliver_events();
            m->nrn_current();
            PL();
        }
        state_.deliverable_events.drop_marked_events();
        PL(); // (current)

        // Update event list and integration step times.

        backend::update_time_to(state_.time_to, state_.time, dt_max, tfinal);
        state_.deliverable_events.event_time_if_before(state_.time_to);
        backend::set_dt(state_.dt, state_.dt_comp, state_.time_to, state_.time, state_.cv_to_cell);

        // Take samples at cell time if sample time in this step interval.

        sample_events_.mark_until(state_.time_to);
        backend::take_samples(sample_events_.marked_events(), state_.time, sample_time_, sample_value_);
        sample_events_.drop_marked_events();

        // Integrate voltage by matrix solve.

        PE("matrix", "setup");
        matrix_.assemble(state_.dt, state_.voltage, state_.current_density);
        PL(); PE("solve");
        matrix_.solve();
        PL();
        memory::copy(matrix_.solution(), state_.voltage);
        PL();

        // Integrate mechanism state.

        PE("state");
        for (auto& m: mechanisms_) {
            PE(m->internal_name().c_str());
            m->nrn_state();
            PL();
        }
        PL();

        // Update ion concentrations.

        PE("ion-update");
        update_ion_state();
        PL();

        // Update time and test for spike threshold crossings.

        memory::copy(state_.time_to, state_.time);
        threshold_watcher_.test();

        // Check for non-physical solutions:

        if (check_physical) {
            assert_voltage_bounded(1000.);
        }

        // Check for end of integration.

        if (!--remaining_steps) {
            tmin_ = backend::minmax_value(state_.time).first;
            remaining_steps = dt_steps(tmin_, tfinal, dt_max);
        }
    }

    set_tmin(tfinal);

    const auto& crossings = threshold_watcher_.crossings();
    sample_time_host_ = host_view(sample_time_);
    sample_value_host_ = host_view(sample_value_);

    return fvm_integration_result{
        util::range_pointer_view(crossings),
        util::range_pointer_view(as_const(sample_time_host_)),
        util::range_pointer_view(as_const(sample_value_host_))
    };
}

template <typename B>
void fvm_lowered_cell<B>::update_ion_state() {
    for (auto& i: state_.ion_data) {
        i.second.init_concentration();
    }
    for (auto& m: mechanisms_) {
        m->write_ions();
    }
    for (auto& i: state_.ion_data) {
        i.second.nernst_reversal_potential(temperature_);
    }
}

template <typename B>
void fvm_lowered_cell<B>::assert_voltage_bounded(fvm_value_type bound) {
    auto v_minmax = backend::minmax_value(state_.voltage);
    if (v_minmax.first>=-bound && v_minmax.second<=bound) {
        return;
    }

    auto t_minmax = backend::minmax_value(state_.time);
    throw std::out_of_range("voltage solution out of bounds for t in ["+
        std::to_string(t_minmax.first)+", "+std::to_string(t_minmax.second)+"]");
}

template <typename B>
void fvm_lowered_cell<B>::initialize(
    const std::vector<cell_gid_type>& gids,
    const recipe& rec,
    std::vector<target_handle>& target_handles,
    probe_association_map<probe_handle>& probe_map)
{
    using util::any_cast;
    using util::count_along;
    using util::make_span;
    using util::value_by_key;

    std::vector<cell> cells;
    const std::size_t ncell = gids.size();

    cells.reserve(ncell);
    for (auto gid: gids) {
        cells.push_back(any_cast<cell>(rec.get_cell_description(gid)));
    }

    auto rec_props = rec.get_global_properties(cable1d_neuron);
    auto global_props = rec_props.has_value()? any_cast<cell_global_properties>(rec_props): cell_global_properties{};

    const mechanism_catalogue* catalogue = global_props.catalogue;
    initial_voltage_ = global_props.init_membrane_potential_mV;
    temperature_ = global_props.temperature_K;

    // Discretize cells, build matrix.

    fvm_discretization D = fvm_discretize(cells);
    EXPECTS(D.ncell == ncell);
    matrix_ = matrix<backend>(D.parent_cv, D.cell_cv_bounds, D.cv_capacitance, D.face_conductance, D.cv_area);
    state_ = shared_state(ncell, D.cv_to_cell);
    sample_events_ = sample_event_stream(ncell);

    // Instantiate mechanisms and ions.

    fvm_mechanism_data mech_data = fvm_build_mechanism_data(*catalogue, cells, D);

    for (auto& i: mech_data.ions) {
        ionKind kind = i.first;

        ion<backend> ion(i.second.cv);
        if (auto ion_def = value_by_key(global_props.ion_default, to_string(kind))) {
            ion.charge = ion_def->charge;
            ion.default_int_concentration = ion_def->default_int_concentration;
            ion.default_ext_concentration = ion_def->default_ext_concentration;
        }

        ion.set_weights(i.second.iconc_norm_area, i.second.econc_norm_area);
        state_.add_ion(kind, std::move(ion));
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

        auto cat = builtin_mechanisms().has(name)? &builtin_mechanisms(): catalogue;
        auto mech = cat->instance<backend>(name);
        mech->instantiate(mech_id, state_, layout);

        for (auto& pv: config.param_values) {
            mech->set_parameter(pv.first, pv.second);
        }
        mechanisms_.push_back(mechanism_ptr(mech.release()));
    }

    // Collect detectors, probe handles.

    std::vector<size_type> detector_cv;
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
                handle = state_.voltage.data()+cv;
                break;
            case cell_probe_address::membrane_current:
                handle = state_.current_density.data()+cv;
                break;
            default:
                throw std::logic_error("unrecognized probeKind");
            }

            probe_map.insert({pi.id, {handle, pi.tag}});
        }
    }

    threshold_watcher_ = threshold_watcher(state_.cv_to_cell, state_.time,
        state_.time_to, state_.voltage, detector_cv, detector_threshold);

    reset();
}

} // namespace arb
