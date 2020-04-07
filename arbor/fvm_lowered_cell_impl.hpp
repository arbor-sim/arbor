#pragma once

// Implementations for fvm_lowered_cell are parameterized
// on the back-end class.
//
// Classes here are exposed in a header only so that
// implementation details may be tested in the unit tests.
// It should otherwise only be used in `fvm_lowered_cell.cpp`.

#include <cmath>
#include <iterator>
#include <memory>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>
#include <unordered_set>

#include <arbor/assert.hpp>
#include <arbor/common_types.hpp>
#include <arbor/cable_cell_param.hpp>
#include <arbor/recipe.hpp>

#include "builtin_mechanisms.hpp"
#include "execution_context.hpp"
#include "fvm_layout.hpp"
#include "fvm_lowered_cell.hpp"
#include "matrix.hpp"
#include "profile/profiler_macro.hpp"
#include "sampler_map.hpp"
#include "util/maputil.hpp"
#include "util/meta.hpp"
#include "util/range.hpp"
#include "util/rangeutil.hpp"
#include "util/strprintf.hpp"
#include "util/transform.hpp"

namespace arb {

template <class Backend>
class fvm_lowered_cell_impl: public fvm_lowered_cell {
public:
    using backend = Backend;
    using value_type = fvm_value_type;
    using index_type = fvm_index_type;
    using size_type = fvm_size_type;

    fvm_lowered_cell_impl(execution_context ctx): context_(ctx), threshold_watcher_(ctx) {};

    void reset() override;

    void initialize(
        const std::vector<cell_gid_type>& gids,
        const recipe& rec,
        std::vector<fvm_index_type>& cell_to_intdom,
        std::vector<target_handle>& target_handles,
        probe_association_map<probe_handle>& probe_map) override;

    fvm_integration_result integrate(
        value_type tfinal,
        value_type max_dt,
        std::vector<deliverable_event> staged_events,
        std::vector<sample_event> staged_samples) override;

    std::vector<fvm_gap_junction> fvm_gap_junctions(
        const std::vector<cable_cell>& cells,
        const std::vector<cell_gid_type>& gids,
        const recipe& rec,
        const fvm_cv_discretization& D);

    // Generates indom index for every gid, guarantees that gids belonging to the same supercell are in the same intdom
    // Fills cell_to_intdom map; returns number of intdoms
    fvm_size_type fvm_intdom(
        const recipe& rec,
        const std::vector<cell_gid_type>& gids,
        std::vector<fvm_index_type>& cell_to_intdom);

    value_type time() const override { return tmin_; }

    //Exposed for testing purposes
    std::vector<mechanism_ptr>& mechanisms() {
        return mechanisms_;
    }

private:
    // Host or GPU-side back-end dependent storage.
    using array = typename backend::array;
    using shared_state = typename backend::shared_state;
    using sample_event_stream = typename backend::sample_event_stream;
    using threshold_watcher = typename backend::threshold_watcher;

    execution_context context_;

    std::unique_ptr<shared_state> state_; // Cell state shared across mechanisms.

    // TODO: Can we move the backend-dependent data structures below into state_?
    sample_event_stream sample_events_;
    array sample_time_;
    array sample_value_;
    matrix<backend> matrix_;
    threshold_watcher threshold_watcher_;

    value_type tmin_ = 0;
    std::vector<mechanism_ptr> mechanisms_; // excludes reversal potential calculators.
    std::vector<mechanism_ptr> revpot_mechanisms_;

    // Non-physical voltage check threshold, 0 => no check.
    value_type check_voltage_mV = 0;

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
        arb_assert((assert_tmin(), true));
    }

    static unsigned dt_steps(value_type t0, value_type t1, value_type dt) {
        return t0>=t1? 0: 1+(unsigned)((t1-t0)/dt);
    }

    // Sets the GPU used for CUDA calls from the thread that calls it.
    // The GPU will be the one in the execution context context_.
    // If not called, the thread may attempt to launch on a different GPU,
    // leading to crashes.
    void set_gpu() {
        if (context_.gpu->has_gpu()) context_.gpu->set_gpu();
    }
};

template <typename Backend>
void fvm_lowered_cell_impl<Backend>::assert_tmin() {
    auto time_minmax = state_->time_bounds();
    if (time_minmax.first != time_minmax.second) {
        throw arbor_internal_error("fvm_lowered_cell: inconsistent times across cells");
    }
    if (time_minmax.first != tmin_) {
        throw arbor_internal_error("fvm_lowered_cell: out of synchronziation with cell state time");
    }
}

template <typename Backend>
void fvm_lowered_cell_impl<Backend>::reset() {
    state_->reset();
    set_tmin(0);

    for (auto& m: revpot_mechanisms_) {
        m->initialize();
    }

    for (auto& m: mechanisms_) {
        m->initialize();
    }

    update_ion_state();

    state_->zero_currents();

    // Note: mechanisms must be initialized again after the ion state is updated,
    // as mechanisms can read/write the ion_state within the initialize block
    for (auto& m: revpot_mechanisms_) {
        m->initialize();
    }

    for (auto& m: mechanisms_) {
        m->initialize();
    }

    // NOTE: Threshold watcher reset must come after the voltage values are set,
    // as voltage is implicitly read by watcher to set initial state.
    threshold_watcher_.reset();
}

template <typename Backend>
fvm_integration_result fvm_lowered_cell_impl<Backend>::integrate(
    value_type tfinal,
    value_type dt_max,
    std::vector<deliverable_event> staged_events,
    std::vector<sample_event> staged_samples)
{
    using util::as_const;

    set_gpu();

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

    arb_assert((assert_tmin(), true));
    unsigned remaining_steps = dt_steps(tmin_, tfinal, dt_max);
    PL();

    // TODO: Consider devolving more of this to back-end routines (e.g.
    // per-compartment dt probably not a win on GPU), possibly rumbling
    // complete fvm state into shared state object.

    while (remaining_steps) {
        // Update any required reversal potentials based on ionic concs.

        for (auto& m: revpot_mechanisms_) {
            m->nrn_current();
        }


        // Deliver events and accumulate mechanism current contributions.

        PE(advance_integrate_events);
        state_->deliverable_events.mark_until_after(state_->time);
        PL();

        PE(advance_integrate_current_zero);
        state_->zero_currents();
        PL();
        for (auto& m: mechanisms_) {
            m->deliver_events();
            m->nrn_current();
        }

        // Add current contribution from gap_junctions
        state_->add_gj_current();

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
        matrix_.assemble(state_->dt_intdom, state_->voltage, state_->current_density, state_->conductivity);
        PL();
        PE(advance_integrate_matrix_solve);
        matrix_.solve();
        memory::copy(matrix_.solution(), state_->voltage);
        PL();

        // Integrate mechanism state.

        for (auto& m: mechanisms_) {
            m->nrn_state();
        }

        // Update ion concentrations.

        PE(advance_integrate_ionupdate);
        update_ion_state();
        PL();

        // Update time and test for spike threshold crossings.

        PE(advance_integrate_threshold);
        threshold_watcher_.test();
        memory::copy(state_->time_to, state_->time);
        PL();

        // Check for non-physical solutions:

        if (check_voltage_mV>0) {
            PE(advance_integrate_physicalcheck);
            assert_voltage_bounded(check_voltage_mV);
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
}

template <typename B>
void fvm_lowered_cell_impl<B>::assert_voltage_bounded(fvm_value_type bound) {
    auto v_minmax = state_->voltage_bounds();
    if (v_minmax.first>=-bound && v_minmax.second<=bound) {
        return;
    }

    auto t_minmax = state_->time_bounds();
    throw range_check_failure(
        util::pprintf("voltage solution out of bounds for t in [{}, {}]", t_minmax.first, t_minmax.second),
        v_minmax.first<-bound? v_minmax.first: v_minmax.second);
}

template <typename B>
void fvm_lowered_cell_impl<B>::initialize(
    const std::vector<cell_gid_type>& gids,
    const recipe& rec,
    std::vector<fvm_index_type>& cell_to_intdom,
    std::vector<target_handle>& target_handles,
    probe_association_map<probe_handle>& probe_map)
{
    using util::any_cast;
    using util::count_along;
    using util::make_span;
    using util::value_by_key;
    using util::keys;

    set_gpu();

    std::vector<cable_cell> cells;
    const std::size_t ncell = gids.size();

    cells.resize(ncell);
    threading::parallel_for::apply(0, gids.size(), context_.thread_pool.get(),
           [&](cell_size_type i) {
               auto gid = gids[i];
               try {
                   cells[i] = any_cast<cable_cell&&>(rec.get_cell_description(gid));
               }
               catch (util::bad_any_cast&) {
                   throw bad_cell_description(rec.get_cell_kind(gid), gid);
               }
           });

    cable_cell_global_properties global_props;
    try {
        util::any rec_props = rec.get_global_properties(cell_kind::cable);
        if (rec_props.has_value()) {
            global_props = any_cast<cable_cell_global_properties>(rec_props);
        }
    }
    catch (util::bad_any_cast&) {
        throw bad_global_property(cell_kind::cable);
    }

    // Assert that all global default parameters have been set.
    // (Throws cable_cell_error on failure.)
    check_global_properties(global_props);

    const mechanism_catalogue* catalogue = global_props.catalogue;

    // Mechanism instantiator helper.
    auto mech_instance = [&catalogue](const std::string& name) {
        auto cat = builtin_mechanisms().has(name)? &builtin_mechanisms(): catalogue;
        return cat->instance<backend>(name);
    };

    // Check for physically reasonable membrane volages?

    check_voltage_mV = global_props.membrane_voltage_limit_mV;

    auto num_intdoms = fvm_intdom(rec, gids, cell_to_intdom);

    // Discretize cells, build matrix.

    fvm_cv_discretization D = fvm_cv_discretize(cells, global_props.default_parameters, context_);

    std::vector<index_type> cv_to_intdom(D.size());
    std::transform(D.geometry.cv_to_cell.begin(), D.geometry.cv_to_cell.end(), cv_to_intdom.begin(),
                   [&cell_to_intdom](index_type i){ return cell_to_intdom[i]; });

    arb_assert(D.n_cell() == ncell);
    matrix_ = matrix<backend>(D.geometry.cv_parent, D.geometry.cell_cv_divs,
                              D.cv_capacitance, D.face_conductance, D.cv_area, cell_to_intdom);
    sample_events_ = sample_event_stream(num_intdoms);

    // Discretize mechanism data.

    fvm_mechanism_data mech_data = fvm_build_mechanism_data(global_props, cells, D, context_);

    // Discretize and build gap junction info.

    auto gj_vector = fvm_gap_junctions(cells, gids, rec, D);

    // Create shared cell state.
    // (SIMD padding requires us to check each mechanism for alignment/padding constraints.)

    unsigned data_alignment = util::max_value(
        util::transform_view(keys(mech_data.mechanisms),
            [&](const std::string& name) { return mech_instance(name).mech->data_alignment(); }));

    state_ = std::make_unique<shared_state>(
                num_intdoms, cv_to_intdom, gj_vector, D.init_membrane_potential, D.temperature_K, D.diam_um,
                data_alignment? data_alignment: 1u);

    // Instantiate mechanisms and ions.

    for (auto& i: mech_data.ions) {
        const std::string& ion_name = i.first;

        if (auto charge = value_by_key(global_props.ion_species, ion_name)) {
            state_->add_ion(ion_name, *charge, i.second);
        }
        else {
            throw cable_cell_error("unrecognized ion '"+ion_name+"' in mechanism");
        }
    }

    target_handles.resize(mech_data.n_target);

    unsigned mech_id = 0;
    for (auto& m: mech_data.mechanisms) {
        auto& name = m.first;
        auto& config = m.second;

        mechanism_layout layout;
        layout.cv = config.cv;
        layout.multiplicity = config.multiplicity;
        layout.weight.resize(layout.cv.size());

        std::vector<fvm_index_type> multiplicity_divs;
        auto multiplicity_part = util::make_partition(multiplicity_divs, layout.multiplicity);

        // Mechanism weights are F·α where α ∈ [0, 1] is the proportional
        // contribution in the CV, and F is the scaling factor required
        // to convert from the mechanism current contribution units to A/m².

        switch (config.kind) {
        case mechanismKind::point:
            // Point mechanism contributions are in [nA]; CV area A in [µm^2].
            // F = 1/A * [nA/µm²] / [A/m²] = 1000/A.

            for (auto i: count_along(config.cv)) {
                auto cv = layout.cv[i];
                layout.weight[i] = 1000/D.cv_area[cv];

                // (builtin stimulus, for example, has no targets)

                if (!config.target.empty()) {
                    if(!config.multiplicity.empty()) {
                        for (auto j: make_span(multiplicity_part[i])) {
                            target_handles[config.target[j]] = target_handle(mech_id, i, cv_to_intdom[cv]);
                        }
                    } else {
                        target_handles[config.target[i]] = target_handle(mech_id, i, cv_to_intdom[cv]);
                    };
                }
            }
            break;
        case mechanismKind::density:
            // Current density contributions from mechanism are already in [A/m²].

            for (auto i: count_along(layout.cv)) {
                layout.weight[i] = config.norm_area[i];
            }
            break;
        case mechanismKind::revpot:
            // Mechanisms that set reversal potential should not be contributing
            // to any currents, so leave weights as zero.
            break;
        }

        auto minst = mech_instance(name);
        minst.mech->instantiate(mech_id++, *state_, minst.overrides, layout);

        for (auto& pv: config.param_values) {
            minst.mech->set_parameter(pv.first, pv.second);
        }

        if (config.kind==mechanismKind::revpot) {
            revpot_mechanisms_.push_back(mechanism_ptr(minst.mech.release()));
        }
        else {
            mechanisms_.push_back(mechanism_ptr(minst.mech.release()));
        }
    }

    // Collect detectors, probe handles.

    std::vector<index_type> detector_cv;
    std::vector<value_type> detector_threshold;

    for (auto cell_idx: make_span(ncell)) {
        cell_gid_type gid = gids[cell_idx];

        for (auto entry: cells[cell_idx].detectors()) {
            detector_cv.push_back(D.geometry.location_cv(cell_idx, entry.loc, cv_prefer::cv_empty));
            detector_threshold.push_back(entry.item.threshold);
        }

        for (cell_lid_type j: make_span(rec.num_probes(gid))) {
            probe_info pi = rec.get_probe({gid, j});
            auto where = any_cast<cell_probe_address>(pi.address);

            fvm_size_type cv;
            probe_handle handle;

            switch (where.kind) {
            case cell_probe_address::membrane_voltage:
                cv = D.geometry.location_cv(cell_idx, where.location, cv_prefer::cv_empty);
                handle = state_->voltage.data()+cv;
                break;
            case cell_probe_address::membrane_current:
                cv = D.geometry.location_cv(cell_idx, where.location, cv_prefer::cv_nonempty);
                handle = state_->current_density.data()+cv;
                break;
            default:
                throw arbor_internal_error("fvm_lowered_cell: unrecognized probeKind");
            }

            probe_map.insert({pi.id, {handle, pi.tag}});
        }
    }

    threshold_watcher_ = backend::voltage_watcher(*state_, detector_cv, detector_threshold, context_);

    reset();
}

// Get vector of gap_junctions
template <typename B>
std::vector<fvm_gap_junction> fvm_lowered_cell_impl<B>::fvm_gap_junctions(
        const std::vector<cable_cell>& cells,
        const std::vector<cell_gid_type>& gids,
        const recipe& rec, const fvm_cv_discretization& D) {

    std::vector<fvm_gap_junction> v;

    std::unordered_map<cell_gid_type, std::vector<unsigned>> gid_to_cvs;
    for (auto cell_idx: util::make_span(0, D.n_cell())) {
        if (!rec.num_gap_junction_sites(gids[cell_idx])) continue;

        gid_to_cvs[gids[cell_idx]].reserve(rec.num_gap_junction_sites(gids[cell_idx]));
        const auto& cell_gj = cells[cell_idx].gap_junction_sites();

        for (auto gj : cell_gj) {
            auto cv = D.geometry.location_cv(cell_idx, gj.loc, cv_prefer::cv_nonempty);
            gid_to_cvs[gids[cell_idx]].push_back(cv);
        }
    }

    for (auto gid: gids) {
        auto gj_list = rec.gap_junctions_on(gid);
        for (auto g: gj_list) {
            if (gid != g.local.gid && gid != g.peer.gid) {
                throw arb::bad_cell_description(cell_kind::cable, gid);
            }
            cell_gid_type cv0, cv1;
            try {
                cv0 = gid_to_cvs[g.local.gid].at(g.local.index);
                cv1 = gid_to_cvs[g.peer.gid].at(g.peer.index);
            }
            catch (std::out_of_range&) {
                throw arb::bad_cell_description(cell_kind::cable, gid);
            }
            if (gid != g.local.gid) {
                std::swap(cv0, cv1);
            }
            v.push_back(fvm_gap_junction(std::make_pair(cv0, cv1), g.ggap * 1e3 / D.cv_area[cv0]));
        }
    }

    return v;
}

template <typename B>
fvm_size_type fvm_lowered_cell_impl<B>::fvm_intdom(
        const recipe& rec,
        const std::vector<cell_gid_type>& gids,
        std::vector<fvm_index_type>& cell_to_intdom) {

    cell_to_intdom.resize(gids.size());

    std::unordered_map<cell_gid_type, cell_size_type> gid_to_loc;
    for (auto i: util::count_along(gids)) {
        gid_to_loc[gids[i]] = i;
    }

    std::unordered_set<cell_gid_type> visited;
    std::queue<cell_gid_type> intdomq;
    cell_size_type intdom_id = 0;

    for (auto gid: gids) {
        if (visited.count(gid)) continue;
        visited.insert(gid);

        intdomq.push(gid);
        while (!intdomq.empty()) {
            auto g = intdomq.front();
            intdomq.pop();

            cell_to_intdom[gid_to_loc[g]] = intdom_id;

            for (auto gj: rec.gap_junctions_on(g)) {
                cell_gid_type peer =
                        gj.local.gid==g? gj.peer.gid:
                        gj.peer.gid==g?  gj.local.gid:
                        throw bad_cell_description(cell_kind::cable, g);

                if (!gid_to_loc.count(peer)) {
                    throw gj_unsupported_domain_decomposition(g, peer);
                }

                if (!visited.count(peer)) {
                    visited.insert(peer);
                    intdomq.push(peer);
                }
            }
        }
        intdom_id++;
    }

    return intdom_id;
}

} // namespace arb
