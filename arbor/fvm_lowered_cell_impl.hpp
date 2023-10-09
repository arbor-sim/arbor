#pragma once

// Implementations for fvm_lowered_cell are parameterized
// on the back-end class.
//
// Classes here are exposed in a header only so that
// implementation details may be tested in the unit tests.
// It should otherwise only be used in `fvm_lowered_cell.cpp`.

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <arbor/assert.hpp>
#include <arbor/common_types.hpp>
#include <arbor/cable_cell_param.hpp>
#include <arbor/recipe.hpp>
#include <arbor/util/any_visitor.hpp>

#include "execution_context.hpp"
#include "fvm_layout.hpp"
#include "fvm_lowered_cell.hpp"
#include "label_resolution.hpp"
#include "profile/profiler_macro.hpp"
#include "util/maputil.hpp"
#include "util/meta.hpp"
#include "util/rangeutil.hpp"
#include "util/strprintf.hpp"
#include "util/transform.hpp"

namespace arb {

template <class Backend>
class fvm_lowered_cell_impl: public fvm_lowered_cell {
public:
    using backend = Backend;
    using value_type = arb_value_type;
    using index_type = arb_index_type;
    using size_type = arb_size_type;

    fvm_lowered_cell_impl(execution_context ctx, arb_seed_type seed = 0):
        context_(ctx),
        seed_{seed}
    {};

    void reset() override;

    fvm_initialization_data initialize(
        const std::vector<cell_gid_type>& gids,
        const recipe& rec) override;

    fvm_integration_result integrate(
        const timestep_range& dts,
        const std::vector<std::vector<std::vector<deliverable_event>>>& staged_events_per_mech_id,
        const std::vector<std::vector<sample_event>>& staged_samples) override;

    value_type time() const override { return state_->time; }

    //Exposed for testing purposes
    std::vector<mechanism_ptr>& mechanisms() {
        return mechanisms_;
    }

    ARB_SERDES_ENABLE(fvm_lowered_cell_impl<Backend>, seed_, state_);

    void t_serialize(serializer& ser, const std::string& k) const override { serialize(ser, k, *this); }
    void t_deserialize(serializer& ser, const std::string& k) override { deserialize(ser, k, *this); }

private:
    // Host or GPU-side back-end dependent storage.
    using array               = typename backend::array;
    using shared_state        = typename backend::shared_state;

    execution_context context_;

    std::unique_ptr<shared_state> state_; // Cell state shared across mechanisms.

    std::vector<mechanism_ptr> mechanisms_; // excludes reversal potential calculators.
    std::vector<mechanism_ptr> revpot_mechanisms_;
    std::vector<mechanism_ptr> voltage_mechanisms_;

    // Optional non-physical voltage check threshold
    std::optional<double> check_voltage_mV_;

    // random number generator seed value
    arb_seed_type seed_;

    // Flag indicating that at least one of the mechanisms implements the post_events procedure
    bool post_events_ = false;

    void update_ion_state();

    // Throw if absolute value of membrane voltage exceeds bounds.
    void assert_voltage_bounded(arb_value_type bound);

    // Sets the GPU used for CUDA calls from the thread that calls it.
    // The GPU will be the one in the execution context context_.
    // If not called, the thread may attempt to launch on a different GPU,
    // leading to crashes.
    void set_gpu() {
        if (context_.gpu->has_gpu()) context_.gpu->set_gpu();
    }

    // Translate cell probe descriptions into probe handles etc.
    void resolve_probe_address(
        std::vector<fvm_probe_data>& probe_data, // out parameter
        const std::vector<cable_cell>& cells,
        std::size_t cell_idx,
        const std::any& paddr,
        const fvm_cv_discretization& D,
        const fvm_mechanism_data& M,
        const std::vector<target_handle>& handles,
        const std::unordered_map<std::string, mechanism*>& mech_instance_by_name);

        // Add probes to fvm_info::probe_map
        void add_probes(const std::vector<cell_gid_type>& gids,
                        const std::vector<cable_cell>& cells,
                        const recipe& rec,
                        const fvm_cv_discretization& D,
                        const std::unordered_map<std::string, mechanism*>& mechptr_by_name,
                        const fvm_mechanism_data& mech_data,
                        const std::vector<target_handle>& target_handles,
                        probe_association_map& probe_map);
};

template <typename Backend>
void fvm_lowered_cell_impl<Backend>::reset() {
    state_->reset();

    for (auto& m: voltage_mechanisms_) {
        m->initialize();
    }

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

    for (auto& m: voltage_mechanisms_) {
        m->initialize();
    }

    // NOTE: Threshold watcher reset must come after the voltage values are set,
    // as voltage is implicitly read by watcher to set initial state.
    state_->reset_thresholds();
}

template <typename Backend>
fvm_integration_result fvm_lowered_cell_impl<Backend>::integrate(
    const timestep_range& dts,
    const std::vector<std::vector<std::vector<deliverable_event>>>& staged_events_per_mech_id,
    const std::vector<std::vector<sample_event>>& staged_samples)
{
    arb_assert(state_->time == dts.t_begin());
    set_gpu();

    // Integration setup
    PE(advance:integrate:setup);
    // Push samples and events down to the state and reset the spike thresholds.
    state_->begin_epoch(staged_events_per_mech_id, staged_samples, dts);
    PL();

    // loop over timesteps
    for (const auto& ts : dts) {
        state_->update_time_to(ts);
        arb_assert(state_->time == ts.t_begin());

        // Update integration step time information visible to mechanisms.
        for (auto& m: mechanisms_) {
            m->set_dt(state_->dt);
        }
        for (auto& m: revpot_mechanisms_) {
            m->set_dt(state_->dt);
        }
        for (auto& m: voltage_mechanisms_) {
            m->set_dt(state_->dt);
        }

        // Update any required reversal potentials based on ionic concentrations
        for (auto& m: revpot_mechanisms_) {
            m->update_current();
        }

        PE(advance:integrate:current:zero);
        state_->zero_currents();
        PL();

        // Deliver events and accumulate mechanism current contributions.

        // Mark all events due before (but not including) the end of this time step (state_->time_to) for delivery
        state_->mark_events();
        for (auto& m: mechanisms_) {
            // apply the events and drop them afterwards
            state_->deliver_events(*m);
            m->update_current();
        }

        // Add stimulus current contributions.
        // NOTE: performed after dt, time_to calculation, in case we want to
        // use mean current contributions as opposed to point sample.
        PE(advance:integrate:stimuli)
        state_->add_stimulus_current();
        PL();

        // Take samples at cell time if sample time in this step interval.
        PE(advance:integrate:samples);
        state_->take_samples();
        PL();

        // Integrate voltage and diffusion
        PE(advance:integrate:cable);
        state_->integrate_cable_state();
        PL();

        // Integrate mechanism state for density
        for (auto& m: mechanisms_) {
            state_->update_prng_state(*m);
            m->update_state();
        }

        // Update ion concentrations.
        PE(advance:integrate:ionupdate);
        update_ion_state();
        PL();

        // voltage mechs run now; after the cable_solver, but before the
        // threshold test
        for (auto& m: voltage_mechanisms_) {
            m->update_current();
        }
        for (auto& m: voltage_mechanisms_) {
            state_->update_prng_state(*m);
            m->update_state();
        }

        // Update time and test for spike threshold crossings.
        PE(advance:integrate:threshold);
        state_->test_thresholds();
        PL();

        PE(advance:integrate:post)
        if (post_events_) {
            for (auto& m: mechanisms_) {
                m->post_event();
            }
        }
        PL();

        // Advance epoch
        state_->next_time_step();

        // Check for non-physical solutions:
        if (check_voltage_mV_) {
            PE(advance:integrate:physicalcheck);
            assert_voltage_bounded(check_voltage_mV_.value());
            PL();
        }
    }

    return state_->get_integration_result();
}

template <typename Backend>
void fvm_lowered_cell_impl<Backend>::update_ion_state() {
    state_->ions_init_concentration();
    for (auto& m: mechanisms_) {
        m->update_ions();
    }
}

template <typename Backend>
void fvm_lowered_cell_impl<Backend>::assert_voltage_bounded(arb_value_type bound) {
    auto v_minmax = state_->voltage_bounds();
    if (v_minmax.first>=-bound && v_minmax.second<=bound) {
        return;
    }

    throw range_check_failure(
        util::pprintf("voltage solution out of bounds for at t = {}", state_->time),
        v_minmax.first<-bound? v_minmax.first: v_minmax.second);
}

inline
fvm_detector_info get_detector_info(arb_size_type max,
                                    arb_size_type ncell,
                                    const std::vector<cable_cell>& cells,
                                    const fvm_cv_discretization& D,
                                    execution_context ctx) {
    std::vector<arb_index_type> cv;
    std::vector<arb_value_type> threshold;
    for (auto cell_idx: util::make_span(ncell)) {
        for (auto entry: cells[cell_idx].detectors()) {
            cv.push_back(D.geometry.location_cv(cell_idx, entry.loc, cv_prefer::cv_empty));
            threshold.push_back(entry.item.threshold);
        }
    }
    return { max, std::move(cv), std::move(threshold), ctx };
}

template <typename Backend>
void fvm_lowered_cell_impl<Backend>::add_probes(const std::vector<cell_gid_type>& gids,
                                                const std::vector<cable_cell>& cells,
                                                const recipe& rec,
                                                const fvm_cv_discretization& D,
                                                const std::unordered_map<std::string, mechanism*>& mechptr_by_name,
                                                const fvm_mechanism_data& mech_data,
                                                const std::vector<target_handle>& target_handles,
                                                probe_association_map& probe_map) {
    auto ncell = gids.size();

    std::vector<fvm_probe_data> probe_data;
    for (auto cell_idx: util::make_span(ncell)) {
        cell_gid_type gid = gids[cell_idx];
        const auto& rec_probes = rec.get_probes(gid);
        for (const auto& pi: rec_probes) {
            resolve_probe_address(probe_data, cells, cell_idx, pi.address, D, mech_data, target_handles, mechptr_by_name);
            if (!probe_data.empty()) {
                cell_address_type addr{gid, pi.tag};
                if (probe_map.count(addr)) throw dup_cell_probe(cell_kind::lif, gid, pi.tag);
                for (auto& data: probe_data) {
                    probe_map.insert(addr, std::move(data));
                }
            }
        }
    }
}

template <typename Backend>
fvm_initialization_data fvm_lowered_cell_impl<Backend>::initialize(const std::vector<cell_gid_type>& gids,
                                                                   const recipe& rec) {
    using std::any_cast;
    using util::count_along;
    using util::make_span;
    using util::value_by_key;
    using util::keys;

    fvm_initialization_data fvm_info;

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
               catch (std::bad_any_cast&) {
                   throw bad_cell_description(rec.get_cell_kind(gid), gid);
               }
           });

    // Populate source, target and gap_junction data vectors.
    for (auto i : util::make_span(ncell)) {
        auto gid = gids[i];
        const auto& c = cells[i];

        fvm_info.source_data.add_cell();
        fvm_info.target_data.add_cell();
        fvm_info.gap_junction_data.add_cell();

        unsigned count = 0;
        for (const auto& [label, range]: c.detector_ranges()) {
            fvm_info.source_data.add_label(label, range);
            count+=(range.end - range.begin);
        }
        fvm_info.num_sources[gid] = count;

        count = 0;
        for (const auto& [label, range]: c.synapse_ranges()) {
            fvm_info.target_data.add_label(label, range);
            count+=(range.end - range.begin);
        }
        fvm_info.num_targets[gid] = count;

        for (const auto& [label, range]: c.junction_ranges()) {
            fvm_info.gap_junction_data.add_label(label, range);
        }
    }

    cable_cell_global_properties global_props;
    try {
        std::any rec_props = rec.get_global_properties(cell_kind::cable);
        if (rec_props.has_value()) {
            global_props = any_cast<cable_cell_global_properties>(rec_props);
        }
    }
    catch (std::bad_any_cast&) {
        throw bad_global_property(cell_kind::cable);
    }

    // Assert that all global default parameters have been set.
    // (Throws cable_cell_error on failure.)
    check_global_properties(global_props);

    const auto& catalogue = global_props.catalogue;

    // Mechanism instantiator helper.
    auto mech_instance = [&catalogue](const std::string& name) {
        return catalogue.instance(backend::kind, name);
    };

    // Check for physically reasonable membrane volages?

    check_voltage_mV_ = global_props.membrane_voltage_limit_mV;

    // Discretize cells, build matrix.

    fvm_cv_discretization D = fvm_cv_discretize(cells, global_props.default_parameters, context_);

    arb_assert(D.n_cell() == ncell);

    // Discretize and build gap junction info.

    auto gj_cvs = fvm_build_gap_junction_cv_map(cells, gids, D);
    auto gj_conns = fvm_resolve_gj_connections(gids, fvm_info.gap_junction_data, gj_cvs, rec);

    // Discretize mechanism data.

    fvm_mechanism_data mech_data = fvm_build_mechanism_data(global_props, cells, gids, gj_conns, D, context_);

    // Fill src_to_spike and cv_to_cell vectors only if mechanisms with post_events implemented are present.
    post_events_ = mech_data.post_events;
    auto max_detector = 0;
    if (post_events_) {
        auto it = util::max_element_by(fvm_info.num_sources, [](auto elem) {return util::second(elem);});
        max_detector = it->second;
    }
    std::vector<arb_index_type> src_to_spike, cv_to_cell;

    if (post_events_) {
        for (auto cell_idx: make_span(ncell)) {
            for (auto lid: make_span(fvm_info.num_sources[gids[cell_idx]])) {
                src_to_spike.push_back(cell_idx * max_detector + lid);
            }
        }
        src_to_spike.shrink_to_fit();
        cv_to_cell = D.geometry.cv_to_cell;
    }

    // map control volume ids to global cell ids
    std::vector<arb_index_type> cv_to_gid(D.geometry.cv_to_cell.size());
    std::transform(D.geometry.cv_to_cell.begin(), D.geometry.cv_to_cell.end(),
                   cv_to_gid.begin(),
                   [&gids](auto i){return gids[i]; });

    // Create shared cell state.
    // Shared state vectors should accommodate each mechanism's data alignment requests.

    unsigned data_alignment = util::max_value(
        util::transform_view(keys(mech_data.mechanisms),
            [&](const std::string& name) { return mech_instance(name).mech->data_alignment(); }));

    auto d_info = get_detector_info(max_detector, ncell, cells, D, context_);
    state_ = std::make_unique<shared_state>(context_.thread_pool,
                                            ncell,
                                            std::move(cv_to_cell),
                                            D,
                                            std::move(src_to_spike),
                                            d_info,
                                            mech_data.ions,
                                            mech_data.stimuli,
                                            data_alignment? data_alignment: 1u,
                                            seed_);

    fvm_info.target_handles.resize(mech_data.n_target);

    // Keep track of mechanisms by name for probe lookup.
    std::unordered_map<std::string, mechanism*> mechptr_by_name;

    unsigned mech_id = 0;
    for (const auto& [name, config]: mech_data.mechanisms) {
        mechanism_layout layout;
        layout.cv = config.cv;
        layout.multiplicity = config.multiplicity;
        layout.peer_cv = config.peer_cv;
        layout.weight.resize(layout.cv.size());

        std::vector<arb_index_type> multiplicity_divs;
        auto multiplicity_part = util::make_partition(multiplicity_divs, layout.multiplicity);

        // Mechanism weights are F·α where α ∈ [0, 1] is the proportional
        // contribution in the CV, and F is the scaling factor required
        // to convert from the mechanism current contribution units to A/m².

        arb_size_type idx_offset = 0;
        switch (config.kind) {
        case arb_mechanism_kind_point:
            // Point mechanism contributions are in [nA]; CV area A in [µm^2].
            // F = 1/A * [nA/µm²] / [A/m²] = 1000/A.

            layout.gid.resize(config.cv.size());
            layout.idx.resize(layout.gid.size());
            for (auto i: count_along(config.cv)) {
                auto cv = layout.cv[i];
                layout.weight[i] = 1000/D.cv_area[cv];
                layout.gid[i] = cv_to_gid[cv];
                if (i>0 && (layout.gid[i-1] != layout.gid[i])) idx_offset = i;
                layout.idx[i] = i - idx_offset;

                if (config.target.empty()) continue;

                target_handle handle(mech_id, i);
                if (config.multiplicity.empty()) {
                    fvm_info.target_handles[config.target[i]] = handle;
                }
                else {
                    for (auto j: make_span(multiplicity_part[i])) {
                        fvm_info.target_handles[config.target[j]] = handle;
                    }
                }
            }
            fvm_info.num_targets_per_mech_id[mech_id] = config.target.size();
            break;
        case arb_mechanism_kind_gap_junction:
            // Junction mechanism contributions are in [nA] (µS * mV); CV area A in [µm^2].
            // F = 1/A * [nA/µm²] / [A/m²] = 1000/A.

            for (auto i: count_along(layout.cv)) {
                auto cv = layout.cv[i];
                layout.weight[i] = config.local_weight[i] * 1000/D.cv_area[cv];
            }
            break;
        case arb_mechanism_kind_voltage:
        case arb_mechanism_kind_density:
            // Current density contributions from mechanism are already in [A/m²].

            layout.gid.resize(layout.cv.size());
            layout.idx.resize(layout.gid.size());
            for (auto i: count_along(layout.cv)) {
                layout.weight[i] = config.norm_area[i];
                layout.gid[i] = cv_to_gid[i];
                if (i>0 && (layout.gid[i-1] != layout.gid[i])) idx_offset = i;
                layout.idx[i] = i - idx_offset;
            }
            break;
        case arb_mechanism_kind_reversal_potential:
            // Mechanisms that set reversal potential should not be contributing
            // to any currents, so leave weights as zero.
            break;
        }

        auto [mech, over] = mech_instance(name);
        state_->instantiate(*mech, mech_id, over, layout, config.param_values);
        mechptr_by_name[name] = mech.get();
        ++mech_id;

        switch (config.kind) {
            case arb_mechanism_kind_gap_junction:
            case arb_mechanism_kind_point:
            case arb_mechanism_kind_density: {
                mechanisms_.emplace_back(mech.release());
                break;
            }
            case arb_mechanism_kind_reversal_potential: {
                revpot_mechanisms_.emplace_back(mech.release());
                break;
            }
            case arb_mechanism_kind_voltage: {
                voltage_mechanisms_.emplace_back(mech.release());
                break;
            }
            default:;
                throw invalid_mechanism_kind(config.kind);
        }
    }

    add_probes(gids, cells, rec, D, mechptr_by_name, mech_data, fvm_info.target_handles, fvm_info.probe_map);

    reset();
    return fvm_info;
}

// Resolution of probe addresses into a specific fvm_probe_data draws upon data
// from the cable cell, the discretization, the target handle map, and the
// back-end shared state.
//
// `resolve_probe_address` collates this data into a `probe_resolution_data`
// struct which is then passed on to the specific resolution procedure
// determined by the type of the user-supplied probe address.

template <typename Backend>
struct probe_resolution_data {
    std::vector<fvm_probe_data>& result;
    typename Backend::shared_state* state;
    const cable_cell& cell;
    const std::size_t cell_idx;
    const fvm_cv_discretization& D;
    const fvm_mechanism_data& M;
    const std::vector<target_handle>& handles;
    const std::unordered_map<std::string, mechanism*>& mech_instance_by_name;

    // Backend state data for a given mechanism and state variable.
    const arb_value_type* mechanism_state(const std::string& name, const std::string& state_var) const {
        mechanism* m = util::value_by_key(mech_instance_by_name, name).value_or(nullptr);
        if (!m) return nullptr;

        const arb_value_type* data = state->mechanism_state_data(*m, state_var);
        if (!data) throw cable_cell_error("no state variable '"+state_var+"' in mechanism '"+name+"'");

        return data;
    }

    // Extent of density mechanism on cell.
    mextent mechanism_support(const std::string& name) const {
        auto& mech_map = cell.region_assignments().template get<density>();
        auto opt_mm = util::value_by_key(mech_map, name);

        return opt_mm? opt_mm->support(): mextent{};
    };

    // Index into ion data from location.
    std::optional<arb_index_type> ion_location_index(const std::string& ion, mlocation loc) const {
        if (state->ion_data.count(ion)) {
            return util::binary_search_index(M.ions.at(ion).cv,
                arb_index_type(D.geometry.location_cv(cell_idx, loc, cv_prefer::cv_nonempty)));
        }
        return std::nullopt;
    }
};

template <typename Backend>
void fvm_lowered_cell_impl<Backend>::resolve_probe_address(std::vector<fvm_probe_data>& probe_data,
                                                           const std::vector<cable_cell>& cells,
                                                           std::size_t cell_idx,
                                                           const std::any& paddr,
                                                           const fvm_cv_discretization& D,
                                                           const fvm_mechanism_data& M,
                                                           const std::vector<target_handle>& handles,
                                                           const std::unordered_map<std::string, mechanism*>& mech_instance_by_name) {
    probe_data.clear();
    probe_resolution_data<Backend> prd{
        probe_data, state_.get(), cells[cell_idx], cell_idx, D, M, handles, mech_instance_by_name};

    using V = util::any_visitor<
        cable_probe_membrane_voltage,
        cable_probe_membrane_voltage_cell,
        cable_probe_axial_current,
        cable_probe_total_ion_current_density,
        cable_probe_total_ion_current_cell,
        cable_probe_total_current_cell,
        cable_probe_stimulus_current_cell,
        cable_probe_density_state,
        cable_probe_density_state_cell,
        cable_probe_point_state,
        cable_probe_point_state_cell,
        cable_probe_ion_current_density,
        cable_probe_ion_current_cell,
        cable_probe_ion_int_concentration,
        cable_probe_ion_int_concentration_cell,
        cable_probe_ion_diff_concentration,
        cable_probe_ion_diff_concentration_cell,
        cable_probe_ion_ext_concentration,
        cable_probe_ion_ext_concentration_cell>;

    auto visitor = util::overload(
        [&prd](auto& probe_addr) { resolve_probe(probe_addr, prd); },
        [] { throw cable_cell_error("unrecognized probe type"), fvm_probe_data{}; });

    return V::visit(visitor, paddr);
}

template <typename B>
void resolve_probe(const cable_probe_membrane_voltage& p, probe_resolution_data<B>& R) {
    const arb_value_type* data = R.state->voltage.data();

    for (mlocation loc: thingify(p.locations, R.cell.provider())) {
        fvm_voltage_interpolant in = fvm_interpolate_voltage(R.cell, R.D, R.cell_idx, loc);

        R.result.push_back(fvm_probe_interpolated{
            {data+in.proximal_cv, data+in.distal_cv},
            {in.proximal_coef, in.distal_coef},
            loc});
    }
}

template <typename B>
void resolve_probe(const cable_probe_membrane_voltage_cell& p, probe_resolution_data<B>& R) {
    fvm_probe_multi r;
    mcable_list cables;

    for (auto cv: R.D.geometry.cell_cvs(R.cell_idx)) {
        const double* ptr = R.state->voltage.data()+cv;
        for (auto cable: R.D.geometry.cables(cv)) {
            r.raw_handles.push_back(ptr);
            cables.push_back(cable);
        }
    }
    r.metadata = std::move(cables);
    r.shrink_to_fit();

    R.result.push_back(std::move(r));
}

template <typename B>
void resolve_probe(const cable_probe_axial_current& p, probe_resolution_data<B>& R) {
    const arb_value_type* data = R.state->voltage.data();

    for (mlocation loc: thingify(p.locations, R.cell.provider())) {
        fvm_voltage_interpolant in = fvm_axial_current(R.cell, R.D, R.cell_idx, loc);

        R.result.push_back(fvm_probe_interpolated{
            {data+in.proximal_cv, data+in.distal_cv},
            {in.proximal_coef, in.distal_coef},
            loc});
    }
}

template <typename B>
void resolve_probe(const cable_probe_total_ion_current_density& p, probe_resolution_data<B>& R) {
    // Use interpolated probe with coeffs 1, -1 to represent difference between accumulated current density and stimulus.
    for (mlocation loc: thingify(p.locations, R.cell.provider())) {
        arb_index_type cv = R.D.geometry.location_cv(R.cell_idx, loc, cv_prefer::cv_nonempty);
        const double* current_cv_ptr = R.state->current_density.data() + cv;

        auto opt_i = util::binary_search_index(R.M.stimuli.cv_unique, cv);
        const double* stim_cv_ptr = opt_i? R.state->stim_data.accu_stim_.data()+*opt_i: nullptr;

        R.result.push_back(fvm_probe_interpolated{
            {current_cv_ptr, stim_cv_ptr},
            {1., -1.},
            loc});
    }
}

template <typename B>
void resolve_probe(const cable_probe_total_ion_current_cell& p, probe_resolution_data<B>& R) {
    fvm_probe_interpolated_multi r;
    std::vector<const double*> stim_handles;

    for (auto cv: R.D.geometry.cell_cvs(R.cell_idx)) {
        const double* current_cv_ptr = R.state->current_density.data()+cv;
        auto opt_i = util::binary_search_index(R.M.stimuli.cv_unique, cv);
        const double* stim_cv_ptr = opt_i? R.state->stim_data.accu_stim_.data()+*opt_i: nullptr;

        for (auto cable: R.D.geometry.cables(cv)) {
            double area = R.cell.embedding().integrate_area(cable); // [µm²]
            if (area>0) {
                r.raw_handles.push_back(current_cv_ptr);
                stim_handles.push_back(stim_cv_ptr);
                r.coef[0].push_back(0.001*area); // Scale from [µm²·A/m²] to [nA].
                r.coef[1].push_back(-r.coef[0].back());
                r.metadata.push_back(cable);
            }
        }
    }

    util::append(r.raw_handles, stim_handles);
    r.shrink_to_fit();
    R.result.push_back(std::move(r));
}

template <typename B>
void resolve_probe(const cable_probe_total_current_cell& p, probe_resolution_data<B>& R) {
    fvm_probe_membrane_currents r;

    auto cell_cv_ival = R.D.geometry.cell_cv_interval(R.cell_idx);
    auto cv0 = cell_cv_ival.first;

    util::assign(r.cv_parent, util::transform_view(util::subrange_view(R.D.geometry.cv_parent, cell_cv_ival),
        [cv0](auto cv) { return cv+1==0? cv: cv-cv0; }));
    util::assign(r.cv_parent_cond, util::subrange_view(R.D.face_conductance, cell_cv_ival));

    const auto& stim_cvs = R.M.stimuli.cv_unique;
    const arb_value_type* stim_src = R.state->stim_data.accu_stim_.data();

    r.cv_cables_divs = {0};
    for (auto cv: R.D.geometry.cell_cvs(R.cell_idx)) {
        r.raw_handles.push_back(R.state->voltage.data()+cv);
        double oo_cv_area = R.D.cv_area[cv]>0? 1./R.D.cv_area[cv]: 0;

        for (auto cable: R.D.geometry.cables(cv)) {
            double area = R.cell.embedding().integrate_area(cable); // [µm²]
            if (area>0) {
                r.weight.push_back(area*oo_cv_area);
                r.metadata.push_back(cable);
            }
        }
        r.cv_cables_divs.push_back(r.metadata.size());
    }
    for (auto cv: R.D.geometry.cell_cvs(R.cell_idx)) {
        auto opt_i = util::binary_search_index(stim_cvs, cv);
        if (!opt_i) continue;

        r.raw_handles.push_back(stim_src+*opt_i);
        r.stim_cv.push_back(cv-cv0);
        r.stim_scale.push_back(0.001*R.D.cv_area[cv]); // Scale from [µm²·A/m²] to [nA].
    }
    r.shrink_to_fit();
    R.result.push_back(std::move(r));
}

template <typename B>
void resolve_probe(const cable_probe_stimulus_current_cell& p, probe_resolution_data<B>& R) {
    fvm_probe_weighted_multi r;

    const auto& stim_cvs = R.M.stimuli.cv_unique;
    const arb_value_type* src = R.state->stim_data.accu_stim_.data();

    for (auto cv: R.D.geometry.cell_cvs(R.cell_idx)) {
        auto opt_i = util::binary_search_index(stim_cvs, cv);
        const double* ptr = opt_i? src+*opt_i: nullptr;

        for (auto cable: R.D.geometry.cables(cv)) {
            double area = R.cell.embedding().integrate_area(cable); // [µm²]
            if (area>0) {
                r.raw_handles.push_back(ptr);
                r.weight.push_back(0.001*area); // Scale from [µm²·A/m²] to [nA].
                r.metadata.push_back(cable);
            }
        }
    }

    r.shrink_to_fit();
    R.result.push_back(std::move(r));
}

template <typename B>
void resolve_probe(const cable_probe_density_state& p, probe_resolution_data<B>& R) {
    const auto& mech = p.mechanism;
    if (!R.mech_instance_by_name.count(mech)) return;
    const arb_value_type* data = R.mechanism_state(mech, p.state);
    if (!data) return;

    auto support = R.mechanism_support(mech);
    for (mlocation loc: thingify(p.locations, R.cell.provider())) {
        if (!support.intersects(loc)) continue;

        arb_index_type cv = R.D.geometry.location_cv(R.cell_idx, loc, cv_prefer::cv_nonempty);
        auto opt_i = util::binary_search_index(R.M.mechanisms.at(mech).cv, cv);
        if (!opt_i) continue;

        R.result.push_back(fvm_probe_scalar{{data+*opt_i}, loc});
    }
}

template <typename B>
void resolve_probe(const cable_probe_density_state_cell& p, probe_resolution_data<B>& R) {
    fvm_probe_multi r;

    const arb_value_type* data = R.mechanism_state(p.mechanism, p.state);
    if (!data) return;

    mextent support = R.mechanism_support(p.mechanism);
    if (!R.M.mechanisms.count(p.mechanism)) return;
    auto& mech_cvs = R.M.mechanisms.at(p.mechanism).cv;
    mcable_list cables;

    for (auto i: util::count_along(mech_cvs)) {
        auto cv = mech_cvs[i];
        auto cv_cables = R.D.geometry.cables(cv);
        mextent cv_extent = mcable_list(cv_cables.begin(), cv_cables.end());
        for (auto cable: intersect(cv_extent, support)) {
            if (cable.prox_pos==cable.dist_pos) continue;

            r.raw_handles.push_back(data+i);
            cables.push_back(cable);
        }
    }
    r.metadata = std::move(cables);
    r.shrink_to_fit();
    R.result.push_back(std::move(r));
}

inline
auto point_info_of(cell_lid_type target,
                   int mech_index,
                   const mlocation_map<synapse>& instances,
                   const std::vector<arb_index_type>& multiplicity) {

    auto opt_i = util::binary_search_index(instances, target, [](auto& item) { return item.lid; });
    if (!opt_i) throw arbor_internal_error("inconsistent mechanism state");

    return cable_probe_point_info {target,
                                   multiplicity.empty() ? 1u: multiplicity.at(mech_index),
                                   instances[*opt_i].loc};
}

template <typename B>
void resolve_probe(const cable_probe_point_state& p, probe_resolution_data<B>& R) {
    arb_assert(R.handles.size()==R.M.target_divs.back());
    arb_assert(R.handles.size()==R.M.n_target);

    const auto& mech   = p.mechanism;
    const auto& state  = p.state;
    const auto& target = p.target;
    const auto& data   = R.mechanism_state(mech, state);
    if (!R.mech_instance_by_name.count(mech)) return;
    const auto  mech_id = R.mech_instance_by_name.at(mech)->mechanism_id();
    const auto& synapses = R.cell.synapses();
    if (!synapses.count(mech)) return;
    if (!data) return;

    // Convert cell-local target number to cellgroup target number.
    const auto& divs = R.M.target_divs;
    auto cell = R.cell_idx;
    auto cg  = target + divs.at(cell);
    if (cg >= divs.at(cell + 1)) return;

    const auto& handle = R.handles.at(cg);
    if (handle.mech_id != mech_id) return;
    auto mech_index = handle.mech_index;
    R.result.push_back(fvm_probe_scalar{{data + mech_index},
                       point_info_of(target,
                                     mech_index,
                                     synapses.at(mech),
                                     R.M.mechanisms.at(mech).multiplicity)});
}

template <typename B>
void resolve_probe(const cable_probe_point_state_cell& p, probe_resolution_data<B>& R) {
    const auto& mech  = p.mechanism;
    const auto& state = p.state;
    const auto& data  = R.mechanism_state(mech, state);

    if (!data) return;
    if (!R.mech_instance_by_name.count(mech)) return;
    auto mech_id = R.mech_instance_by_name.at(mech)->mechanism_id();
    const auto& multiplicity = R.M.mechanisms.at(mech).multiplicity;

    const auto& synapses = R.cell.synapses();
    if (!synapses.count(mech)) return;
    const auto& placed_instances = synapses.at(mech);

    auto cell_targets_beg = R.M.target_divs.at(R.cell_idx);
    auto cell_targets_end = R.M.target_divs.at(R.cell_idx + 1);

    fvm_probe_multi r;
    std::vector<cable_probe_point_info> metadata;

    for (auto target: util::make_span(cell_targets_beg, cell_targets_end)) {
        const auto& handle = R.handles.at(target);
        if (handle.mech_id != mech_id) continue;

        auto mech_index = handle.mech_index;
        r.raw_handles.push_back(data + mech_index);

        metadata.push_back(point_info_of(target - cell_targets_beg, // Convert to cell-local target index.
                                         mech_index,
                                         placed_instances,
                                         multiplicity));
    }

    r.metadata = std::move(metadata);
    r.shrink_to_fit();
    R.result.push_back(std::move(r));
}

template <typename B>
void resolve_probe(const cable_probe_ion_current_density& p, probe_resolution_data<B>& R) {
    for (mlocation loc: thingify(p.locations, R.cell.provider())) {
        auto opt_i = R.ion_location_index(p.ion, loc);
        if (!opt_i) continue;

        R.result.push_back(fvm_probe_scalar{{R.state->ion_data.at(p.ion).iX_.data()+*opt_i}, loc});
    }
}

template <typename B>
void resolve_probe(const cable_probe_ion_current_cell& p, probe_resolution_data<B>& R) {
    if (!R.state->ion_data.count(p.ion)) return;
    auto& ion_cvs = R.M.ions.at(p.ion).cv;
    const arb_value_type* src = R.state->ion_data.at(p.ion).iX_.data();

    fvm_probe_weighted_multi r;
    for (auto cv: R.D.geometry.cell_cvs(R.cell_idx)) {
        auto opt_i = util::binary_search_index(ion_cvs, cv);
        if (!opt_i) continue;

        const double* ptr = src+*opt_i;
        for (auto cable: R.D.geometry.cables(cv)) {
            double area = R.cell.embedding().integrate_area(cable); // [µm²]
            if (area>0) {
                r.raw_handles.push_back(ptr);
                r.weight.push_back(0.001*area); // Scale from [µm²·A/m²] to [nA].
                r.metadata.push_back(cable);
            }
        }
    }
    r.metadata.shrink_to_fit();
    R.result.push_back(std::move(r));
}

template <typename B>
void resolve_probe(const cable_probe_ion_int_concentration& p, probe_resolution_data<B>& R) {
    for (mlocation loc: thingify(p.locations, R.cell.provider())) {
        auto opt_i = R.ion_location_index(p.ion, loc);
        if (!opt_i) continue;

        R.result.push_back(fvm_probe_scalar{{R.state->ion_data.at(p.ion).Xi_.data()+*opt_i}, loc});
    }
}

template <typename B>
void resolve_probe(const cable_probe_ion_ext_concentration& p, probe_resolution_data<B>& R) {
    for (mlocation loc: thingify(p.locations, R.cell.provider())) {
        auto opt_i = R.ion_location_index(p.ion, loc);
        if (!opt_i) continue;

        R.result.push_back(fvm_probe_scalar{{R.state->ion_data.at(p.ion).Xo_.data()+*opt_i}, loc});
    }
}

template <typename B>
void resolve_probe(const cable_probe_ion_diff_concentration& p, probe_resolution_data<B>& R) {
    for (mlocation loc: thingify(p.locations, R.cell.provider())) {
        auto opt_i = R.ion_location_index(p.ion, loc);
        if (!opt_i) continue;

        R.result.push_back(fvm_probe_scalar{{R.state->ion_data.at(p.ion).Xd_.data()+*opt_i}, loc});
    }
}

// Common implementation for int and ext concentrations across whole cell:
template <typename B>
void resolve_ion_conc_common(const std::vector<arb_index_type>& ion_cvs, const arb_value_type* src, probe_resolution_data<B>& R) {
    fvm_probe_multi r;
    mcable_list cables;

    for (auto i: util::count_along(ion_cvs)) {
        for (auto cable: R.D.geometry.cables(ion_cvs[i])) {
            if (cable.prox_pos!=cable.dist_pos) {
                r.raw_handles.push_back(src+i);
                cables.push_back(cable);
            }
        }
    }
    r.metadata = std::move(cables);
    r.shrink_to_fit();
    R.result.push_back(std::move(r));
}

template <typename B>
void resolve_probe(const cable_probe_ion_int_concentration_cell& p, probe_resolution_data<B>& R) {
    if (!R.state->ion_data.count(p.ion)) return;
    resolve_ion_conc_common<B>(R.M.ions.at(p.ion).cv, R.state->ion_data.at(p.ion).Xi_.data(), R);
}

template <typename B>
void resolve_probe(const cable_probe_ion_ext_concentration_cell& p, probe_resolution_data<B>& R) {
    if (!R.state->ion_data.count(p.ion)) return;
    resolve_ion_conc_common<B>(R.M.ions.at(p.ion).cv, R.state->ion_data.at(p.ion).Xo_.data(), R);
}

template <typename B>
void resolve_probe(const cable_probe_ion_diff_concentration_cell& p, probe_resolution_data<B>& R) {
    if (!R.state->ion_data.count(p.ion)) return;
    resolve_ion_conc_common<B>(R.M.ions.at(p.ion).cv, R.state->ion_data.at(p.ion).Xd_.data(), R);
}

} // namespace arb
