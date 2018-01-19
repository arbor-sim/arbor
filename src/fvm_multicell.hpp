#pragma once

#include <algorithm>
#include <iterator>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <algorithms.hpp>
#include <backends/event.hpp>
#include <backends/fvm_types.hpp>
#include <cell.hpp>
#include <compartment.hpp>
#include <constants.hpp>
#include <event_queue.hpp>
#include <ion.hpp>
#include <math.hpp>
#include <matrix.hpp>
#include <memory/memory.hpp>
#include <profiling/profiler.hpp>
#include <recipe.hpp>
#include <sampler_map.hpp>
#include <segment.hpp>
#include <stimulus.hpp>
#include <util/meta.hpp>
#include <util/partition.hpp>
#include <util/rangeutil.hpp>
#include <util/span.hpp>

namespace arb {
namespace fvm {

inline int find_cv_index(const segment_location& loc, const compartment_model& graph) {
    const auto& si = graph.segment_index;
    const auto seg = loc.segment;

    auto first = si[seg];
    auto n = si[seg+1] - first;

    int index = static_cast<int>(n*loc.position+0.5);
    index = index==0? graph.parent_index[first]: first+(index-1);

    return index;
};

template<class Backend>
class fvm_multicell {
public:
    using backend = Backend;

    /// the real number type
    using value_type = fvm_value_type;

    /// the integral index type
    using size_type = fvm_size_type;

    /// the container used for values
    using array = typename backend::array;
    using host_array = typename backend::host_array;

    /// the container used for indexes
    using iarray = typename backend::iarray;

    using view = typename array::view_type;
    using const_view = typename array::const_view_type;

    /// the type (view or copy) for a const host-side view of an array
    using host_view = decltype(memory::on_host(std::declval<array>()));

    // handles and events are currently common across implementations;
    // re-expose definitions from `backends/event.hpp`.
    using target_handle = ::arb::target_handle;
    using probe_handle = ::arb::probe_handle;
    using deliverable_event = ::arb::deliverable_event;

    fvm_multicell() = default;

    void resting_potential(value_type potential_mV) {
        resting_potential_ = potential_mV;
    }

    // Set up data structures for a fixed collection of cells identified by `gids`
    // with descriptions taken from the recipe `rec`.
    //
    // Lowered-cell specific handles for targets and probes are stored in the
    // caller-provided vector `target_handles` and map `probe_map`.
    void initialize(
        const std::vector<cell_gid_type>& gids,
        const recipe& rec,
        std::vector<target_handle>& target_handles,
        probe_association_map<probe_handle>& probe_map);

    void reset();

    // fvm_multicell::deliver_event is used only for testing.
    void deliver_event(target_handle h, value_type weight) {
        mechanisms_[h.mech_id]->net_receive(h.mech_index, weight);
    }

    // fvm_multicell::probe is used only for testing.
    value_type probe(probe_handle h) const {
        return backend::dereference(h); // h is a pointer, but might be device-side.
    }

    // Initialize state prior to a sequence of integration steps.
    // `staged_events` and `staged_samples` are expected to be
    // sorted by event time.
    void setup_integration(
        value_type tfinal, value_type dt_max,
        const std::vector<deliverable_event>& staged_events,
        const std::vector<sample_event>& staged_samples)
    {
        EXPECTS(dt_max>0);

        tfinal_ = tfinal;
        dt_max_ = dt_max;

        compute_min_remaining();

        EXPECTS(!has_pending_events());

        n_samples_ = staged_samples.size();

        events_.init(staged_events);
        sample_events_.init(staged_samples);

        // Reallocate sample buffers if necessary.
        if (sample_value_.size()<n_samples_) {
            sample_value_ = array(n_samples_);
            sample_time_ = array(n_samples_);
        }
    }

    // Advance one integration step.
    void step_integration();

    // Query integration completion state.
    bool integration_complete() const {
        return min_remaining_steps_==0;
    }

    // Access to sample data post-integration.
    decltype(memory::make_const_view(std::declval<host_view>())) sample_value() const {
        EXPECTS(sample_events_.empty());
        host_sample_value_ = memory::on_host(sample_value_);
        return host_sample_value_;
    }

    decltype(memory::make_const_view(std::declval<host_view>())) sample_time() const {
        EXPECTS(sample_events_.empty());
        host_sample_time_ = memory::on_host(sample_time_);
        return host_sample_time_;
    }

    // Query per-cell time state.
    // Placeholder: external time queries will no longer be required when
    // complete integration loop is in lowered cell.
    value_type time(size_type cell_idx) const {
        refresh_time_cache();
        return cached_time_[cell_idx];
    }

    value_type min_time() const {
        return backend::minmax_value(time_).first;
    }

    value_type max_time() const {
        return backend::minmax_value(time_).second;
    }

    bool state_synchronized() const {
        auto mm = backend::minmax_value(time_);
        return mm.first==mm.second;
    }

    /// Set times for all cells (public for testing purposes only).
    void set_time_global(value_type t) {
        memory::fill(time_, t);
        invalidate_time_cache();
    }

    void set_time_to_global(value_type t) {
        memory::fill(time_to_, t);
        invalidate_time_cache();
    }

    /// Following types and methods are public only for testing:

    /// the type used to store matrix information
    using matrix_type = matrix<backend>;

    /// mechanism type
    using mechanism = typename backend::mechanism;
    using mechanism_ptr = typename backend::mechanism_ptr;

    /// stimulus type
    using stimulus = typename backend::stimulus;

    /// ion species storage
    using ion_type = typename backend::ion_type;

    /// view into index container
    using iview = typename backend::iview;
    using const_iview = typename backend::const_iview;

    const matrix_type& jacobian() { return matrix_; }

    /// return list of CV areas in :
    ///          um^2
    ///     1e-6.mm^2
    ///     1e-8.cm^2
    const_view cv_areas() const { return cv_areas_; }

    /// return the voltage in each CV
    view       voltage()       { return voltage_; }
    const_view voltage() const { return voltage_; }

    /// return the current density in each CV: A.m^-2
    view       current()       { return current_; }
    const_view current() const { return current_; }

    std::size_t size() const { return matrix_.size(); }

    /// return reference to in iterable container of the mechanisms
    std::vector<mechanism_ptr>& mechanisms() { return mechanisms_; }

    /// return reference to list of ions
    std::map<ionKind, ion_type>&       ions()       { return ions_; }
    std::map<ionKind, ion_type> const& ions() const { return ions_; }

    /// return reference to sodium ion
    ion_type&       ion_na()       { return ions_[ionKind::na]; }
    ion_type const& ion_na() const { return ions_[ionKind::na]; }

    /// return reference to calcium ion
    ion_type&       ion_ca()       { return ions_[ionKind::ca]; }
    ion_type const& ion_ca() const { return ions_[ionKind::ca]; }

    /// return reference to pottasium ion
    ion_type&       ion_k()       { return ions_[ionKind::k]; }
    ion_type const& ion_k() const { return ions_[ionKind::k]; }

    /// flags if solution is physically realistic.
    /// here we define physically realistic as the voltage being within reasonable bounds.
    /// use a simple test of the voltage at the soma is reasonable, i.e. in the range
    ///     v_soma \in (-1000mv, 1000mv)
    bool is_physical_solution() const {
        auto v = voltage_[0];
        return (v>-1000.) && (v<1000.);
    }

    /// Return reference to the mechanism that matches name.
    /// The reference is const, because this information should not be
    /// modified by the caller, however it is needed for unit testing.
    util::optional<const mechanism_ptr&> find_mechanism(const std::string& name) const {
        auto it = std::find_if(
            std::begin(mechanisms_), std::end(mechanisms_),
            [&name](const mechanism_ptr& m) {return m->name()==name;});
        return it==mechanisms_.end() ? util::nullopt: util::just(*it);
    }

    //
    // Threshold crossing interface.
    // Used by calling code to perform spike detection
    //

    /// types defined by the back end for threshold detection
    using threshold_watcher = typename backend::threshold_watcher;
    using crossing_list     = typename backend::threshold_watcher::crossing_list;

    /// Forward the list of threshold crossings from the back end.
    /// The list is passed by value, because we don't want the calling code
    /// to depend on references to internal state of the solver, and because
    /// for some backends the results might have to be collated before returning.
    crossing_list get_spikes() const {
       return threshold_watcher_.crossings();
    }

    /// clear all spikes: aka threshold crossings.
    void clear_spikes() {
       threshold_watcher_.clear_crossings();
    }

private:
    /// number of distinct cells (integration domains)
    size_type ncell_;

    threshold_watcher threshold_watcher_;

    /// resting potential (initial voltage condition)
    value_type resting_potential_ = -65;

    /// final time in integration round [ms]
    value_type tfinal_ = 0;

    /// max time step for integration [ms]
    value_type dt_max_ = 0;

    /// minimum number of integration steps left in integration period.
    // zero => integration complete.
    unsigned min_remaining_steps_ = 0;

    void compute_min_remaining() {
        auto tmin = min_time();
        min_remaining_steps_ = tmin>=tfinal_? 0: 1 + (unsigned)((tfinal_-tmin)/dt_max_);
    }

    void decrement_min_remaining() {
        EXPECTS(min_remaining_steps_>0);
        if (!--min_remaining_steps_) {
            compute_min_remaining();
        }
    }

    /// event queue for integration period
    using deliverable_event_stream = typename backend::deliverable_event_stream;
    deliverable_event_stream events_;

    bool has_pending_events() const {
        return !events_.empty();
    }

    /// sample events for integration period
    using sample_event_stream = typename backend::sample_event_stream;
    sample_event_stream sample_events_;

    /// sample buffers
    size_type n_samples_ = 0;
    array sample_value_;
    array sample_time_;

    mutable host_view host_sample_value_; // host-side views/copies of sample data
    mutable host_view host_sample_time_;

    /// the linear system for implicit time stepping of cell state
    matrix_type matrix_;

    /// cv_areas_[i] is the surface area of CV i [µm^2]
    array cv_areas_;

    /// the map from compartment index to cell index
    iarray cv_to_cell_;

    /// the per-cell simulation time
    array time_;

    /// the per-cell integration period end point
    array time_to_;

    // the per-compartment dt
    // (set to dt_cell_[j] for each compartment in cell j).
    array dt_comp_;

    // the per-cell dt
    // (set to time_to_[j]-time_[j] for each cell j).
    array dt_cell_;

    // Maintain cached copy of time vector for querying by
    // cell_group. This will no longer be necessary when full
    // integration loop is in lowered cell.
    mutable std::vector<value_type> cached_time_;
    mutable bool cached_time_valid_ = false;

    void invalidate_time_cache() { cached_time_valid_ = false; }
    void refresh_time_cache() const {
        if (!cached_time_valid_) {
            memory::copy(time_, memory::make_view(cached_time_));
        }
        cached_time_valid_ = true;
    }

    /// the transmembrane current density over the surface of each CV [A.m^-2]
    ///     I = i_m - I_e/area
    array current_;

    /// the potential in each CV [mV]
    array voltage_;

    /// the set of mechanisms present in the cell
    std::vector<mechanism_ptr> mechanisms_;

    /// the ion species
    std::map<ionKind, ion_type> ions_;

    /// Compact representation of the control volumes into which a segment is
    /// decomposed. Used to reconstruct the weights used to convert current
    /// densities to currents for density channels.
    struct segment_cv_range {
        // the contribution to the surface area of the CVs that
        // are at the beginning and end of the segment
        std::pair<value_type, value_type> areas;

        // the range of CVs in the segment, excluding the parent CV
        std::pair<size_type, size_type> segment_cvs;

        // The last CV in the parent segment, which corresponds to the
        // first CV in this segment.
        // Set to npos() if there is no parent (i.e. if soma)
        size_type parent_cv;

        static constexpr size_type npos() {
            return std::numeric_limits<size_type>::max();
        }

        // the number of CVs (including the parent)
        std::size_t size() const {
            return segment_cvs.second-segment_cvs.first + (parent_cv==npos() ? 0 : 1);
        }

        bool has_parent() const {
            return parent_cv != npos();
        }
    };

    // perform area and capacitance calculation on initialization
    segment_cv_range compute_cv_area_capacitance(
        std::pair<size_type, size_type> comp_ival,
        const segment* seg,
        const std::vector<size_type>& parent,
        std::vector<value_type>& face_conductance,
        std::vector<value_type>& tmp_cv_areas,
        std::vector<value_type>& cv_capacitance
    );

    // TODO: This process should be simpler when we can deal with mechanism prototypes and have
    // separate initialization.
    //
    // Create possibly-specialized mechanism and add to mechanism set.
    // Weights are unset, and should be set specifically with mechanism::set_weights().
    mechanism& make_mechanism(
        const std::string& name,
        const std::map<std::string, specialized_mechanism>& special_mechs,
        const std::vector<size_type>& node_indices)
    {
        std::string impl_name = name;
        std::vector<std::pair<std::string, double>> global_params;

        if (special_mechs.count(name)) {
            const auto& spec_mech = special_mechs.at(name);
            impl_name = spec_mech.mech_name;
            global_params = spec_mech.parameters;
        }

        size_type mech_id = mechanisms_.size();
        auto m = backend::make_mechanism(impl_name, mech_id, cv_to_cell_, time_, time_to_, dt_comp_, voltage_, current_, {}, node_indices);
        if (impl_name!=name) {
            m->set_alias(name);
        }

        for (const auto& pv: global_params) {
            auto field = m->field_value_ptr(pv.first);
            if (!field) {
                throw std::invalid_argument("no scalar parameter "+pv.first+" in mechanism "+m->name());
            }
            m.get()->*field = pv.second;
        }

        mechanisms_.push_back(std::move(m));
        return *mechanisms_.back();
    }

    // Throwing-wrapper around mechanism (range) parameter look up.
    static view mech_field(mechanism& m, const std::string& param_name) {
        auto p = m.field_view_ptr(param_name);
        if (!p) {
            throw std::invalid_argument("no parameter "+param_name+" in mechanism "+m.name());
        }
        return m.*p;
    }
};

////////////////////////////////////////////////////////////////////////////////
//////////////////////////////// Implementation ////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
template <typename Backend>
typename fvm_multicell<Backend>::segment_cv_range
fvm_multicell<Backend>::compute_cv_area_capacitance(
    std::pair<size_type, size_type> comp_ival,
    const segment* seg,
    const std::vector<size_type>& parent,
    std::vector<value_type>& face_conductance,
    std::vector<value_type>& tmp_cv_areas,
    std::vector<value_type>& cv_capacitance)
{
    // precondition: group_parent_index[j] holds the correct value for
    // j in [base_comp, base_comp+segment.num_compartments()].

    auto ncomp = comp_ival.second-comp_ival.first;

    segment_cv_range cv_range;

    auto cm = seg->cm;
    auto rL = seg->rL;

    if (auto soma = seg->as_soma()) {
        // confirm assumption that there is one compartment in soma
        if (ncomp!=1) {
            throw std::logic_error("soma allocated more than one compartment");
        }
        auto i = comp_ival.first;
        auto area = math::area_sphere(soma->radius());

        tmp_cv_areas[i] += area;
        cv_capacitance[i] += area*cm;

        cv_range.segment_cvs = {comp_ival.first, comp_ival.first+1};
        cv_range.areas = {0.0, area};
        cv_range.parent_cv = segment_cv_range::npos();
    }
    else if (auto cable = seg->as_cable()) {
        // Loop over each compartment in the cable
        //
        // Each compartment i straddles the ith control volume on the right
        // and the jth control volume on the left, where j is the parent index
        // of i.
        //
        // Dividing the comparment into two halves, the centre face C
        // corresponds to the shared face between the two control volumes,
        // the surface areas in each half contribute to the surface area of
        // the respective control volumes, and the volumes and lengths of
        // each half are used to calculate the flux coefficients that
        // for the connection between the two control volumes and which
        // is stored in `face_conductance[i]`.
        //
        //
        //  +------- cv j --------+------- cv i -------+
        //  |                     |                    |
        //  v                     v                    v
        //  ____________________________________________
        //  | ........ | ........ |          |         |
        //  | ........ L ........ C          R         |
        //  |__________|__________|__________|_________|
        //             ^                     ^
        //             |                     |
        //             +--- compartment i ---+
        //
        // The first control volume of any cell corresponds to the soma
        // and the first half of the first cable compartment of that cell.

        auto divs = div_compartments<div_compartment_integrator>(cable, ncomp);

        // assume that this segment has a parent, which is the case so long
        // as the soma is the root of all cell trees.
        cv_range.parent_cv = parent[comp_ival.first];
        cv_range.segment_cvs = comp_ival;
        cv_range.areas = {divs(0).left.area, divs(ncomp-1).right.area};

        for (auto i: util::make_span(comp_ival)) {
            const auto& div = divs(i-comp_ival.first);
            auto j = parent[i];

            // Conductance approximated by weighted harmonic mean of mean
            // conductances in each half.
            //
            // Mean conductances:
            // g₁ = 1/h₁ ∫₁ A(x)/R dx
            // g₂ = 1/h₂ ∫₂ A(x)/R dx
            //
            // where A(x) is the cross-sectional area, R is the bulk
            // resistivity, h is the length of the interval and the
            // integrals are taken over the intervals respectively.
            // Equivalently, in terms of the semi-compartment volumes
            // V₁ and V₂:
            //
            // g₁ = 1/R·V₁/h₁
            // g₂ = 1/R·V₂/h₂
            //
            // Weighted harmonic mean, with h = h₁+h₂:
            //
            // g = (h₁/h·g₁¯¹+h₂/h·g₂¯¹)¯¹
            //   = 1/R · hV₁V₂/(h₂²V₁+h₁²V₂)
            //
            // the following units are used
            //  lengths : μm
            //  areas   : μm^2
            //  volumes : μm^3

            auto h1 = div.left.length;
            auto V1 = div.left.volume;
            auto h2 = div.right.length;
            auto V2 = div.right.volume;
            auto h = h1+h2;

            auto conductance = 1/rL*h*V1*V2/(h2*h2*V1+h1*h1*V2);
            // the scaling factor of 10^2 is to convert the quantity
            // to micro Siemens [μS]
            face_conductance[i] =  1e2 * conductance / h;

            auto al = div.left.area;
            auto ar = div.right.area;

            tmp_cv_areas[j] += al;
            tmp_cv_areas[i] += ar;
            cv_capacitance[j] += al * cm;
            cv_capacitance[i] += ar * cm;
        }
    }
    else {
        throw std::domain_error("FVM lowering encountered unsuported segment type");
    }

    return cv_range;
}

template <typename Backend>
void fvm_multicell<Backend>::initialize(
    const std::vector<cell_gid_type>& gids,
    const recipe& rec,
    std::vector<target_handle>& target_handles,
    probe_association_map<probe_handle>& probe_map)
{
    using memory::make_const_view;
    using util::any_cast;
    using util::assign_by;
    using util::make_partition;
    using util::make_span;
    using util::size;
    using util::sort_by;
    using util::transform_view;
    using util::subrange_view;

    ncell_ = size(gids);
    std::size_t targets_count = 0u;

    // Handle any global parameters for these cell groups.
    // (Currently: just specialized mechanisms).
    std::map<std::string, specialized_mechanism> special_mechs;
    util::any gprops = rec.get_global_properties(cable1d_neuron);
    if (gprops.has_value()) {
        special_mechs = util::any_cast<cell_global_properties&>(gprops).special_mechs;
    }

    // Take cell descriptions from recipe. These are used initially
    // to count compartments for allocation of data structures, and
    // then interrogated again for the details for each cell in turn.

    std::vector<cell> cells;
    cells.reserve(gids.size());
    for (auto gid: gids) {
        cells.push_back(std::move(any_cast<cell>(rec.get_cell_description(gid))));
    }

    auto cell_num_compartments =
        transform_view(cells, [](const cell& c) { return c.num_compartments(); });

    std::vector<cell_lid_type> cell_comp_bounds;
    auto cell_comp_part = make_partition(cell_comp_bounds, cell_num_compartments);
    auto ncomp = cell_comp_part.bounds().second;

    // initialize storage from total compartment count
    current_ = array(ncomp, 0);
    voltage_ = array(ncomp, resting_potential_);
    cv_to_cell_ = iarray(ncomp, 0);
    time_ = array(ncell_, 0);
    time_to_ = array(ncell_, 0);
    cached_time_.resize(ncell_);
    cached_time_valid_ = false;
    dt_cell_ = array(ncell_, 0);
    dt_comp_ = array(ncomp, 0);

    // initialize cv_to_cell_ values from compartment partition
    std::vector<size_type> cv_to_cell_tmp(ncomp);
    for (size_type i = 0; i<ncell_; ++i) {
        util::fill(util::subrange_view(cv_to_cell_tmp, cell_comp_part[i]), i);
    }
    memory::copy(cv_to_cell_tmp, cv_to_cell_);

    // TODO: mechanism parameters are currently indexed by string keys; more efficient
    // to use the mechanism member pointers, when these become easily accessible via
    // the mechanism catalogue interface.

    // look up table: mechanism name -> list of cv_range objects and parameter settings.

    struct mech_area_contrib {
        size_type index;
        value_type area;
    };

    struct mech_info {
        segment_cv_range cv_range;
        // Note: owing to linearity constraints, the only parameters for which it is
        // sensible to modify are those which make a linear contribution to currents
        // (or ion fluxes, etc.)
        std::map<std::string, value_type> param_map;
        std::vector<mech_area_contrib> contributions;
    };

    std::map<std::string, std::vector<mech_info>> mech_map;

    // look up table: point mechanism (synapse) name -> list of CV indices, target numbers, parameters.
    struct syn_info {
        cell_lid_type cv;
        cell_lid_type target;
        std::map<std::string, value_type> param_map;
    };

    std::map<std::string, std::vector<syn_info>> syn_mech_map;

    // initialize vector used for matrix creation.
    std::vector<size_type> group_parent_index(ncomp);

    // setup per-cell event stores.
    events_ = deliverable_event_stream(ncell_);
    sample_events_ = sample_event_stream(ncell_);

    // Create each cell:

    // Allocate scratch storage for calculating quantities used to build the
    // linear system: these will later be copied into target-specific storage.

    // face_conductance_[i] = area_face  / (rL * delta_x);
    std::vector<value_type> face_conductance(ncomp); // [µS]
    /// cv_capacitance_[i] is the capacitance of CV membrane
    std::vector<value_type> cv_capacitance(ncomp);   // [µm^2*F*m^-2 = pF]
    /// membrane area of each cv
    std::vector<value_type> tmp_cv_areas(ncomp);     // [µm^2]

    // used to build the information required to construct spike detectors
    std::vector<size_type> spike_detector_index;
    std::vector<value_type> thresholds;

    // Iterate over the input cells and build the indexes etc that descrbe the
    // fused cell group. On completion:
    //  - group_paranet_index contains the full parent index for the fused cells.
    //  - mech_to_cv_range and syn_mech_map provide a map from mechanism names to an
    //    iterable container of compartment ranges, which are used later to
    //    generate the node index for each mechanism kind.
    //  - the tmp_* vectors contain compartment-specific information for each
    //    compartment in the fused cell group (areas, capacitance, etc).
    //  - each probe, stimulus and detector is attached to its compartment.
    for (auto i: make_span(0, ncell_)) {
        const auto& c = cells[i];
        auto gid = gids[i];
        auto comp_ival = cell_comp_part[i];

        auto graph = c.model();

        for (auto k: make_span(comp_ival)) {
            group_parent_index[k] = graph.parent_index[k-comp_ival.first]+comp_ival.first;
        }

        auto seg_num_compartments =
            transform_view(c.segments(), [](const segment_ptr& s) { return s->num_compartments(); });
        const auto nseg = seg_num_compartments.size();

        std::vector<cell_lid_type> seg_comp_bounds;
        auto seg_comp_part =
            make_partition(seg_comp_bounds, seg_num_compartments, comp_ival.first);

        for (size_type j = 0; j<nseg; ++j) {
            const auto& seg = c.segment(j);
            const auto& seg_comp_ival = seg_comp_part[j];

            auto cv_range = compute_cv_area_capacitance(
                seg_comp_ival, seg, group_parent_index,
                face_conductance, tmp_cv_areas, cv_capacitance);

            for (const auto& mech: seg->mechanisms()) {
                mech_map[mech.name()].push_back({cv_range, mech.values()});
            }
        }

        for (const auto& syn: c.synapses()) {
            const auto& name = syn.mechanism.name();

            cell_lid_type syn_cv = comp_ival.first + find_cv_index(syn.location, graph);
            cell_lid_type target_index = targets_count++;

            syn_mech_map[name].push_back({syn_cv, target_index, syn.mechanism.values()});
        }

        //
        // add the stimuli
        //

        // TODO: use same process as for synapses!
        // step 1: pack the index and parameter information into flat vectors
        std::vector<size_type> stim_index;
        std::vector<value_type> stim_durations;
        std::vector<value_type> stim_delays;
        std::vector<value_type> stim_amplitudes;
        std::vector<value_type> stim_weights;
        for (const auto& stim: c.stimuli()) {
            auto idx = comp_ival.first+find_cv_index(stim.location, graph);
            stim_index.push_back(idx);
            stim_durations.push_back(stim.clamp.duration());
            stim_delays.push_back(stim.clamp.delay());
            stim_amplitudes.push_back(stim.clamp.amplitude());
            stim_weights.push_back(1e3/tmp_cv_areas[idx]);
        }

        // step 2: create the stimulus mechanism and initialize the stimulus
        //         parameters
        // NOTE: the indexes and associated metadata (durations, delays,
        //       amplitudes) have not been permuted to ascending cv index order,
        //       as is the case with other point processes.
        //       This is because the hard-coded stimulus mechanism makes no
        //       optimizations that rely on this assumption.
        if (stim_index.size()) {
            auto stim = new stimulus(
                cv_to_cell_, time_, time_to_, dt_comp_,
                voltage_, current_, memory::make_const_view(stim_index));
            stim->set_parameters(stim_amplitudes, stim_durations, stim_delays);
            stim->set_weights(memory::make_const_view(stim_weights));
            mechanisms_.push_back(mechanism_ptr(stim));
        }

        // calculate spike detector handles are their corresponding compartment indices
        for (const auto& detector: c.detectors()) {
            auto comp = comp_ival.first+find_cv_index(detector.location, graph);
            spike_detector_index.push_back(comp);
            thresholds.push_back(detector.threshold);
        }

        // Retrieve probe addresses and tags from recipe for this cell.
        for (cell_lid_type j: make_span(0, rec.num_probes(gid))) {
            probe_info pi = rec.get_probe({gid, j});
            auto where = any_cast<cell_probe_address>(pi.address);

            auto comp = comp_ival.first+find_cv_index(where.location, graph);
            probe_handle handle;

            switch (where.kind) {
            case cell_probe_address::membrane_voltage:
                handle = fvm_multicell::voltage_.data()+comp;
                break;
            case cell_probe_address::membrane_current:
                handle = fvm_multicell::current_.data()+comp;
                break;
            default:
                throw std::logic_error("unrecognized probeKind");
            }

            probe_map.insert({pi.id, {handle, pi.tag}});
        }
    }

    // set a back-end supplied watcher on the voltage vector
    threshold_watcher_ =
        threshold_watcher(cv_to_cell_, time_, time_to_, voltage_, spike_detector_index, thresholds);

    // store the geometric information in target-specific containers
    cv_areas_ = make_const_view(tmp_cv_areas);

    // initalize matrix
    matrix_ = matrix_type(
        group_parent_index, cell_comp_bounds, cv_capacitance, face_conductance, tmp_cv_areas);

    // Keep cv index list for each mechanism for ion set up below.
    std::map<std::string, std::vector<size_type>> mech_to_cv_index;
    // Keep area of each cv occupied by each mechanism, which may be less than
    // the total area of the cv.
    std::map<std::string, std::vector<value_type>> mech_to_area;

    // Working vectors (re-used per mechanism).
    std::vector<size_type> mech_cv(ncomp);
    std::vector<value_type> mech_weight(ncomp);

    for (auto& entry: mech_map) {
        const auto& mech_name = entry.first;
        auto& segments = entry.second;

        mech_cv.clear();
        mech_weight.clear();

        // Three passes are performed over the segment list:
        //   1. Compute the CVs and area contributions where the mechanism is instanced.
        //   2. Build table of modified parameters together with default values.
        //   3. Compute weights and parameters.
        // The mechanism is instantiated after the first pass, in order to gain
        // access to default mechanism parameter values.

        for (auto& seg: segments) {
            const auto& rng = seg.cv_range;
            seg.contributions.reserve(rng.size());

            if (rng.has_parent()) {
                auto cv = rng.parent_cv;

                auto it = algorithms::binary_find(mech_cv, cv);
                size_type pos = it - mech_cv.begin();

                if (it == mech_cv.end()) {
                    mech_cv.push_back(cv);
                }

                seg.contributions.push_back({pos, rng.areas.first});
            }

            for (auto cv: make_span(rng.segment_cvs)) {
                size_type pos = mech_cv.size();
                mech_cv.push_back(cv);
                seg.contributions.push_back({pos, tmp_cv_areas[cv]});
            }

            // Last CV contribution may be only partial, so adjust.
            seg.contributions.back().area = rng.areas.second;
        }

        auto nindex = mech_cv.size();

        EXPECTS(std::is_sorted(mech_cv.begin(), mech_cv.end()));
        EXPECTS(nindex>0);

        auto& mech = make_mechanism(mech_name, special_mechs, mech_cv);

        // Save the indices for ion set up below.

        mech_to_cv_index[mech_name] = mech_cv;

        // Build modified (non-global) parameter table.

        struct param_tbl_entry {
            std::vector<value_type> values; // staged for writing to mechanism
            view data;                      // view to corresponding data in mechanism
            value_type dflt;                // default value for parameter
        };

        std::map<std::string, param_tbl_entry> param_tbl;

        for (const auto& seg: segments) {
            for (const auto& pv: seg.param_map) {
                if (param_tbl.count(pv.first)) {
                    continue;
                }

                // Grab default value from mechanism data.
                auto& entry = param_tbl[pv.first];
                entry.data = mech_field(mech, pv.first);
                entry.dflt = entry.data[0];
                entry.values.assign(nindex, 0.);
            }
        }

        // Perform another pass of segment list to compute weights and (non-global) parameters.

        mech_weight.assign(nindex, 0.);

        for (const auto& seg: segments) {
            for (auto cw: seg.contributions) {
                mech_weight[cw.index] += cw.area;

                for (auto& entry: param_tbl) {
                    value_type v = entry.second.dflt;
                    const auto& name = entry.first;

                    auto it = seg.param_map.find(name);
                    if (it != seg.param_map.end()) {
                        v = it->second;
                    }

                    entry.second.values[cw.index] += cw.area*v;
                }
            }
        }

        // Save the areas for ion setup below.
        mech_to_area[mech_name] = mech_weight;

        for (auto& entry: param_tbl) {
            for (size_type i = 0; i<nindex; ++i) {
                entry.second.values[i] /= mech_weight[i];
            }
            memory::copy(entry.second.values, entry.second.data);
        }

        // Scale the weights by the CV area to get the proportion of the CV surface
        // on which the mechanism is present. After scaling, the current will have
        // units A.m^-2.
        for (auto i: make_span(0, mech_weight.size())) {
            mech_weight[i] *= 10/tmp_cv_areas[mech_cv[i]];
        }
        mech.set_weights(memory::make_const_view(mech_weight));
    }

    target_handles.resize(targets_count);

    // Create point (synapse) mechanisms.
    for (auto& map_entry: syn_mech_map) {
        size_type mech_id = mechanisms_.size();

        const auto& mech_name = map_entry.first;
        auto& syn_data = map_entry.second;
        auto n_instance = syn_data.size();

        // Build permutation p such that p[j] is the index into
        // syn_data for the jth synapse of this mechanism type as ordered by cv index.

        auto cv_of = [&](cell_lid_type i) { return syn_data[i].cv; };

        std::vector<cell_lid_type> p(n_instance);
        std::iota(p.begin(), p.end(), 0u);
        util::sort_by(p, cv_of);

        std::vector<cell_lid_type> mech_cv;
        std::vector<value_type> mech_weight;
        mech_cv.reserve(n_instance);
        mech_weight.reserve(n_instance);

        // Build mechanism cv index vector, weights and targets.
        for (auto i: make_span(0u, n_instance)) {
            const auto& syn = syn_data[p[i]];
            mech_cv.push_back(syn.cv);
            // The weight for each synapses is 1/cv_area, scaled by 100 to match the units
            // of 10.A.m^-2 used to store current densities in current_.
            mech_weight.push_back(1e3/tmp_cv_areas[syn.cv]);
            target_handles[syn.target] = target_handle(mech_id, i, cv_to_cell_tmp[syn.cv]);
        }

        auto& mech = make_mechanism(mech_name, special_mechs, mech_cv);
        mech.set_weights(memory::make_const_view(mech_weight));

        // Save the indices for ion set up below.
        mech_to_cv_index[mech_name] = mech_cv;

        // Update the mechanism parameters.
        std::map<std::string, std::vector<std::pair<cell_lid_type, value_type>>> param_assigns;
        for (auto i: make_span(0u, n_instance)) {
            for (const auto& pv: syn_data[p[i]].param_map) {
                param_assigns[pv.first].push_back({i, pv.second});
            }
        }

        for (const auto& pa: param_assigns) {
            view field_data = mech_field(mech, pa.first);
            host_array field_values = field_data;
            for (const auto &iv: pa.second) {
                field_values[iv.first] = iv.second;
            }
            memory::copy(field_values, field_data);
        }
    }

    // build the ion species
    for (auto ion : ion_kinds()) {
        // find the compartment indexes of all compartments that have a
        // mechanism that depends on/influences ion
        std::set<size_type> index_set;
        for (auto const& mech : mechanisms_) {
            if(mech->uses_ion(ion).uses) {
                auto const& ni = mech_to_cv_index[mech->name()];
                index_set.insert(ni.begin(), ni.end());
            }
        }
        std::vector<size_type> indexes(index_set.begin(), index_set.end());
        const auto n = indexes.size();

        if (n==0u) continue;

        // create the ion state
        ions_[ion] = indexes;

        std::vector<value_type> w_int;
        w_int.reserve(n);
        for (auto i: indexes) {
            w_int.push_back(tmp_cv_areas[i]);
        }
        std::vector<value_type> w_out = w_int;

        // Join the ion reference in each mechanism into the cell-wide ion state.
        for (auto& mech : mechanisms_) {
            const auto spec = mech->uses_ion(ion);
            if (spec.uses) {
                const auto& ni = mech_to_cv_index[mech->name()];
                const auto m = ni.size(); // number of CVs
                const std::vector<size_type> sub_index =
                    util::assign_from(algorithms::index_into(ni, indexes));
                mech->set_ion(ion, ions_[ion], sub_index);

                const auto& ai = mech_to_area[mech->name()];
                if (spec.write_concentration_in) {
                    for (auto i: make_span(0, m)) {
                        w_int[sub_index[i]] -= ai[i];
                    }
                }
                if (spec.write_concentration_out) {
                    for (auto i: make_span(0, m)) {
                        w_out[sub_index[i]] -= ai[i];
                    }
                }
            }
        }
        // Normalise the weights.
        for (auto i: make_span(0, n)) {
            w_int[i] /= tmp_cv_areas[indexes[i]];
            w_out[i] /= tmp_cv_areas[indexes[i]];
        }
        ions_[ion].set_weights(w_int, w_out);
    }

    // Note: NEURON defined default values for reversal potential as follows,
    //       with units mV:
    //
    // const auto DEF_vrest = -65.0
    // ena = 115.0 + DEF_vrest
    // ek  = -12.0 + DEF_vrest
    // eca = 12.5*std::log(2.0/5e-5)
    //
    // Whereas we use the Nernst equation to calculate reversal potentials at
    // the start of each time step.

    ion_na().default_int_concentration = 10;
    ion_na().default_ext_concentration =140;
    ion_na().valency = 1;

    ion_k().default_int_concentration =54.4;
    ion_k().default_ext_concentration = 2.5;
    ion_k().valency = 1;

    ion_ca().default_int_concentration =5e-5;
    ion_ca().default_ext_concentration = 2.0;
    ion_ca().valency = 2;

    // initialize mechanism and voltage state
    reset();
}

template <typename Backend>
void fvm_multicell<Backend>::reset() {
    memory::fill(voltage_, resting_potential_);

    set_time_global(0);
    set_time_to_global(0);

    // Update ion species:
    //   - clear currents
    //   - reset concentrations to defaults
    //   - recalculate reversal potentials
    for (auto& i: ions_) {
        i.second.reset();
    }

    for (auto& m : mechanisms_) {
        m->set_params();
        m->nrn_init();
        m->write_back();
    }

    // Update reversal potential to account for changes to concentrations made
    // by calls to nrn_init() in mechansisms.
    for (auto& i: ions_) {
        i.second.nernst_reversal_potential(constant::hh_squid_temp); // TODO: use temperature specfied in model
    }

    // Reset state of the threshold watcher.
    // NOTE: this has to come after the voltage_ values have been reinitialized,
    // because these values are used by the watchers to set their initial state.
    threshold_watcher_.reset();

    // Reset integration state.
    tfinal_ = 0;
    dt_max_ = 0;
    min_remaining_steps_ = 0;
    events_.clear();
    sample_events_.clear();

    EXPECTS(integration_complete());
    EXPECTS(!has_pending_events());
}

template <typename Backend>
void fvm_multicell<Backend>::step_integration() {
    EXPECTS(!integration_complete());

    // mark pending events for delivery
    events_.mark_until_after(time_);

    PE("current");
    memory::fill(current_, 0.);

    // clear currents and recalculate reversal potentials for all ion channels
    for (auto& i: ions_) {
        auto& ion = i.second;
        memory::fill(ion.current(), 0.);
        ion.nernst_reversal_potential(constant::hh_squid_temp); // TODO: use temperature specfied in model
    }

    // deliver pending events and update current contributions from mechanisms
    for (auto& m: mechanisms_) {
        PE(m->name().c_str());
        m->deliver_events(events_.marked_events());
        m->nrn_current();
        PL();
    }

    // remove delivered events from queue and set time_to_
    events_.drop_marked_events();

    backend::update_time_to(time_to_, time_, dt_max_, tfinal_);
    invalidate_time_cache();
    events_.event_time_if_before(time_to_);
    PL();

    // set per-cell and per-compartment dt (constant within a cell)
    backend::set_dt(dt_cell_, dt_comp_, time_to_, time_, cv_to_cell_);

    // take samples if they lie within the integration step; they will be provided
    // with the values (post-event delivery) at the beginning of the interval.
    sample_events_.mark_until(time_to_);
    backend::take_samples(sample_events_.marked_events(), time_, sample_time_, sample_value_);
    sample_events_.drop_marked_events();

    // solve the linear system
    PE("matrix", "setup");
    matrix_.assemble(dt_cell_, voltage_, current_);

    PL(); PE("solve");
    matrix_.solve();
    PL();
    memory::copy(matrix_.solution(), voltage_);
    PL();

    // integrate state of gating variables etc.
    PE("state");
    for(auto& m: mechanisms_) {
        PE(m->name().c_str());
        m->nrn_state();
        PL();
    }
    PL();

    PE("ion-update");
    for(auto& i: ions_) {
        i.second.init_concentration();
    }
    for(auto& m: mechanisms_) {
        m->write_back();
    }
    PL();

    memory::copy(time_to_, time_);
    invalidate_time_cache();

    // update spike detector thresholds
    threshold_watcher_.test();

    // are we there yet?
    decrement_min_remaining();

    EXPECTS(!integration_complete() || !has_pending_events());
}

} // namespace fvm
} // namespace arb
