#pragma once

#include <algorithm>
#include <iterator>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <algorithms.hpp>
#include <backends/fvm.hpp>
#include <cell.hpp>
#include <compartment.hpp>
#include <event_queue.hpp>
#include <ion.hpp>
#include <math.hpp>
#include <matrix.hpp>
#include <memory/memory.hpp>
#include <profiling/profiler.hpp>
#include <segment.hpp>
#include <stimulus.hpp>
#include <util/debug.hpp>
#include <util/meta.hpp>
#include <util/partition.hpp>
#include <util/rangeutil.hpp>
#include <util/span.hpp>

namespace nest {
namespace mc {
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
    using value_type = typename backend::value_type;

    /// the integral index type
    using size_type = typename backend::size_type;

    /// the container used for values
    using array = typename backend::array;
    using host_array = typename backend::host_array;

    /// the container used for indexes
    using iarray = typename backend::iarray;

    using target_handle = std::pair<size_type, size_type>;
    using probe_handle = std::pair<const array fvm_multicell::*, size_type>;

    fvm_multicell() = default;

    void resting_potential(value_type potential_mV) {
        resting_potential_ = potential_mV;
    }

    template <typename Cells, typename Targets, typename Probes>
    void initialize(
        const Cells& cells,           // collection of nest::mc::cell descriptions
        Targets& target_handles,      // (write) where to store target handles
        Probes& probe_handles);       // (write) where to store probe handles

    void reset();

    void deliver_event(target_handle h, value_type weight) {
        mechanisms_[h.first]->net_receive(h.second, weight);
    }

    value_type probe(probe_handle h) const {
        return (this->*h.first)[h.second];
    }

    /// Initialize state prior to a sequence of integration steps.
    void start_integration(value_type tfinal, value_type dt_max) {
        EXPECTS(tfinal>t_);
        EXPECTS(dt_max>0);

        tfinal_ = tfinal;
        dt_max_ = dt_max;
        integration_running_ = true;

        // TODO: Placeholder; construct backend event delivery
        // data structure from events added.
        EXPECTS(events_.empty());
        for (auto& ev: staged_events_) {
            EXPECTS(ev.time<tfinal);
            events_.push(std::move(ev));
        }

        staged_events_.clear();
    }

    /// Advance one integration step.
    void step_integration();

    /// Query integration completion state.
    bool integration_complete() const {
        return !integration_running_;
    }

    /// Query per-cell time state. (Placeholders)
    value_type time(size_type cell_index) const {
        return t_;
    }

    value_type max_time() const {
        return t_;
    }

    bool state_synchronized() const {
        return true;
    }

    /// Add an event for processing in next integration stage.
    void add_event(value_type ev_time, target_handle h, value_type weight) {
        EXPECTS(!integration_running_);

        // TODO: Placeholder; either add to backend event list structure
        // incrementally here, or store by cell gid offset for construction
        // all at once in `start_integration()`.
        staged_events_.push_back({ev_time, h, weight});
    }

    /// Following types and methods are public only for testing:

    /// the type used to store matrix information
    using matrix_type = matrix<backend>;

    /// mechanism type
    using mechanism = typename backend::mechanism;

    /// stimulus type
    using stimulus = typename backend::stimulus;

    /// ion species storage
    using ion = typename backend::ion;

    /// view into index container
    using iview = typename backend::iview;
    using const_iview = typename backend::const_iview;

    /// view into value container
    using view = typename backend::view;
    using const_view = typename backend::const_view;

    /// which requires const_view in the vector library
    const matrix_type& jacobian() { return matrix_; }

    /// return list of CV areas in :
    ///          um^2
    ///     1e-6.mm^2
    ///     1e-8.cm^2
    const_view cv_areas() const { return cv_areas_; }

    /// return the voltage in each CV
    view       voltage()       { return voltage_; }
    const_view voltage() const { return voltage_; }

    /// return the current in each CV
    view       current()       { return current_; }
    const_view current() const { return current_; }

    std::size_t size() const { return matrix_.size(); }

    /// return reference to in iterable container of the mechanisms
    std::vector<mechanism>& mechanisms() { return mechanisms_; }

    /// return reference to list of ions
    std::map<mechanisms::ionKind, ion>&       ions()       { return ions_; }
    std::map<mechanisms::ionKind, ion> const& ions() const { return ions_; }

    /// return reference to sodium ion
    ion&       ion_na()       { return ions_[mechanisms::ionKind::na]; }
    ion const& ion_na() const { return ions_[mechanisms::ionKind::na]; }

    /// return reference to calcium ion
    ion&       ion_ca()       { return ions_[mechanisms::ionKind::ca]; }
    ion const& ion_ca() const { return ions_[mechanisms::ionKind::ca]; }

    /// return reference to pottasium ion
    ion&       ion_k()       { return ions_[mechanisms::ionKind::k]; }
    ion const& ion_k() const { return ions_[mechanisms::ionKind::k]; }

    /// flags if solution is physically realistic.
    /// here we define physically realistic as the voltage being within reasonable bounds.
    /// use a simple test of the voltage at the soma is reasonable, i.e. in the range
    ///     v_soma \in (-1000mv, 1000mv)
    bool is_physical_solution() const {
        auto v = voltage_[0];
        return (v>-1000.) && (v<1000.);
    }

    /// Return reference to the mechanism that matches name.
    /// The reference is const, because it this information should not be
    /// modified by the caller, however it is needed for unit testing.
    util::optional<const mechanism&> find_mechanism(const std::string& name) const {
        auto it = std::find_if(
            std::begin(mechanisms_), std::end(mechanisms_),
            [&name](const mechanism& m) {return m->name()==name;});
        return it==mechanisms_.end() ? util::nothing: util::just(*it);
    }

    std::size_t num_probes() const { return probes_.size(); }

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
    threshold_watcher threshold_watcher_;

    /// current time [ms]
    value_type t_ = 0;

    /// resting potential (initial voltage condition)
    value_type resting_potential_ = -65;

    /// final time in integration round [ms]
    value_type tfinal_ = 0;

    /// max time step for integration [ms]
    value_type dt_max_ = 0;

    /// flag: true after a call to `setup_integration()`; reset
    /// once integration to `tfinal_` is complete.
    bool integration_running_ = false;

    struct deliverable_event {
        double time;
        target_handle handle;
        double weight;
    };

    /// events staged for upcoming integration stage
    std::vector<deliverable_event> staged_events_;

    /// event queue for integration period
    /// TODO: Placeholder; this will change when event lists are moved to
    /// a backend structure.
    event_queue<deliverable_event> events_;

    /// the linear system for implicit time stepping of cell state
    matrix_type matrix_;

    /// cv_areas_[i] is the surface area of CV i [µm^2]
    array cv_areas_;

    /// the transmembrane current over the surface of each CV [nA]
    ///     I = area*i_m - I_e
    array current_;

    /// the potential in each CV [mV]
    array voltage_;

    /// the set of mechanisms present in the cell
    std::vector<mechanism> mechanisms_;

    /// the ion species
    std::map<mechanisms::ionKind, ion> ions_;

    std::vector<std::pair<const array fvm_multicell::*, size_type>> probes_;

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

    if (auto soma = seg->as_soma()) {
        // confirm assumption that there is one compartment in soma
        if (ncomp!=1) {
            throw std::logic_error("soma allocated more than one compartment");
        }
        auto i = comp_ival.first;
        auto area = math::area_sphere(soma->radius());
        auto c_m = soma->mechanism("membrane").get("c_m").value;

        tmp_cv_areas[i] += area;
        cv_capacitance[i] += area*c_m;

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

        auto c_m = cable->mechanism("membrane").get("c_m").value;
        auto r_L = cable->mechanism("membrane").get("r_L").value;

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

            auto conductance = 1/r_L*h*V1*V2/(h2*h2*V1+h1*h1*V2);
            // the scaling factor of 10^2 is to convert the quantity
            // to micro Siemens [μS]
            face_conductance[i] =  1e2 * conductance / h;

            auto al = div.left.area;
            auto ar = div.right.area;

            tmp_cv_areas[j] += al;
            tmp_cv_areas[i] += ar;
            cv_capacitance[j] += al * c_m;
            cv_capacitance[i] += ar * c_m;
        }
    }
    else {
        throw std::domain_error("FVM lowering encountered unsuported segment type");
    }

    return cv_range;
}

template <typename Backend>
template <typename Cells, typename Targets, typename Probes>
void fvm_multicell<Backend>::initialize(
    const Cells& cells,
    Targets& target_handles,
    Probes& probe_handles)
{
    using memory::make_const_view;
    using util::assign_by;
    using util::make_partition;
    using util::make_span;
    using util::size;
    using util::sort_by;
    using util::transform_view;
    using util::subrange_view;

    // count total targets and probes for validation of handle container sizes
    std::size_t targets_count = 0u;
    std::size_t probes_count = 0u;
    auto targets_size = size(target_handles);
    auto probes_size = size(probe_handles);

    auto ncell = size(cells);
    auto cell_num_compartments =
        transform_view(cells, [](const cell& c) { return c.num_compartments(); });

    std::vector<cell_lid_type> cell_comp_bounds;
    auto cell_comp_part = make_partition(cell_comp_bounds, cell_num_compartments);
    auto ncomp = cell_comp_part.bounds().second;

    // initialize storage from total compartment count
    current_ = array(ncomp, 0);
    voltage_ = array(ncomp, resting_potential_);

    // create maps for mechanism initialization.
    std::map<std::string, std::vector<segment_cv_range>> mech_map;
    std::vector<std::vector<cell_lid_type>> syn_mech_map;
    std::map<std::string, std::size_t> syn_mech_indices;

    // initialize vector used for matrix creation.
    std::vector<size_type> group_parent_index(ncomp);

    // create each cell:
    auto target_hi = target_handles.begin();
    auto probe_hi = probe_handles.begin();

    // Allocate scratch storage for calculating quantities used to build the
    // linear system: these will later be copied into target-specific storage

    // face_conductance_[i] = area_face  / (r_L * delta_x);
    std::vector<value_type> face_conductance(ncomp); // [µS]
    /// cv_capacitance_[i] is the capacitance of CV membrane
    std::vector<value_type> cv_capacitance(ncomp);   // [µm^2*F*m^-2 = pF]
    /// membrane area of each cv
    std::vector<value_type> tmp_cv_areas(ncomp);         // [µm^2]

    // used to build the information required to construct spike detectors
    std::vector<size_type> spike_detector_index;
    std::vector<value_type> thresholds;

    // Iterate over the input cells and build the indexes etc that descrbe the
    // fused cell group. On completion:
    //  - group_paranet_index contains the full parent index for the fused cells.
    //  - mech_map and syn_mech_map provide a map from mechanism names to an
    //    iterable container of compartment ranges, which are used later to
    //    generate the node index for each mechanism kind.
    //  - the tmp_* vectors contain compartment-specific information for each
    //    compartment in the fused cell group (areas, capacitance, etc).
    //  - each probe, stimulus and detector is attached to its compartment.
    for (auto i: make_span(0, ncell)) {
        const auto& c = cells[i];
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
                if (mech.name()!="membrane") {
                    mech_map[mech.name()].push_back(cv_range);
                }
            }
        }

        for (const auto& syn: c.synapses()) {
            EXPECTS(targets_count < targets_size);

            const auto& name = syn.mechanism.name();
            std::size_t syn_mech_index = 0;
            if (syn_mech_indices.count(name)==0) {
                syn_mech_index = syn_mech_map.size();
                syn_mech_indices[name] = syn_mech_index;
                syn_mech_map.push_back({});
            }
            else {
                syn_mech_index = syn_mech_indices[name];
            }

            auto& map_entry = syn_mech_map[syn_mech_index];

            auto syn_cv = comp_ival.first + find_cv_index(syn.location, graph);
            map_entry.push_back(syn_cv);
        }

        //
        // add the stimuli
        //

        // step 1: pack the index and parameter information into flat vectors
        std::vector<size_type> stim_index;
        std::vector<value_type> stim_durations;
        std::vector<value_type> stim_delays;
        std::vector<value_type> stim_amplitudes;
        for (const auto& stim: c.stimuli()) {
            auto idx = comp_ival.first+find_cv_index(stim.location, graph);
            stim_index.push_back(idx);
            stim_durations.push_back(stim.clamp.duration());
            stim_delays.push_back(stim.clamp.delay());
            stim_amplitudes.push_back(stim.clamp.amplitude());
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
                voltage_, current_, memory::make_const_view(stim_index));
            stim->set_parameters(stim_amplitudes, stim_durations, stim_delays);
            mechanisms_.push_back(mechanism(stim));
        }

        // calculate spike detector handles are their corresponding compartment indices
        for (const auto& detector: c.detectors()) {
            auto comp = comp_ival.first+find_cv_index(detector.location, graph);
            spike_detector_index.push_back(comp);
            thresholds.push_back(detector.threshold);
        }

        // record probe locations by index into corresponding state vector
        for (const auto& probe: c.probes()) {
            EXPECTS(probes_count < probes_size);

            auto comp = comp_ival.first+find_cv_index(probe.location, graph);
            switch (probe.kind) {
            case probeKind::membrane_voltage:
                *probe_hi++ = {&fvm_multicell::voltage_, comp};
                break;
            case probeKind::membrane_current:
                *probe_hi++ = {&fvm_multicell::current_, comp};
                break;
            default:
                throw std::logic_error("unrecognized probeKind");
            }
            ++probes_count;
        }
    }

    // set a back-end supplied watcher on the voltage vector
    threshold_watcher_ =
        threshold_watcher(voltage_, spike_detector_index, thresholds, 0);

    // confirm user-supplied container probes were appropriately sized.
    EXPECTS(probes_size==probes_count);

    // store the geometric information in target-specific containers
    cv_areas_ = make_const_view(tmp_cv_areas);

    // initalize matrix
    matrix_ = matrix_type(
        group_parent_index, cell_comp_bounds, cv_capacitance, face_conductance);

    // For each density mechanism build the full node index, i.e the list of
    // compartments with that mechanism, then build the mechanism instance.
    std::vector<size_type> mech_cv_index(ncomp);
    std::vector<value_type> mech_cv_weight(ncomp);
    std::map<std::string, std::vector<size_type>> mech_index_map;
    for (auto const& mech: mech_map) {
        // Clear the pre-allocated storage for mechanism indexes and weights.
        // Reuse the same vectors each time to have only one malloc and free
        // outside of the loop for each
        mech_cv_index.clear();
        mech_cv_weight.clear();

        const auto& seg_cv_ranges = mech.second;
        for (auto& rng: seg_cv_ranges) {
            if (rng.has_parent()) {
                // locate the parent cv in the partially constructed list of cv indexes
                auto it = algorithms::binary_find(mech_cv_index, rng.parent_cv);
                if (it == mech_cv_index.end()) {
                    mech_cv_index.push_back(rng.parent_cv);
                    mech_cv_weight.push_back(0);
                }
                auto pos = std::distance(std::begin(mech_cv_index), it);

                // add area contribution to the parent cv for the segment
                mech_cv_weight[pos] += rng.areas.first;
            }
            util::append(mech_cv_index, make_span(rng.segment_cvs));
            util::append(mech_cv_weight, subrange_view(tmp_cv_areas, rng.segment_cvs));

            // adjust the last CV
            mech_cv_weight.back() = rng.areas.second;

            EXPECTS(mech_cv_weight.size()==mech_cv_index.size());
        }

        // Scale the weights to get correct units (see w_i^d in formulation docs)
        // The units for the density channel weights are [10^2 μm^2 = 10^-10 m^2],
        // which requires that we scale the areas [μm^2] by 10^-2
        for (auto& w: mech_cv_weight) {
            w *= 1e-2;
        }

        mechanisms_.push_back(
            backend::make_mechanism(mech.first, voltage_, current_, mech_cv_weight, mech_cv_index)
        );

        // save the indices for easy lookup later in initialization
        mech_index_map[mech.first] = mech_cv_index;
    }

    // Create point (synapse) mechanisms
    for (const auto& syni: syn_mech_indices) {
        const auto& mech_name = syni.first;
        size_type mech_index = mechanisms_.size();

        auto cv_map = syn_mech_map[syni.second];
        size_type n_indices = size(cv_map);

        // sort indices but keep track of their original order for assigning
        // target handles
        using index_pair = std::pair<cell_lid_type, size_type>;
        auto cv_index = [](index_pair x) { return x.first; };
        auto target_index = [](index_pair x) { return x.second; };

        std::vector<index_pair> permute;
        assign_by(permute, make_span(0u, n_indices),
            [&](size_type i) { return index_pair(cv_map[i], i); });

        // sort the cv information in order of cv index
        sort_by(permute, cv_index);

        std::vector<cell_lid_type> cv_indices =
            assign_from(transform_view(permute, cv_index));

        // Create the mechanism.
        // An empty weight vector is supplied, because there are no weights applied to point
        // processes, because their currents are calculated with the target units of [nA]
        mechanisms_.push_back(
            backend::make_mechanism(mech_name, voltage_, current_, {}, cv_indices));

        // save the compartment indexes for this synapse type
        mech_index_map[mech_name] = cv_indices;

        // make target handles
        std::vector<target_handle> handles(n_indices);
        for (auto i: make_span(0u, n_indices)) {
            handles[target_index(permute[i])] = {mech_index, i};
        }
        target_hi = std::copy_n(std::begin(handles), n_indices, target_hi);
        targets_count += n_indices;
    }

    // confirm user-supplied containers for targets are appropriately sized
    EXPECTS(targets_size==targets_count);

    // build the ion species
    for (auto ion : mechanisms::ion_kinds()) {
        // find the compartment indexes of all compartments that have a
        // mechanism that depends on/influences ion
        std::set<size_type> index_set;
        for (auto const& mech : mechanisms_) {
            if(mech->uses_ion(ion)) {
                auto const& ni = mech_index_map[mech->name()];
                index_set.insert(ni.begin(), ni.end());
            }
        }
        std::vector<size_type> indexes(index_set.begin(), index_set.end());

        // create the ion state
        if(indexes.size()) {
            ions_[ion] = indexes;
        }

        // join the ion reference in each mechanism into the cell-wide ion state
        for (auto& mech : mechanisms_) {
            if (mech->uses_ion(ion)) {
                auto const& ni = mech_index_map[mech->name()];
                mech->set_ion(ion, ions_[ion],
                    util::make_copy<std::vector<size_type>> (algorithms::index_into(ni, indexes)));
            }
        }
    }

    // FIXME: Hard code parameters for now.
    //        Take defaults for reversal potential of sodium and potassium from
    //        the default values in Neuron.
    //        Neuron's defaults are defined in the file
    //          nrn/src/nrnoc/membdef.h
    constexpr value_type DEF_vrest = -65.0; // same name as #define in Neuron

    memory::fill(ion_na().reversal_potential(),     115+DEF_vrest); // mV
    memory::fill(ion_na().internal_concentration(),  10.0);         // mM
    memory::fill(ion_na().external_concentration(), 140.0);         // mM

    memory::fill(ion_k().reversal_potential(),     -12.0+DEF_vrest);// mV
    memory::fill(ion_k().internal_concentration(),  54.4);          // mM
    memory::fill(ion_k().external_concentration(),  2.5);           // mM

    memory::fill(ion_ca().reversal_potential(),     12.5*std::log(2.0/5e-5));// mV
    memory::fill(ion_ca().internal_concentration(), 5e-5);          // mM
    memory::fill(ion_ca().external_concentration(), 2.0);           // mM

    // initialise mechanism and voltage state
    reset();
}

template <typename Backend>
void fvm_multicell<Backend>::reset() {
    memory::fill(voltage_, resting_potential_);
    t_ = 0.;
    for (auto& m : mechanisms_) {
        // TODO : the parameters have to be set before the nrn_init
        // for now use a dummy value of dt.
        m->set_params(t_, 0.025);
        m->nrn_init();
    }

    // Reset state of the threshold watcher.
    // NOTE: this has to come after the voltage_ values have been reinitialized,
    // because these values are used by the watchers to set their initial state.
    threshold_watcher_.reset(t_);

    // Reset integration state.
    tfinal_ = 0;
    dt_max_ = 0;
    integration_running_ = false;
    staged_events_.clear();
    events_.clear();
}


template <typename Backend>
void fvm_multicell<Backend>::step_integration() {
    EXPECTS(integration_running_);

    while (auto ev = events_.pop_if_not_after(t_)) {
        deliver_event(ev->handle, ev->weight);
    }

    value_type t_to = std::min(t_+dt_max_, tfinal_);

    if (auto t = events_.time_if_before(t_to)) {
        t_to = *t;
    }

    value_type dt = t_to-t_;
    EXPECTS(dt>0);

    PE("current");
    memory::fill(current_, 0.);

    // update currents from ion channels
    for(auto& m : mechanisms_) {
        PE(m->name().c_str());
        m->set_params(t_, dt);
        m->nrn_current();
        PL();
    }
    PL();

    // solve the linear system
    PE("matrix", "setup");
    matrix_.assemble(dt, voltage_, current_);

    PL(); PE("solve");
    matrix_.solve();
    PL();
    memory::copy(matrix_.solution(), voltage_);
    PL();

    // integrate state of gating variables etc.
    PE("state");
    for(auto& m : mechanisms_) {
        PE(m->name().c_str());
        m->nrn_state();
        PL();
    }
    PL();

    t_ = t_to;

    // update spike detector thresholds
    threshold_watcher_.test(t_);

    // are we there yet?
    integration_running_ = t_<tfinal_;

    EXPECTS(integration_running_ || events_.empty());
}

} // namespace fvm
} // namespace mc
} // namespace nest
