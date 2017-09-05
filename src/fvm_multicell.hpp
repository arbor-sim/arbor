#pragma once

#include <algorithm>
#include <iterator>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <algorithms.hpp>
#include <backends/fvm_types.hpp>
#include <cell.hpp>
#include <compartment.hpp>
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
    using value_type = fvm_value_type;

    /// the integral index type
    using size_type = fvm_size_type;

    /// the container used for values
    using array = typename backend::array;
    using host_array = typename backend::host_array;

    /// the container used for indexes
    using iarray = typename backend::iarray;

    using probe_handle = std::pair<const array fvm_multicell::*, size_type>;

    using target_handle = typename backend::target_handle;
    using deliverable_event = typename backend::deliverable_event;

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

    // fvm_multicell::deliver_event is used only for testing
    void deliver_event(target_handle h, value_type weight) {
        mechanisms_[h.mech_id]->net_receive(h.index, weight);
    }

    value_type probe(probe_handle h) const {
        return (this->*h.first)[h.second];
    }

    // Initialize state prior to a sequence of integration steps.
    void setup_integration(value_type tfinal, value_type dt_max) {
        EXPECTS(dt_max>0);

        tfinal_ = tfinal;
        dt_max_ = dt_max;

        compute_min_remaining();

        EXPECTS(!has_pending_events());

        util::stable_sort_by(staged_events_, [](const deliverable_event& ev) { return event_index(ev); });
        events_->init(staged_events_);
        staged_events_.clear();
    }

    // Advance one integration step.
    void step_integration();

    // Query integration completion state.
    bool integration_complete() const {
        return min_remaining_steps_==0;
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

    /// Add an event for processing in next integration stage.
    void add_event(value_type ev_time, target_handle h, value_type weight) {
        EXPECTS(integration_complete());
        staged_events_.push_back(deliverable_event(ev_time, h, weight));
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

    /// events staged for upcoming integration stage
    std::vector<deliverable_event> staged_events_;

    /// event queue for integration period
    using deliverable_event_stream = typename backend::deliverable_event_stream;
    std::unique_ptr<deliverable_event_stream> events_;

    bool has_pending_events() const {
        return events_ && !events_->empty();
    }

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

    /// the transmembrane current over the surface of each CV [nA]
    ///     I = area*i_m - I_e
    array current_;

    /// the potential in each CV [mV]
    array voltage_;

    /// the set of mechanisms present in the cell
    std::vector<mechanism> mechanisms_;

    /// the ion species
    std::map<mechanisms::ionKind, ion> ions_;

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

    // Take cell descriptions from recipe. These are used initially
    // to count compartments for allocation of data structures, and
    // then interrogated again for the details for each cell in turn.

    std::vector<cell> cells;
    cells.reserve(gids.size());
    for (auto gid: gids) {
        cells.push_back(any_cast<cell>(rec.get_cell_description(gid)));
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

    // look up table: mechanism name -> list of cv_range objects
    std::map<std::string, std::vector<segment_cv_range>> mech_to_cv_range;

    // look up table: point mechanism (synapse) name -> CV indices and target numbers.
    struct syn_cv_and_target {
        cell_lid_type cv;
        cell_lid_type target;
    };
    std::map<std::string, std::vector<syn_cv_and_target>> syn_mech_map;

    // initialize vector used for matrix creation.
    std::vector<size_type> group_parent_index(ncomp);

    // setup per-cell event stores.
    events_ = util::make_unique<deliverable_event_stream>(ncell_);

    // Create each cell:

    // Allocate scratch storage for calculating quantities used to build the
    // linear system: these will later be copied into target-specific storage

    // face_conductance_[i] = area_face  / (r_L * delta_x);
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
                if (mech.name()!="membrane") {
                    mech_to_cv_range[mech.name()].push_back(cv_range);
                }
            }
        }

        for (const auto& syn: c.synapses()) {
            const auto& name = syn.mechanism.name();
            auto& map_entry = syn_mech_map[name];

            cell_lid_type syn_cv = comp_ival.first + find_cv_index(syn.location, graph);
            cell_lid_type target_index = targets_count++;

            map_entry.push_back(syn_cv_and_target{syn_cv, target_index});
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
                cv_to_cell_, time_, time_to_, dt_comp_,
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

        // Retrieve probe addresses and tags from recipe for this cell.
        for (cell_lid_type j: make_span(0, rec.num_probes(gid))) {
            probe_info pi = rec.get_probe({gid, j});
            auto where = any_cast<cell_probe_address>(pi.address);

            auto comp = comp_ival.first+find_cv_index(where.location, graph);
            probe_handle handle;

            switch (where.kind) {
            case cell_probe_address::membrane_voltage:
                handle = {&fvm_multicell::voltage_, comp};
                break;
            case cell_probe_address::membrane_current:
                handle = {&fvm_multicell::current_, comp};
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
        group_parent_index, cell_comp_bounds, cv_capacitance, face_conductance);

    // For each density mechanism build the full node index, i.e the list of
    // compartments with that mechanism, then build the mechanism instance.
    std::vector<size_type> mech_cv_index(ncomp);
    std::vector<value_type> mech_cv_weight(ncomp);
    std::map<std::string, std::vector<size_type>> mech_to_cv_index;
    for (auto const& entry: mech_to_cv_range) {
        const auto& mech_name = entry.first;
        const auto& seg_cv_ranges = entry.second;

        // Clear the pre-allocated storage for mechanism indexes and weights.
        // Reuse the same vectors each time to have only one malloc and free
        // outside of the loop for each
        mech_cv_index.clear();
        mech_cv_weight.clear();

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

        size_type mech_id = mechanisms_.size();
        mechanisms_.push_back(
            backend::make_mechanism(mech_name, mech_id, cv_to_cell_, time_, time_to_, dt_comp_, voltage_, current_, mech_cv_weight, mech_cv_index));

        // Save the indices for ion set up below.
        mech_to_cv_index[mech_name] = mech_cv_index;
    }

    target_handles.resize(targets_count);

    // Create point (synapse) mechanisms.
    for (auto& syni: syn_mech_map) {
        size_type mech_id = mechanisms_.size();

        const auto& mech_name = syni.first;
        auto& cv_assoc = syni.second;

        // Sort CV indices but keep track of their corresponding targets.
        auto cv_index = [](syn_cv_and_target x) { return x.cv; };
        util::stable_sort_by(cv_assoc, cv_index);
        std::vector<cell_lid_type> cv_indices = assign_from(transform_view(cv_assoc, cv_index));

        // Create the mechanism.
        // An empty weight vector is supplied, because there are no weights applied to point
        // processes, because their currents are calculated with the target units of [nA]
        mechanisms_.push_back(
            backend::make_mechanism(mech_name, mech_id, cv_to_cell_, time_, time_to_, dt_comp_, voltage_, current_, {}, cv_indices));

        // Save the indices for ion set up below.
        mech_to_cv_index[mech_name] = cv_indices;

        // Make the target handles.
        cell_lid_type instance = 0;
        for (auto entry: cv_assoc) {
            target_handles[entry.target] = target_handle(mech_id, instance++, cv_to_cell_tmp[entry.cv]);
        }
    }

    // build the ion species
    for (auto ion : mechanisms::ion_kinds()) {
        // find the compartment indexes of all compartments that have a
        // mechanism that depends on/influences ion
        std::set<size_type> index_set;
        for (auto const& mech : mechanisms_) {
            if(mech->uses_ion(ion)) {
                auto const& ni = mech_to_cv_index[mech->name()];
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
                auto const& ni = mech_to_cv_index[mech->name()];
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

    // initialize mechanism and voltage state
    reset();
}

template <typename Backend>
void fvm_multicell<Backend>::reset() {
    memory::fill(voltage_, resting_potential_);

    set_time_global(0);
    set_time_to_global(0);

    for (auto& m : mechanisms_) {
        // TODO : the parameters have to be set before the nrn_init
        // for now use a dummy value of dt.
        m->set_params();
        m->nrn_init();
    }

    // Reset state of the threshold watcher.
    // NOTE: this has to come after the voltage_ values have been reinitialized,
    // because these values are used by the watchers to set their initial state.
    threshold_watcher_.reset();

    // Reset integration state.
    tfinal_ = 0;
    dt_max_ = 0;
    min_remaining_steps_ = 0;
    staged_events_.clear();
    events_->clear();

    EXPECTS(integration_complete());
    EXPECTS(!has_pending_events());
}


template <typename Backend>
void fvm_multicell<Backend>::step_integration() {
    EXPECTS(!integration_complete());

    PE("current");
    memory::fill(current_, 0.);

    // mark pending events for delivery
    events_->mark_until_after(time_);

    // deliver pending events and update current contributions from mechanisms
    for(auto& m: mechanisms_) {
        PE(m->name().c_str());
        m->deliver_events(*events_);
        m->nrn_current();
        PL();
    }

    // remove delivered events from queue and set time_to_
    events_->drop_marked_events();

    backend::update_time_to(time_to_, time_, dt_max_, tfinal_);
    invalidate_time_cache();
    events_->event_time_if_before(time_to_);
    PL();

    // set per-cell and per-compartment dt (constant within a cell)
    backend::set_dt(dt_cell_, dt_comp_, time_to_, time_, cv_to_cell_);

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

    memory::copy(time_to_, time_);
    invalidate_time_cache();

    // update spike detector thresholds
    threshold_watcher_.test();

    // are we there yet?
    decrement_min_remaining();

    EXPECTS(!integration_complete() || !has_pending_events());
}

} // namespace fvm
} // namespace mc
} // namespace nest
