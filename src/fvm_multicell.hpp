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
#include <event_queue.hpp>
#include <ion.hpp>
#include <math.hpp>
#include <matrix.hpp>
#include <profiling/profiler.hpp>
#include <segment.hpp>
#include <stimulus.hpp>
#include <util/debug.hpp>
#include <util/meta.hpp>
#include <util/partition.hpp>
#include <util/rangeutil.hpp>
#include <util/span.hpp>

#include <memory/memory.hpp>

namespace nest {
namespace mc {

// FIXME : these will go when we merge the complex compartment stuff
namespace util {
    /// helper for taking the first value in a std::pair
    template <typename L, typename R>
    L const& left(std::pair<L, R> const& p) {
        return p.first;
    }

    /// helper for taking the second value in a std::pair
    template <typename L, typename R>
    R const& right(std::pair<L, R> const& p) {
        return p.second;
    }
}


namespace fvm {

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

    using matrix_assembler = typename backend::matrix_assembler;

    /// API for cell_group (see above):
    using detector_handle = size_type;
    using target_handle = std::pair<size_type, size_type>;
    using probe_handle = std::pair<const array fvm_multicell::*, size_type>;

    using stimulus_store_type = std::vector<std::pair<size_type, i_clamp>>;

    fvm_multicell() = default;

    void resting_potential(value_type potential_mV) {
        resting_potential_ = potential_mV;
    }

    template <typename Cells, typename Detectors, typename Targets, typename Probes>
    void initialize(
        const Cells& cells,           // collection of nest::mc::cell descriptions
        Detectors& detector_handles,  // (write) where to store detector handles
        Targets& target_handles,      // (write) where to store target handles
        Probes& probe_handles);       // (write) where to store probe handles

    void reset();

    void deliver_event(target_handle h, value_type weight) {
        mechanisms_[h.first]->net_receive(h.second, weight);
    }

    value_type detector_voltage(detector_handle h) const {
        return voltage_[h]; // detector_handle is just the compartment index
    }

    value_type probe(probe_handle h) const {
        return (this->*h.first)[h.second];
    }

    void advance(double dt);

    /// Following types and methods are public only for testing:

    /// the type used to store matrix information
    using matrix_type = matrix<backend>;

    /// mechanism type
    using mechanism = typename backend::mechanism;

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

    /// return the capacitance of each CV surface
    /// this is the total capacitance, not per unit area,
    /// i.e. equivalent to sigma_i * c_m
    const_view cv_capacitance() const { return cv_capacitance_; }

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

    /// return reference to the stimuli
    stimulus_store_type& stimuli() {
        return stimuli_;
    }

    stimulus_store_type const& stimuli() const {
        return stimuli_;
    }

    value_type time() const { return t_; }

    std::size_t num_probes() const { return probes_.size(); }

private:
    /// current time [ms]
    value_type t_ = value_type{0};

    /// resting potential (initial voltage condition)
    value_type resting_potential_ = -65;

    /// the linear system for implicit time stepping of cell state
    matrix_type matrix_;

    /// the helper used to construct the matrix
    matrix_assembler matrix_assembler_;

    /// cv_areas_[i] is the surface area of CV i [µm^2]
    array cv_areas_;

    /// alpha_[i] is the following value at the CV face between
    /// CV i and its parent, required when constructing linear system
    ///     face_alpha_[i] = area_face  / (c_m * r_L * delta_x);
    array face_alpha_; // [µm·m^2/cm/s ≡ 10^5 µm^2/ms]

    /// cv_capacitance_[i] is the capacitance of CV i per unit area (i.e. c_m) [F/m^2]
    array cv_capacitance_;

    /// the average current density over the surface of each CV [mA/cm^2]
    /// current_ = i_m - i_e
    array current_;

    /// the potential in each CV [mV]
    array voltage_;

    /// the set of mechanisms present in the cell
    std::vector<mechanism> mechanisms_;

    /// the ion species
    std::map<mechanisms::ionKind, ion> ions_;

    stimulus_store_type stimuli_;

    std::vector<std::pair<const array fvm_multicell::*, size_type>> probes_;

    // perform area and capacitance calculation on initialization
    void compute_cv_area_unnormalized_capacitance(
        std::pair<size_type, size_type> comp_ival,
        const segment* seg,
        const std::vector<size_type>& parent,
        std::vector<value_type>& tmp_face_alpha,
        std::vector<value_type>& tmp_cv_areas,
        std::vector<value_type>& tmp_cv_capacitance
    );
};

////////////////////////////////////////////////////////////////////////////////
//////////////////////////////// Implementation ////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
template <typename Backend>
void fvm_multicell<Backend>::compute_cv_area_unnormalized_capacitance(
    std::pair<size_type, size_type> comp_ival,
    const segment* seg,
    const std::vector<size_type>& parent,
    std::vector<value_type>& tmp_face_alpha,
    std::vector<value_type>& tmp_cv_areas,
    std::vector<value_type>& tmp_cv_capacitance)
{
    using util::left;
    using util::right;
    using util::size;

    // precondition: group_parent_index[j] holds the correct value for
    // j in [base_comp, base_comp+segment.num_compartments()].

    auto ncomp = comp_ival.second-comp_ival.first;

    if (auto soma = seg->as_soma()) {
        // confirm assumption that there is one compartment in soma
        if (ncomp!=1) {
            throw std::logic_error("soma allocated more than one compartment");
        }
        auto i = comp_ival.first;
        auto area = math::area_sphere(soma->radius());

        tmp_cv_areas[i] += area;
        tmp_cv_capacitance[i] += area * soma->mechanism("membrane").get("c_m").value;
    }
    else if (auto cable = seg->as_cable()) {
        // loop over each compartment in the cable
        // each compartment has the face between two CVs at its centre
        // the centers of the CVs are the end points of the compartment
        //
        //  __________________________________
        //  | ........ | .cvleft. |    cv    |
        //  | ........ L ........ C          R
        //  |__________|__________|__________|
        //
        //  The compartment has end points marked L and R (left and right).
        //  The left compartment is assumed to be closer to the soma
        //  (i.e. it follows the minimal degree ordering)
        //  The face is at the center, marked C.
        //  The full control volume to the left (marked with .)

        auto c_m = cable->mechanism("membrane").get("c_m").value;
        auto r_L = cable->mechanism("membrane").get("r_L").value;
        const auto& compartments = cable->compartments();

        EXPECTS(size(compartments)==ncomp);

        for (auto i: util::make_span(comp_ival)) {
            const auto& c = compartments[i-comp_ival.first];
            auto j = parent[i];

            auto radius_center = math::mean(c.radius);
            auto area_face = math::area_circle(radius_center);
            tmp_face_alpha[i] = area_face / (c_m * r_L * c.length);

            auto halflen = c.length/2;
            auto al = math::area_frustrum(halflen, left(c.radius), radius_center);
            auto ar = math::area_frustrum(halflen, right(c.radius), radius_center);

            tmp_cv_areas[j] += al;
            tmp_cv_areas[i] += ar;
            tmp_cv_capacitance[j] += al * c_m;
            tmp_cv_capacitance[i] += ar * c_m;
        }
    }
    else {
        throw std::domain_error("FVM lowering encountered unsuported segment type");
    }
}

template <typename Backend>
template <typename Cells, typename Detectors, typename Targets, typename Probes>
void fvm_multicell<Backend>::initialize(
    const Cells& cells,
    Detectors& detector_handles,
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

    // count total detectors, targets and probes for validation of handle container sizes
    std::size_t detectors_count = 0u;
    std::size_t targets_count = 0u;
    std::size_t probes_count = 0u;
    auto detectors_size = size(detector_handles);
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
    std::map<std::string, std::vector<std::pair<size_type, size_type>>> mech_map;
    std::vector<std::vector<cell_lid_type>> syn_mech_map;
    std::map<std::string, std::size_t> syn_mech_indices;

    // initialize vector used for matrix creation.
    std::vector<size_type> group_parent_index(ncomp);

    // create each cell:
    auto target_hi = target_handles.begin();
    auto detector_hi = detector_handles.begin();
    auto probe_hi = probe_handles.begin();

    // allocate scratch vectors
    std::vector<value_type> tmp_face_alpha(ncomp);
    std::vector<value_type> tmp_cv_areas(ncomp);
    std::vector<value_type> tmp_cv_capacitance(ncomp);

    for (auto i: make_span(0, ncell)) {
        const auto& c = cells[i];
        auto comp_ival = cell_comp_part[i];

        auto graph = c.model();

        for (auto k: make_span(comp_ival)) {
            group_parent_index[k] = graph.parent_index[k-comp_ival.first]+comp_ival.first;
        }

        auto seg_num_compartments =
            transform_view(c.segments(), [](const segment_ptr& s) { return s->num_compartments(); });
        auto nseg = seg_num_compartments.size();

        std::vector<cell_lid_type> seg_comp_bounds;
        auto seg_comp_part =
            make_partition(seg_comp_bounds, seg_num_compartments, comp_ival.first);

        for (size_type j = 0; j<nseg; ++j) {
            const auto& seg = c.segment(j);
            const auto& seg_comp_ival = seg_comp_part[j];

            compute_cv_area_unnormalized_capacitance(
                seg_comp_ival, seg, group_parent_index,
                tmp_face_alpha, tmp_cv_areas, tmp_cv_capacitance);

            for (const auto& mech: seg->mechanisms()) {
                if (mech.name()!="membrane") {
                    mech_map[mech.name()].push_back(seg_comp_ival);
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

            size_type syn_comp = comp_ival.first+find_compartment_index(syn.location, graph);
            map_entry.push_back(syn_comp);
        }

        // add the stimuli
        for (const auto& stim: c.stimuli()) {
            auto idx = comp_ival.first+find_compartment_index(stim.location, graph);
            stimuli_.push_back({idx, stim.clamp});
        }

        // detector handles are just their corresponding compartment indices
        for (const auto& detector: c.detectors()) {
            EXPECTS(detectors_count < detectors_size);

            auto comp = comp_ival.first+find_compartment_index(detector.location, graph);
            *detector_hi++ = comp;
            ++detectors_count;
        }

        // record probe locations by index into corresponding state vector
        for (const auto& probe: c.probes()) {
            EXPECTS(probes_count < probes_size);

            auto comp = comp_ival.first+find_compartment_index(probe.location, graph);
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

    // normalize capacitance across cell
    for (auto i: util::make_span(0, ncomp)) {
        tmp_cv_capacitance[i] /= tmp_cv_areas[i];
    }

    // store the geometric information in target-specific containers
    face_alpha_     = make_const_view(tmp_face_alpha);
    cv_areas_       = make_const_view(tmp_cv_areas);
    cv_capacitance_ = make_const_view(tmp_cv_capacitance);

    // initalize matrix
    matrix_ = matrix_type(group_parent_index, cell_comp_bounds);

    matrix_assembler_ = matrix_assembler(
        matrix_.d(), matrix_.u(), matrix_.rhs(), matrix_.p(),
        cv_areas_, face_alpha_, voltage_, current_, cv_capacitance_);

    // create density mechanisms
    std::vector<size_type> mech_comp_indices(ncomp);

    std::map<std::string, std::vector<size_type>> mech_index_map;
    for (auto& mech: mech_map) {
        mech_comp_indices.clear();
        for (auto comp_ival: mech.second) {
            util::append(mech_comp_indices, make_span(comp_ival));
        }

        mechanisms_.push_back(
            backend::make_mechanism(mech.first, voltage_, current_, mech_comp_indices)
        );

        // save the indices for easy lookup later in initialization
        mech_index_map[mech.first] = mech_comp_indices;
    }

    // create point (synapse) mechanisms
    for (const auto& syni: syn_mech_indices) {
        const auto& mech_name = syni.first;
        size_type mech_index = mechanisms_.size();

        auto comp_indices = syn_mech_map[syni.second];
        size_type n_indices = size(comp_indices);

        // sort indices but keep track of their original order for assigning
        // target handles

        using index_pair = std::pair<cell_lid_type, size_type>;
        auto compartment_index = [](index_pair x) { return x.first; };
        auto target_index = [](index_pair x) { return x.second; };

        std::vector<index_pair> permute;
        assign_by(permute, make_span(0u, n_indices),
            [&](size_type i) { return index_pair(comp_indices[i], i); });

        sort_by(permute, compartment_index);
        assign_by(comp_indices, permute, compartment_index);

        // make target handles
        std::vector<target_handle> handles(n_indices);
        for (auto i: make_span(0u, n_indices)) {
            handles[target_index(permute[i])] = {mech_index, i};
        }
        target_hi = std::copy_n(std::begin(handles), n_indices, target_hi);
        targets_count += n_indices;

        //auto mech = mechanism_catalogue::make(
        auto mech = backend::make_mechanism(
            mech_name, voltage_, current_, comp_indices);
        mech->set_areas(cv_areas_);
        mechanisms_.push_back(std::move(mech));
    }

    // confirm write-parameters were appropriately sized
    EXPECTS(detectors_size==detectors_count);
    EXPECTS(targets_size==targets_count);
    EXPECTS(probes_size==probes_count);

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
}

template <typename Backend>
void fvm_multicell<Backend>::advance(double dt) {
    PE("current");
    memory::fill(current_, 0.);

    // update currents from ion channels
    for(auto& m : mechanisms_) {
        PE(m->name().c_str());
        m->set_params(t_, dt);
        m->nrn_current();
        PL();
    }

    // add current contributions from stimuli
    for (auto& stim : stimuli_) {
        auto ie = stim.second.amplitude(t_); // [nA]
        auto loc = stim.first;

        // TODO KERNEL
        // is a kernel actually needed?
        // for now I only make the update if the injected current in nonzero to
        // avoid a redundant host->device copy on the gpu
        //
        // note: current_ in [mA/cm^2], ie in [nA], cv_areas_ in [µm^2].
        // unit scale factor: [nA/µm^2]/[mA/cm^2] = 100
        //current_[loc] -= 100*ie/cv_areas_[loc];
        if (ie!=0.) {
            current_[loc] = current_[loc] - 100*ie/cv_areas_[loc];
        }
    }
    PL();

    // solve the linear system
    PE("matrix", "setup");
    matrix_assembler_.assemble(dt);
    PL(); PE("solve");
    matrix_.solve();
    PL();
    memory::copy(matrix_.rhs(), voltage_);
    PL();

    // integrate state of gating variables etc.
    PE("state");
    for(auto& m : mechanisms_) {
        PE(m->name().c_str());
        m->nrn_state();
        PL();
    }
    PL();

    t_ += dt;
}

} // namespace fvm
} // namespace mc
} // namespace nest
