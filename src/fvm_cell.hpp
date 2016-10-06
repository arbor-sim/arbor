#pragma once

#include <algorithm>
#include <iterator>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <algorithms.hpp>
#include <cell.hpp>
#include <event_queue.hpp>
#include <ion.hpp>
#include <math.hpp>
#include <matrix.hpp>
#include <mechanism.hpp>
#include <mechanism_catalogue.hpp>
#include <profiling/profiler.hpp>
#include <segment.hpp>
#include <stimulus.hpp>
#include <util.hpp>
#include <util/meta.hpp>

#include <memory/memory.hpp>

/*
 * Lowered cell implementation based on finite volume method.
 *
 * TODO: Move following description of internal API
 * to a better location or document.
 *
 * Lowered cells are managed by the `cell_group` class.
 * A `cell_group` manages one lowered cell instance, which
 * may in turn simulate one or more cells.
 *
 * The outward facing interface for `cell_group` is defined
 * in terms of cell gids and member indices; the interface
 * between `cell_group` and a lowered cell is described below.
 *
 * The motivation for the following interface is to
 *   1. Provide more flexibility in lowered cell implementation
 *   2. Save memory and duplicated index maps by associating
 *      externally visible objects with opaque handles.
 *
 * In the following, `lowered_cell` represents the lowered
 * cell class, and `lowered` an instance thereof.
 *
 * `lowered_cell::detector_handle`
 *       Type of handles used by spike detectors to query
 *       membrane voltage.
 *
 * `lowered_cell::probe_handle`
 *       Type of handle used to query the value of a probe.
 *
 * `lowered_cell::target_handle`
 *       Type of handle used to identify the target of a
 *       postsynaptic spike event.
 *
 * `lowered_cell::value_type`
 *       Floating point type used for internal states
 *       and simulation time.
 *
 * `lowered_cell()`
 *       Default constructor; performs no cell-specific
 *       initialization.
 *
 * `lowered.initialize(const Cells& cells, ...)`
 *       Allocate and initalize data structures to simulate
 *       the cells described in the collection `cells`,
 *       where each item is of type `nest::mc::cell`.
 *
 *       Remaining arguments consist of references to
 *       collections for storing:
 *         1. Detector handles
 *         2. Target handles
 *         3. Probe handles
 *
 *       Handles are written in the same order as they
 *       appear in the provided cell descriptions.
 *
 *  `lowered.reset()`
 *
 *       Resets state to initial conditiions and sets
 *       internal simulation time to 0.
 *
 *  `lowered.advance(value_type dt)`
 *
 *       Advanece simulation state by `dt` (value in
 *       milliseconds). For `fvm_cell` at least,
 *       this corresponds to one integration step.
 *
 *  `lowered.deliver_event(target_handle target, value_type weight)`
 *
 *       Update target-specifc state based on the arrival
 *       of a postsynaptic spike event with given weight.
 *
 *  `lowered.detect_voltage(detector_handle)`
 *
 *       Return membrane voltage at detector site as specified
 *       by handle.
 *
 *  `lowered.probe(probe_handle)`
 *
 *       Return value of corresponding probe.
 *
 *  `lowered.resting_potential(value_type potential)`
 *
 *       Set the steady-state membrane voltage used for the
 *       cell initial condition. (Defaults to -65 mV currently.)
 */

namespace nest {
namespace mc {
namespace fvm {

template <typename Value, typename Index>
class fvm_cell {
public:
    fvm_cell() = default;

    /// the real number type
    using value_type = Value;

    /// the integral index type
    using size_type = Index;

    /// the container used for indexes
    using index_type = memory::HostVector<size_type>;

    /// the container used for values
    using vector_type = memory::HostVector<value_type>;

    /// API for cell_group (see above):

    using detector_handle = size_type;
    using target_handle = std::pair<size_type, size_type>;
    using probe_handle = std::pair<const vector_type fvm_cell::*, size_type>;

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
        mechanisms_[synapse_base_+h.first]->net_receive(h.second, weight);
    }

    value_type detector_voltage(detector_handle h) const {
        return voltage_[h]; // detector_handle is just the compartment index
    }

    value_type probe(probe_handle h) const {
        return (this->*h.first)[h.second];
    }

    void advance(value_type dt);

    /// Following types and methods are public only for testing:

    /// the type used to store matrix information
    using matrix_type =
        matrix<value_type, size_type, backends::host_matrix_policy>;

    /// mechanism type
    using mechanism_type =
        nest::mc::mechanisms::mechanism_ptr<value_type, size_type>;

    /// ion species storage
    using ion_type = mechanisms::ion<value_type, size_type>;

    /// view into index container
    using index_view = typename index_type::view_type;
    using const_index_view = typename index_type::const_view_type;

    /// view into value container
    using vector_view = typename vector_type::view_type;
    using const_vector_view = typename vector_type::const_view_type;

    /// build the matrix for a given time step
    void setup_matrix(value_type dt);

    /// which requires const_view in the vector library
    const matrix_type& jacobian() {
        return matrix_;
    }

    /// return list of CV areas in :
    ///          um^2
    ///     1e-6.mm^2
    ///     1e-8.cm^2
    const_vector_view cv_areas() const {
        return cv_areas_;
    }

    /// return the capacitance of each CV surface
    /// this is the total capacitance, not per unit area,
    /// i.e. equivalent to sigma_i * c_m
    const_vector_view cv_capacitance() const {
        return cv_capacitance_;
    }

    /// return the voltage in each CV
    vector_view       voltage()       { return voltage_; }
    const_vector_view voltage() const { return voltage_; }

    /// return the current in each CV
    vector_view       current()       { return current_; }
    const_vector_view current() const { return current_; }

    std::size_t size() const { return matrix_.size(); }

    /// return reference to in iterable container of the mechanisms
    std::vector<mechanism_type>& mechanisms() { return mechanisms_; }

    /// return reference to list of ions
    std::map<mechanisms::ionKind, ion_type>&       ions()       { return ions_; }
    std::map<mechanisms::ionKind, ion_type> const& ions() const { return ions_; }

    /// return reference to sodium ion
    ion_type&       ion_na()       { return ions_[mechanisms::ionKind::na]; }
    ion_type const& ion_na() const { return ions_[mechanisms::ionKind::na]; }

    /// return reference to calcium ion
    ion_type&       ion_ca()       { return ions_[mechanisms::ionKind::ca]; }
    ion_type const& ion_ca() const { return ions_[mechanisms::ionKind::ca]; }

    /// return reference to pottasium ion
    ion_type&       ion_k()       { return ions_[mechanisms::ionKind::k]; }
    ion_type const& ion_k() const { return ions_[mechanisms::ionKind::k]; }

    /// flags if solution is physically realistic.
    /// here we define physically realistic as the voltage being within reasonable bounds.
    /// use a simple test of the voltage at the soma is reasonable, i.e. in the range
    ///     v_soma \in (-1000mv, 1000mv)
    bool is_physical_solution() const {
        auto v = voltage_[0];
        return (v>-1000.) && (v<1000.);
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

    /// index for fast lookup of compartment index ranges of segments
    index_type segment_index_;

    /// cv_areas_[i] is the surface area of CV i [µm^2]
    vector_type cv_areas_;

    /// alpha_[i] is the following value at the CV face between
    /// CV i and its parent, required when constructing linear system
    ///     face_alpha_[i] = area_face  / (c_m * r_L * delta_x);
    vector_type face_alpha_; // [µm·m^2/cm/s ≡ 10^5 µm^2/ms]

    /// cv_capacitance_[i] is the capacitance of CV i per unit area (i.e. c_m) [F/m^2]
    vector_type cv_capacitance_;

    /// the average current density over the surface of each CV [mA/cm^2]
    /// current_ = i_m - i_e
    vector_type current_;

    /// the potential in each CV [mV]
    vector_type voltage_;

    /// Where point mechanisms start in the mechanisms_ list.
    std::size_t synapse_base_;

    /// the set of mechanisms present in the cell
    std::vector<mechanism_type> mechanisms_;

    /// the ion species
    std::map<mechanisms::ionKind, ion_type> ions_;

    std::vector<std::pair<uint32_t, i_clamp>> stimuli_;

    std::vector<std::pair<const vector_type fvm_cell::*, uint32_t>> probes_;

    // mechanism factory
    using mechanism_catalogue = nest::mc::mechanisms::catalogue<value_type, size_type>;
};

////////////////////////////////////////////////////////////////////////////////
//////////////////////////////// Implementation ////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template <typename T, typename I>
template <typename Cells, typename Detectors, typename Targets, typename Probes>
void fvm_cell<T, I>::initialize(
    const Cells& cells,
    Detectors& detector_handles,
    Targets& target_handles,
    Probes& probe_handles)
{
    if (util::size(cells)!=1u) {
        throw std::invalid_argument("fvm_cell accepts only one cell");
    }

    const nest::mc::cell& cell = *(std::begin(cells));
    size_type ncomp = cell.num_compartments();

    // confirm write-parameters have enough room to store handles
    EXPECTS(util::size(detector_handles)==cell.detectors().size());
    EXPECTS(util::size(target_handles)==cell.synapses().size());
    EXPECTS(util::size(probe_handles)==cell.probes().size());

    // initialize storage
    cv_areas_   = vector_type{ncomp, T{0}};
    face_alpha_ = vector_type{ncomp, T{0}};
    cv_capacitance_ = vector_type{ncomp, T{0}};
    current_    = vector_type{ncomp, T{0}};
    voltage_    = vector_type{ncomp, T{resting_potential_}};

    using util::left;
    using util::right;

    const auto graph = cell.model();
    matrix_ = matrix_type(graph.parent_index);

    auto parent_index = matrix_.p();
    segment_index_ = graph.segment_index;

    auto seg_idx = 0;
    for(auto const& s : cell.segments()) {
        if(auto soma = s->as_soma()) {
            // assert the assumption that the soma is at 0
            if(seg_idx!=0) {
                throw std::domain_error(
                        "FVM lowering encountered soma with non-zero index"
                );
            }
            auto area = math::area_sphere(soma->radius());
            cv_areas_[0] += area;
            cv_capacitance_[0] += area * soma->mechanism("membrane").get("c_m").value;
        }
        else if(auto cable = s->as_cable()) {
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
            for(auto c : cable->compartments()) {
                auto i = segment_index_[seg_idx] + c.index;
                auto j = parent_index[i];

                auto radius_center = math::mean(c.radius);
                auto area_face = math::area_circle( radius_center );
                face_alpha_[i] = area_face  / (c_m * r_L * c.length);

                auto halflen = c.length/2;

                auto al = math::area_frustrum(halflen, left(c.radius), radius_center);
                auto ar = math::area_frustrum(halflen, right(c.radius), radius_center);
                cv_areas_[j] += al;
                cv_areas_[i] += ar;
                cv_capacitance_[j] += al * c_m;
                cv_capacitance_[i] += ar * c_m;
            }
        }
        else {
            throw std::domain_error("FVM lowering encountered unsuported segment type");
        }
        ++seg_idx;
    }

    // normalize the capacitance by cv_area
    for(auto i=0u; i<size(); ++i) {
        cv_capacitance_[i] /= cv_areas_[i];
    }

    /////////////////////////////////////////////
    //  create mechanisms
    /////////////////////////////////////////////

    // FIXME : candidate for a private member function

    // for each mechanism in the cell record the indexes of the segments that
    // contain the mechanism
    std::map<std::string, std::vector<unsigned>> mech_map;

    for (unsigned i=0; i<cell.num_segments(); ++i) {
        for (const auto& mech : cell.segment(i)->mechanisms()) {
            // FIXME : Membrane has to be a proper mechanism,
            //         because it is exposed via the public interface.
            //         This if statement is bad
            if(mech.name() != "membrane") {
                mech_map[mech.name()].push_back(i);
            }
        }
    }

    // Create the mechanism implementations with the state for each mechanism
    // instance.
    // TODO : this works well for density mechanisms (e.g. ion channels), but
    // does it work for point processes (e.g. synapses)?
    for (auto& mech : mech_map) {
        // calculate the number of compartments that contain the mechanism
        auto num_comp = 0u;
        for (auto seg : mech.second) {
            num_comp += segment_index_[seg+1] - segment_index_[seg];
        }

        // build a vector of the indexes of the compartments that contain
        // the mechanism
        index_type compartment_index(num_comp);
        auto pos = 0u;
        for (auto seg : mech.second) {
            auto seg_size = segment_index_[seg+1] - segment_index_[seg];
            std::iota(
                compartment_index.data() + pos,
                compartment_index.data() + pos + seg_size,
                segment_index_[seg]
            );
            pos += seg_size;
        }

        // instantiate the mechanism
        mechanisms_.push_back(
            mechanism_catalogue::make(mech.first, voltage_, current_, compartment_index)
        );
    }

    synapse_base_ = mechanisms_.size();

    // Create the synapse mechanism implementations together with the target handles
    std::vector<std::vector<cell_lid_type>> syn_mech_map;
    std::map<std::string, std::size_t> syn_mech_indices;
    auto target_hi = target_handles.begin();

    for (const auto& syn : cell.synapses()) {
        const auto& name = syn.mechanism.name();
        std::size_t index = 0;
        if (syn_mech_indices.count(name)==0) {
            index = syn_mech_map.size();
            syn_mech_indices[name] = index;
            syn_mech_map.push_back(std::vector<cell_lid_type>{});
        }
        else {
            index = syn_mech_indices[name];
        }

        size_type comp = find_compartment_index(syn.location, graph);
        *target_hi++ = target_handle{index, syn_mech_map[index].size()};
        syn_mech_map[index].push_back(comp);
    }

    for (const auto& syni : syn_mech_indices) {
        const auto& mech_name = syni.first;

        index_type compartment_index(syn_mech_map[syni.second]);
        auto mech = mechanism_catalogue::make(mech_name, voltage_, current_, compartment_index);
        mech->set_areas(cv_areas_);
        mechanisms_.push_back(std::move(mech));
    }


    /////////////////////////////////////////////
    // build the ion species
    /////////////////////////////////////////////
    for(auto ion : mechanisms::ion_kinds()) {
        // find the compartment indexes of all compartments that have a
        // mechanism that depends on/influences ion
        std::set<int> index_set;
        for(auto& mech : mechanisms_) {
            if(mech->uses_ion(ion)) {
                for(auto idx : mech->node_index()) {
                    index_set.insert(idx);
                }
            }
        }
        std::vector<cell_lid_type> indexes(index_set.begin(), index_set.end());

        // create the ion state
        if(indexes.size()) {
            ions_.emplace(ion, index_type(indexes));
        }

        // join the ion reference in each mechanism into the cell-wide ion state
        for(auto& mech : mechanisms_) {
            if(mech->uses_ion(ion)) {
                mech->set_ion(ion, ions_[ion]);
            }
        }
    }

    // FIXME: Hard code parameters for now.
    //        Take defaults for reversal potential of sodium and potassium from
    //        the default values in Neuron.
    //        Neuron's defaults are defined in the file
    //          nrn/src/nrnoc/membdef.h
    auto all = memory::all;

    constexpr value_type DEF_vrest = -65.0; // same name as #define in Neuron

    ion_na().reversal_potential()(all)     = 115+DEF_vrest; // mV
    ion_na().internal_concentration()(all) =  10.0;         // mM
    ion_na().external_concentration()(all) = 140.0;         // mM

    ion_k().reversal_potential()(all)     = -12.0+DEF_vrest;// mV
    ion_k().internal_concentration()(all) =  54.4;          // mM
    ion_k().external_concentration()(all) =  2.5;           // mM

    ion_ca().reversal_potential()(all)     = 12.5 * std::log(2.0/5e-5);// mV
    ion_ca().internal_concentration()(all) = 5e-5;          // mM
    ion_ca().external_concentration()(all) = 2.0;           // mM

    // add the stimuli
    for(const auto& stim : cell.stimuli()) {
        auto idx = find_compartment_index(stim.location, graph);
        stimuli_.push_back( {idx, stim.clamp} );
    }

    // record probe locations by index into corresponding state vector
    auto probe_hi = probe_handles.begin();
    for (auto probe : cell.probes()) {
        auto comp = find_compartment_index(probe.location, graph);
        switch (probe.kind) {
        case probeKind::membrane_voltage:
            *probe_hi = {&fvm_cell::voltage_, comp};
            break;
        case probeKind::membrane_current:
            *probe_hi = {&fvm_cell::current_, comp};
            break;
        default:
            throw std::logic_error("unrecognized probeKind");
        }
        ++probe_hi;
    }

    // detector handles are just their corresponding compartment indices
    auto detector_hi = detector_handles.begin();
    for (auto detector : cell.detectors()) {
        auto comp = find_compartment_index(detector.location, graph);
        *detector_hi++ = comp;
    }

    // initialise mechanism and voltage state
    reset();
}

template <typename T, typename I>
void fvm_cell<T, I>::setup_matrix(T dt) {
    using memory::all;

    // convenience accesors to matrix storage
    auto l = matrix_.l();
    auto d = matrix_.d();
    auto u = matrix_.u();
    auto p = matrix_.p();
    auto rhs = matrix_.rhs();

    //  The matrix has the following layout in memory
    //  where j is the parent index of i, i.e. i<j
    //
    //      d[i] is the diagonal entry at a_ii
    //      u[i] is the upper triangle entry at a_ji
    //      l[i] is the lower triangle entry at a_ij
    //
    //       d[j] . . u[i]
    //        .  .     .
    //        .     .  .
    //       l[i] . . d[i]
    //

    d(all) = cv_areas_; // [µm^2]
    for (auto i=1u; i<d.size(); ++i) {
        auto a = 1e5*dt * face_alpha_[i];

        d[i] +=  a;
        l[i]  = -a;
        u[i]  = -a;

        // add contribution to the diagonal of parent
        d[p[i]] += a;
    }

    // the RHS of the linear system is
    //      V[i] - dt/cm*(im - ie)
    auto factor = 10.*dt; //  units: 10·ms/(F/m^2)·(mA/cm^2) ≡ mV
    for(auto i=0u; i<d.size(); ++i) {
        rhs[i] = cv_areas_[i]*(voltage_[i] - factor/cv_capacitance_[i]*current_[i]);
    }
}

template <typename T, typename I>
void fvm_cell<T, I>::reset() {
    voltage_(memory::all) = resting_potential_;
    t_ = 0.;
    for (auto& m : mechanisms_) {
        m->nrn_init();
    }
}

template <typename T, typename I>
void fvm_cell<T, I>::advance(T dt) {
    using memory::all;

    PE("current");
    current_(all) = 0.;

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

        // note: current_ in [mA/cm^2], ie in [nA], cv_areas_ in [µm^2].
        // unit scale factor: [nA/µm^2]/[mA/cm^2] = 100
        current_[loc] -= 100*ie/cv_areas_[loc];
    }
    PL();

    // solve the linear system
    PE("matrix", "setup");
    setup_matrix(dt);
    PL(); PE("solve");
    matrix_.solve();
    PL();
    voltage_(all) = matrix_.rhs();
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

