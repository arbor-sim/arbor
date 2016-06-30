#pragma once

#include <algorithm>
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
#include <mechanism_interface.hpp>
#include <segment.hpp>
#include <stimulus.hpp>
#include <util.hpp>
#include <profiling/profiler.hpp>

#include <vector/include/Vector.hpp>
#include <mechanisms/expsyn.hpp>


namespace nest {
namespace mc {
namespace fvm {

template <typename T, typename I>
class fvm_cell {
public:

    fvm_cell() = default;

    /// the real number type
    using value_type = T;
    /// the integral index type
    using size_type  = I;

    /// the type used to store matrix information
    using matrix_type = matrix<value_type, size_type>;

    /// mechanism type
    using mechanism_type =
        nest::mc::mechanisms::mechanism_ptr<value_type, size_type>;

    /// ion species storage
    using ion_type = mechanisms::ion<value_type, size_type>;

    /// the container used for indexes
    using index_type = memory::HostVector<size_type>;
    /// view into index container
    using index_view = typename index_type::view_type;
    using const_index_view = typename index_type::const_view_type;

    /// the container used for values
    using vector_type = memory::HostVector<value_type>;
    /// view into value container
    using vector_view = typename vector_type::view_type;
    using const_vector_view = typename vector_type::const_view_type;

    /// constructor
    fvm_cell(nest::mc::cell const& cell);

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

    std::size_t size() const { return matrix_.size(); }

    /// return reference to in iterable container of the mechanisms
    std::vector<mechanism_type>& mechanisms() { return mechanisms_; }

    /// return reference to list of ions
    //std::map<mechanisms::ionKind, ion_type> ions_;
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

    /// make a time step
    void advance(value_type dt);

    /// pass an event to the appropriate synapse and call net_receive
    void apply_event(postsynaptic_spike_event e) {
        mechanisms_[synapse_index_]->net_receive(e.target, e.weight);
    }

    mechanism_type& synapses() {
        return mechanisms_[synapse_index_];
    }

    /// set initial states
    void initialize();

    /// returns the compartment index of a segment location
    int compartment_index(segment_location loc) const;

    /// returns voltage at a segment location
    value_type voltage(segment_location loc) const;

    value_type time() const { return t_; }

    value_type probe(uint32_t i) const {
        auto p = probes_[i];
        return (this->*p.first)[p.second];
    }

    std::size_t num_probes() const { return probes_.size(); }

private:

    /// current time
    value_type t_ = value_type{0};

    /// the linear system for implicit time stepping of cell state
    matrix_type matrix_;

    /// index for fast lookup of compartment index ranges of segments
    index_type segment_index_;

    /// cv_areas_[i] is the surface area of CV i
    vector_type cv_areas_;

    /// alpha_[i] is the following value at the CV face between
    /// CV i and its parent, required when constructing linear system
    ///     face_alpha_[i] = area_face  / (c_m * r_L * delta_x);
    vector_type face_alpha_;

    /// cv_capacitance_[i] is the capacitance of CV i per unit area (i.e. c_m)
    vector_type cv_capacitance_;

    /// the average current over the surface of each CV
    /// current_ = i_m - i_e
    /// so the total current over the surface of CV i is
    ///     current_[i] * cv_areas_
    vector_type current_;

    /// the potential in mV in each CV
    vector_type voltage_;

    /// synapses
    using synapse_type =
        mechanisms::ExpSyn::mechanism_ExpSyn<value_type, size_type>;
    std::size_t synapse_index_;

    /// the set of mechanisms present in the cell
    std::vector<mechanism_type> mechanisms_;

    /// the ion species
    std::map<mechanisms::ionKind, ion_type> ions_;

    std::vector<std::pair<uint32_t, i_clamp>> stimulii_;

    std::vector<std::pair<const vector_type fvm_cell::*, uint32_t>> probes_;
};

////////////////////////////////////////////////////////////////////////////////
//////////////////////////////// Implementation ////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template <typename T, typename I>
fvm_cell<T, I>::fvm_cell(nest::mc::cell const& cell)
:   cv_areas_      {cell.num_compartments(), T(0)}
,   face_alpha_    {cell.num_compartments(), T(0)}
,   cv_capacitance_{cell.num_compartments(), T(0)}
,   current_       {cell.num_compartments(), T(0)}
,   voltage_       {cell.num_compartments(), T(0)}
{
    using util::left;
    using util::right;

    // TODO: potential code stink
    // matrix_ is a member, but it is not initialized with the other members
    // above because it requires the parent_index, which is calculated
    // "on the fly" by cell.model().
    // cell.model() is quite expensive, and the information it calculates is
    // used elsewhere, so we defer the intialization to inside the constructor
    // body.
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
    std::map<std::string, std::vector<int>> mech_map;

    for(auto i=0; i<cell.num_segments(); ++i) {
        for(const auto& mech : cell.segment(i)->mechanisms()) {
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
    for(auto& mech : mech_map) {
        auto& helper = nest::mc::mechanisms::get_mechanism_helper(mech.first);

        // calculate the number of compartments that contain the mechanism
        auto num_comp = 0u;
        for(auto seg : mech.second) {
            num_comp += segment_index_[seg+1] - segment_index_[seg];
        }

        // build a vector of the indexes of the compartments that contain
        // the mechanism
        index_type compartment_index(num_comp);
        auto pos = 0u;
        for(auto seg : mech.second) {
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
            helper->new_mechanism(voltage_, current_, compartment_index)
        );
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
        std::vector<int> indexes(index_set.begin(), index_set.end());

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

    // add the stimulii
    for(const auto& stim : cell.stimulii()) {
        auto idx = find_compartment_index(stim.location, graph);
        stimulii_.push_back( {idx, stim.clamp} );
    }

    // add the synapses
    std::vector<size_type> synapse_indexes;
    synapse_indexes.reserve(cell.synapses().size());
    for(auto loc : cell.synapses()) {
        synapse_indexes.push_back(
            find_compartment_index(loc, graph)
        );
    }

    mechanisms_.push_back(
        mechanisms::make_mechanism<synapse_type>(
            voltage_, current_, index_view(synapse_indexes)
        )
    );
    synapse_index_ = mechanisms_.size()-1;
    // don't forget to give point processes access to cv_areas_
    mechanisms_[synapse_index_]->set_areas(cv_areas_);

    // record probe locations by index into corresponding state vector
    for (auto probe : cell.probes()) {
        uint32_t comp = find_compartment_index(probe.location, graph);
        switch (probe.kind) {
            case probeKind::membrane_voltage:
                probes_.push_back({&fvm_cell::voltage_, comp});
                break;
            case probeKind::membrane_current:
                probes_.push_back({&fvm_cell::current_, comp});
                break;
            default:
                throw std::logic_error("unrecognized probeKind");
        }
    }
}

template <typename T, typename I>
void fvm_cell<T, I>::setup_matrix(T dt)
{
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
    //d(all) = 1.0;
    d(all) = cv_areas_;
    for(auto i=1u; i<d.size(); ++i) {
        auto a = 1e5*dt * face_alpha_[i];

        d[i] +=  a;
        l[i]  = -a;
        u[i]  = -a;

        // add contribution to the diagonal of parent
        d[p[i]] += a;
    }

    // the RHS of the linear system is
    //      V[i] - dt/cm*(im - ie)
    auto factor = 10.*dt;
    for(auto i=0u; i<d.size(); ++i) {
        //rhs[i] = voltage_[i] - factor/cv_capacitance_[i]*current_[i];
        rhs[i] = cv_areas_[i]*(voltage_[i] - factor/cv_capacitance_[i]*current_[i]);
    }
}
template <typename T, typename I>
int fvm_cell<T, I>::compartment_index(segment_location loc) const
{
    EXPECTS(loc.segment < segment_index_.size());

    const auto seg = loc.segment;

    auto first = segment_index_[seg];
    auto n = segment_index_[seg+1] - first;
    auto index = std::floor(n*loc.position);
    return index<n ? first+index : first+n-1;
}

template <typename T, typename I>
T fvm_cell<T, I>::voltage(segment_location loc) const
{
    return voltage_[compartment_index(loc)];
}

template <typename T, typename I>
void fvm_cell<T, I>::initialize()
{
    t_ = 0.;

    for(auto& m : mechanisms_) {
        m->nrn_init();
    }
}

template <typename T, typename I>
void fvm_cell<T, I>::advance(T dt)
{
    using memory::all;

        mc::util::profiler_enter("current");
    current_(all) = 0.;

    // update currents from ion channels
    for(auto& m : mechanisms_) {
            mc::util::profiler_enter(m->name().c_str());
        m->set_params(t_, dt);
        m->nrn_current();
            mc::util::profiler_leave();
    }

    // add current contributions from stimulii
    for(auto& stim : stimulii_) {
        auto ie = stim.second.amplitude(t_);
        auto loc = stim.first;

        // the factor of 100 scales the injected current to 10^2.nA
        current_[loc] -= 100.*ie/cv_areas_[loc];
    }
        mc::util::profiler_leave();

        mc::util::profiler_enter("matrix", "setup");
    // solve the linear system
    setup_matrix(dt);
        mc::util::profiler_leave(); mc::util::profiler_enter("solve");
    matrix_.solve();
        mc::util::profiler_leave();
    voltage_(all) = matrix_.rhs();
        mc::util::profiler_leave();

        mc::util::profiler_enter("state");
    // integrate state of gating variables etc.
    for(auto& m : mechanisms_) {
            mc::util::profiler_enter(m->name().c_str());
        m->nrn_state();
            mc::util::profiler_leave();
    }
        mc::util::profiler_leave();

    t_ += dt;
}

} // namespace fvm
} // namespace mc
} // namespace nest


