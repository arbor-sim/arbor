#pragma once

#include <algorithm>
#include <map>
#include <string>
#include <vector>

#include <algorithms.hpp>
#include <cell.hpp>
#include <ion.hpp>
#include <math.hpp>
#include <matrix.hpp>
#include <mechanism.hpp>
#include <mechanism_interface.hpp>
#include <util.hpp>
#include <segment.hpp>

#include <vector/include/Vector.hpp>

namespace nest {
namespace mc {
namespace fvm {

template <typename T, typename I>
class fvm_cell {
    public :

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

    /// the container used for values
    using vector_type = memory::HostVector<value_type>;
    /// view into value container
    using vector_view = typename vector_type::view_type;

    /// constructor
    fvm_cell(nest::mc::cell const& cell);

    /// build the matrix for a given time step
    void setup_matrix(value_type dt);

    /// TODO this should be const
    /// which requires const_view in the vector library
    matrix_type& jacobian() {
        return matrix_;
    }

    /// TODO this should be const
    /// return list of CV areas in :
    ///          um^2
    ///     1e-6.mm^2
    ///     1e-8.cm^2
    vector_view cv_areas() {
        return cv_areas_;
    }

    /// TODO this should be const
    /// return the capacitance of each CV surface
    /// this is the total capacitance, not per unit area,
    /// i.e. equivalent to sigma_i * c_m
    vector_view cv_capacitance() {
        return cv_capacitance_;
    }

    /// return the voltage in each CV
    vector_view voltage() {
        return voltage_;
    }

    std::size_t size() const {
        return matrix_.size();
    }

    /// return reference to in iterable container of the mechanisms
    std::vector<mechanism_type>& mechanisms() {
        return mechanisms_;
    }

    /// return reference to list of ions
    //std::map<mechanisms::ionKind, ion_type> ions_;
    std::map<mechanisms::ionKind, ion_type>& ions() {
        return ions_;
    }
    std::map<mechanisms::ionKind, ion_type> const& ions() const {
        return ions_;
    }

    /// return reference to sodium ion
    ion_type& ion_na() {
        return ions_[mechanisms::ionKind::na];
    }
    ion_type const& ion_na() const {
        return ions_[mechanisms::ionKind::na];
    }

    /// return reference to calcium ion
    ion_type& ion_ca() {
        return ions_[mechanisms::ionKind::ca];
    }
    ion_type const& ion_ca() const {
        return ions_[mechanisms::ionKind::ca];
    }

    /// return reference to pottasium ion
    ion_type& ion_k() {
        return ions_[mechanisms::ionKind::k];
    }
    ion_type const& ion_k() const {
        return ions_[mechanisms::ionKind::k];
    }

    /// make a time step
    void advance(value_type dt);

    /// set initial states
    void initialize();

    private:

    /// current time
    value_type t_ = value_type{0};

    /// the linear system for implicit time stepping of cell state
    matrix_type matrix_;

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

    /// the set of mechanisms present in the cell
    std::vector<mechanism_type> mechanisms_;

    /// the ion species
    std::map<mechanisms::ionKind, ion_type> ions_;
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
    auto const& segment_index = graph.segment_index;

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
                auto i = segment_index[seg_idx] + c.index;
                auto j = parent_index[i];

                auto radius_center = math::mean(c.radius);
                auto area_face = math::area_circle( radius_center );
                face_alpha_[i] = area_face  / (c_m * r_L * c.length);
                cv_capacitance_[i] = c_m;

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

    std::cout << "capacitance " << cv_capacitance_ << "\n";

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
            num_comp += segment_index[seg+1] - segment_index[seg];
        }

        // build a vector of the indexes of the compartments that contain
        // the mechanism
        std::vector<int> compartment_index(num_comp);
        auto pos = 0u;
        for(auto seg : mech.second) {
            auto seg_size = segment_index[seg+1] - segment_index[seg];
            std::iota(
                compartment_index.data() + pos,
                compartment_index.data() + pos + seg_size,
                segment_index[seg]
            );
            pos += seg_size;
        }

        // instantiate the mechanism
        index_view node_index(compartment_index.data(), compartment_index.size());
        mechanisms_.push_back(
            helper->new_mechanism(voltage_, current_, node_index)
        );
    }

    /////////////////////////////////////////////
    // build the ion species
    // FIXME : this should be a private member function
    /////////////////////////////////////////////
    for(auto ion : mechanisms::ion_kinds()) {
        auto indexes =
            std::make_pair(std::vector<int>(size()), std::vector<int>(size()));
        auto ends =
            std::make_pair(indexes.first.begin(), indexes.second.begin());

        // after the loop the range
        //      [indexes.first.begin(), ends.first)
        // will hold the indexes of the compartments that require ion
        for(auto& mech : mechanisms_) {
            if(mech->uses_ion(ion)) {
                ends.second =
                    std::set_union(
                        mech->node_index().begin(), mech->node_index().end(),
                        indexes.first.begin(), ends.first,
                        indexes.second.begin()
                    );
                std::swap(indexes.first, indexes.second);
                std::swap(ends.first, ends.second);
            }
        }

        // create the ion state
        if(auto n = std::distance(indexes.first.begin(), ends.first)) {
            ions_.emplace(ion, index_view(indexes.first.data(), n));
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
    using memory::all;

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
        // TODO get this right
        // probably requires scaling a by cv_areas_[i] and cv_areas_[p[i]]
        auto a = 1e7*dt * face_alpha_[i];

        d[i] +=  a;
        l[i]  = -a;
        u[i]  = -a;

        // add contribution to the diagonal of parent
        d[p[i]] += a;
    }
    //std::cout << "d " << d << " l " << l << " u " << u << "\n";

    // the RHS of the linear system is
    //      V[i] - dt/cm*(im - ie)
    auto factor = 10.*dt;
    for(auto i=0u; i<d.size(); ++i) {
        //rhs[i] = voltage_[i] - factor/cv_capacitance_[i]*current_[i];
        rhs[i] = cv_areas_[i]*(voltage_[i] - factor/cv_capacitance_[i]*current_[i]);
    }
}

template <typename T, typename I>
void fvm_cell<T, I>::initialize()
{
    // initialize mechanism states
    for(auto& m : mechanisms_) {
        m->nrn_init();
    }
}

template <typename T, typename I>
void fvm_cell<T, I>::advance(T dt)
{
    using memory::all;

    current_(all) = 0.;

    // update currents
    for(auto& m : mechanisms_) {
        m->set_params(t_, dt);
        m->nrn_current();
    }

    // the factor scales the injected current to 10^2.nA
    auto ie_factor = 100.;
    auto ie = 0.1;
    auto loc = size()-1;
    //auto loc = 0;
    if(t_>=5. && t_<8.)
      current_[loc] -= ie_factor*ie/cv_areas_[loc];

    //std::cout << "t " << t_ << " current " << current_;

    // set matrix diagonals and rhs
    setup_matrix(dt);

    //printf("rhs %18.14f    d %18.14f\n", matrix_.rhs()[0], matrix_.d()[0]);

    // solve the linear system
    matrix_.solve();

    voltage_(all) = matrix_.rhs();

    //printf("v solve %18.14f\n", voltage_[0]);

    //std::cout << " v " << voltage_ << "\n";

    // update states
    for(auto& m : mechanisms_) {
        m->nrn_state();
    }

    t_ += dt;
    //std::cout << "******************\n";
}

} // namespace fvm
} // namespace mc
} // namespace nest


