#pragma once

#include <cmath>
#include <limits>

#include <mechanism.hpp>
#include <mechanism_interface.hpp>
#include <algorithms.hpp>

namespace nest {
namespace mc {
namespace mechanisms {
namespace expsyn_test {

template<typename T, typename I>
class mechanism_expsyn : public mechanism<T, I> {
public:
    using base = mechanism<T, I>;
    using value_type  = typename base::value_type;
    using size_type   = typename base::size_type;
    using vector_type = typename base::vector_type;
    using view_type   = typename base::view_type;
    using index_type  = typename base::index_type;
    using index_view  = typename base::index_view;
    using const_index_view  = typename base::const_index_view;
    using indexed_view_type= typename base::indexed_view_type;
    using ion_type = typename base::ion_type;


    mechanism_expsyn(view_type vec_v, view_type vec_i, const_index_view node_index)
    :   base(vec_v, vec_i, node_index)
    {
        size_type num_fields = 3;

        // calculate the padding required to maintain proper alignment of sub arrays
        auto alignment  = data_.alignment();
        auto field_size_in_bytes = sizeof(value_type)*size();
        auto remainder  = field_size_in_bytes % alignment;
        auto padding    = remainder ? (alignment - remainder)/sizeof(value_type) : 0;
        auto field_size = size()+padding;

        // allocate memory
        data_ = vector_type(field_size * num_fields);
        data_(memory::all) = std::numeric_limits<value_type>::quiet_NaN();

        // asign the sub-arrays
        tau             = data_(0*field_size, 1*size());
        e               = data_(1*field_size, 2*size());
        g               = data_(2*field_size, 3*size());

        // set initial values for variables and parameters
        std::fill(tau.data(), tau.data()+size(), 2);
        std::fill(e.data(), e.data()+size(), 0);

    }

    using base::size;

    std::size_t memory() const override {
        auto s = std::size_t{0};
        s += data_.size()*sizeof(value_type);
        return s;
    }

    void set_params(value_type t_, value_type dt_) override {
        t = t_;
        dt = dt_;
    }

    std::string name() const override {
        return "expsyn";
    }

    mechanismKind kind() const override {
        return mechanismKind::point;
    }

    bool uses_ion(ionKind k) const override {
        switch(k) {
            case ionKind::na : return false;
            case ionKind::ca : return false;
            case ionKind::k  : return false;
        }
        return false;
    }

    void set_ion(ionKind k, ion_type& i) override {
        using nest::mc::algorithms::index_into;
        throw std::domain_error(nest::mc::util::pprintf("mechanism % does not support ion type\n", name()));
    }

    void nrn_init() override {
        int n_ = node_index_.size();
        for(int i_=0; i_<n_; ++i_) {
            g[i_] =  0;
        }
    }

    void net_receive(int i_, value_type weight) override {
        g[i_] = g[i_]+weight;
    }

    void nrn_current() override {
        const indexed_view_type vec_area(vec_area_, node_index_);
        indexed_view_type vec_i(vec_i_, node_index_);
        const indexed_view_type vec_v(vec_v_, node_index_);
        int n_ = node_index_.size();
        for(int i_=0; i_<n_; ++i_) {
            value_type area_ = vec_area[i_];
            value_type v = vec_v[i_];
            value_type current_, i;
            i = g[i_]*(v-e[i_]);
            current_ = i;
            current_ = ( 100*current_)/area_;
            vec_i[i_] += current_;
        }
    }

    void nrn_state() override {
        int n_ = node_index_.size();
        for(int i_=0; i_<n_; ++i_) {
            value_type ba_, a_;
            a_ =  -1/tau[i_];
            ba_ =  0/a_;
            g[i_] =  -ba_+(g[i_]+ba_)*exp(a_*dt);
        }
    }

    vector_type data_;
    view_type tau;
    view_type e;
    view_type g;
    value_type t = 0;
    value_type dt = 0;

    using base::vec_v_;
    using base::vec_i_;
    using base::vec_area_;
    using base::node_index_;

};

template<typename T, typename I>
struct helper : public mechanism_helper<T, I> {
    using base = mechanism_helper<T, I>;
    using index_view  = typename base::index_view;
    using view_type  = typename base::view_type;
    using mechanism_ptr_type  = typename base::mechanism_ptr_type;
    using mechanism_type = mechanism_expsyn<T, I>;

    std::string
    name() const override
    {
        return "expsyn";
    }

    mechanism_ptr<T,I>
    new_mechanism(view_type vec_v, view_type vec_i, index_view node_index) const override
    {
        return nest::mc::mechanisms::make_mechanism<mechanism_type>(vec_v, vec_i, node_index);
    }

    void
    set_parameters(mechanism_ptr_type&, parameter_list const&) const override
    {
    }

};

}}}} // namespaces
