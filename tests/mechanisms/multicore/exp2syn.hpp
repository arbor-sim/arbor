#pragma once

#include <cmath>
#include <limits>

#include <mechanism.hpp>
#include <algorithms.hpp>
#include <util/pprintf.hpp>

namespace nest{ namespace mc{ namespace mechanisms{ namespace exp2syn_proto{

template<class Backend>
class mechanism_exp2syn : public mechanism<Backend> {
public:
    using base = mechanism<Backend>;
    using value_type  = typename base::value_type;
    using size_type   = typename base::size_type;

    using array = typename base::array;
    using iarray  = typename base::iarray;
    using view   = typename base::view;
    using iview  = typename base::iview;
    using const_iview = typename base::const_iview;
    using indexed_view_type= typename base::indexed_view_type;
    using ion_type = typename base::ion_type;


    mechanism_exp2syn(view vec_v, view vec_i, array&& weights, iarray&& node_index)
    :   base(vec_v, vec_i, std::move(node_index))
    {
        size_type num_fields = 6;

        // calculate the padding required to maintain proper alignment of sub arrays
        auto alignment  = data_.alignment();
        auto field_size_in_bytes = sizeof(value_type)*size();
        auto remainder  = field_size_in_bytes % alignment;
        auto padding    = remainder ? (alignment - remainder)/sizeof(value_type) : 0;
        auto field_size = size()+padding;

        // allocate memory
        data_ = array(field_size*num_fields, std::numeric_limits<value_type>::quiet_NaN());

        // asign the sub-arrays
        tau1            = data_(0*field_size, 1*size());
        A               = data_(1*field_size, 2*size());
        e               = data_(2*field_size, 3*size());
        tau2            = data_(3*field_size, 4*size());
        factor          = data_(4*field_size, 5*size());
        B               = data_(5*field_size, 6*size());

        // set initial values for variables and parameters
        std::fill(tau1.data(), tau1.data()+size(), 0.5);
        std::fill(e.data(), e.data()+size(), 0);
        std::fill(tau2.data(), tau2.data()+size(), 2);

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
        return "exp2syn";
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

    void set_ion(ionKind k, ion_type& i, std::vector<size_type>const& index) override {
        using nest::mc::algorithms::index_into;
        throw std::domain_error(nest::mc::util::pprintf("mechanism % does not support ion type\n", name()));
    }

    void nrn_current() override {
        const indexed_view_type vec_v(vec_v_, node_index_);
        indexed_view_type vec_i(vec_i_, node_index_);
        int n_ = node_index_.size();
        for(int i_=0; i_<n_; ++i_) {
            value_type v = vec_v[i_];
            value_type current_, i;
            i = (B[i_]-A[i_])*(v-e[i_]);
            current_ = i;
            vec_i[i_] += current_;
        }
    }

    void nrn_init() override {
        int n_ = node_index_.size();
        for(int i_=0; i_<n_; ++i_) {
            value_type tp;
            A[i_] =  0;
            B[i_] =  0;
            tp = (tau1[i_]*tau2[i_])/(tau2[i_]-tau1[i_])*log(tau2[i_]/tau1[i_]);
            factor[i_] =  1/( -exp( -tp/tau1[i_])+exp( -tp/tau2[i_]));
        }
    }

    void net_receive(int i_, value_type weight) override {
        A[i_] = A[i_]+weight*factor[i_];
        B[i_] = B[i_]+weight*factor[i_];
    }

    void nrn_state() override {
        int n_ = node_index_.size();
        for(int i_=0; i_<n_; ++i_) {
            value_type ba_, a_;
            a_ =  -1/tau1[i_];
            ba_ =  0/a_;
            A[i_] =  -ba_+(A[i_]+ba_)*exp(a_*dt);
            a_ =  -1/tau2[i_];
            ba_ =  0/a_;
            B[i_] =  -ba_+(B[i_]+ba_)*exp(a_*dt);
        }
    }

    array data_;
    view tau1;
    view A;
    view e;
    view tau2;
    view factor;
    view B;
    value_type dt = 0;
    value_type t = 0;

    using base::vec_v_;
    using base::vec_i_;
    using base::node_index_;

};

}}}} // namespaces
