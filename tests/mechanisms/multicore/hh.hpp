#pragma once

#include <cmath>
#include <limits>

#include <mechanism.hpp>
#include <algorithms.hpp>
#include <util/pprintf.hpp>

namespace nest{ namespace mc{ namespace mechanisms{ namespace hh_proto{

template<class Backend>
class mechanism_hh : public mechanism<Backend> {
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

    struct Ionna {
        view ena;
        view ina;
        iarray index;
        std::size_t memory() const { return sizeof(size_type)*index.size(); }
        std::size_t size() const { return index.size(); }
    };
    Ionna ion_na;
    struct Ionk {
        view ek;
        view ik;
        iarray index;
        std::size_t memory() const { return sizeof(size_type)*index.size(); }
        std::size_t size() const { return index.size(); }
    };
    Ionk ion_k;

    mechanism_hh(view vec_v, view vec_i, array&& weights, iarray&& node_index)
    :   base(vec_v, vec_i, std::move(node_index))
    {
        size_type num_fields = 16;

        // calculate the padding required to maintain proper alignment of sub arrays
        auto alignment  = data_.alignment();
        auto field_size_in_bytes = sizeof(value_type)*size();
        auto remainder  = field_size_in_bytes % alignment;
        auto padding    = remainder ? (alignment - remainder)/sizeof(value_type) : 0;
        auto field_size = size()+padding;

        // allocate memory
        data_ = array(field_size*num_fields, std::numeric_limits<value_type>::quiet_NaN());

        // asign the sub-arrays
        ntau            = data_(0*field_size, 1*size());
        htau            = data_(1*field_size, 2*size());
        gna             = data_(2*field_size, 3*size());
        mtau            = data_(3*field_size, 4*size());
        ninf            = data_(4*field_size, 5*size());
        minf            = data_(5*field_size, 6*size());
        hinf            = data_(6*field_size, 7*size());
        gk              = data_(7*field_size, 8*size());
        el              = data_(8*field_size, 9*size());
        n               = data_(9*field_size, 10*size());
        h               = data_(10*field_size, 11*size());
        weights_        = data_(11*field_size, 12*size());
        gkbar           = data_(12*field_size, 13*size());
        m               = data_(13*field_size, 14*size());
        gl              = data_(14*field_size, 15*size());
        gnabar          = data_(15*field_size, 16*size());

        // add the user-supplied weights for converting from current density
        // to per-compartment current in nA
        memory::copy(weights, weights_(0, size()));

        // set initial values for variables and parameters
        std::fill(el.data(), el.data()+size(), -54.299999999999997);
        std::fill(gkbar.data(), gkbar.data()+size(), 0.035999999999999997);
        std::fill(gl.data(), gl.data()+size(), 0.00029999999999999997);
        std::fill(gnabar.data(), gnabar.data()+size(), 0.12);
    }

    using base::size;

    std::size_t memory() const override {
        auto s = std::size_t{0};
        s += data_.size()*sizeof(value_type);
        s += ion_na.memory();
        s += ion_k.memory();
        return s;
    }

    void set_params(value_type t_, value_type dt_) override {
        t = t_;
        dt = dt_;
    }

    std::string name() const override {
        return "hh";
    }

    mechanismKind kind() const override {
        return mechanismKind::density;
    }

    bool uses_ion(ionKind k) const override {
        switch(k) {
            case ionKind::na : return true;
            case ionKind::ca : return false;
            case ionKind::k  : return true;
        }
        return false;
    }

    void set_ion(ionKind k, ion_type& i, std::vector<size_type>const& index) override {
        using nest::mc::algorithms::index_into;
        if(k==ionKind::na) {
            ion_na.index = iarray(memory::make_const_view(index));
            ion_na.ina = i.current();
            ion_na.ena = i.reversal_potential();
            return;
        }
        if(k==ionKind::k) {
            ion_k.index = iarray(memory::make_const_view(index));
            ion_k.ik = i.current();
            ion_k.ek = i.reversal_potential();
            return;
        }
        throw std::domain_error(nest::mc::util::pprintf("mechanism % does not support ion type\n", name()));
    }

    void nrn_current() override {
        const indexed_view_type ion_ek(ion_k.ek, ion_k.index);
        indexed_view_type vec_i(vec_i_, node_index_);
        const indexed_view_type vec_v(vec_v_, node_index_);
        indexed_view_type ion_ina(ion_na.ina, ion_na.index);
        indexed_view_type ion_ik(ion_k.ik, ion_k.index);
        const indexed_view_type ion_ena(ion_na.ena, ion_na.index);
        int n_ = node_index_.size();
        for(int i_=0; i_<n_; ++i_) {
            value_type ek = ion_ek[i_];
            value_type v = vec_v[i_];
            value_type ena = ion_ena[i_];
            value_type il, current_, ina, ik;
            gna[i_] = gnabar[i_]*m[i_]*m[i_]*m[i_]*h[i_];
            ina = gna[i_]*(v-ena);
            current_ = ina;
            gk[i_] = gkbar[i_]*n[i_]*n[i_]*n[i_]*n[i_];
            ik = gk[i_]*(v-ek);
            current_ = current_+ik;
            il = gl[i_]*(v-el[i_]);
            current_ = current_+il;
            current_ = weights_[i_]*current_;
            vec_i[i_] += current_;
            ion_ina[i_] += ina;
            ion_ik[i_] += ik;
        }
    }

    void nrn_state() override {
        const indexed_view_type vec_v(vec_v_, node_index_);
        int n_ = node_index_.size();
        for(int i_=0; i_<n_; ++i_) {
            value_type v = vec_v[i_];
            rates(i_, v);
            m[i_] = minf[i_]+(m[i_]-minf[i_])*exp( -dt/mtau[i_]);
            h[i_] = hinf[i_]+(h[i_]-hinf[i_])*exp( -dt/htau[i_]);
            n[i_] = ninf[i_]+(n[i_]-ninf[i_])*exp( -dt/ntau[i_]);
        }
    }

    void rates(int i_, value_type v) {
        value_type ll0_, sum, ll3_, q10, ll2_, ll1_, beta, alpha;
        q10 = std::pow( 3, (celsius- 6.3)/ 10);
        ll2_ =  -(v+ 40);
        ll0_ = ll2_/(exp(ll2_/ 10)- 1);
        alpha =  0.1*ll0_;
        beta =  4*exp( -((v+ 65))/ 18);
        sum = alpha+beta;
        mtau[i_] =  1/(q10*sum);
        minf[i_] = alpha/sum;
        alpha =  0.07*exp( -((v+ 65))/ 20);
        beta =  1/(exp( -((v+ 35))/ 10)+ 1);
        sum = alpha+beta;
        htau[i_] =  1/(q10*sum);
        hinf[i_] = alpha/sum;
        ll3_ =  -(v+ 55);
        ll1_ = ll3_/(exp(ll3_/ 10)- 1);
        alpha =  0.01*ll1_;
        beta =  0.125*exp( -((v+ 65))/ 80);
        sum = alpha+beta;
        ntau[i_] =  1/(q10*sum);
        ninf[i_] = alpha/sum;
    }

    void nrn_init() override {
        const indexed_view_type vec_v(vec_v_, node_index_);
        int n_ = node_index_.size();
        for(int i_=0; i_<n_; ++i_) {
            value_type v = vec_v[i_];
            rates(i_, v);
            m[i_] = minf[i_];
            h[i_] = hinf[i_];
            n[i_] = ninf[i_];
        }
    }

    array data_;
    view ntau;
    view htau;
    view gna;
    view mtau;
    view ninf;
    view minf;
    view hinf;
    view gk;
    view el;
    view n;
    view h;
    view weights_;
    view gkbar;
    view m;
    view gl;
    view gnabar;
    value_type celsius = 6.2999999999999998;
    value_type t = 0;
    value_type dt = 0;

    using base::vec_v_;
    using base::vec_i_;
    using base::node_index_;

};

}}}} // namespaces
