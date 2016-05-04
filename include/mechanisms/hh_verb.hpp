#pragma once

#include <cmath>
#include <limits>

#include <mechanism.hpp>
#include <mechanism_interface.hpp>
#include <algorithms.hpp>

namespace nest{ namespace mc{ namespace mechanisms{ namespace hh{

template<typename T, typename I>
class mechanism_hh : public mechanism<T, I> {
public:
    using base = mechanism<T, I>;
    using value_type  = typename base::value_type;
    using size_type   = typename base::size_type;
    using vector_type = typename base::vector_type;
    using view_type   = typename base::view_type;
    using index_type  = typename base::index_type;
    using index_view  = typename index_type::view_type;
    using indexed_view_type= typename base::indexed_view_type;
    using ion_type = typename base::ion_type;

    struct Ionna {
        view_type ena;
        view_type ina;
        index_type index;
        std::size_t memory() const { return sizeof(size_type)*index.size(); }
        std::size_t size() const { return index.size(); }
    };
    Ionna ion_na;
    struct Ionk {
        view_type ek;
        view_type ik;
        index_type index;
        std::size_t memory() const { return sizeof(size_type)*index.size(); }
        std::size_t size() const { return index.size(); }
    };
    Ionk ion_k;

    mechanism_hh(view_type vec_v, view_type vec_i, index_view node_index)
    :   base(vec_v, vec_i, node_index)
    {
        size_type num_fields = 15;

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
        gnabar          = data_(0*field_size, 1*size());
        minf            = data_(1*field_size, 2*size());
        h               = data_(2*field_size, 3*size());
        m               = data_(3*field_size, 4*size());
        gl              = data_(4*field_size, 5*size());
        gkbar           = data_(5*field_size, 6*size());
        el              = data_(6*field_size, 7*size());
        ninf            = data_(7*field_size, 8*size());
        mtau            = data_(8*field_size, 9*size());
        gna             = data_(9*field_size, 10*size());
        gk              = data_(10*field_size, 11*size());
        n               = data_(11*field_size, 12*size());
        hinf            = data_(12*field_size, 13*size());
        ntau            = data_(13*field_size, 14*size());
        htau            = data_(14*field_size, 15*size());

        // set initial values for variables and parameters
        std::fill(gnabar.data(), gnabar.data()+size(), 1.2);
        std::fill(gl.data(), gl.data()+size(), 0.0029999999999999997);
        std::fill(gkbar.data(), gkbar.data()+size(), 0.35999999999999997);
        std::fill(el.data(), el.data()+size(), -54.299999999999997);

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

    void set_ion(ionKind k, ion_type& i) override {
        using nest::mc::algorithms::index_into;
        if(k==ionKind::na) {
            ion_na.index = index_into(i.node_index(), node_index_);
            ion_na.ina = i.current();
            ion_na.ena = i.reversal_potential();
            return;
        }
        if(k==ionKind::k) {
            ion_k.index = index_into(i.node_index(), node_index_);
            ion_k.ik = i.current();
            ion_k.ek = i.reversal_potential();
            return;
        }
        throw std::domain_error(nest::mc::util::pprintf("mechanism % does not support ion type\n", name()));
    }

    void nrn_state() {
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

    void rates(const int i_, value_type v) {
        value_type ll3_, ll1_, ll0_, alpha, beta, ll2_, sum, q10;
        q10 = std::pow( 3, (celsius- 6.2999999999999998)/ 10);
        //ll2_ =  -v+ 40;
        ll2_ =  -(v+ 40);
        ll0_ = ll2_/(exp(ll2_/ 10)- 1);
        alpha =  0.10000000000000001*ll0_;
        beta =  4*exp( -(v+ 65)/ 18);

        std::cout << "v  " << v << "\n";
        //std::cout << "m (alpha, beta) : " << alpha << " " << beta << "\n";

        sum = alpha+beta;
        mtau[i_] =  1/q10*sum;
        minf[i_] = alpha/sum;

        //////////////////////////////////////////////////////////

        alpha =  0.070000000000000007*exp( -(v+ 65)/ 20);
        beta =  1/(exp( -(v+ 35)/ 10)+ 1);

        //std::cout << "h (alpha, beta) : " << alpha << " " << beta << "\n";

        sum = alpha+beta;
        htau[i_] =  1/q10*sum;
        hinf[i_] = alpha/sum;

        //////////////////////////////////////////////////////////

        //ll3_ =  -v+ 55; // TODO : inlining is breaking
        ll3_ =  -(v+ 55);
        ll1_ = ll3_/(exp(ll3_/ 10)- 1);
        alpha =  0.01*ll1_;
        beta =  0.125*exp( -(v+ 65)/ 80);

        //std::cout << "n (alpha, beta) : " << alpha << " " << beta << "\n";

        sum = alpha+beta;
        ntau[i_] =  1/q10*sum;
        ninf[i_] = alpha/sum;
    }

    void nrn_current() {
        const indexed_view_type ion_ek(ion_k.ek, ion_k.index);
        indexed_view_type ion_ina(ion_na.ina, ion_na.index);
        const indexed_view_type ion_ena(ion_na.ena, ion_na.index);
        indexed_view_type ion_ik(ion_k.ik, ion_k.index);
        indexed_view_type vec_i(vec_i_, node_index_);
        const indexed_view_type vec_v(vec_v_, node_index_);
        int n_ = node_index_.size();
        for(int i_=0; i_<n_; ++i_) {
            value_type ek = ion_ek[i_];
            value_type ena = ion_ena[i_];
            value_type v = vec_v[i_];
            value_type il, ina, ik, current_;
            gna[i_] = gnabar[i_]*m[i_]*m[i_]*m[i_]*h[i_];
            ina = gna[i_]*(v-ena);
            current_ = ina;
            gk[i_] = gkbar[i_]*n[i_]*n[i_]*n[i_]*n[i_];
            ik = gk[i_]*(v-ek);
            current_ = current_+ik;
            il = gl[i_]*(v-el[i_]);
            current_ = current_+il;
            ion_ina[i_] += ina;
            ion_ik[i_] += ik;
            vec_i[i_] += current_;
            printf("i = (l+k+na) %18.16f = %18.16f %18.16f %18.16f\n",
                   current_, il, ik, ina);
        }
    }

    void nrn_init() {
        const indexed_view_type vec_v(vec_v_, node_index_);
        int n_ = node_index_.size();
        for(int i_=0; i_<n_; ++i_) {
            value_type v = vec_v[i_];
            rates(i_, v);
            m[i_] = minf[i_];
            h[i_] = hinf[i_];
            n[i_] = ninf[i_];
            printf("initial conditions for m, h, n : %16.14f %16.14f %16.14f\n", m[0], h[0], n[0]);
        }
    }

    vector_type data_;
    view_type gnabar;
    view_type minf;
    view_type h;
    view_type m;
    view_type gl;
    view_type gkbar;
    view_type el;
    view_type ninf;
    view_type mtau;
    view_type gna;
    view_type gk;
    view_type n;
    view_type hinf;
    view_type ntau;
    view_type htau;
    value_type t = std::numeric_limits<value_type>::quiet_NaN();
    value_type dt = std::numeric_limits<value_type>::quiet_NaN();
    value_type celsius = 6.3; // TODO change from 37

    using base::vec_v_;
    using base::vec_i_;
    using base::node_index_;

};

template<typename T, typename I>
struct helper : public mechanism_helper<T, I> {
    using base = mechanism_helper<T, I>;
    using index_view  = typename base::index_view;
    using view_type  = typename base::view_type;
    using mechanism_ptr_type  = typename base::mechanism_ptr_type;
    using mechanism_type = mechanism_hh<T, I>;

    std::string
    name() const override
    {
        return "hh";
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
