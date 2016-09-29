#pragma once

#include <cmath>
#include <limits>

#include <mechanism.hpp>
#include <mechanism_interface.hpp>

namespace nest{ namespace mc{ namespace mechanisms{ namespace gpu{ namespace hh{

    template <typename T, typename I>
    struct hh_ParamPack {
        // array parameters
        T* gnabar;
        T* minf;
        T* h;
        T* m;
        T* gl;
        T* gkbar;
        T* el;
        T* ninf;
        T* mtau;
        T* gna;
        T* gk;
        T* n;
        T* hinf;
        T* ntau;
        T* htau;
        // scalar parameters
        T dt;
        T t;
        T celsius;
        // ion channel dependencies
        T* ion_ena;
        T* ion_ina;
        I* ion_na_idx_;
        T* ion_ek;
        T* ion_ik;
        I* ion_k_idx_;
        // voltage and current state within the cell
        T* vec_v;
        T* vec_i;
        T* vec_area;
        // node index information
        I* ni;
        unsigned long n_;
    };

    namespace kernels {
        __device__
        inline double atomicAdd(double* address, double val) {
            using I = unsigned long long int;
            I* address_as_ull = (I*)address;
            I old = *address_as_ull, assumed;
            do {
            assumed = old;
            old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val+__longlong_as_double(assumed)));
            } while (assumed != old);
            return __longlong_as_double(old);
        }

        template <typename T, typename I>
        __device__
        void rates(hh_ParamPack<T,I> const& params_,const int tid_, T v);

        template <typename T, typename I>
        __global__
        void nrn_state(hh_ParamPack<T,I> params_) {
            using value_type = T;
            using index_type = I;

            auto tid_ = threadIdx.x + blockDim.x*blockIdx.x;
            auto const n_ = params_.n_;

            if(tid_<n_) {
                auto gid_ __attribute__((unused)) = params_.ni[tid_];
                value_type v = params_.vec_v[gid_]; // indexed load

                // the kernel computation
                rates<T,I>(params_, tid_, v);
                params_.m[tid_] = params_.minf[tid_]+(params_.m[tid_]-params_.minf[tid_])*exp( -params_.dt/params_.mtau[tid_]);
                params_.h[tid_] = params_.hinf[tid_]+(params_.h[tid_]-params_.hinf[tid_])*exp( -params_.dt/params_.htau[tid_]);
                params_.n[tid_] = params_.ninf[tid_]+(params_.n[tid_]-params_.ninf[tid_])*exp( -params_.dt/params_.ntau[tid_]);
            }
        }

        template <typename T, typename I>
        __device__
        void rates(hh_ParamPack<T,I> const& params_,const int tid_, T v) {
            using value_type = T;
            using index_type = I;

            value_type ll3_;
            value_type ll1_;
            value_type ll0_;
            value_type ll2_;
            value_type sum;
            value_type q10;
            value_type alpha;
            value_type beta;
            q10 = std::pow( 3, (params_.celsius- 6.2999999999999998)/ 10);
            ll2_ =  -(v+ 40);
            ll0_ = ll2_/(exp(ll2_/ 10)- 1);
            alpha =  0.10000000000000001*ll0_;
            beta =  4*exp( -((v+ 65))/ 18);
            sum = alpha+beta;
            params_.mtau[tid_] =  1/(q10*sum);
            params_.minf[tid_] = alpha/sum;
            alpha =  0.070000000000000007*exp( -((v+ 65))/ 20);
            beta =  1/(exp( -((v+ 35))/ 10)+ 1);
            sum = alpha+beta;
            params_.htau[tid_] =  1/(q10*sum);
            params_.hinf[tid_] = alpha/sum;
            ll3_ =  -(v+ 55);
            ll1_ = ll3_/(exp(ll3_/ 10)- 1);
            alpha =  0.01*ll1_;
            beta =  0.125*exp( -((v+ 65))/ 80);
            sum = alpha+beta;
            params_.ntau[tid_] =  1/(q10*sum);
            params_.ninf[tid_] = alpha/sum;
        }

        template <typename T, typename I>
        __global__
        void nrn_current(hh_ParamPack<T,I> params_) {
            using value_type = T;
            using index_type = I;

            auto tid_ = threadIdx.x + blockDim.x*blockIdx.x;
            auto const n_ = params_.n_;

            if(tid_<n_) {
                auto gid_ __attribute__((unused)) = params_.ni[tid_];
                auto kid_  = params_.ion_k_idx_[tid_];
                auto naid_ = params_.ion_na_idx_[tid_];
                value_type ek = params_.ion_ek[kid_]; // indexed load
                value_type ik;
                value_type current_;
                value_type ena = params_.ion_ena[naid_]; // indexed load
                value_type v = params_.vec_v[gid_]; // indexed load
                value_type ina;

                // the kernel computation
                value_type il;
                params_.gna[tid_] = params_.gnabar[tid_]*params_.m[tid_]*params_.m[tid_]*params_.m[tid_]*params_.h[tid_];
                ina = params_.gna[tid_]*(v-ena);
                current_ = ina;
                params_.gk[tid_] = params_.gkbar[tid_]*params_.n[tid_]*params_.n[tid_]*params_.n[tid_]*params_.n[tid_];
                ik = params_.gk[tid_]*(v-ek);
                current_ = current_+ik;
                il = params_.gl[tid_]*(v-params_.el[tid_]);
                current_ = current_+il;

                // stores to indexed global memory
                params_.ion_ik[kid_] += ik;
                params_.vec_i[gid_] += current_;
                params_.ion_ina[naid_] += ina;
            }
        }

        template <typename T, typename I>
        __global__
        void nrn_init(hh_ParamPack<T,I> params_) {
            using value_type = T;
            using index_type = I;

            auto tid_ = threadIdx.x + blockDim.x*blockIdx.x;
            auto const n_ = params_.n_;

            if(tid_<n_) {
                auto gid_ __attribute__((unused)) = params_.ni[tid_];
                value_type v = params_.vec_v[gid_]; // indexed load

                // the kernel computation
                rates<T,I>(params_, tid_, v);
                params_.m[tid_] = params_.minf[tid_];
                params_.h[tid_] = params_.hinf[tid_];
                params_.n[tid_] = params_.ninf[tid_];
            }
        }

    } // namespace kernels

    template<typename T, typename I>
    class mechanism_hh : public ::nest::mc::mechanisms::gpu::mechanism<T, I> {
    public:
        using base = ::nest::mc::mechanisms::gpu::mechanism<T, I>;
        using value_type  = typename base::value_type;
        using size_type   = typename base::size_type;
        using vector_type = typename base::vector_type;
        using view_type   = typename base::view_type;
        using index_type  = typename base::index_type;
        using index_view  = typename base::index_view;
        using const_index_view  = typename base::const_index_view;
        using indexed_view_type= typename base::indexed_view_type;
        using ion_type = typename base::ion_type;
        using param_pack_type = hh_ParamPack<T,I>;
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


        template <typename IVT>
        mechanism_hh(view_type vec_v, view_type vec_i, IVT node_index) :
           base(vec_v, vec_i, node_index)
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
            gnabar = data_(0*field_size, 1*field_size);
            minf = data_(1*field_size, 2*field_size);
            h = data_(2*field_size, 3*field_size);
            m = data_(3*field_size, 4*field_size);
            gl = data_(4*field_size, 5*field_size);
            gkbar = data_(5*field_size, 6*field_size);
            el = data_(6*field_size, 7*field_size);
            ninf = data_(7*field_size, 8*field_size);
            mtau = data_(8*field_size, 9*field_size);
            gna = data_(9*field_size, 10*field_size);
            gk = data_(10*field_size, 11*field_size);
            n = data_(11*field_size, 12*field_size);
            hinf = data_(12*field_size, 13*field_size);
            ntau = data_(13*field_size, 14*field_size);
            htau = data_(14*field_size, 15*field_size);
            gnabar(memory::all) = 0.120000;
            gl(memory::all) = 0.000300;
            gkbar(memory::all) = 0.036000;
            el(memory::all) = -54.300000;

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
            param_pack_ =
                param_pack_type {
                    gnabar.data(),
                    minf.data(),
                    h.data(),
                    m.data(),
                    gl.data(),
                    gkbar.data(),
                    el.data(),
                    ninf.data(),
                    mtau.data(),
                    gna.data(),
                    gk.data(),
                    n.data(),
                    hinf.data(),
                    ntau.data(),
                    htau.data(),
                    dt,
                    t,
                    celsius,
                    ion_na.ena.data(),
                    ion_na.ina.data(),
                    ion_na.index.data(),
                    ion_k.ek.data(),
                    ion_k.ik.data(),
                    ion_k.index.data(),
                    vec_v_.data(),
                    vec_i_.data(),
                    vec_area_.data(),
                    node_index_.data(),
                    node_index_.size(),
                };
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
                ion_na.index = index_into(memory::on_host(i.node_index()), memory::on_host(node_index_));
                ion_na.ina = i.current();
                ion_na.ena = i.reversal_potential();
                return;
            }
            if(k==ionKind::k) {
                ion_k.index = index_into(memory::on_host(i.node_index()), memory::on_host(node_index_));
                for(auto idx : memory::on_host(ion_k.index)) {
                    std::cout << "\n index " << idx << " \n";
                }
                ion_k.ik = i.current();
                ion_k.ek = i.reversal_potential();
                return;
            }
            throw std::domain_error(nest::mc::util::pprintf("mechanism % does not support ion type\n", name()));
        }

        void nrn_state() {
            auto n = size();
            auto thread_dim = 192;
            dim3 dim_block(thread_dim);
            dim3 dim_grid(n/dim_block.x + (n%dim_block.x ? 1 : 0) );

            kernels::nrn_state<T,I><<<dim_grid, dim_block>>>(param_pack_);
        }

        void nrn_current() {
            auto n = size();
            auto thread_dim = 192;
            dim3 dim_block(thread_dim);
            dim3 dim_grid(n/dim_block.x + (n%dim_block.x ? 1 : 0) );

            kernels::nrn_current<T,I><<<dim_grid, dim_block>>>(param_pack_);
        }

        void nrn_init() {
            auto n = size();
            auto thread_dim = 192;
            dim3 dim_block(thread_dim);
            dim3 dim_grid(n/dim_block.x + (n%dim_block.x ? 1 : 0) );

            kernels::nrn_init<T,I><<<dim_grid, dim_block>>>(param_pack_);
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
        value_type dt = value_type{0};
        value_type t = value_type{0};
        value_type celsius = 6.300000;
        using base::vec_v_;
        using base::vec_i_;
        using base::vec_area_;
        using base::node_index_;

        param_pack_type param_pack_;
    };
}}}}} // namespaces
