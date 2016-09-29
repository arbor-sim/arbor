#pragma once

#include <cmath>
#include <limits>

#include <mechanism.hpp>
#include <mechanism_interface.hpp>

namespace nest{ namespace mc{ namespace mechanisms{ namespace gpu{ namespace exp2syn{

    template <typename T, typename I>
    struct exp2syn_ParamPack {
        // array parameters
        T* factor;
        T* tau2;
        T* e;
        T* A;
        T* B;
        T* tau1;
        // scalar parameters
        T t;
        T dt;
        // ion channel dependencies
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
        __global__
        void nrn_current(exp2syn_ParamPack<T,I> params_) {
            using value_type = T;
            using index_type = I;

            auto tid_ = threadIdx.x + blockDim.x*blockIdx.x;
            auto const n_ = params_.n_;

            if(tid_<n_) {
                auto gid_ __attribute__((unused)) = params_.ni[tid_];
                value_type area_ = params_.vec_area[gid_]; // indexed load
                value_type current_;
                value_type v = params_.vec_v[gid_]; // indexed load

                // the kernel computation
                value_type i;
                i = (params_.B[tid_]-params_.A[tid_])*(v-params_.e[tid_]);
                current_ = i;
                current_ = ( 100*current_)/area_;

                // stores to indexed global memory
                atomicAdd(&params_.vec_i[gid_], current_);
            }
        }

        template <typename T, typename I>
        __global__
        void nrn_state(exp2syn_ParamPack<T,I> params_) {
            using value_type = T;
            using index_type = I;

            auto tid_ = threadIdx.x + blockDim.x*blockIdx.x;
            auto const n_ = params_.n_;

            if(tid_<n_) {
                auto gid_ __attribute__((unused)) = params_.ni[tid_];

                // the kernel computation
                value_type ba_;
                value_type a_;
                a_ =  -1/params_.tau1[tid_];
                ba_ =  0/a_;
                params_.A[tid_] =  -ba_+(params_.A[tid_]+ba_)*exp(a_*params_.dt);
                a_ =  -1/params_.tau2[tid_];
                ba_ =  0/a_;
                params_.B[tid_] =  -ba_+(params_.B[tid_]+ba_)*exp(a_*params_.dt);
            }
        }

        template <typename T, typename I>
        __global__
        void nrn_init(exp2syn_ParamPack<T,I> params_) {
            using value_type = T;
            using index_type = I;

            auto tid_ = threadIdx.x + blockDim.x*blockIdx.x;
            auto const n_ = params_.n_;

            if(tid_<n_) {
                auto gid_ __attribute__((unused)) = params_.ni[tid_];

                // the kernel computation
                value_type tp;
                params_.A[tid_] =  0;
                params_.B[tid_] =  0;
                tp = (params_.tau1[tid_]*params_.tau2[tid_])/(params_.tau2[tid_]-params_.tau1[tid_])*log(params_.tau2[tid_]/params_.tau1[tid_]);
                params_.factor[tid_] =  -exp( -tp/params_.tau1[tid_])+exp( -tp/params_.tau2[tid_]);
                params_.factor[tid_] =  1/params_.factor[tid_];
            }
        }

    } // namespace kernels

    template<typename T, typename I>
    class mechanism_exp2syn : public ::nest::mc::mechanisms::gpu::mechanism<T, I> {
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
        using param_pack_type = exp2syn_ParamPack<T,I>;

        template <typename IVT>
        mechanism_exp2syn(view_type vec_v, view_type vec_i, IVT node_index) :
           base(vec_v, vec_i, node_index)
        {
            size_type num_fields = 6;

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
            factor = data_(0*field_size, 1*field_size);
            tau2 = data_(1*field_size, 2*field_size);
            e = data_(2*field_size, 3*field_size);
            A = data_(3*field_size, 4*field_size);
            B = data_(4*field_size, 5*field_size);
            tau1 = data_(5*field_size, 6*field_size);
            tau2(memory::all) = 2.000000;
            e(memory::all) = 0.000000;
            tau1(memory::all) = 0.500000;

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
            param_pack_ =
                param_pack_type {
                    factor.data(),
                    tau2.data(),
                    e.data(),
                    A.data(),
                    B.data(),
                    tau1.data(),
                    t,
                    dt,
                    vec_v_.data(),
                    vec_i_.data(),
                    vec_area_.data(),
                    node_index_.data(),
                    node_index_.size(),
                };
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

        void set_ion(ionKind k, ion_type& i) override {
            using nest::mc::algorithms::index_into;
            throw std::domain_error(nest::mc::util::pprintf("mechanism % does not support ion type\n", name()));
        }

        void nrn_current() {
            auto n = size();
            auto thread_dim = 192;
            dim3 dim_block(thread_dim);
            dim3 dim_grid(n/dim_block.x + (n%dim_block.x ? 1 : 0) );

            kernels::nrn_current<T,I><<<dim_grid, dim_block>>>(param_pack_);
        }

        void nrn_state() {
            auto n = size();
            auto thread_dim = 192;
            dim3 dim_block(thread_dim);
            dim3 dim_grid(n/dim_block.x + (n%dim_block.x ? 1 : 0) );

            kernels::nrn_state<T,I><<<dim_grid, dim_block>>>(param_pack_);
        }

        void nrn_init() {
            auto n = size();
            auto thread_dim = 192;
            dim3 dim_block(thread_dim);
            dim3 dim_grid(n/dim_block.x + (n%dim_block.x ? 1 : 0) );

            kernels::nrn_init<T,I><<<dim_grid, dim_block>>>(param_pack_);
        }

        vector_type data_;
        view_type factor;
        view_type tau2;
        view_type e;
        view_type A;
        view_type B;
        view_type tau1;
        value_type t = value_type{0};
        value_type dt = value_type{0};
        using base::vec_v_;
        using base::vec_i_;
        using base::vec_area_;
        using base::node_index_;

        param_pack_type param_pack_;
    };
}}}}} // namespaces
