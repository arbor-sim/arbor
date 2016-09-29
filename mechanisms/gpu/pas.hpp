#pragma once

#include <cmath>
#include <limits>

#include <mechanism.hpp>
#include <mechanism_interface.hpp>

namespace nest{ namespace mc{ namespace mechanisms{ namespace gpu{ namespace pas{

    template <typename T, typename I>
    struct pas_ParamPack {
        // array parameters
        T* e;
        T* g;
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
        void nrn_current(pas_ParamPack<T,I> params_) {
            using value_type = T;
            using index_type = I;

            auto tid_ = threadIdx.x + blockDim.x*blockIdx.x;
            auto const n_ = params_.n_;

            if(tid_<n_) {
                auto gid_ __attribute__((unused)) = params_.ni[tid_];
                value_type current_;
                value_type v = params_.vec_v[gid_]; // indexed load

                // the kernel computation
                value_type i;
                i = params_.g[tid_]*(v-params_.e[tid_]);
                current_ = i;

                // stores to indexed global memory
                params_.vec_i[gid_] += current_;
            }
        }

        template <typename T, typename I>
        __global__
        void nrn_state(pas_ParamPack<T,I> params_) {
            using value_type = T;
            using index_type = I;

            auto tid_ = threadIdx.x + blockDim.x*blockIdx.x;
            auto const n_ = params_.n_;

            if(tid_<n_) {
                auto gid_ __attribute__((unused)) = params_.ni[tid_];

                // the kernel computation
            }
        }

        template <typename T, typename I>
        __global__
        void nrn_init(pas_ParamPack<T,I> params_) {
            using value_type = T;
            using index_type = I;

            auto tid_ = threadIdx.x + blockDim.x*blockIdx.x;
            auto const n_ = params_.n_;

            if(tid_<n_) {
                auto gid_ __attribute__((unused)) = params_.ni[tid_];

                // the kernel computation
            }
        }

    } // namespace kernels

    template<typename T, typename I>
    class mechanism_pas : public ::nest::mc::mechanisms::gpu::mechanism<T, I> {
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
        using param_pack_type = pas_ParamPack<T,I>;

        template <typename IVT>
        mechanism_pas(view_type vec_v, view_type vec_i, IVT node_index) :
           base(vec_v, vec_i, node_index)
        {
            size_type num_fields = 2;

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
            e = data_(0*field_size, 1*field_size);
            g = data_(1*field_size, 2*field_size);
            e(memory::all) = -65.000000;
            g(memory::all) = 0.001000;

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
                    e.data(),
                    g.data(),
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
            return "pas";
        }

        mechanismKind kind() const override {
            return mechanismKind::density;
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
        view_type e;
        view_type g;
        value_type t = value_type{0};
        value_type dt = value_type{0};
        using base::vec_v_;
        using base::vec_i_;
        using base::vec_area_;
        using base::node_index_;

        param_pack_type param_pack_;
    };
}}}}} // namespaces
