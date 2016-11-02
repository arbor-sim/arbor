#pragma once

#include "memory_traits.hpp"

#include <util/span.hpp>

namespace nest {
namespace mc {
namespace gpu {

/// Parameter pack for passing matrix fields and dimensions to the
/// Hines matrix solver implemented on the GPU backend.
template <typename T, typename I>
struct matrix_solve_param_pack {
    T* d;
    T* u;
    T* rhs;
    const I* p;
    const I* cell_index;
    I n;
    I ncells;
};

/// GPU implementation of Hines Matrix solver.
/// Naiive implementation with one CUDA thread per matrix.
template <typename T, typename I>
__global__
void matrix_solve(matrix_solve_param_pack<T, I> params) {
    auto tid = threadIdx.x + blockDim.x*blockIdx.x;
    auto d   = params.d;
    auto u   = params.u;
    auto rhs = params.rhs;
    auto p   = params.p;

    if(tid < params.ncells) {
        // get range of this thread's cell matrix
        auto first = params.cell_index[tid];
        auto last  = params.cell_index[tid+1];

        // backward sweep
        for(auto i=last-1; i>first; --i) {
            auto factor = u[i] / d[i];
            d[p[i]]   -= factor * u[i];
            rhs[p[i]] -= factor * rhs[i];
        }

        __syncthreads();
        rhs[first] /= d[first];

        // forward sweep
        for(auto i=first+1; i<last; ++i) {
            rhs[i] -= u[i] * rhs[p[i]];
            rhs[i] /= d[i];
        }
    }
}

/// Policy class that calls the GPU implementation of the Hines matrix solver
struct matrix_solver : public memory_traits {
    using solve_param_pack = matrix_solve_param_pack<value_type, size_type>;

    void solve(view d, view u, view rhs, const_iview p, const_iview cell_index) {
        // pack the parameters into a single struct for kernel launch
        auto params = solve_param_pack{
             d.data(), u.data(), rhs.data(),
             p.data(), cell_index.data(),
             size_type(d.size()), size_type(cell_index.size()-1)
        };

        // determine the grid dimensions for the kernel
        auto const n = params.ncells;
        auto const block_dim = 96;
        auto const grid_dim = (n+block_dim-1)/block_dim;

        // perform solve on gpu
        matrix_solve<value_type, size_type><<<grid_dim, block_dim>>>(params);
    }
};

/// Parameter pack for passing matrix and fvm fields and dimensions to the
/// FVM matrix generator implemented on the GPU
template <typename T, typename I>
struct matrix_update_param_pack {
    T* d;
    T* u;
    T* rhs;
    const T* sigma;
    const T* alpha_d;
    const T* alpha;
    const T* voltage;
    const T* current;
    const T* cv_capacitance;
    I n;
};

template <typename T, typename I>
__global__
void matrix_update(matrix_update_param_pack<T, I> params, T dt) {
    auto tid = threadIdx.x + blockDim.x*blockIdx.x;

    T factor_lhs = 1e5*dt;
    T factor_rhs = 10.*dt;
    if(tid < params.n) {
        params.d[tid] = params.sigma[tid] + factor_lhs*params.alpha_d[tid];
        params.u[tid] = -factor_lhs*params.alpha[tid];
        params.rhs[tid] = params.sigma[tid] *
            (params.voltage[tid] - factor_rhs/params.cv_capacitance[tid]*params.current[tid]);
    }
}

struct fvm_matrix_builder : public memory_traits {
    matrix_update_param_pack<value_type, size_type> params;
    array alpha_d;

    fvm_matrix_builder() = default;

    fvm_matrix_builder(
        view d, view u, view rhs, const_iview p,
        const_view sigma, const_view alpha,
        const_view voltage, const_view current, const_view cv_capacitance)
    {
        auto n = d.size();
        host_array alpha_d_tmp(n, 0);
        for(auto i: util::make_span(1u, n)) {
            alpha_d_tmp[i] += alpha[i];

            // add contribution to the diagonal of parent
            alpha_d_tmp[p[i]] += alpha[i];
        }
        alpha_d = alpha_d_tmp;

        params = {
            d.data(), u.data(), rhs.data(),
            sigma.data(), alpha_d.data(), alpha.data(),
            voltage.data(), current.data(), cv_capacitance.data(), size_type(n)};
    }

    void build(value_type dt) {
        // determine the grid dimensions for the kernel
        auto const n = params.n;
        auto const block_dim = 96;
        auto const grid_dim = (n+block_dim-1)/block_dim;

        matrix_update<value_type, size_type><<<grid_dim, block_dim>>>(params, dt);
    }

};

} // namespace gpu
} // namespace mc
} // namespace nest
