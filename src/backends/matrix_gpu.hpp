#pragma once

#include <memory/memory.hpp>

namespace nest {
namespace mc {
namespace gpu {

template <typename T, typename I>
struct matrix_param_pack {
    T* l;
    T* d;
    T* u;
    T* rhs;
    const I* p;
    const I* cell_index;
    I n;
    I ncells;
};

template <typename T, typename I>
__global__
void matrix_solve(matrix_param_pack<T, I> params) {
    auto tid = threadIdx.x + blockDim.x*blockIdx.x;
    auto l   = params.l;
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
            auto factor = l[i] / d[i];
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

template <typename T, typename I>
struct matrix_policy {
    // define basic types
    using value_type = T;
    using size_type  = I;

    // define storage types
    using array  = memory::DeviceVector<value_type>;
    using iarray   = memory::DeviceVector<size_type>;

    using view = typename array::view_type;
    using const_view = typename array::const_view_type;
    using iview = typename iarray::view_type;
    using const_iview = typename iarray::const_view_type;

    using param_pack_type = matrix_param_pack<value_type, size_type>;

    void solve(
        view l, view d, view u, view rhs,
        const_iview p, const_iview cell_index)
    {
        // pack the parameters into a single struct for kernel launch
        auto params = param_pack_type{
             l.data(), d.data(), u.data(), rhs.data(),
             p.data(), cell_index.data(),
             size_type(d.size()), size_type(cell_index.size())
        };

        // determine the grid dimensions for the kernel
        auto const n = params.ncells;
        auto const block_dim = 96;
        auto const grid_dim = (n+block_dim-1)/block_dim;

        // perform solve on gpu
        matrix_solve<T,I><<<grid_dim, block_dim>>>(params);
    }
};

} // namespace gpu
} // namespace mc
} // namespace nest
