#pragma once

#include <map>
#include <string>

#include <common_types.hpp>
#include <mechanism.hpp>
#include <memory/memory.hpp>
#include <util/span.hpp>

#include "stimulus_gpu.hpp"

namespace nest {
namespace mc {
namespace gpu {

/// Parameter pack for passing matrix fields and dimensions to the
/// Hines matrix solver implemented on the GPU backend.
template <typename T, typename I>
struct matrix_solve_param_pack {
    T* d;
    const T* u;
    T* rhs;
    const I* p;
    const I* cell_index;
    I n;
    I ncells;
};

/// Parameter pack for passing matrix and fvm fields and dimensions to the
/// FVM matrix generator implemented on the GPU
template <typename T, typename I>
struct matrix_update_param_pack {
    T* d;
    const T* u;
    T* rhs;
    const T* invariant_d;
    const T* cv_capacitance;
    const T* face_conductance;
    const T* voltage;
    const T* current;
    I n;
};

// forward declarations of the matrix solver implementation
// see the bottom of the file for implementation

template <typename T, typename I>
__global__ void matrix_solve(matrix_solve_param_pack<T, I> params);

template <typename T, typename I>
__global__ void assemble_matrix(matrix_update_param_pack<T, I> params, T dt);

struct backend {
    /// define the real and index types
    using value_type = double;
    using size_type  = nest::mc::cell_lid_type;

    /// define storage types
    using array  = memory::device_vector<value_type>;
    using iarray = memory::device_vector<size_type>;

    using view       = typename array::view_type;
    using const_view = typename array::const_view_type;

    using iview       = typename iarray::view_type;
    using const_iview = typename iarray::const_view_type;

    using host_array  = typename memory::host_vector<value_type>;
    using host_iarray = typename memory::host_vector<size_type>;

    using host_view   = typename host_array::view_type;
    using host_iview  = typename host_iarray::const_view_type;

    static std::string name() {
        return "gpu";
    }

    //
    // matrix infrastructure
    //

    /// Hines matrix assembly interface
    struct matrix_assembler {
        matrix_update_param_pack<value_type, size_type> params;

        // the invariant part of the matrix diagonal
        array invariant_d;  // [μS]

        matrix_assembler() = default;

        matrix_assembler(
            view d, view u, view rhs, const_iview p,
            const_view cv_capacitance,
            const_view face_conductance,
            const_view voltage,
            const_view current)
        {
            auto n = d.size();
            host_array invariant_d_tmp(n, 0);
            // make a copy of the conductance on the host
            host_array face_conductance_tmp = face_conductance;
            for(auto i: util::make_span(1u, n)) {
                auto gij = face_conductance_tmp[i];

                u[i] = -gij;
                invariant_d_tmp[i] += gij;
                invariant_d_tmp[p[i]] += gij;
            }
            invariant_d = invariant_d_tmp;

            params = {
                d.data(), u.data(), rhs.data(),
                invariant_d.data(), cv_capacitance.data(), face_conductance.data(),
                voltage.data(), current.data(), size_type(n)};
        }

        void assemble(value_type dt) {
            // determine the grid dimensions for the kernel
            auto const n = params.n;
            auto const block_dim = 96;
            auto const grid_dim = (n+block_dim-1)/block_dim;

            assemble_matrix<value_type, size_type><<<grid_dim, block_dim>>>(params, dt);
        }

    };

    /// Hines solver interface
    static void hines_solve(
        view d, view u, view rhs,
        const_iview p, const_iview cell_index)
    {
        using solve_param_pack = matrix_solve_param_pack<value_type, size_type>;

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

    //
    // mechanism infrastructure
    //
    using ion = mechanisms::ion<backend>;

    using mechanism = mechanisms::mechanism_ptr<backend>;

    using stimulus = mechanisms::gpu::stimulus<backend>;

    static mechanism make_mechanism(
        const std::string& name,
        view vec_v, view vec_i,
        const std::vector<value_type>& weights,
        const std::vector<size_type>& node_indices)
    {
        if (!has_mechanism(name)) {
            throw std::out_of_range("no mechanism in database : " + name);
        }

        return mech_map_.find(name)->
            second(vec_v, vec_i, memory::make_const_view(weights), memory::make_const_view(node_indices));
    }

    static bool has_mechanism(const std::string& name) { return mech_map_.count(name)>0; }

private:

    using maker_type = mechanism (*)(view, view, array&&, iarray&&);
    static std::map<std::string, maker_type> mech_map_;

    template <template <typename> class Mech>
    static mechanism maker(view vec_v, view vec_i, array&& weights, iarray&& node_indices) {
        return mechanisms::make_mechanism<Mech<backend>>
            (vec_v, vec_i, std::move(weights), std::move(node_indices));
    }
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

/// GPU implementatin of Hines matrix assembly
/// For a given time step size dt
///     - use the precomputed alpha and alpha_d values to construct the diagonal
///       and off diagonal of the symmetric Hines matrix.
///     - compute the RHS of the linear system to solve
template <typename T, typename I>
__global__
void assemble_matrix(matrix_update_param_pack<T, I> params, T dt) {
    auto tid = threadIdx.x + blockDim.x*blockIdx.x;

    T factor = 1e-3/dt;
    if(tid < params.n) {
        auto gi = factor * params.cv_capacitance[tid];

        params.d[tid] = gi + params.invariant_d[tid];

        params.rhs[tid] = gi*params.voltage[tid] - params.current[tid];
    }
}

} // namespace multicore
} // namespace mc
} // namespace nest
