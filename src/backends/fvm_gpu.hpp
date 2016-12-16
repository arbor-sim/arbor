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

template <typename T, typename I>
struct gpu_threshold_watch {
    const T* values;
    const I* index;
    const T* threshold;
    I* is_spiking;
    T t_prev;
    I n_watches;

    I allocated_size_;
    I size_;

    __device__
    void test(T t) {
        auto tid = 100;


    }
};

template <typename T, typename I>
__global__ void gpu_threshold_test(gpu_threshold_watch<T, I>* watch, T t) {
    auto i = threadIdx.x + blockIdx.x*blockDim.x;


}

// forward declarations of the matrix solver implementation
// see the bottom of the file for implementation

template <typename T, typename I>
__global__ void matrix_solve(matrix_solve_param_pack<T, I> params);

template <typename T, typename I>
__global__ void assemble_matrix(matrix_update_param_pack<T, I> params, T dt);

template <typename T, typename I>
__global__ void test_thresholds(T t_prev, T t_now, T* );

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

    /// threshold crossing logic
    /// used as part of spike detection back end
    class threshold_watcher {
    public:
        /// stores a single crossing event
        struct threshold_crossing {
            size_type index;    // index of variable
            value_type time;    // time of crossing
            friend bool operator==
                (const threshold_crossing& lhs, const threshold_crossing& rhs)
            {
                return lhs.index==rhs.index && lhs.time==rhs.time;
            }
        };

        threshold_watcher() = default;

        threshold_watcher(
                const_view vals,
                const std::vector<size_type>& indxs,
                const std::vector<value_type>& thresh,
                value_type t=0):
            values_(vals),
            index_(memory::make_const_view(indxs)),
            thresholds_(memory::make_const_view(thresh)),
            v_prev_(vals)
        {
            is_spiking_ = iarray(size());
            reset(t);
        }

        /// Remove all stored crossings that were detected in previous calls
        /// to the test() member function.
        void clear_crossings() {
            // TODO: KERNEL clear gpu buffer
            crossings_.clear();
        }

        /// Reset state machine for each detector.
        /// Assume that the values in values_ have been set correctly before
        /// calling, because the values are used to determine the initial state
        void reset(value_type t=0) {
            // TODO: KERNEL
            clear_crossings();
            for (auto i=0u; i<size(); ++i) {
                is_spiking_[i] = values_[index_[i]]>=thresholds_[i];
            }
            t_prev_ = t;
        }

        const std::vector<threshold_crossing>& crossings() const {
            // TODO: get crossings from GPU first
            return crossings_;
        }

        /// The time at which the last test was performed
        value_type last_test_time() const {
            return t_prev_;
        }

        /// Tests each target for changed threshold state
        /// Crossing events are recorded for each threshold that
        /// is crossed since the last call to test
        void test(value_type t) {
            // TODO: KERNEL
            for (auto i=0u; i<size(); ++i) {
                auto v = values_[index_[i]];
                auto thresh = thresholds_[i];
                auto v_prev = v_prev_[i];
                if (!is_spiking_[i]) {
                    if (v>=thresh) {
                        // the threshold has been passed, so estimate the time using
                        // linear interpolation
                        auto pos = (thresh - v_prev)/(v - v_prev);
                        auto crossing_time = t_prev_ + pos*(t - t_prev_);
                        crossings_.push_back({i, crossing_time});

                        is_spiking_[i] = true;
                    }
                }
                else {
                    if (v<thresh) {
                        is_spiking_[i] = false;
                    }
                }

                v_prev_[i] = v;
            }

            // this is stored on host
            t_prev_ = t;
        }

        bool is_spiking(size_type i) const {
            return is_spiking_[i];
        }

        /// the number of threashold values that are being monitored
        std::size_t size() const {
            return index_.size();
        }

        /// Data type used to store the crossings.
        /// Provided to make type-generic calling code.
        using crossing_list =  std::vector<threshold_crossing>;

    private:
        const_view values_;         // values to watch: on gpu
        iarray index_;              // indexes of values to watch: on gpu

        array thresholds_;          // threshold for each watch: on gpu
        value_type t_prev_;         // time of previous sample: on host
        array v_prev_;              // values at previous sample time: on host
        crossing_list crossings_;   // buffer of crossings: on host
        iarray is_spiking_;         // bool flag for state of each watch: on gpu
    };

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
