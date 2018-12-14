// CUDA kernels and wrappers for shared state methods.

#include <cstdint>

#include <backends/event.hpp>
#include <backends/multi_event_stream_state.hpp>

#include "cuda_atomic.hpp"
#include "cuda_common.hpp"

namespace arb {
namespace gpu {

namespace kernel {

template <typename T>
__global__
void nernst_impl(unsigned n, T factor, const T* Xo, const T* Xi, T* eX) {
    unsigned i = threadIdx.x+blockIdx.x*blockDim.x;

    if (i<n) {
        eX[i] = factor*std::log(Xo[i]/Xi[i]);
    }
}

template <typename T>
__global__
void init_concentration_impl(unsigned n, T* Xi, T* Xo, const T* weight_Xi, const T* weight_Xo, T c_int, T c_ext) {
    unsigned i = threadIdx.x+blockIdx.x*blockDim.x;

    if (i<n) {
        Xi[i] = c_int*weight_Xi[i];
        Xo[i] = c_ext*weight_Xo[i];
    }
}

template <typename T>
__global__ void update_time_to_impl(unsigned n, T* time_to, const T* time, T dt, T tmax) {
    unsigned i = threadIdx.x+blockIdx.x*blockDim.x;
    if (i<n) {
        auto t = time[i]+dt;
        time_to[i] = t<tmax? t: tmax;
    }
}

template <typename T, typename I>
__global__ void sync_time_to_impl(unsigned n, T* time_to, const I* time_deps) {
    unsigned i = threadIdx.x+blockIdx.x*blockDim.x;
    if (i<n) {
        if (time_deps[i] > 0) {
            auto min_t = time_to[i];
            for(int j = 1; j < time_deps[i]; j++) {
                if (time_to[i+j] < min_t) {
                    min_t = time_to[i+j];
                }
            }
            for(int j = 0; j < time_deps[i]; j++) {
                time_to[i+j] = min_t;
            }
        }
    }
}

template <typename T, typename I>
__global__ void update_gj_state_impl(unsigned n, const T* gj_info, const I* voltage, I* current_density) {
    unsigned i = threadIdx.x+blockIdx.x*blockDim.x;
    if (i<n) {
        auto gj = gj_info[i];
        auto curr = gj.weight * (voltage[gj.loc.second] - voltage[gj.loc.first]); // nA

        cuda_atomic_sub(current_density + gj.loc.first, curr);
    }
}

// Vector minus: x = y - z
template <typename T>
__global__ void vec_minus(unsigned n, T* x, const T* y, const T* z) {
    unsigned i = threadIdx.x+blockIdx.x*blockDim.x;
    if (i<n) {
        x[i] = y[i]-z[i];
    }
}

// Vector gather: x[i] = y[index[i]]
template <typename T, typename I>
__global__ void gather(unsigned n, T* x, const T* y, const I* index) {
    unsigned i = threadIdx.x+blockIdx.x*blockDim.x;
    if (i<n) {
        x[i] = y[index[i]];
    }
}

__global__ void take_samples_impl(
    multi_event_stream_state<raw_probe_info> s,
    const fvm_value_type* time, fvm_value_type* sample_time, fvm_value_type* sample_value)
{
    unsigned i = threadIdx.x+blockIdx.x*blockDim.x;
    if (i<s.n) {
        auto begin = s.ev_data+s.begin_offset[i];
        auto end = s.ev_data+s.end_offset[i];
        for (auto p = begin; p!=end; ++p) {
            sample_time[p->offset] = time[i];
            sample_value[p->offset] = *p->handle;
        }
    }
}

} // namespace kernel

using impl::block_count;

void nernst_impl(
    std::size_t n, fvm_value_type factor,
    const fvm_value_type* Xo, const fvm_value_type* Xi, fvm_value_type* eX)
{
    if (!n) return;

    constexpr int block_dim = 128;
    int nblock = block_count(n, block_dim);
    kernel::nernst_impl<<<nblock, block_dim>>>(n, factor, Xo, Xi, eX);
}

void init_concentration_impl(
    std::size_t n, fvm_value_type* Xi, fvm_value_type* Xo, const fvm_value_type* weight_Xi,
    const fvm_value_type* weight_Xo, fvm_value_type c_int, fvm_value_type c_ext)
{
    if (!n) return;

    constexpr int block_dim = 128;
    int nblock = block_count(n, block_dim);
    kernel::init_concentration_impl<<<nblock, block_dim>>>(n, Xi, Xo, weight_Xi, weight_Xo, c_int, c_ext);
}

void sync_time_to_impl(std::size_t n, fvm_value_type* time_to, const fvm_index_type* time_deps)
{
    if (!n) return;

    constexpr int block_dim = 128;
    int nblock = block_count(n, block_dim);
    kernel::sync_time_to_impl<<<nblock, block_dim>>>(n, time_to, time_deps);
}

void update_time_to_impl(
    std::size_t n, fvm_value_type* time_to, const fvm_value_type* time,
    fvm_value_type dt, fvm_value_type tmax)
{
    if (!n) return;

    constexpr int block_dim = 128;
    const int nblock = block_count(n, block_dim);
    kernel::update_time_to_impl<<<nblock, block_dim>>>(n, time_to, time, dt, tmax);
}

void set_dt_impl(
    fvm_size_type ncell, fvm_size_type ncomp, fvm_value_type* dt_cell, fvm_value_type* dt_comp,
    const fvm_value_type* time_to, const fvm_value_type* time, const fvm_index_type* cv_to_cell)
{
    if (!ncell || !ncomp) return;

    constexpr int block_dim = 128;
    int nblock = block_count(ncell, block_dim);
    kernel::vec_minus<<<nblock, block_dim>>>(ncell, dt_cell, time_to, time);

    nblock = block_count(ncomp, block_dim);
    kernel::gather<<<nblock, block_dim>>>(ncomp, dt_comp, dt_cell, cv_to_cell);
}

void update_gj_state_impl(
    fvm_size_type n_gj, const gap_junction* gj_info, const fvm_value_type* voltage, fvm_value_type* current_density)
{
    if(!n_gj) return;

    constexpr int block_dim = 128;
    int nblock = block_count(n_gj, block_dim);
    kernel::update_gj_state_impl<<<nblock, block_dim>>>(n_gj, gj_info, voltage, current_density);
}

void take_samples_impl(
    const multi_event_stream_state<raw_probe_info>& s,
    const fvm_value_type* time, fvm_value_type* sample_time, fvm_value_type* sample_value)
{
    if (!s.n_streams()) return;

    constexpr int block_dim = 128;
    const int nblock = block_count(s.n_streams(), block_dim);
    kernel::take_samples_impl<<<nblock, block_dim>>>(s, time, sample_time, sample_value);
}

} // namespace gpu
} // namespace arb
