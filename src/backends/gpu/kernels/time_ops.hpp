#pragma once

#include <type_traits>

#include <cuda.h>

#include <memory/wrappers.hpp>

namespace nest {
namespace mc {
namespace gpu {

namespace kernels {
    template <typename T, typename I>
    __global__ void update_time_to(I n, T* time_to, const T* time, T dt, T tmax) {
        int i = threadIdx.x+blockIdx.x*blockDim.x;
        if (i<n) {
            auto t = time[i]+dt;
            time_to[i] = t<tmax? t: tmax;
        }
    }

    // array-array comparison
    template <typename T, typename I, typename Pred>
    __global__ void array_reduce_any(I n, const T* x, const T* y, Pred p, int* rptr) {
        int i = threadIdx.x+blockIdx.x*blockDim.x;
        int cmp = i<n? p(x[i], y[i]): 0;
        if (__syncthreads_or(cmp) && !threadIdx.x) {
            *rptr=1;
        }
    }

    // array-scalar comparison
    template <typename T, typename I, typename Pred>
    __global__ void array_reduce_any(I n, const T* x, T y, Pred p, int* rptr) {
        int i = threadIdx.x+blockIdx.x*blockDim.x;
        int cmp = i<n? p(x[i], y): 0;
        if (__syncthreads_or(cmp) && !threadIdx.x) {
            *rptr=1;
        }
    }

    template <typename T>
    struct less {
        __device__ __host__
        bool operator()(const T& a, const T& b) const { return a<b; }
    };

    // vector minus: x = y - z
    template <typename T, typename I, typename Pred>
    __global__ void vec_minus(I n, T* x, const T* y, const T* z) {
        int i = threadIdx.x+blockIdx.x*blockDim.x;
        if (i<n) {
            x[i] = y[i]-z[i];
        }
    }

    // vector gather: x[i] = y[index[i]]
    template <typename T, typename I, typename Pred>
    __global__ void gather(I n, T* x, const T* y, const I* index) {
        int i = threadIdx.x+blockIdx.x*blockDim.x;
        if (i<n) {
            x[i] = y[index[i]];
        }
    }
}

template <typename T, typename I>
void update_time_to(I n, T* time_to, const T* time, T dt, T tmax) {
    if (!n) {
        return;
    }

    constexpr int blockwidth = 128;
    int nblock = 1+(n-1)/blockwidth;
    kernels::update_time_to<<<nblock, blockwidth>>>(n, time_to, time, dt, tmax);
}

template <typename T, typename U, typename I>
bool any_time_before(I n, T* t0, U t1) {
    static_assert(std::is_convertible<T*, U>::value || std::is_convertible<T, U>::value,
        "third-argument must be a compatible scalar or pointer type");

    static thread_local auto r = memory::device_vector<int>(1);
    if (!n) {
        return false;
    }

    constexpr int blockwidth = 128;
    int nblock = 1+(n-1)/blockwidth;

    r[0] = 0;
    kernels::array_reduce_any<<<nblock, blockwidth>>>(n, t0, t1, kernels::less<T>(), r.data());
    return r[0];
}

template <typename T, typename I>
void set_dt(I ncell, I ncomp, T* dt_cell, T& dt_comp, const T* time_to, const T* time, const I* cv_to_cell) {
    if (!ncell || !ncomp) {
        return;
    }

    constexpr int blockwidth = 128;
    int nblock = 1+(n-1)/blockwidth;
    kernels::vec_minus<<<nblock, blockwidth>>>(ncell, dt_cell, time_to, time);
    kernels::gather<<<nblock, blockwidth>>>(ncomp, dt_comp, dt_cell, cv_to_cell);
}

} // namespace gpu
} // namespace mc
} // namespace nest
