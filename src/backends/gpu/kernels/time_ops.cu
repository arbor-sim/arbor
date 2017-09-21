#include <backends/fvm_types.hpp>

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

    template <typename T>
    struct less {
        __device__ __host__
        bool operator()(const T& a, const T& b) const { return a<b; }
    };

    // vector minus: x = y - z
    template <typename T, typename I>
    __global__ void vec_minus(I n, T* x, const T* y, const T* z) {
        int i = threadIdx.x+blockIdx.x*blockDim.x;
        if (i<n) {
            x[i] = y[i]-z[i];
        }
    }

    // vector gather: x[i] = y[index[i]]
    template <typename T, typename I>
    __global__ void gather(I n, T* x, const T* y, const I* index) {
        int i = threadIdx.x+blockIdx.x*blockDim.x;
        if (i<n) {
            x[i] = y[index[i]];
        }
    }
}

void update_time_to(fvm_size_type n,
                    fvm_value_type* time_to,
                    const fvm_value_type* time,
                    fvm_value_type dt,
                    fvm_value_type tmax)
{
    if (!n) {
        return;
    }

    constexpr int blockwidth = 128;
    int nblock = 1+(n-1)/blockwidth;
    kernels::update_time_to<<<nblock, blockwidth>>>
        (n, time_to, time, dt, tmax);
}

void set_dt(fvm_size_type ncell,
            fvm_size_type ncomp,
            fvm_value_type* dt_cell,
            fvm_value_type* dt_comp,
            const fvm_value_type* time_to,
            const fvm_value_type* time,
            const fvm_size_type* cv_to_cell)
{
    if (!ncell || !ncomp) {
        return;
    }

    constexpr int blockwidth = 128;
    int nblock = 1+(ncell-1)/blockwidth;
    kernels::vec_minus<<<nblock, blockwidth>>>(ncell, dt_cell, time_to, time);

    nblock = 1+(ncomp-1)/blockwidth;
    kernels::gather<<<nblock, blockwidth>>>(ncomp, dt_comp, dt_cell, cv_to_cell);
}

} // namespace gpu
} // namespace mc
} // namespace nest
