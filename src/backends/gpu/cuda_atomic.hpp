#pragma once

// Wrappers around CUDA addition functions.
// CUDA 8 introduced support for atomicAdd with double precision, but only for
// Pascal GPUs (__CUDA_ARCH__ >= 600). These wrappers provide a portable
// atomic addition interface that chooses the appropriate implementation.

#if __CUDA_ARCH__ < 600 // Maxwell or older (no native double precision atomic addition)
    __device__
    inline double cuda_atomic_add(double* address, double val) {
        using I = unsigned long long int;
        I* address_as_ull = (I*)address;
        I old = *address_as_ull, assumed;
        do {
            assumed = old;
            old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val+__longlong_as_double(assumed)));
        } while (assumed != old);
        return __longlong_as_double(old);
    }
#else // use build in atomicAdd for double precision from Pascal onwards
    __device__
    inline double cuda_atomic_add(double* address, double val) {
        return atomicAdd(address, val);
    }
#endif

__device__
inline double cuda_atomic_sub(double* address, double val) {
    return cuda_atomic_add(address, -val);
}

__device__
inline float cuda_atomic_add(float* address, float val) {
    return atomicAdd(address, val);
}

__device__
inline float cuda_atomic_sub(float* address, float val) {
    return atomicAdd(address, -val);
}

