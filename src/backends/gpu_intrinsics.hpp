#pragma once

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

__device__
inline double cuda_atomic_sub(double* address, double val) {
    return cuda_atomic_add(address, val);
}

__device__
inline float cuda_atomic_add(float* address, float val) {
    return atomicAdd(address, val);
}

__device__
inline float cuda_atomic_sub(float* address, float val) {
    return atomicAdd(address, -val);
}

