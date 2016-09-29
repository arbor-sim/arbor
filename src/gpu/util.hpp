#pragma once

// the atomic update overloads for double are in the root namespace
// so that they match the namespace of the CUDA builtin equivalents
// for 32 bit float and 32/64 bit int
/*
__device__
inline double atomicAdd(double* address, double val)
{
    using I = unsigned long long int;
    I* address_as_ull = (I*)address;
    I old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(
                address_as_ull,
                assumed,
                __double_as_longlong(val + __longlong_as_double(assumed))
        );
    } while (assumed != old);
    return __longlong_as_double(old);
}

__device__
static double atomicSub(double* address, double val)
{
    return atomicAdd(address, -val);
}

__device__
static float atomicSub(float* address, float val)
{
    return atomicAdd(address, -val);
}
*/

namespace nest {
namespace mc {
namespace util {
namespace gpu {



} // namespace gpu
} // namespace util
} // namespace mc
} // namespace nest
