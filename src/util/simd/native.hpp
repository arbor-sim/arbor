#pragma once

// Native SIMD representations based on architecture.

namespace arb {
namespace simd_abi {

template <typename Value, unsigned N>
struct native {
    using type = void;
};

} // namespace simd_abi
} // namespace arb

#define ARB_DEF_NATIVE_SIMD_(T, N, A)\
namespace arb { namespace simd_abi {\
template <> struct native<T, N> { using type = typename A<T, N>::type; };\
}}


#if defined(__AVX2__)

#include <util/simd/avx.hpp>
ARB_DEF_NATIVE_SIMD_(int, 4, avx2)
ARB_DEF_NATIVE_SIMD_(double, 4, avx2)

#elif defined(__AVX__)

#include <util/simd/avx.hpp>
ARB_DEF_NATIVE_SIMD_(int, 4, avx)
ARB_DEF_NATIVE_SIMD_(double, 4, avx)

// #elif...

#endif


