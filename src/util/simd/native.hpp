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

#endif

#if defined(__AVX512F__)

#include <util/simd/avx512.hpp>
ARB_DEF_NATIVE_SIMD_(int, 8, avx512)
ARB_DEF_NATIVE_SIMD_(double, 8, avx512)

#endif


namespace arb {
namespace simd_abi {

// Define native widths based on largest native vector implementation
// of corresponding type. Presume power of 2, no larger than largest
// possible over implemented architectures.

template <typename Value, int k = 64>
struct native_width;

template <typename Value, int k>
struct native_width {
   static constexpr int value =
        std::is_same<void, typename native<Value, k>::type>::value?
        native_width<Value, k/2>::value:
        k;
};

template <typename Value>
struct native_width<Value, 1> {
    static constexpr int value = 1;
};

} // namespace simd_abi
} // namespace arb
