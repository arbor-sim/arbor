#pragma once

// Native SIMD representations based on architecture.

// Predefined macros from the compiler are used to guard
// architecture-specific code here and in the source code
// for the concrete implementation classes.
//
// For x86 vector extensions, the same preprocessor defines
// are used across gcc, clang and icpc: __AVX__, __AVX2__,
// __FMA__, __AVX512F__.
//
// Note that the FMA extensions for x86 are strictly speaking
// independent of AVX2, and instructing gcc (for example)
// to generate AVX2 code with '-mavx2' will not enable FMA
// instructions unless '-mfma' is also given. It is generally
// safer to explicitly request the target using
// '-march', '-mcpu' or '-x', depending on the compiler and
// architecure.

namespace arb {
namespace simd {
namespace simd_abi {

template <typename Value, unsigned N>
struct native {
    using type = void;
};

} // namespace simd_abi
} // namespace simd
} // namespace arb

#define ARB_DEF_NATIVE_SIMD_(T, N, A)\
namespace arb {\
namespace simd {\
namespace simd_abi {\
template <> struct native<T, N> {\
    using type = typename A<T, N>::type;\
};\
}\
}\
}


#if defined(__AVX2__) && defined(__FMA__)

#include <arbor/simd/avx.hpp>
ARB_DEF_NATIVE_SIMD_(int, 4, avx2)
ARB_DEF_NATIVE_SIMD_(double, 4, avx2)

#elif defined(__AVX__)

#include <arbor/simd/avx.hpp>
ARB_DEF_NATIVE_SIMD_(int, 4, avx)
ARB_DEF_NATIVE_SIMD_(double, 4, avx)

#endif

#if defined(__AVX512F__)

#include <arbor/simd/avx512.hpp>
ARB_DEF_NATIVE_SIMD_(int, 8, avx512)
ARB_DEF_NATIVE_SIMD_(double, 8, avx512)

#endif

#if defined(__ARM_FEATURE_SVE)

#include "sve.hpp"
ARB_DEF_NATIVE_SIMD_(int, 0, sve)
ARB_DEF_NATIVE_SIMD_(double, 0, sve)

#elif defined(__ARM_NEON)

#include <arbor/simd/neon.hpp>
ARB_DEF_NATIVE_SIMD_(int, 2, neon)
ARB_DEF_NATIVE_SIMD_(double, 2, neon)

#endif

namespace arb {
namespace simd {
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
    static constexpr int value = std::is_same<void, typename native<Value, 0>::type>::value;
};

} // namespace simd_abi
} // namespace simd
} // namespace arb
