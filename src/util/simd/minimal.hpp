#pragma once

#include <array>
#include <cstring>
#include <cmath>

#include <util/simd/implbase.hpp>

// Minimal implementation of a concrete SIMD representation.o
//
// This class is used only for testing the simd_detail::implbase
// class and as a guide for implementors.

namespace arb {
namespace simd_detail {

template <typename T, unsigned N>
struct minimal;

template <typename T, unsigned N>
struct simd_traits<minimal<T, N>> {
    static constexpr unsigned width = N;
    using scalar_type = T;
    using vector_type = std::array<T, N>;
    using mask_impl = minimal<bool, N>;
};

template <typename T, unsigned N>
struct minimal: implbase<minimal<T, N>> {
    using base = implbase<minimal<T, N>>;
    using typename base::scalar_type;
    using typename base::vector_type;

    static vector_type copy_from(const scalar_type* p) {
        vector_type result;
        std::memcpy(&result, p, sizeof(result));
        return result;
    }

    static void copy_to(const vector_type& v, scalar_type *p) {
        std::memcpy(p, &v, sizeof(v));
    }

    static void mask_copy_to(const vector_type& v, bool* w) {
        std::copy(v.begin(), v.end(), w);
    }

    static vector_type mask_copy_from(const bool* y) {
        vector_type v;
        std::copy(y, y+N, v.data());
        return v;
    }

    static bool mask_element(const vector_type& v, int i) {
        return static_cast<bool>(v[i]);
    }

    static void mask_set_element(vector_type& v, int i, bool x) {
        v[i] = x;
    }
};

} // namespace simd_detail

namespace simd_abi {

    template <typename T, unsigned N> struct minimal {
        using type = simd_detail::minimal<T, N>;
    };

} // namespace simd_abi

} // namespace arb
