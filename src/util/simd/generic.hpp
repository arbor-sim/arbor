#pragma once

#include <array>
#include <cstring>
#include <cmath>

// std::array-backed implementations of simd interface.

namespace arb {
namespace simd_detail {

template <typename T, unsigned N>
struct generic {
    using scalar_type = T;
    using vector_type = std::array<T, N>;

    using mask_impl = generic<int, N>;
    using mask_type = typename mask_impl::vector_type;

    constexpr static unsigned width = N;

    static vector_type broadcast(scalar_type v) {
        vector_type result;
        result.fill(v);
        return result;
    }

    template <typename... V>
    static vector_type immediate(V... vs) {
        vector_type result({static_cast<scalar_type>(vs)...});
        return result;
    }

    static vector_type copy_from(const scalar_type* p) {
        vector_type result;
        std::memcpy(&result, p, sizeof(result));
        return result;
    }

    static void copy_to(const vector_type& v, scalar_type *p) {
        std::memcpy(p, &v, sizeof(v));
    }

    static scalar_type element(const vector_type& v, int i) {
        return v[i];
    }

    static bool bool_element(const vector_type& v, int i) {
        return static_cast<bool>(v[i]);
    }

    static void set_element(vector_type& v, int i, scalar_type x) {
        v[i] = x;
    }

    static vector_type add(const vector_type& u, const vector_type& v) {
        vector_type result;
        for (unsigned i = 0; i<N; ++i) {
            result[i] = u[i]+v[i];
        }
        return result;
    }

    static vector_type mul(const vector_type& u, const vector_type& v) {
        vector_type result;
        for (unsigned i = 0; i<N; ++i) {
            result[i] = u[i]*v[i];
        }
        return result;
    }

    static vector_type sub(const vector_type& u, const vector_type& v) {
        vector_type result;
        for (unsigned i = 0; i<N; ++i) {
            result[i] = u[i]-v[i];
        }
        return result;
    }

    static vector_type div(const vector_type& u, const vector_type& v) {
        vector_type result;
        for (unsigned i = 0; i<N; ++i) {
            result[i] = u[i]/v[i];
        }
        return result;
    }

    static vector_type fma(const vector_type& u, const vector_type& v, const vector_type& w) {
        vector_type result;
        for (unsigned i = 0; i<N; ++i) {
            result[i] = std::fma(u[i], v[i], w[i]);
        }
        return result;
    }

    static mask_type cmp_eq(const vector_type& u, const vector_type& v) {
        mask_type result;
        for (unsigned i = 0; i<N; ++i) {
            result[i] = u[i]==v[i];
        }
        return result;
    }

    static mask_type cmp_not_eq(const vector_type& u, const vector_type& v) {
        mask_type result;
        for (unsigned i = 0; i<N; ++i) {
            result[i] = u[i]!=v[i];
        }
        return result;
    }

    static vector_type logical_not(const vector_type& u) {
        vector_type result;
        for (unsigned i = 0; i<N; ++i) {
            result[i] = !u[i];
        }
        return result;
    }

    static vector_type logical_and(const vector_type& u, const vector_type& v) {
        vector_type result;
        for (unsigned i = 0; i<N; ++i) {
            result[i] = u[i] && v[i];
        }
        return result;
    }

    static vector_type logical_or(const vector_type& u, const vector_type& v) {
        vector_type result;
        for (unsigned i = 0; i<N; ++i) {
            result[i] = u[i] || v[i];
        }
        return result;
    }

    static vector_type select(const mask_type& mask, const vector_type& u, const vector_type& v) {
        vector_type result;
        for (unsigned i = 0; i<N; ++i) {
            result[i] = mask[i]? v[i]: u[i];
        }
        return result;
    }
};


} // namespace simd_detail
} // namespace arb
