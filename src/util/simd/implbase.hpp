#pragma once

// Base class (via CRTP) for concrete implementation
// classes, with default implementations based on
// copy_to/copy_from.
//
// Also provides simd_detail::simd_traits type map.

#include <cstring>
#include <cmath>
#include <algorithm>
#include <iterator>

// Derived class I must at minimum provide:
//
// * specialization of simd_traits.
//
// * implementations (static) for copy_to, copy_from:
//
//     void I::copy_to(const vector_type&, scalar_type*)
//     vector_type I::copy_from(const scalar_type*)
//
//     void I::mask_copy_to(const vector_type&, scalar_type*)
//     vector_type I::mask_copy_from(const bool*)
//
// * implementations (static) for mask element get/set:
//
//     bool I::mask_element(const vector_type& v, int i);
//     void I::mask_set_element(vector_type& v, int i, bool x);

namespace arb {
namespace simd_detail {

// The simd_traits class provides the mapping between a concrete SIMD
// implementation I and its associated classes. This must be specialized
// for each concrete implementation.

template <typename I>
struct simd_traits {
    static constexpr unsigned width = 0;
    using scalar_type = void;
    using vector_type = void;
    using mask_impl = void;
};

template <typename I>
struct implbase {
    constexpr static unsigned width = simd_traits<I>::width;
    using scalar_type = typename simd_traits<I>::scalar_type;
    using vector_type = typename simd_traits<I>::vector_type;

    using mask_impl = typename simd_traits<I>::mask_impl;
    using mask_type = typename simd_traits<mask_impl>::vector_type;

    using store = scalar_type [width];
    using mask_store = bool [width];

    static vector_type broadcast(scalar_type x) {
        store a;
        std::fill(std::begin(a), std::end(a), x);
        return I::copy_from(a);
    }

    static scalar_type element(const vector_type& v, int i) {
        store a;
        I::copy_to(v, a);
        return a[i];
    }

    static void set_element(vector_type& v, int i, scalar_type x) {
        store a;
        I::copy_to(v, a);
        a[i] = x;
        v = I::copy_from(a);
    }

    static vector_type add(const vector_type& u, const vector_type& v) {
        store a, b, r;
        I::copy_to(u, a);
        I::copy_to(v, b);

        for (unsigned i = 0; i<width; ++i) {
            r[i] = a[i]+b[i];
        }
        return I::copy_from(r);
    }

    static vector_type mul(const vector_type& u, const vector_type& v) {
        store a, b, r;
        I::copy_to(u, a);
        I::copy_to(v, b);

        for (unsigned i = 0; i<width; ++i) {
            r[i] = a[i]*b[i];
        }
        return I::copy_from(r);
    }

    static vector_type sub(const vector_type& u, const vector_type& v) {
        store a, b, r;
        I::copy_to(u, a);
        I::copy_to(v, b);

        for (unsigned i = 0; i<width; ++i) {
            r[i] = a[i]-b[i];
        }
        return I::copy_from(r);
    }

    static vector_type div(const vector_type& u, const vector_type& v) {
        store a, b, r;
        I::copy_to(u, a);
        I::copy_to(v, b);

        for (unsigned i = 0; i<width; ++i) {
            r[i] = a[i]/b[i];
        }
        return I::copy_from(r);
    }

    static vector_type fma(const vector_type& u, const vector_type& v, const vector_type& w) {
        store a, b, c, r;
        I::copy_to(u, a);
        I::copy_to(v, b);
        I::copy_to(w, c);

        for (unsigned i = 0; i<width; ++i) {
            r[i] = std::fma(a[i], b[i], c[i]);
        }
        return I::copy_from(r);
    }

    static mask_type cmp_eq(const vector_type& u, const vector_type& v) {
        store a, b;
        mask_store r;
        I::copy_to(u, a);
        I::copy_to(v, b);

        for (unsigned i = 0; i<width; ++i) {
            r[i] = a[i]==b[i];
        }
        return mask_impl::mask_copy_from(r);
    }

    static mask_type cmp_neq(const vector_type& u, const vector_type& v) {
        store a, b;
        mask_store r;
        I::copy_to(u, a);
        I::copy_to(v, b);

        for (unsigned i = 0; i<width; ++i) {
            r[i] = a[i]!=b[i];
        }
        return mask_impl::mask_copy_from(r);
    }

    static mask_type cmp_gt(const vector_type& u, const vector_type& v) {
        store a, b;
        mask_store r;
        I::copy_to(u, a);
        I::copy_to(v, b);

        for (unsigned i = 0; i<width; ++i) {
            r[i] = a[i]>b[i];
        }
        return mask_impl::mask_copy_from(r);
    }

    static mask_type cmp_geq(const vector_type& u, const vector_type& v) {
        store a, b;
        mask_store r;
        I::copy_to(u, a);
        I::copy_to(v, b);

        for (unsigned i = 0; i<width; ++i) {
            r[i] = a[i]>=b[i];
        }
        return mask_impl::mask_copy_from(r);
    }

    static mask_type cmp_lt(const vector_type& u, const vector_type& v) {
        store a, b;
        mask_store r;
        I::copy_to(u, a);
        I::copy_to(v, b);

        for (unsigned i = 0; i<width; ++i) {
            r[i] = a[i]<b[i];
        }
        return mask_impl::mask_copy_from(r);
    }

    static mask_type cmp_leq(const vector_type& u, const vector_type& v) {
        store a, b;
        mask_store r;
        I::copy_to(u, a);
        I::copy_to(v, b);

        for (unsigned i = 0; i<width; ++i) {
            r[i] = a[i]<=b[i];
        }
        return mask_impl::mask_copy_from(r);
    }

    static vector_type logical_not(const vector_type& u) {
        store a, r;
        I::copy_to(u, a);

        for (unsigned i = 0; i<width; ++i) {
            r[i] = !a[i];
        }
        return I::copy_from(r);
    }

    static vector_type logical_and(const vector_type& u, const vector_type& v) {
        store a, b, r;
        I::copy_to(u, a);
        I::copy_to(v, b);

        for (unsigned i = 0; i<width; ++i) {
            r[i] = a[i] && b[i];
        }
        return I::copy_from(r);
    }

    static vector_type logical_or(const vector_type& u, const vector_type& v) {
        store a, b, r;
        I::copy_to(u, a);
        I::copy_to(v, b);

        for (unsigned i = 0; i<width; ++i) {
            r[i] = a[i] || b[i];
        }
        return I::copy_from(r);
    }

    static vector_type select(const mask_type& mask, const vector_type& u, const vector_type& v) {
        store a, b, r;
        mask_store m;
        I::mask_copy_to(mask, m);
        I::copy_to(u, a);
        I::copy_to(v, b);

        for (unsigned i = 0; i<width; ++i) {
            r[i] = m[i]? b[i]: a[i];
        }
        return I::copy_from(r);
    }

    static vector_type mask_broadcast(bool v) {
        mask_store m;
        std::fill(std::begin(m), std::end(m), v);
        return I::mask_copy_from(m);
    }

    template <typename ImplIndex>
    static vector_type gather(ImplIndex, const scalar_type* p, const typename ImplIndex::vector_type& index) {
        typename ImplIndex::scalar_type o[width];
        ImplIndex::copy_to(index, o);

        scalar_type data[width];
        for (unsigned i = 0; i<width; ++i) {
            data[i] = p[o[i]];
        }
        return I::copy_from(data);
    }
};


} // namespace simd_detail
} // namespace arb
