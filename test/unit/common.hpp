#pragma once

/*
 * Convenience functions, structs used across
 * more than one unit test.
 */

#include <cmath>
#include <string>
#include <type_traits>
#include <utility>

#include "../gtest.h"

// Pair printer.

namespace std {
    template <typename A, typename B>
    std::ostream& operator<<(std::ostream& out, const std::pair<A, B>& p) {
        return out << '(' << p.first << ',' << p.second << ')';
    }
}

namespace testing {

// Sentinel for C-style strings, for use with range-related tests.

struct null_terminated_t {
    bool operator==(null_terminated_t) const { return true; }
    bool operator!=(null_terminated_t) const { return false; }

    bool operator==(const char *p) const { return !*p; }
    bool operator!=(const char *p) const { return !!*p; }

    friend bool operator==(const char *p, null_terminated_t x) {
        return x==p;
    }

    friend bool operator!=(const char *p, null_terminated_t x) {
        return x!=p;
    }

    constexpr null_terminated_t() {}
};

constexpr null_terminated_t null_terminated;

template <typename... A>
struct matches_cvref_impl: std::false_type {};

template <typename X>
struct matches_cvref_impl<X, X>: std::true_type {};

template <typename... A>
using matches_cvref = matches_cvref_impl<std::remove_cv_t<std::remove_reference_t<A>>...>;

// Wrap a value type, with copy operations disabled.

template <typename V>
struct nocopy {
    V value;

    template <typename... A>
    using is_self = matches_cvref<nocopy, A...>;

    template <typename... A, typename = std::enable_if_t<!is_self<A...>::value>>
    nocopy(A&&... a): value(std::forward<A>(a)...) {}

    nocopy(nocopy& n) = delete;
    nocopy(const nocopy& n) = delete;

    nocopy(nocopy&& n): value(std::move(n.value)) {
        n.clear();
        ++move_ctor_count;
    }

    nocopy& operator=(const nocopy& n) = delete;
    nocopy& operator=(nocopy&& n) {
        value = std::move(n.value);
        n.clear();
        ++move_assign_count;
        return *this;
    }

    template <typename U = V>
    std::enable_if_t<std::is_default_constructible<U>::value> clear() { value = V{}; }

    template <typename U = V>
    std::enable_if_t<!std::is_default_constructible<U>::value> clear() {}

    bool operator==(const nocopy& them) const { return them.value==value; }
    bool operator!=(const nocopy& them) const { return !(*this==them); }

    static int move_ctor_count;
    static int move_assign_count;
    static void reset_counts() {
        move_ctor_count = 0;
        move_assign_count = 0;
    }
};

template <typename V>
int nocopy<V>::move_ctor_count;

template <typename V>
int nocopy<V>::move_assign_count;

// Wrap a value type, with move operations disabled.

template <typename V>
struct nomove {
    V value;

    template <typename... A>
    using is_self = matches_cvref<nomove, A...>;

    template <typename... A, typename = std::enable_if_t<!is_self<A...>::value>>
    nomove(A&&... a): value(std::forward<A>(a)...) {}

    nomove(nomove& n): value(n.value) { ++copy_ctor_count; }
    nomove(const nomove& n): value(n.value) { ++copy_ctor_count; }

    nomove& operator=(nomove&& n) = delete;

    nomove& operator=(const nomove& n) {
        value = n.value;
        ++copy_assign_count;
        return *this;
    }

    bool operator==(const nomove& them) const { return them.value==value; }
    bool operator!=(const nomove& them) const { return !(*this==them); }

    static int copy_ctor_count;
    static int copy_assign_count;
    static void reset_counts() {
        copy_ctor_count = 0;
        copy_assign_count = 0;
    }
};

template <typename V>
int nomove<V>::copy_ctor_count;

template <typename V>
int nomove<V>::copy_assign_count;


// Subvert class access protections. Demo:
//
//     class foo {
//         int secret = 7;
//     };
//
//     int foo::* secret_mptr;
//     template class access::bind<int foo::*, secret_mptr, &foo::secret>;
//
//     int seven = foo{}.*secret_mptr;
//
// Or with shortcut define (places global in anonymous namespace):
//
//     ACCESS_BIND(int foo::*, secret_mptr, &foo::secret)
//
//     int seven = foo{}.*secret_mptr;

namespace access {
    template <typename V, V& store, V value>
    struct bind {
        static struct binder {
            binder() { store = value; }
        } init;
    };

    template <typename V, V& store, V value>
    typename bind<V, store, value>::binder bind<V, store, value>::init;
} // namespace access

#define ACCESS_BIND(type, global, value)\
namespace { using global ## _type_ = type; global ## _type_ global; }\
template struct ::testing::access::bind<type, global, value>;


// Google Test assertion-returning predicates:

// Assert two values are 'almost equal', with exact test for non-floating point types.
// (Uses internal class `FloatingPoint` from gtest.)

template <typename FPType>
::testing::AssertionResult almost_eq_(FPType a, FPType b, std::true_type) {
    using FP = testing::internal::FloatingPoint<FPType>;

    if ((std::isnan(a) && std::isnan(b)) || FP{a}.AlmostEquals(FP{b})) {
        return ::testing::AssertionSuccess();
    }

    return ::testing::AssertionFailure() << "floating point numbers " << a << " and " << b << " differ";
}

template <typename X>
::testing::AssertionResult almost_eq_(const X& a, const X& b, std::false_type) {
    if (a==b) {
        return ::testing::AssertionSuccess();
    }

    return ::testing::AssertionFailure() << "values " << a << " and " << b << " differ";
}

template <typename X>
::testing::AssertionResult almost_eq(const X& a, const X& b) {
    return almost_eq_(a, b, typename std::is_floating_point<X>::type{});
}

// Assert two sequences of floating point values are almost equal, with explicit
// specification of floating point type.

template <typename FPType, typename Seq1, typename Seq2>
::testing::AssertionResult seq_almost_eq(Seq1&& seq1, Seq2&& seq2) {
    using std::begin;
    using std::end;

    auto i1 = begin(seq1);
    auto i2 = begin(seq2);

    auto e1 = end(seq1);
    auto e2 = end(seq2);

    for (std::size_t j = 0; i1!=e1 && i2!=e2; ++i1, ++i2, ++j) {

        auto v1 = *i1;
        auto v2 = *i2;

        // Cast to FPType to avoid warnings about lowering conversion
        // if FPType has lower precision than Seq{12}::value_type.

        auto status = almost_eq((FPType)(v1), (FPType)(v2));
        if (!status) return status << " at index " << j;
    }

    if (i1!=e1 || i2!=e2) {
        return ::testing::AssertionFailure() << "sequences differ in length";
    }
    return ::testing::AssertionSuccess();
}

template <typename V>
inline bool generic_isnan(const V& x) { return false; }
inline bool generic_isnan(float x) { return std::isnan(x); }
inline bool generic_isnan(double x) { return std::isnan(x); }
inline bool generic_isnan(long double x) { return std::isnan(x); }

template <typename U, typename V>
static bool equiv(const U& u, const V& v) {
    return u==v || (generic_isnan(u) && generic_isnan(v));
}

template <typename Seq1, typename Seq2>
::testing::AssertionResult seq_eq(Seq1&& seq1, Seq2&& seq2) {
    using std::begin;
    using std::end;

    auto i1 = begin(seq1);
    auto i2 = begin(seq2);

    auto e1 = end(seq1);
    auto e2 = end(seq2);

    for (std::size_t j = 0; i1!=e1 && i2!=e2; ++i1, ++i2, ++j) {
        auto v1 = *i1;
        auto v2 = *i2;

        if (!equiv(v1, v2)) {
            return ::testing::AssertionFailure() << "values " << v1 << " and " << v2 << " differ at index " << j;
        }
    }

    if (i1!=e1 || i2!=e2) {
        return ::testing::AssertionFailure() << "sequences differ in length";
    }
    return ::testing::AssertionSuccess();
}

// Assert elements 0..n-1 inclusive of two indexed collections are exactly equal.

template <typename Arr1, typename Arr2>
::testing::AssertionResult indexed_eq_n(int n, Arr1&& a1, Arr2&& a2) {
    for (int i = 0; i<n; ++i) {
        auto v1 = a1[i];
        auto v2 = a2[i];

        if (!equiv(v1,v2)) {
            return ::testing::AssertionFailure() << "values " << v1 << " and " << v2 << " differ at index " << i;
        }
    }

    return ::testing::AssertionSuccess();
}

// Assert elements 0..n-1 inclusive of two indexed collections are almost equal.

template <typename Arr1, typename Arr2>
::testing::AssertionResult indexed_almost_eq_n(int n, Arr1&& a1, Arr2&& a2) {
    for (int i = 0; i<n; ++i) {
        auto v1 = a1[i];
        auto v2 = a2[i];

        auto status = almost_eq(v1, v2);
        if (!status) return status << " at index " << i;
    }

    return ::testing::AssertionSuccess();
}

// Assert two floating point values are within a relative tolerance.

inline ::testing::AssertionResult near_relative(double a, double b, double relerr) {
    double tol = relerr*std::max(std::abs(a), std::abs(b));
    if (std::abs(a-b)>tol) {
        return ::testing::AssertionFailure() << "relative error between floating point numbers " << a << " and " << b << " exceeds tolerance " << relerr;
    }
    return ::testing::AssertionSuccess();
}

} // namespace testing
