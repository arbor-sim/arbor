#pragma once

/*
 * Convenience functions, structs used across
 * more than one unit test.
 */

#include <cmath>
#include <string>
#include <utility>

#include "../gtest.h"

namespace testing {

// string ctor suffix (until C++14!)

namespace string_literals {
    inline std::string operator ""_s(const char* s, std::size_t n) {
        return std::string(s, n);
    }
}

// sentinel for use with range-related tests

struct null_terminated_t {
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

// wrap a value type, with copy operations disabled

template <typename V>
struct nocopy {
    V value;

    nocopy(): value{} {}
    nocopy(V v): value(v) {}
    nocopy(const nocopy& n) = delete;

    nocopy(nocopy&& n) {
        value=n.value;
        n.value=V{};
        ++move_ctor_count;
    }

    nocopy& operator=(const nocopy& n) = delete;
    nocopy& operator=(nocopy&& n) {
        value=n.value;
        n.value=V{};
        ++move_assign_count;
        return *this;
    }

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

// wrap a value type, with move operations disabled

template <typename V>
struct nomove {
    V value;

    nomove(): value{} {}
    nomove(V v): value(v) {}
    nomove(nomove&& n) = delete;

    nomove(const nomove& n): value(n.value) {
        ++copy_ctor_count;
    }

    nomove& operator=(nomove&& n) = delete;

    nomove& operator=(const nomove& n) {
        value=n.value;
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

// Google Test assertion-returning predicates:

// Assert two sequences of floating point values are almost equal.
// (Uses internal class `FloatingPoint` from gtest.)
template <typename FPType, typename Seq1, typename Seq2>
::testing::AssertionResult seq_almost_eq(Seq1&& seq1, Seq2&& seq2) {
    using std::begin;
    using std::end;

    auto i1 = begin(seq1);
    auto i2 = begin(seq2);

    auto e1 = end(seq1);
    auto e2 = end(seq2);

    for (std::size_t j = 0; i1!=e1 && i2!=e2; ++i1, ++i2, ++j) {
        using FP = testing::internal::FloatingPoint<FPType>;

        // cast to FPType to avoid warnings about lowering conversion
        // if FPType has lower precision than Seq{12}::value_type
        auto v1 = *i1;
        auto v2 = *i2;

        if (!FP{v1}.AlmostEquals(FP{v2})) {
            return ::testing::AssertionFailure() << "floating point numbers " << v1 << " and " << v2 << " differ at index " << j;
        }

    }

    if (i1!=e1 || i2!=e2) {
        return ::testing::AssertionFailure() << "sequences differ in length";
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
