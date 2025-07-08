#pragma once

#include "backends/multicore/multicore_common.hpp"
#include "util/rangeutil.hpp"

#include <arbor/simd/simd.hpp>
#include <arbor/serdes.hpp>

namespace arb {
namespace multicore {

namespace S = ::arb::simd;
using S::index_constraint;

struct constraint_partition {
    using iarray = arb::multicore::iarray;
    // sequence of _ranges_ of contiguous index block of width N
    iarray contiguous;
    // sequence of constant index blocks of width N
    iarray constant;
    // sequence of independent index blocks of width N
    iarray independent;
    // sequence of unconstrained index blocks of width N
    iarray none;

    ARB_SERDES_ENABLE(constraint_partition, contiguous, constant, independent, none);
};

// contiguous over width n
//
// all n values are strictly monotonic
template <typename It>
bool is_contiguous_n(It first, unsigned width) {
    while (--width) {
        It next = first;
        if ((*first) +1 != *++next) {
            return false;
        }
        first = next;
    }
    return true;
}

// constant over width n
//
// all n values are equal to the first
template <typename It>
bool is_constant_n(It first, unsigned width) {
    while (--width) {
        It next = first;
        if (*first != *++next) {
            return false;
        }
        first = next;
    }
    return true;
}

// independent over width n
//
// no repetitions?? so 0 1 0 1 qualifies, but not 0 0 1 1
template <typename It>
bool is_independent_n(It first, unsigned width) {
    while (--width) {
        It next = first;
        if (*first == *++next) {
            return false;
        }
        first = next;
    }
    return true;
}

template <typename It>
index_constraint idx_constraint(It it, unsigned simd_width) {
    if (is_contiguous_n(it, simd_width)) {
        return index_constraint::contiguous;
    }
    else if (is_constant_n(it, simd_width)) {
        return index_constraint::constant;
    }
    else if (is_independent_n(it, simd_width)) {
        return index_constraint::independent;
    }
    else {
        return index_constraint::none;
    }
}

template <typename T>
constraint_partition make_constraint_partition(const T& node_index, unsigned width, unsigned simd_width) {
    if (!simd_width) return {};
    arb_assert(util::is_sorted(node_index));
    constraint_partition part;
    unsigned idx = 0;
    while (idx < width) {
        auto ptr = &node_index[idx];
        // First, greedily absorb the contiguous part of the index array. As the
        // array is sorted, the greedy strategy produces the maximum ranges. If
        // the contiguous prefix is exhaust, try the other options in decreasing
        // order of strength.
        auto beg = idx;
        while (idx < width && is_contiguous_n(&node_index[idx], simd_width)) idx += simd_width;
        if (idx > beg) {
            // NB. This one is different from the others
            // 1. we already _have_ bumped idx as far as possible
            // 2. we need to push the start _and_ the end (=idx) of the range
            part.contiguous.push_back(beg);
            part.contiguous.push_back(idx);
        }
        else if (is_constant_n(ptr, simd_width)) {
            part.constant.push_back(idx);
            idx += simd_width;
        }
        else if (is_independent_n(ptr, simd_width)) {
            part.independent.push_back(idx);
            idx += simd_width;
        }
        else {
            part.none.push_back(idx);
            idx += simd_width;
        }
    }
    return part;
}

bool constexpr is_constraint_stronger(index_constraint a, index_constraint b) {
    return a==b ||
           a==index_constraint::none ||
           (a==index_constraint::independent && b==index_constraint::contiguous);
}

template <typename T, typename U>
bool compatible_index_constraints(const T& node_index, const U& ion_index, unsigned simd_width){
    for (unsigned i = 0; i < node_index.size(); i+= simd_width) {
        auto nc = idx_constraint(&node_index[i], simd_width);
        auto ic = idx_constraint(&ion_index[i], simd_width);
        if (!is_constraint_stronger(nc, ic)) return false;
    }
    return true;
}

} // namespace util
} // namespace arb



