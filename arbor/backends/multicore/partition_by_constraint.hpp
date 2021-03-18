#pragma once

#include <vector>

#include <arbor/simd/simd.hpp>
#include <arbor/mechanism_ppack.hpp>

namespace arb {
namespace multicore {

namespace S = ::arb::simd;
using S::index_constraint;

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
    std::vector<fvm_index_type> contiguous, constant, independent, none;

    for (unsigned i = 0; i < width; i+= simd_width) {
        auto ptr = &node_index[i];
        if (is_contiguous_n(ptr, simd_width)) {
            contiguous.push_back(i);
        }
        else if (is_constant_n(ptr, simd_width)) {
            constant.push_back(i);
        }
        else if (is_independent_n(ptr, simd_width)) {
            independent.push_back(i);
        }
        else {
            none.push_back(i);
        }
    }
    auto mv = [](const auto& in) {
        auto ptr = new fvm_index_type[in.size()];
        std::copy(in.begin(), in.end(), ptr);
        return ptr;
    };
    return { contiguous.size(), constant.size(), independent.size(), none.size(),
             mv(contiguous),    mv(constant),    mv(independent),    mv(none) };
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

        if (!is_constraint_stronger(nc, ic)) {
            return false;
        }
    }
    return true;
}

// TODO temporary fix to delete constraints
inline void clear_constraint_partition(constraint_partition& p) {
    delete[] p.contiguous;  p.contiguous  = nullptr; p.n_contiguous  = 0;
    delete[] p.constant;    p.constant    = nullptr; p.n_constant    = 0;
    delete[] p.independent; p.independent = nullptr; p.n_independent = 0;
    delete[] p.none;        p.none        = nullptr; p.n_none        = 0;
}

} // namespace util
} // namespace arb



