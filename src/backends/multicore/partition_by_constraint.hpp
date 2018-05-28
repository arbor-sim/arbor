#pragma once

#include<vector>
#include<simd/simd.hpp>

namespace arb {
namespace multicore {

namespace S = ::arb::simd;
static constexpr unsigned simd_width_ = S::simd_abi::native_width<fvm_value_type>::value;

struct constraint_partitions {
    using iarray = arb::multicore::iarray;

    static constexpr int num_compartments = 4;
    iarray contiguous;
    iarray constant;
    iarray independent;
    iarray none;
};

template <typename T>
bool is_contiguous(const T& node_index, unsigned i) {
    for (unsigned j = i + 1; j < i + simd_width_; j++) {
        if(node_index[j] != node_index[j - 1] + 1)
            return false;
    }
    return true;
}

template <typename T>
bool is_constant(const T& node_index, unsigned i) {
    for (unsigned j = i + 1; j < i + simd_width_; j++) {
        if(node_index[j] != node_index[j - 1])
            return false;
    }
    return true;
}

template <typename T>
bool is_independent(const T& node_index, unsigned i) {
    for (unsigned j = i + 1; j < i + simd_width_; j++) {
        if(node_index[j] == node_index[j - 1])
            return false;
    }
    return true;
}

template <typename T>
index_constraint get_subvector_index_constraint(const T& node_index, unsigned i) {

    if (is_contiguous(node_index, i))
        return index_constraint::contiguous;

    else if (is_constant(node_index, i))
        return index_constraint::constant;

    else if (is_independent(node_index, i))
        return index_constraint::independent;

    else
        return index_constraint::none;
}

template <typename T>
void generate_index_constraint_partitions(const T& node_index, constraint_partitions& partitions, unsigned width) {

    for (unsigned i = 0; i < node_index.size(); i+= simd_width_) {
        index_constraint constraint = get_subvector_index_constraint(node_index, i);
        switch(constraint) {
            case index_constraint::none: {
                partitions.none.push_back(i);
            }
            break;
            case index_constraint::independent: {
                partitions.independent.push_back(i);
            }
            break;
            case index_constraint::constant: {
                partitions.constant.push_back(i);
            }
            break;
            case index_constraint::contiguous: {
                partitions.contiguous.push_back(i);
            }
            break;
        }
    }

    if(simd_width_ != 1) {
        unsigned size_of_constant_section = ((width + (simd_width_ - 1))/ simd_width_) -
                                            (partitions.none.size() +
                                             partitions.independent.size() +
                                             partitions.contiguous.size() );

        partitions.constant.resize(size_of_constant_section);
    }
    else {
        partitions.contiguous.resize(width);
    }

}

template <typename T>
bool compatible_index_constraints(const T& node_index, const T& ion_index){
    for (unsigned i = 0; i < node_index.size(); i+= simd_width_) {
        index_constraint node_constraint = get_subvector_index_constraint(node_index, i);
        index_constraint ion_constraint = get_subvector_index_constraint(ion_index, i);
        if(node_constraint != ion_constraint) {
            if(!((node_constraint == index_constraint::none) ||
               (node_constraint == index_constraint::independent &&
                ion_constraint == index_constraint::contiguous))) {
                return false;
            }
        }
    }
    return true;
}

} // namespace util
} // namespace arb



