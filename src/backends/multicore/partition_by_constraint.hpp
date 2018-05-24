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
        iarray contiguous_indices;
        iarray constant_indices;
        iarray independent_indices;
        iarray serialized_indices;
        int size;
        //std::vector<int> compartment_sizes;
        //std::vector<int> compartment_starts_and_ends;
    };


    template <typename T>
    index_constraint get_subvector_index_constraint(const T& node_index, unsigned i) {
        index_constraint con = index_constraint::contiguous;
        if(simd_width_ != 1) {
            if(node_index[i] == node_index[i + 1])
                con = index_constraint::constant;
            for (unsigned j = i + 1; j < i + simd_width_; j++) {
                switch (con) {
                    case index_constraint::independent: {
                        if (node_index[j] == node_index[j - 1])
                            con = index_constraint::none;
                    }
                    break;
                    case index_constraint::constant: {
                        if (node_index[j] != node_index[j - 1])
                            con = index_constraint::none;
                    }
                    break;
                    case index_constraint::contiguous: {
                        if (node_index[j] != node_index[j - 1] + 1) {
                            con = index_constraint::independent;
                            if(node_index[j] == node_index[j - 1])
                                con = index_constraint::none;
                        }
                    }
                    break;
                    default: {
                    }
                }
            }
        }
        return con;
    }

    template <typename T>
    void gen_constraint(const T& node_index, constraint_partitions& partitioned_indices, unsigned width) {

        for (unsigned i = 0; i < node_index.size(); i+= simd_width_) {
            index_constraint con = get_subvector_index_constraint(node_index, i);
            switch(con) {
                case index_constraint::none: {
                    partitioned_indices.serialized_indices.push_back(i);
                }
                break;
                case index_constraint::independent: {
                    partitioned_indices.independent_indices.push_back(i);
                }
                break;
                case index_constraint::constant: {
                    partitioned_indices.constant_indices.push_back(i);
                }
                break;
                case index_constraint::contiguous: {
                    partitioned_indices.contiguous_indices.push_back(i);
                }
                break;
            }
        }
 
        if(simd_width_ != 1) {
            unsigned size_of_constant_section = ((width + (simd_width_ - 1))/ simd_width_) -
                                                (partitioned_indices.serialized_indices.size() +
                                                 partitioned_indices.independent_indices.size() +
                                                 partitioned_indices.contiguous_indices.size() );
 
            partitioned_indices.constant_indices.resize(size_of_constant_section);
        }
        else {
            partitioned_indices.contiguous_indices.resize(width);
        }
 
        partitioned_indices.size = node_index.size();
 
    }
  
    template <typename T>
    bool compatible_constraint_indices(const T& node_index, const T& ion_index){
        for (unsigned i = 0; i < node_index.size(); i+= simd_width_) {
            index_constraint node_constraint = get_subvector_index_constraint(node_index, i);
            index_constraint ion_constraint = get_subvector_index_constraint(ion_index, i);
            if(node_constraint != ion_constraint) {
                if(!((node_constraint == index_constraint::none) ||
                   (node_constraint == index_constraint::independent &&
                    ion_constraint == index_constraint::contiguous))) {
                    std::cout << static_cast<std::underlying_type<index_constraint>::type>(node_constraint)<<std::endl;
                    std::cout << static_cast<std::underlying_type<index_constraint>::type>(ion_constraint)<<std::endl;
                    return false;
                }
            }
 
        }
        return true;
    }

} // namespace util
} // namespace arb



