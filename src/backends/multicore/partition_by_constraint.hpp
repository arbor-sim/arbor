#pragma once

#include<vector>
#include<simd/simd.hpp>

namespace arb {
namespace multicore {
    
namespace S = ::arb::simd;
static constexpr unsigned simd_width_ = S::simd_abi::native_width<fvm_value_type>::value;

   struct constraint_partition {

       using iarray = arb::multicore::iarray;

       static constexpr int num_compartments = 4;
       iarray full_index_compartments;
       std::vector<int> compartment_sizes;
   };


   template <typename T>
   void gen_constraint(const T& node_index, constraint_partition& partitioned_index) {
       using iarray = arb::multicore::iarray;

       iarray serial_part;
       iarray independent_part;
       iarray contiguous_part;
       iarray constant_part;

       for (unsigned i = 0; i < node_index.size(); i+= simd_width_) {
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
                           if (node_index[j] != node_index[j - 1] + 1)
                               con = index_constraint::independent;
                       }
                           break;
                       default: {
                       }
                   }
               }
           }
           switch(con) {
               case index_constraint::none: {
                   for (unsigned j = 0; j < simd_width_; j++)
                       serial_part.push_back(node_index[i + j]);
               }
               break;
               case index_constraint::independent: {
                   for (unsigned j = 0; j < simd_width_; j++)
                       independent_part.push_back(node_index[i + j]);
               }
               break;
               case index_constraint::constant: {
                   for (unsigned j = 0; j < simd_width_; j++)
                       constant_part.push_back(node_index[i + j]);
               }
               break;
               case index_constraint::contiguous: {
                   for (unsigned j = 0; j < simd_width_; j++)
                       contiguous_part.push_back(node_index[i + j]);
               }
               break;
           }
       }

       partitioned_index.full_index_compartments.reserve(
               serial_part.size() + independent_part.size() +
               contiguous_part.size() + constant_part.size() );// preallocate memory

       partitioned_index.full_index_compartments.insert(
               partitioned_index.full_index_compartments.end(),
               contiguous_part.begin(), contiguous_part.end() );
       partitioned_index.full_index_compartments.insert(
               partitioned_index.full_index_compartments.end(),
               independent_part.begin(), independent_part.end() );
       partitioned_index.full_index_compartments.insert(
               partitioned_index.full_index_compartments.end(),
               serial_part.begin(), serial_part.end() );
       partitioned_index.full_index_compartments.insert(
               partitioned_index.full_index_compartments.end(),
               constant_part.begin(), constant_part.end() );

       partitioned_index.compartment_sizes.push_back(contiguous_part.size());
       partitioned_index.compartment_sizes.push_back(independent_part.size());
       partitioned_index.compartment_sizes.push_back(serial_part.size());
       partitioned_index.compartment_sizes.push_back(constant_part.size());
   }

} // namespace util
} // namespace arb



