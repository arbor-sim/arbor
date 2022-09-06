#pragma once

#include <iostream>

#include <arbor/export.hpp>
#include <arbor/fvm_types.hpp>

namespace arb {
namespace gpu {


// Helper type for branch meta data in setup phase of fine grained
// matrix storage+solver.
//
//      leaf
//      .
//      .
//      .
//  -   *
//      |
//  l   *
//  e   |
//  n   *
//  g   |
//  t   *
//  h   |
//  -   start_idx
//      |
//      parent_idx
//      |
//      .
//      .
//      .
//      root
struct branch {
    unsigned id;         // branch id
    unsigned parent_id;  // parent branch id
    unsigned parent_idx; //
    unsigned start_idx;  // the index of the first node in the input parent index
    unsigned length;     // the number of nodes in the branch
};

// order branches by:
//  - descending length
//  - ascending id
inline
bool operator<(const branch& lhs, const branch& rhs) {
    if (lhs.length!=rhs.length) {
        return lhs.length>rhs.length;
    } else {
        return lhs.id<rhs.id;
    }
}

inline
std::ostream& operator<<(std::ostream& o, branch b) {
    return o << "[" << b.id
        << ", len " << b.length
        << ", pid " << b.parent_idx
        << ", sta " << b.start_idx
        << "]";
}

struct level_metadata {
    unsigned num_branches = 0; // Number of branches in a level
    unsigned max_length = 0;   // Length of the longest branch
    unsigned matrix_data_index = 0;   // Index into data values (d, u, rhs) of the first branch
    unsigned level_data_index  = 0;   // Index into data values (lengths, parents) of each level
};

// C wrappers around kernels
ARB_ARBOR_API void gather(
    const arb_value_type* from,
    arb_value_type* to,
    const arb_index_type* p,
    unsigned n);

ARB_ARBOR_API void scatter(
    const arb_value_type* from,
    arb_value_type* to,
    const arb_index_type* p,
    unsigned n);

} // gpu
} // arb
