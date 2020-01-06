#include <arbor/fvm_types.hpp>

#include <ostream>

namespace arb {
namespace gpu {

struct level_metadata {
    unsigned num_branches = 0; // Number of branches in a level
    unsigned max_length = 0;   // Length of the longest branch
    unsigned matrix_data_index = 0;   // Index into data values (d, u, rhs) of the first branch
    unsigned level_data_index  = 0;   // Index into data values (lengths, parents) of each level
};

// C wrappers around kernels
void gather(
    const fvm_value_type* from,
    fvm_value_type* to,
    const fvm_index_type* p,
    unsigned n);

void scatter(
    const fvm_value_type* from,
    fvm_value_type* to,
    const fvm_index_type* p,
    unsigned n);

void assemble_matrix_fine(
    fvm_value_type* d,
    fvm_value_type* rhs,
    const fvm_value_type* invariant_d,
    const fvm_value_type* voltage,
    const fvm_value_type* current,
    const fvm_value_type* conductivity,
    const fvm_value_type* cv_capacitance,
    const fvm_value_type* area,
    const fvm_index_type* cv_to_cell,
    const fvm_value_type* dt_intdom,
    const fvm_index_type* cell_to_intdom,
    const fvm_index_type* perm,
    unsigned n);

void solve_matrix_fine(
    fvm_value_type* rhs,
    fvm_value_type* d,                     // diagonal values
    const fvm_value_type* u,               // upper diagonal (and lower diagonal as the matrix is SPD)
    const level_metadata* level_meta,      // information pertaining to each level
    const fvm_index_type* level_lengths,   // lengths of branches of every level concatenated
    const fvm_index_type* level_parents,   // parents of branches of every level concatenated
    const fvm_index_type* block_index,     // start index (exclusive) into levels for each cuda block
    fvm_index_type* num_cells,             // the number of cells packed into this single matrix
    fvm_index_type* padded_size,           // length of rhs, d, u, including padding
    unsigned num_blocks,                   // number of blocks
    unsigned blocksize);                   // size of each block

} // namespace gpu
} // namespace arb
