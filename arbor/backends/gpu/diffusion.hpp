#include <arbor/fvm_types.hpp>
#include <arbor/export.hpp>

#include <ostream>

#include "fine.hpp"

namespace arb {
namespace gpu {

ARB_ARBOR_API void assemble_diffusion(
    fvm_value_type* d,
    fvm_value_type* rhs,
    const fvm_value_type* invariant_d,
    const fvm_value_type* concentration,
    const fvm_value_type* voltage,
    const fvm_value_type* current,
    const fvm_value_type q,
    const fvm_value_type* conductivity,
    const fvm_value_type* area,
    const fvm_index_type* cv_to_intdom,
    const fvm_value_type* dt_intdom,
    const fvm_index_type* perm,
    unsigned n);

ARB_ARBOR_API void solve_diffusion(
    fvm_value_type* rhs,
    fvm_value_type* d,                     // diagonal values
    const fvm_value_type* u,               // upper diagonal (and lower diagonal as the matrix is SPD)
    const level_metadata* level_meta,      // information pertaining to each level
    const fvm_index_type* level_lengths,   // lengths of branches of every level concatenated
    const fvm_index_type* level_parents,   // parents of branches of every level concatenated
    const fvm_index_type* block_index,     // start index (exclusive) into levels for each gpu block
    fvm_index_type* num_cells,             // the number of cells packed into this single matrix
    fvm_index_type* padded_size,           // length of rhs, d, u, including padding
    unsigned num_blocks,                   // number of blocks
    unsigned blocksize);                   // size of each block

} // namespace gpu
} // namespace arb
