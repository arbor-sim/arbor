#include <arbor/fvm_types.hpp>

#include <ostream>

namespace arb {
namespace gpu {

struct level {
    level() = default;

    level(unsigned branches);
    level(level&& other);
    level(const level& other);

    ~level();

    unsigned num_branches = 0; // Number of branches
    unsigned max_length = 0;   // Length of the longest branch
    unsigned data_index = 0;   // Index into data values of the first branch

    //  The lengths and parents vectors are raw pointers to managed memory,
    //  so there is need for tricksy deep copy of this type to GPU.

    // An array holding the length of each branch in the level.
    // length: num_branches.
    unsigned* lengths = nullptr;

    // An array with the index of the parent branch for each branch on this level.
    // length: num_branches.
    // When performing backward/forward substitution we need to update/read
    // data values for the parent node for each branch.
    // This can be done easily if we know where the parent branch is located
    // on the next level.
    unsigned* parents = nullptr;
};

std::ostream& operator<<(std::ostream& o, const level& l);

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
    fvm_value_type* d,                // diagonal values
    const fvm_value_type* u,          // upper diagonal (and lower diagonal as the matrix is SPD)
    const level* levels,              // pointer to an array containing level meta-data for all blocks
    const unsigned* levels_end,       // end index (exclusive) into levels for each cuda block
    unsigned* num_cells,              // he number of cells packed into this single matrix
    unsigned* padded_size,            // length of rhs, d, u, including padding
    unsigned num_blocks,              // nuber of blocks
    unsigned blocksize);              // size of each block

} // namespace gpu
} // namespace arb
