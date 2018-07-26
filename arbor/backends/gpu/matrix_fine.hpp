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

    //
    //  the lengths and parents vectors are stored in managed memory
    //

    // An array holding the length of each branch for each branch on this level.
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
    const fvm_value_type* cv_capacitance,
    const fvm_value_type* area,
    const fvm_index_type* cv_to_cell,
    const fvm_value_type* dt_cell,
    const fvm_index_type* perm,
    unsigned n);

void solve_matrix_fine(
    fvm_value_type* rhs,
    fvm_value_type* d,
    const fvm_value_type* u,
    const level* levels,              // pointer to an array containing level meta-data
    unsigned num_cells,               // the number of cells packed into this single matrix
    unsigned num_levels,              // depth of the tree (in branches)
    unsigned padded_size,             // length of rhs, d, u, including padding
    unsigned max_branches_per_level); // the maximum number of branches on any level

} // namespace gpu
} // namespace arb
