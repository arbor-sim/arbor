#pragma once

#include <cstring>
#include <vector>

#include <arbor/common_types.hpp>

#include "memory/memory.hpp"
#include "util/partition.hpp"
#include "util/rangeutil.hpp"
#include "util/span.hpp"
#include "tree.hpp"
#include "diffusion.hpp"
#include "forest.hpp"
#include "fine.hpp"

namespace arb {
namespace gpu {

template <typename T, typename I>
struct diffusion_state {
    using value_type = T;
    using size_type = I;

    using array      = memory::device_vector<value_type>;
    using view       = typename array::view_type;
    using const_view = typename array::const_view_type;
    using iarray     = memory::device_vector<size_type>;

    using metadata_array = memory::device_vector<level_metadata>;

    array d;     // [μS]
    array u;     // [μS]
    array rhs;   // [nA]

    // Required for matrix assembly
    array cv_volume;           // [μm^3]

    // Invariant part of the matrix diagonal
    array invariant_d;         // [μS]

    // Solution in unpacked format
    array solution_;

    // Maximum number of branches in each level per block
    unsigned max_branches_per_level;

    // Number of rows in matrix
    unsigned matrix_size;

    // Number of cells
    unsigned num_cells;
    iarray num_cells_in_block;

    // End of the data of each level
    iarray data_partition;
    std::size_t data_size;

    // Metadata for each level
    // Includes indices into d, u, rhs and level_lengths, level_parents
    metadata_array level_meta;

    // Stores the lengths (number of compartments) of the branches of each
    // level sequentially in memory. Indexed by level_metadata::level_data_index
    iarray level_lengths;

    // Stores the indices of the parent of each of the branches in each
    // level sequentially in memory. Indexed by level_metadata::level_data_index
    iarray level_parents;

    // Stores the indices to the first level belonging to each block
    // block b owns { levels[block_index[b]], ..., levels[block_index[b+1] - 1] }
    // there is an additional entry at the end of the vector to make the above
    // computation safe
    iarray block_index;

    // Permutation from front end storage to packed storage
    //      `solver_format[perm[i]] = external_format[i]`
    iarray perm;


    diffusion_state() = default;

    // constructor for fine-grained matrix.
    diffusion_state(const std::vector<size_type>& p,
                    const std::vector<size_type>& cell_cv_divs,
                    const std::vector<value_type>& face_diffusivity,
                    const std::vector<value_type>& volume) {
        using util::make_span;
        constexpr unsigned npos = unsigned(-1);

        max_branches_per_level = 128;

        num_cells = cell_cv_divs.size()-1;

        forest trees(p, cell_cv_divs);
        trees.optimize();

        // Now distribute the cells into gpu blocks.
        // While the total number of branches on each level of theses cells in a
        // block are less than `max_branches_per_level` we add more cells. If
        // one block is full, we start a new gpu block.

        unsigned current_block = 0;
        std::vector<unsigned> block_num_branches_per_depth;
        std::vector<unsigned> block_ix(num_cells);

        // Accumulate num cells in block in a temporary vector to be copied to the device
        std::vector<size_type> temp_ncells_in_block;
        temp_ncells_in_block.resize(1, 0);

        // branch_map = branch_maps[block] is a branch map for each gpu block
        // branch_map[depth] is list of branches is this level
        // each branch branch_map[depth][i] has
        // {id, parent_id, start_idx, parent_idx, length}
        std::vector<std::vector<std::vector<branch>>> branch_maps;
        branch_maps.resize(1);

        unsigned num_branches = 0u;
        for (auto c: make_span(0u, num_cells)) {
            auto cell_start = cell_cv_divs[c];
            auto cell_tree = trees.branch_tree(c);
            auto fine_tree = trees.compartment_tree(c);
            auto branch_starts  = trees.branch_offsets(c);
            auto branch_lengths = trees.branch_lengths(c);

            auto depths = depth_from_root(cell_tree);

            // calculate the number of levels in this cell
            auto cell_num_levels = util::max_value(depths)+1u;

            auto num_cell_branches = cell_tree.num_segments();

            // count number of branches per level
            std::vector<unsigned> cell_num_branches_per_depth(cell_num_levels, 0u);
            for (auto i: make_span(num_cell_branches)) {
                cell_num_branches_per_depth[depths[i]] += 1;
            }
            // resize the block levels if neccessary
            if (cell_num_levels > block_num_branches_per_depth.size()) {
                block_num_branches_per_depth.resize(cell_num_levels, 0);
            }


            // check if we can fit the current cell into the last gpu block
            bool fits_current_block = true;
            for (auto i: make_span(cell_num_levels)) {
                unsigned new_branches_per_depth =
                    block_num_branches_per_depth[i]
                    + cell_num_branches_per_depth[i];
                if (new_branches_per_depth > max_branches_per_level) {
                    fits_current_block = false;
                }
            }
            if (fits_current_block) {
                // put the cell into current block
                block_ix[c] = current_block;
                temp_ncells_in_block[block_ix[c]] += 1;
                // and increment counter
                for (auto i: make_span(cell_num_levels)) {
                    block_num_branches_per_depth[i] += cell_num_branches_per_depth[i];
                }
            } else {
                // otherwise start a new block
                block_ix[c] = current_block + 1;
                temp_ncells_in_block.push_back(1);
                branch_maps.resize(branch_maps.size()+1);
                current_block += 1;
                // and reset counter
                block_num_branches_per_depth = cell_num_branches_per_depth;
                for (auto num: block_num_branches_per_depth) {
                    if (num > max_branches_per_level) {
                        throw std::runtime_error(
                            "Could not fit " + std::to_string(num)
                            + " branches in a block of size "
                            + std::to_string(max_branches_per_level));
                    }
                }
            }
            num_cells_in_block = memory::make_const_view(temp_ncells_in_block);


            // the branch map for the block in which we put the cell
            // maps levels to a list of branches in that level
            auto& branch_map = branch_maps[block_ix[c]];

            // build branch_map:
            // branch_map[i] is a list of branch meta-data for branches with depth i
            if (cell_num_levels > branch_map.size()) {
                branch_map.resize(cell_num_levels);
            }
            for (auto i: make_span(num_cell_branches)) {
                branch b;
                auto depth = depths[i];
                // give the branch a unique id number
                b.id = i + num_branches;
                // take care to mark branches with no parents with npos
                b.parent_id = cell_tree.parent(i)==cell_tree.no_parent ?
                    npos : cell_tree.parent(i) + num_branches;
                b.start_idx = branch_starts[i] + cell_start;
                b.length = branch_lengths[i];
                b.parent_idx = p[b.start_idx] + cell_start;
                branch_map[depth].push_back(b);
            }
            // total number of branches of all cells
            num_branches += num_cell_branches;
        }

        for (auto& branch_map: branch_maps) {
            // reverse the levels
            std::reverse(branch_map.begin(), branch_map.end());

            // Sort all branches on each level in descending order of length.
            // Later, branches will be partitioned over thread blocks, and we will
            // take advantage of the fact that the first branch in a partition is
            // the longest, to determine how to pack all the branches in a block.
            for (auto& branches: branch_map) {
                util::sort(branches);
            }
        }

        // The branches generated above have been assigned contiguous ids.
        // Now generate a vector of branch_loc, one for each branch, that
        // allow for quick lookup by id of the level and index within a level
        // of each branch.
        // This information is only used in the generation of the levels below.

        // Helper for recording location of a branch once packed.
        struct branch_loc {
            unsigned block; // the gpu block containing the cell to which the branch blongs to
            unsigned level; // the level containing the branch
            unsigned index; // the index of the branch on that level
        };

        // branch_locs will hold the location information for each branch.
        std::vector<branch_loc> branch_locs(num_branches);
        for (unsigned b: make_span(branch_maps.size())) {
            const auto& branch_map = branch_maps[b];
            for (unsigned l: make_span(branch_map.size())) {
                const auto& branches = branch_map[l];

                // Record the location information
                for (auto i=0u; i<branches.size(); ++i) {
                    const auto& branch = branches[i];
                    branch_locs[branch.id] = {b, l, i};
                }
            }
        }

        // Construct description for the set of branches on each level for each
        // block. This is later used to sort the branches in each block in each
        // level into conineous chunks which are easier to read for the gpu
        // kernel.

        // Accumulate metadata about the levels, level lengths, level parents,
        // data_partition and block indices in temporary vectors to be copied to the device
        std::vector<level_metadata> temp_meta;
        std::vector<size_type> temp_lengths, temp_parents, temp_data_part, temp_block_index;

        temp_block_index.reserve(branch_maps.size() + 1);
        temp_block_index.push_back(0);
        temp_data_part.reserve(branch_maps.size());

        // Offset into the packed data format, used to apply permutation on data
        auto pos = 0u;
        // Offset into the packed data format, used to access level_lengths and level_parents
        auto data_start = 0u;
        for (const auto& branch_map: branch_maps) {
            for (const auto& lvl_branches: branch_map) {

                level_metadata lvl_meta;
                std::vector<size_type> lvl_lengths(lvl_branches.size()), lvl_parents(lvl_branches.size());

                lvl_meta.num_branches = lvl_branches.size();
                lvl_meta.matrix_data_index = pos;
                lvl_meta.level_data_index = data_start;

                // The length of the first branch is the upper bound on branch
                // length as they are sorted in descending order of length.
                lvl_meta.max_length = lvl_branches.front().length;

                unsigned bi = 0u;
                for (const auto& b: lvl_branches) {
                    // Set the length of the branch.
                    lvl_lengths[bi] = b.length;

                    // Set the parent indexes. During the forward and backward
                    // substitution phases each branch accesses the last node in
                    // its parent branch.
                    auto index = b.parent_id==npos? npos: branch_locs[b.parent_id].index;
                    lvl_parents[bi] = index;
                    ++bi;
                }

                data_start+= lvl_meta.num_branches;
                pos += lvl_meta.max_length*lvl_meta.num_branches;

                temp_meta.push_back(std::move(lvl_meta));
                util::append(temp_lengths, lvl_lengths);
                util::append(temp_parents, lvl_parents);
            }
            auto prev_end = temp_block_index.back();
            temp_block_index.push_back(prev_end + branch_map.size());
            temp_data_part.push_back(pos);
        }
        data_size = pos;

        // set matrix state
        matrix_size = p.size();

        // form the permutation index used to reorder vectors to/from the
        // ordering used by the fine grained matrix storage.
        std::vector<size_type> perm_tmp(matrix_size);
        for (auto block: make_span(branch_maps.size())) {
            const auto& branch_map = branch_maps[block];
            const auto first_level = temp_block_index[block];

            for (auto i: make_span(temp_block_index[block + 1] - first_level)) {
                const auto& l = temp_meta[first_level + i];
                for (auto j: make_span(l.num_branches)) {
                    const auto& b = branch_map[i][j];
                    auto j_lvl_length = temp_lengths[l.level_data_index + j];
                    auto to = l.matrix_data_index + j + l.num_branches*(j_lvl_length-1);
                    auto from = b.start_idx;
                    for (auto k: make_span(b.length)) {
                        perm_tmp[from + k] = to - k*l.num_branches;
                    }
                }
            }
        }

        level_meta     = memory::make_const_view(temp_meta);
        level_lengths  = memory::make_const_view(temp_lengths);
        level_parents  = memory::make_const_view(temp_parents);
        data_partition = memory::make_const_view(temp_data_part);
        block_index    = memory::make_const_view(temp_block_index);

        auto perm_balancing = trees.permutation();

        // apppy permutation form balancing
        std::vector<size_type> perm_tmp2(matrix_size);
        for (auto i: make_span(matrix_size)) {
             // This is CORRECT! verified by using the ring benchmark with root=0 (where the permutation is actually not id)
            perm_tmp2[perm_balancing[i]] = perm_tmp[i];
        }
        // copy permutation to device memory
        perm = memory::make_const_view(perm_tmp2);


        // Summary of fields and their storage format:
        //
        // face_conductance : not needed, don't store
        // d, u, rhs        : packed
        // invariant_d      : flat
        // cv_to_cell       : flat

        // the invariant part of d is stored in in flat form
        std::vector<value_type> invariant_d_tmp(matrix_size, 0);
        std::vector<value_type> temp_u_shuffled(matrix_size, 0);
        array u_shuffled;
        for (auto i: make_span(1u, matrix_size)) {
            auto gij = face_diffusivity[i];

            temp_u_shuffled[i] = -gij;
            invariant_d_tmp[i] += gij;
            if (p[i]!=-1) {
                invariant_d_tmp[p[i]] += gij;
            }
        }
        u_shuffled = memory::make_const_view(temp_u_shuffled);

        // the matrix components u, d and rhs are stored in packed form
        auto nan = std::numeric_limits<double>::quiet_NaN();
        d   = array(data_size, nan);
        u   = array(data_size, nan);
        rhs = array(data_size, nan);

        // transform u_shuffled values into packed u vector.
        flat_to_packed(u_shuffled, u);

        // data in flat form
        cv_volume = memory::make_const_view(volume);
        invariant_d = memory::make_const_view(invariant_d_tmp);

        // calculate the cv -> cell mappings
        std::vector<size_type> cv_to_cell_tmp(matrix_size);
        size_type ci = 0;
        for (auto cv_span: util::partition_view(cell_cv_divs)) {
            util::fill(util::subrange_view(cv_to_cell_tmp, cv_span), ci);
            ++ci;
        }
    }

    // Assemble the matrix
    // Afterwards the diagonal and RHS will have been set given dt, voltage, current, and conductivity.
    //   dt [ms] (scalar)
    //   voltage [mV]
    void assemble(const value_type dt, const_view concentration) {
        assemble_diffusion(d.data(),
                           rhs.data(),
                           invariant_d.data(),
                           concentration.data(),
                           cv_volume.data(),
                           dt,
                           perm.data(),
                           size());
    }

    void solve(array& to) {
        solve_diffusion(rhs.data(),
                        d.data(),
                        u.data(),
                        level_meta.data(),
                        level_lengths.data(),
                        level_parents.data(),
                        block_index.data(),
                        num_cells_in_block.data(),
                        data_partition.data(),
                        num_cells_in_block.size(),
                        max_branches_per_level);
        // unpermute the solution
        packed_to_flat(rhs, to);
    }

    void solve(array& concentration,
               const value_type dt) {
        assemble(dt, concentration);
        solve(concentration);
    }

    std::size_t size() const { return matrix_size; }

private:
    void flat_to_packed(const array& from, array& to ) {
        arb_assert(from.size()==matrix_size);
        arb_assert(to.size()==data_size);
        scatter(from.data(), to.data(), perm.data(), perm.size());
    }

    void packed_to_flat(const array& from, array& to ) {
        arb_assert(from.size()==data_size);
        arb_assert(to.size()==matrix_size);
        gather(from.data(), to.data(), perm.data(), perm.size());
    }
};

} // namespace gpu
} // namespace arb
