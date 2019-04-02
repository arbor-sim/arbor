#pragma once

#include <cstring>

#include <vector>
#include <type_traits>

#include <arbor/common_types.hpp>

#include "algorithms.hpp"
#include "memory/memory.hpp"
#include "util/partition.hpp"
#include "util/rangeutil.hpp"
#include "util/span.hpp"
#include "tree.hpp"

#include "matrix_fine.hpp"
#include "forest.hpp"

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

template <typename T, typename I>
struct matrix_state_fine {
public:
    using value_type = T;
    using size_type = I;

    using array      = memory::device_vector<value_type>;
    using view       = typename array::view_type;
    using const_view = typename array::const_view_type;
    using iarray     = memory::device_vector<size_type>;

    template <typename ValueType>
    using managed_vector = std::vector<ValueType, memory::managed_allocator<ValueType>>;

    iarray cv_to_cell;

    array d;     // [μS]
    array u;     // [μS]
    array rhs;   // [nA]

    // required for matrix assembly

    array cv_area; // [μm^2]

    array cv_capacitance;      // [pF]

    // the invariant part of the matrix diagonal
    array invariant_d;         // [μS]

    // for storing the solution in unpacked format
    array solution_;

    iarray cell_to_intdom;

    // the maximum nuber of branches in each level per block
    unsigned max_branches_per_level;

    // number of rows in matrix
    unsigned matrix_size;

    // number of cells
    unsigned num_cells;
    managed_vector<unsigned> num_cells_in_block;

    // end of the data of each level
    // use data_size.back() to get the total data size
    //      data_size >= size
    managed_vector<unsigned> data_size; // TODO rename

    // the meta data for each level for each block layed out linearly in memory
    managed_vector<level> levels;
    // the start of the levels of each block
    // block b owns { leves[level_start[b]], ..., leves[level_start[b+1] - 1] }
    // there is an additional entry at the end of the vector to make the above
    // compuation save
    managed_vector<unsigned> levels_start;

    // permutation from front end storage to packed storage
    //      `solver_format[perm[i]] = external_format[i]`
    iarray perm;


    matrix_state_fine() = default;

    // constructor for fine-grained matrix.
    matrix_state_fine(const std::vector<size_type>& p,
                 const std::vector<size_type>& cell_cv_divs,
                 const std::vector<value_type>& cap,
                 const std::vector<value_type>& face_conductance,
                 const std::vector<value_type>& area,
                 const std::vector<size_type>& cell_intdom)
    {
        using util::make_span;
        constexpr unsigned npos = unsigned(-1);

        max_branches_per_level = 128;

        // for now we have single cell per cell group
        arb_assert(cell_cv_divs.size()==2);

        num_cells = cell_cv_divs.size()-1;

        forest trees(p, cell_cv_divs);
        trees.optimize();

        // Now distribute the cells into cuda blocks.
        // While the total number of branches on each level of theses cells in a
        // block are less than `max_branches_per_level` we add more cells. If
        // one block is full, we start a new cuda block.

        unsigned current_block = 0;
        std::vector<unsigned> block_num_branches_per_depth;
        std::vector<unsigned> block_ix(num_cells);
        num_cells_in_block.resize(1, 0);

        // branch_map = branch_maps[block] is a branch map for each cuda block
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


            // check if we can fit the current cell into the last cuda block
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
                num_cells_in_block[block_ix[c]] += 1;
                // and increment counter
                for (auto i: make_span(cell_num_levels)) {
                    block_num_branches_per_depth[i] += cell_num_branches_per_depth[i];
                }
            } else {
                // otherwise start a new block
                block_ix[c] = current_block + 1;
                num_cells_in_block.push_back(1);
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
            unsigned block; // the cuda block containing the cell to which the branch blongs to
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

        unsigned total_num_levels = std::accumulate(
            branch_maps.begin(), branch_maps.end(), 0,
            [](unsigned value, decltype(branch_maps[0])& l) {
                return value + l.size();});

        // construct description for the set of branches on each level for each
        // block. This is later used to sort the branches in each block in each
        // level into conineous chunks which are easier to read for the cuda
        // kernel.
        levels.reserve(total_num_levels);
        levels_start.reserve(branch_maps.size() + 1);
        levels_start.push_back(0);
        data_size.reserve(branch_maps.size());
        // offset into the packed data format, used to apply permutation on data
        auto pos = 0u;
        for (const auto& branch_map: branch_maps) {
            for (const auto& lvl_branches: branch_map) {

                level lvl(lvl_branches.size());

                // The length of the first branch is the upper bound on branch
                // length as they are sorted in descending order of length.
                lvl.max_length = lvl_branches.front().length;
                lvl.data_index = pos;

                unsigned bi = 0u;
                for (const auto& b: lvl_branches) {
                    // Set the length of the branch.
                    lvl.lengths[bi] = b.length;

                    // Set the parent indexes. During the forward and backward
                    // substitution phases each branch accesses the last node in
                    // its parent branch.
                    auto index = b.parent_id==npos? npos: branch_locs[b.parent_id].index;
                    lvl.parents[bi] = index;
                    ++bi;
                }

                pos += lvl.max_length*lvl.num_branches;

                levels.push_back(std::move(lvl));
            }
            auto prev_end = levels_start.back();
            levels_start.push_back(prev_end + branch_map.size());
            data_size.push_back(pos);
        }

        // set matrix state
        matrix_size = p.size();

        // form the permutation index used to reorder vectors to/from the
        // ordering used by the fine grained matrix storage.
        std::vector<size_type> perm_tmp(matrix_size);
        for (auto block: make_span(branch_maps.size())) {
            const auto& branch_map = branch_maps[block];
            const auto first_level = levels_start[block];

            for (auto i: make_span(levels_start[block + 1] - first_level)) {
                const auto& l = levels[first_level + i];
                for (auto j: make_span(l.num_branches)) {
                    const auto& b = branch_map[i][j];
                    auto to = l.data_index + j + l.num_branches*(l.lengths[j]-1);
                    auto from = b.start_idx;
                    for (auto k: make_span(b.length)) {
                        perm_tmp[from + k] = to - k*l.num_branches;
                    }
                }
            }
        }

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
        // cv_capacitance   : flat
        // invariant_d      : flat
        // solution_        : flat
        // cv_to_cell       : flat
        // area             : flat

        // the invariant part of d is stored in in flat form
        std::vector<value_type> invariant_d_tmp(matrix_size, 0);
        managed_vector<value_type> u_tmp(matrix_size, 0);
        for (auto i: make_span(1u, matrix_size)) {
            auto gij = face_conductance[i];

            u_tmp[i] = -gij;
            invariant_d_tmp[i] += gij;
            invariant_d_tmp[p[i]] += gij;
        }

        // the matrix components u, d and rhs are stored in packed form
        auto nan = std::numeric_limits<double>::quiet_NaN();
        d   = array(data_size.back(), nan);
        u   = array(data_size.back(), nan);
        rhs = array(data_size.back(), nan);

        // transform u_tmp values into packed u vector.
        flat_to_packed(u_tmp, u);

        // the invariant part of d, cv_area and the solution are in flat form
        solution_ = array(matrix_size, 0);
        cv_area = memory::make_const_view(area);

        // the cv_capacitance can be copied directly because it is
        // to be stored in flat format
        cv_capacitance = memory::make_const_view(cap);
        invariant_d = memory::make_const_view(invariant_d_tmp);
        cell_to_intdom = memory::make_const_view(cell_intdom);

        // calculte the cv -> cell mappings
        std::vector<size_type> cv_to_cell_tmp(matrix_size);
        size_type ci = 0;
        for (auto cv_span: util::partition_view(cell_cv_divs)) {
            util::fill(util::subrange_view(cv_to_cell_tmp, cv_span), ci);
            ++ci;
        }
        cv_to_cell = memory::make_const_view(cv_to_cell_tmp);
    }

    // Assemble the matrix
    // Afterwards the diagonal and RHS will have been set given dt, voltage, current, and conductivity.
    //   dt_intdom [ms] (per cell)
    //   voltage [mV]
    //   current density [A/m²]
    //   conductivity [kS/m²]
    void assemble(const_view dt_intdom, const_view voltage, const_view current, const_view conductivity) {
        assemble_matrix_fine(
            d.data(),
            rhs.data(),
            invariant_d.data(),
            voltage.data(),
            current.data(),
            conductivity.data(),
            cv_capacitance.data(),
            cv_area.data(),
            cv_to_cell.data(),
            dt_intdom.data(),
            cell_to_intdom.data(),
            perm.data(),
            size());
    }

    void solve() {
        solve_matrix_fine(
            rhs.data(), d.data(), u.data(),
            levels.data(), levels_start.data(),
            num_cells_in_block.data(),
            data_size.data(),
            num_cells_in_block.size(), max_branches_per_level);

        // unpermute the solution
        packed_to_flat(rhs, solution_);
    }

    const_view solution() const {
        return solution_;
    }

    template <typename VFrom, typename VTo>
    void flat_to_packed(const VFrom& from, VTo& to ) {
        arb_assert(from.size()==matrix_size);
        arb_assert(to.size()==data_size.back());

        scatter(from.data(), to.data(), perm.data(), perm.size());
    }

    template <typename VFrom, typename VTo>
    void packed_to_flat(const VFrom& from, VTo& to ) {
        arb_assert(from.size()==data_size.back());
        arb_assert(to.size()==matrix_size);

        gather(from.data(), to.data(), perm.data(), perm.size());
    }

private:
    std::size_t size() const {
        return matrix_size;
    }
};

} // namespace gpu
} // namespace arb
