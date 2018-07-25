#pragma once

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
    unsigned depth;      // depth from leaf
    unsigned parent_idx;
    unsigned start_idx;
    unsigned length;
};

// Order branches by
//  - ascending depth depth
//  - descending length
//  - ascending id
inline
bool operator<(const branch& lhs, const branch& rhs) {
    return lhs.depth<rhs.depth || lhs.length>rhs.length || lhs.id<rhs.id;
}

inline
std::ostream& operator<<(std::ostream& o, branch b) {
    return o << "[" << b.id
        << ": dp " << b.depth
        << ", ln " << b.length
        << ", pt " << b.parent_idx
        << ", st " << b.start_idx
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

    // the most branches per level
    unsigned max_branches;

    // number of levels
    unsigned num_levels;

    // number of rows in matrix
    unsigned matrix_size;

    // number of cells
    unsigned num_cells;

    // length or array to store packed vector
    //      data_size >= size
    unsigned data_size;

    // the meta data for each level
    managed_vector<level> levels;

    // permutation from front end storage to packed storage
    //      packed[perm[i]] = flat[i]
    iarray perm;

    matrix_state_fine() = default;

    // constructor for fine-grained matrix.
    matrix_state_fine(const std::vector<size_type>& p,
                 const std::vector<size_type>& cell_cv_divs,
                 const std::vector<value_type>& cap,
                 const std::vector<value_type>& face_conductance,
                 const std::vector<value_type>& area)
    {
        using util::make_span;
        constexpr unsigned npos = unsigned(-1);

        // for now we have single cell per cell group
        arb_assert(cell_cv_divs.size()==2);
        /* TODO: size_type was uint32_t, replaced by int. Will this break us?
         * simplest fix is to make a copy of p converted to unsigned.
        static_assert(
            std::is_same<cell_lid_type, size_type>::value,
            "cell_lid_type required by cell tree");
        */

        num_levels = 0u;
        std::vector<std::vector<branch>> branch_map;

        num_cells = cell_cv_divs.size()-1;
        unsigned num_branches = 0u;
        for (auto c: make_span(0u, num_cells)) {
            // build the parent index for cell c
            auto cell_start = cell_cv_divs[c];
            std::vector<size_type> cell_p =
                util::assign_from(
                    util::transform_view(
                        util::subrange_view(p, cell_cv_divs[c], cell_cv_divs[c+1]),
                        [cell_start](size_type i) {return i-cell_start;}));

            // build tree structure that describes the branch topology from parent index.
            auto cell_tree = tree(cell_p);

            // find the index of the first entry, and depth, of each branch
            auto branch_starts = algorithms::branches(cell_p);
            auto depths = cell_tree.depth_from_root();

            // convert to depth from leaf
            num_levels = std::max(num_levels, util::max_value(depths)+1u);
            for (auto& d: depths) {
                d = num_levels-d-1;
            }

            // build branch_map:
            // branch_map[i] is a list of branch meta-data for branches with depth i
            auto num_cell_branches = cell_tree.num_segments();
            if (num_levels > branch_map.size()) {
                branch_map.resize(num_levels);
            }
            for (auto i: make_span(0, num_cell_branches)) {
                branch b;
                b.depth = depths[i];
                b.id = i + num_branches;
                // take care to mark branches with no parents with npos
                b.parent_id = cell_tree.parent(i)==cell_tree.no_parent?
                    npos: cell_tree.parent(i) + num_branches;
                b.start_idx = branch_starts[i] + cell_start;
                b.parent_idx = p[b.start_idx] + cell_start;
                b.length = branch_starts[i+1] - branch_starts[i];
                branch_map[b.depth].push_back(b);
            }
            num_branches += num_cell_branches;
        }

        // Sort all branches on each level in descending order of length.
        // Later, branches will be partitioned over thread blocks, and we will
        // take advantage of the fact that the first branch in a partition is
        // the longest, to determine how to pack all the branches in a block.
        for (auto& branches: branch_map) {
            util::sort(branches);
        }

        // The branches generated above have been assigned contiguous ids.
        // Now generate a vector of branch_loc, one for each branch, that
        // allow for quick lookup by id of the level and index within a level
        // of each branch.
        // This information is only used in the generation of the levels below.

        // Helper for recording location of a branch once packed.
        struct branch_loc {
            unsigned level;
            unsigned index;
        };

        // branch_locs will hold the location information for each branch.
        std::vector<branch_loc> branch_locs(num_branches);
        for (unsigned l: make_span(0u, num_levels)) {
            const auto& branches = branch_map[l];

            // Record the location information
            for (auto i=0u; i<branches.size(); ++i) {
                const auto& b = branches[i];
                branch_locs[b.id] = {l, i};
            }
        }

        levels.reserve(num_levels);
        auto pos = 0u;
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

        // set matrix state
        matrix_size = p.size();
        data_size = pos;
        max_branches = levels.front().num_branches;

        // form the permutation index used to reorder vectors to/from the
        // ordering used by the fine grained matrix storage.
        std::vector<size_type> perm_tmp(matrix_size);
        for (auto i: make_span(0u, num_levels)) {
            const auto& l = levels[i];
            for (auto j: make_span(0u, l.num_branches)) {
                const auto& b = branch_map[i][j];
                auto to = l.data_index + j + l.num_branches*(l.lengths[j]-1);
                auto from = b.start_idx;
                for (auto k: make_span(0u, b.length)) {
                    perm_tmp[from + k] = to - k*l.num_branches;
                }
            }
        }
        // copy permutation to device memory
        perm = memory::make_const_view(perm_tmp);

        // Summary of fields and their storage format:
        //
        // face_conductance : not needed, don't store
        // d, u, rhs        : packed
        // cv_capacitance   : flat
        // invariant_d      : flat
        // solution_        : flat
        // cv_to_cell       : flat?

        // the invariant part of d is stored in in flat form
        std::vector<value_type> invariant_d_tmp(matrix_size, 0);
        managed_vector<value_type> u_tmp(matrix_size, 0);
        for (auto i: util::make_span(1u, size())) {
            auto gij = face_conductance[i];

            u_tmp[i] = -gij;
            invariant_d_tmp[i] += gij;
            invariant_d_tmp[p[i]] += gij;
        }

        // the matrix components u, d and rhs are stored in packed form
        auto nan = std::numeric_limits<double>::quiet_NaN();
        d   = array(data_size, nan);
        u   = array(data_size, nan);
        rhs = array(data_size, nan);

        // transform u_tmp values into packed u vector.
        flat_to_packed(u_tmp, u);

        // the invariant part of d and the solution are in flat form
        solution_ = array(matrix_size, 0);

        // the cv_capacitance can be copied directly because it is
        // to be stored in flat format
        cv_capacitance = memory::make_const_view(cap);
        invariant_d = memory::make_const_view(invariant_d_tmp);

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
    // Afterwards the diagonal and RHS will have been set given dt, voltage and current
    //   dt_cell [ms] (per cell)
    //   voltage [mV]
    //   current [nA]
    void assemble(const_view dt_cell, const_view voltage, const_view current) {
        const unsigned n = size();

        // TODO: call assemble kernel via C-prototype defined in matrix_flat.hpp
        // TODO: write assemble kernel + C-prototype
        /*
        const unsigned block_dim = 128;
        const unsigned num_blocks = impl::block_count(n, block_dim);
        assemble_matrix_fine<value_type, size_type><<<num_blocks, block_dim>>>(
            d.data(), rhs.data(), invariant_d.data(), voltage.data(),
            current.data(), cv_capacitance.data(), perm.data(),
            cv_to_cell.data(), dt_cell.data(), n);
        */
    }

    void solve() {
        // n = maximum number of blocks on any level
        auto max_branches_per_level = std::accumulate(
                levels.begin(), levels.end(), 0u,
                [](unsigned value, const level& l) {
                    return std::max(value, unsigned(l.num_branches));});

        // TODO: call solve kernel via C-prototype defined in matrix_flat.hpp
        /*
        solve_matrix_fine<<<1, n>>>(
            rhs.data(), d.data(), u.data(), levels.data(),
            unsigned(levels.size()), num_cells, data_size);
        */

        // unpermute the solution
        packed_to_flat(rhs, solution_);
    }

    const_view solution() const {
        return solution_;
    }

    template <typename VFrom, typename VTo>
    void flat_to_packed(const VFrom& from, VTo& to ) {
        arb_assert(from.size()==matrix_size);
        arb_assert(to.size()==data_size);
        const auto n = from.size();

        // TODO: call C wrapper for scatter, defined in ??

        /*
        constexpr unsigned blockdim = 256;
        const unsigned griddim = impl::block_count(n, blockdim);

        //for (i=0:n-1) to[perm[i]] = from[i];
        scatter<<<blockdim, griddim>>>(from.data(), to.data(), perm.data(), n);
        */
    }

    template <typename VFrom, typename VTo>
    void packed_to_flat(const VFrom& from, VTo& to ) {
        arb_assert(from.size()==data_size);
        arb_assert(to.size()==matrix_size);
        const unsigned n = to.size();

        // TODO: call C wrapper for gather, defined in ??

        /*
        constexpr unsigned blockdim = 256;
        const unsigned griddim = impl::block_count(n, blockdim);

        //for (i=0:n-1) to[i] = from[perm[i]];
        gather<<<blockdim, griddim>>>(from.data(), to.data(), perm.data(), n);
        */
    }

private:
    std::size_t size() const {
        return matrix_size;
    }
};

} // namespace gpu
} // namespace arb
