#pragma once

#include <vector>

#include <arbor/export.hpp>
#include "tree.hpp"

namespace arb {
namespace gpu {

using size_type = int;

struct ARB_ARBOR_API forest {
    forest(const std::vector<size_type>& p, const std::vector<size_type>& cell_cv_divs);

    void optimize();

    unsigned num_trees() {
        return fine_trees.size();
    }

    const tree& branch_tree(unsigned tree_index) {
        return trees[tree_index];
    }

    const tree& compartment_tree(unsigned tree_index) {
        return fine_trees[tree_index];
    }

    // Returns the offset into the compartments of a tree where each branch
    // starts. It holds `0 <= offset < tree.num_segments()`
    const std::vector<unsigned>& branch_offsets(unsigned tree_index) {
        return tree_branch_starts[tree_index];
    }

    // Returns vector of the length of each branch in a tree.
    const std::vector<unsigned>& branch_lengths(unsigned tree_index) {
        return tree_branch_lengths[tree_index];
    }

    // Return the permutation that was applied to the compartments in the
    // format: `new[i] = old[perm[i]]`
    const std::vector<size_type>& permutation() {
        return perm_balancing;
    }

    // trees of compartments
    std::vector<tree> fine_trees;
    std::vector<std::vector<unsigned>> tree_branch_starts;
    std::vector<std::vector<unsigned>> tree_branch_lengths;
    // trees of branches
    std::vector<tree> trees;

    // the permutation matrix used by the balancing algorithm
    // format: `solver_format[i] = external_format[perm[i]]`
    std::vector<size_type> perm_balancing;
};


struct level_iterator {
    level_iterator(tree* t, unsigned level) {
        tree_ = t;
        only_on_level = level;
        // due to the ordering of the nodes we know that 0 is the root
        current_node  = 0;
        current_level = 0;
        next_children = 0;
        if (level != 0) {
            next();
        };
    }

    void advance_depth_first() {
        auto children = tree_->children(current_node);
        if (next_children < children.size() && current_level <= only_on_level) {
            // go to next children
            current_level += 1;
            current_node = children[next_children];
            next_children = 0;
        } else {
            // go to parent
            auto parent_node = tree_->parents()[current_node];
            constexpr unsigned npos = unsigned(-1);
            if (parent_node != npos) {
                auto siblings = tree_->children(parent_node);
                // get the index in the child list of the parent
                unsigned index = 0;
                while (siblings[index] != current_node) { // TODO repalce by array lockup: sibling_nr
                    index += 1;
                }

                current_level -= 1;
                current_node = parent_node;
                next_children = index + 1;
            } else {
                // we are done with the iteration
                current_level = -1;
                current_node  = -1;
                next_children = -1;
            }

        }
    }

    unsigned next() {
        constexpr unsigned npos = unsigned(-1);
        if (!valid()) {
            // we are done
            return npos;
        } else {
            advance_depth_first();
            // next_children != 0 means, that we have seen the node before
            while (valid() && (current_level != only_on_level || next_children != 0)) {
                advance_depth_first();
            }
            return current_node;
        }
    }

    bool valid() {
        constexpr unsigned npos = unsigned(-1);
        return this->peek() != npos;
    }

    unsigned peek() {
        return current_node;
    }

private:
    tree* tree_;

    unsigned current_node;
    unsigned current_level;
    unsigned next_children;

    unsigned only_on_level;
};

// Works like std::is_sorted(), but with stronger condition that successive
// elements must be greater than those before them
template <typename C>
bool is_strictly_monotonic_increasing(C const& c)
{
    using value_type = typename C::value_type;
    return std::is_sorted(
        c.begin(),
        c.end(),
        [] (value_type const& lhs, value_type const& rhs) {
          return lhs <= rhs;
        }
    );
}

template<typename C>
bool has_contiguous_compartments(const C& parent_index)
{
    using value_type = typename C::value_type;
    static_assert(
        std::is_integral<value_type>::value,
        "integral type required"
    );

    if (!is_minimal_degree(parent_index)) {
        return false;
    }

    auto num_child = child_count(parent_index);
    for (auto i=1u; i < parent_index.size(); ++i) {
        auto p = parent_index[i];
        if (num_child[p]==1 && p!=value_type(i-1)) {
            return false;
        }
    }

    return true;
}

template<typename C>
std::vector<typename C::value_type> branches(const C& parent_index)
{
    static_assert(
        std::is_integral<typename C::value_type>::value,
        "integral type required"
    );

    arb_assert(has_contiguous_compartments(parent_index));

    std::vector<typename C::value_type> branch_index;
    if (parent_index.empty()) {
        return branch_index;
    }

    auto num_child = child_count(parent_index);
    branch_index.push_back(0);
    for (std::size_t i = 1; i < parent_index.size(); ++i) {
        auto p = parent_index[i];
        if (num_child[p] > 1 || parent_index[p] == p) {
            // `parent_index[p] == p` ~> parent_index[i] is the soma
            branch_index.push_back(i);
        }
    }

    branch_index.push_back(parent_index.size());
    return branch_index;
}

// creates a vector that contains the branch index for each compartment.
// e.g. {0, 1, 5, 9, 10} -> {0, 1, 1, 1, 1, 2, 2, 2, 2, 3}
//                  indices  0  1           5           9
template<typename C>
std::vector<typename C::value_type> expand_branches(const C& branch_index)
{
    static_assert(
        std::is_integral<typename C::value_type>::value,
        "integral type required"
    );

    if (branch_index.empty())
        return {};

    std::vector<typename C::value_type> expanded(branch_index.back());
    for (std::size_t i = 0; i < branch_index.size()-1; ++i) {
        for (auto j = branch_index[i]; j < branch_index[i+1]; ++j) {
            expanded[j] = i;
        }
    }

    return expanded;
}

// Find the reduced form of a tree represented as a parent index.
// The operation of transforming a tree into a homeomorphic pre-image is called
// 'reduction'. The 'reduced' tree is the smallest tree that maps into the tree
// homeomorphically (i.e. preserving labels and child relationships).
// For example, the tree represented by the following index:
//      {0, 0, 1, 2, 0, 4, 0, 6, 7, 8, 9, 8, 11, 12}
// Has reduced form
//      {0, 0, 0, 0, 3, 3}
//
// This transformation can be represented graphically:
//
//        0               0
//       /|\             /|\.
//      1 4 6      =>   1 2 3
//     /  |  \             / \.
//    2   5   7           4   5
//   /         \.
//  3           8
//             / \.
//            9   11
//           /     \.
//          10     12
//                   \.
//                   13
//
template<typename C>
std::vector<typename C::value_type> tree_reduce(
    const C& parent_index, const C& branch_index)
{
    static_assert(
        std::is_integral<typename C::value_type>::value,
        "integral type required"
    );

    if (parent_index.empty() && branch_index.empty()) {
        return {};
    }

    arb_assert(parent_index.size()-branch_index.back() == 0);
    arb_assert(has_contiguous_compartments(parent_index));
    arb_assert(is_strictly_monotonic_increasing(branch_index));

    // expand the branch index to lookup the banch id for each compartment
    auto expanded_branch = expand_branches(branch_index);

    std::vector<typename C::value_type> new_parent_index;
    // push the first element manually as the parent of the root might be -1
    new_parent_index.push_back(expanded_branch[0]);
    for (std::size_t i = 1; i < branch_index.size()-1; ++i) {
        auto p = parent_index[branch_index[i]];
        new_parent_index.push_back(expanded_branch[p]);
    }

    return new_parent_index;
}

} // namespace gpu
} // namespace arb
