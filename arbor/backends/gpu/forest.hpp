#pragma once

#include <vector>

#include "tree.hpp"

namespace arb {
namespace gpu {

using size_type = int;

struct forest {
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


} // namespace gpu
} // namespace arb
