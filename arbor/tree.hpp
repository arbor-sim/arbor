#pragma once

#include <algorithm>
#include <cassert>
#include <fstream>
#include <numeric>
#include <vector>

#include <arbor/export.hpp>
#include <arbor/common_types.hpp>

#include "memory/memory.hpp"
#include "util/rangeutil.hpp"
#include "util/span.hpp"

namespace arb {

class ARB_ARBOR_API tree {
public:
    using int_type   = cell_lid_type;
    using size_type  = cell_local_size_type;

    using iarray = std::vector<int_type>;
    static constexpr int_type no_parent = (int_type)-1;

    tree() = default;

    /// Create the tree from a parent index that lists the parent segment
    /// of each segment in a cell tree.
    tree(std::vector<int_type> parent_index);

    size_type num_children() const;

    size_type num_children(size_t b) const;

    size_type num_segments() const;

    /// return the child index
    const iarray& child_index();

    /// return the list of all children
    const iarray& children() const;

    /// return the list of all children of branch i
    auto children(size_type i) const {
        const auto b = child_index_[i];
        const auto e = child_index_[i+1];
        return util::subrange_view(children_, b, e);
    }

    /// return the list of parents
    const iarray& parents() const;

    /// return the parent of branch b
    const int_type& parent(size_t b) const;
    int_type& parent(size_t b);

    // splits the node in two parts. Returns `ix + 1` which is the new index of
    // the old node.
    // .-------------------.
    // |      P         P  |
    // |     /         /   |
    // |    A   ~~>   N    |
    // |   / \        |    |
    // |  B  C        A    |
    // |             / \   |
    // |            B  C   |
    // '-------------------'
    int_type split_node(int_type ix);

    // Changes the root node of a tree
    // .------------------------.
    // |        A               |
    // |       / \         R    |
    // |      R  B        / \   |
    // |     /     ~~>   A  C   |
    // |    C           /  / \  |
    // |   / \         B  D  E  |
    // |  D  E                  |
    // '------------------------'
    // Returns the permutation applied to the nodes,
    // i.e. `new_node_data[i] = old_node_data[perm[i]]`
    //
    // This function has the additional effect, that branches with only one
    // child branch get merged. That means that `select_new_root(0)` can also
    // lead to an permutation of the indices of the compartments:
    // .------------------------------.
    // |        0               0     |
    // |       / \             / \    |
    // |      1  3            1  3    |
    // |     /    \   ~~>    /    \   |
    // |    2     4         2     4   |
    // |   / \     \       / \     \  |
    // |  5  6     7      6  7     5  |
    // '------------------------------'
    iarray select_new_root(int_type root);

    // Selects a new node such that the depth of the graph is minimal.
    // Returns the permutation applied to the nodes,
    // i.e. `new_node_data[i] = old_node_data[perm[i]]`
    iarray minimize_depth();

    /// memory used to store tree (in bytes)
    std::size_t memory() const;

private:
    void init(size_type nnode);

    // state
    iarray children_;
    iarray child_index_;
    iarray parents_;
};

// Calculates the depth of each branch from the root of a cell segment tree.
// The root has depth 0, it's children have depth 1, and so on.
ARB_ARBOR_API tree::iarray depth_from_root(const tree& t);

// Check if c[0] == 0 and c[i] < 0 holds for i != 0
// Also handle the valid case of c[0]==value_type(-1)
// This means that children of a node always have larger indices than their
// parent.
template <
    typename C,
    typename = typename std::enable_if<std::is_integral<typename C::value_type>::value>
>
bool is_minimal_degree(C const& c)
{
    static_assert(
        std::is_integral<typename C::value_type>::value,
        "is_minimal_degree only applies to integral types"
    );

    if (c.size()==0u) {
        return true;
    }

    using value_type = typename C::value_type;
    if (!(c[0]==value_type(0) || c[0]==value_type(-1))) {
        return false;
    }
    auto i = value_type(1);
    auto it = std::find_if(
        c.begin()+1, c.end(), [&i](value_type v) { return v>=(i++); }
    );
    return it==c.end();
}

// returns a vector containing the number of children for each node.
template<typename C>
std::vector<typename C::value_type> child_count(const C& parent_index)
{
    using value_type = typename C::value_type;
    static_assert(
        std::is_integral<value_type>::value,
        "integral type required"
    );

    std::vector<value_type> count(parent_index.size(), 0);
    for (auto i = 0u; i < parent_index.size(); ++i) {
        auto p = parent_index[i];
        // -1 means no parent
        if (p != value_type(i) && p != value_type(-1)) {
            ++count[p];
        }
    }
    return count;
}

} // namespace arb
