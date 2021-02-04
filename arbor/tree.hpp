#pragma once

#include <algorithm>
#include <cassert>
#include <fstream>
#include <numeric>
#include <vector>

#include <arbor/common_types.hpp>

#include "memory/memory.hpp"
#include "util/index.hpp"
#include "util/rangeutil.hpp"
#include "util/span.hpp"

namespace arb {

class tree {
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
tree::iarray depth_from_root(const tree& t);

template <typename C>
bool all_positive(const C& c) {
    return util::all_of(c, [](auto v) { return v>decltype(v){}; });
}

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

template <typename C>
std::vector<tree::int_type> make_parent_index(tree const& t, C const& counts)
{
    using util::make_span;
    using int_type = tree::int_type;
    constexpr auto no_parent = tree::no_parent;

    if (!all_positive(counts) || counts.size() != t.num_segments()) {
        throw std::domain_error(
            "make_parent_index requires one non-zero count per segment"
        );
    }
    auto index = util::make_index(counts);
    auto num_compartments = index.back();
    std::vector<int_type> parent_index(num_compartments);
    int_type pos = 0;
    for (int_type i : make_span(0, t.num_segments())) {
        // get the parent of this segment
        // taking care for the case where the root node has -1 as its parent
        auto parent = t.parent(i);
        parent = parent!=no_parent ? parent : 0;

        // the index of the first compartment in the segment
        // is calculated differently for the root (i.e when i==parent)
        if (i!=parent) {
            parent_index[pos++] = index[parent+1]-1;
        }
        else {
            parent_index[pos++] = parent;
        }
        // number the remaining compartments in the segment consecutively
        while (pos<index[i+1]) {
            parent_index[pos] = pos-1;
            pos++;
        }
    }

    // if one of these assertions is tripped, we have to improve
    // the input validation above
    assert(pos==num_compartments);
    assert(is_minimal_degree(parent_index));

    return parent_index;
}

} // namespace arb
