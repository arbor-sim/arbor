#include <algorithm>
#include <cassert>
#include <numeric>
#include <vector>

#include <arbor/common_types.hpp>

#include "algorithms.hpp"
#include "memory/memory.hpp"
#include "tree.hpp"
#include "util/span.hpp"

namespace arb {

tree::tree(std::vector<tree::int_type> parent_index) {
    // validate the input
    if(!algorithms::is_minimal_degree(parent_index)) {
        throw std::domain_error(
            "parent index used to build a tree did not satisfy minimal degree ordering"
        );
    }

    // an empty parent_index implies a single-compartment/segment cell
    arb_assert(parent_index.size()!=0u);

    init(parent_index.size());
    memory::copy(parent_index, parents_);
    parents_[0] = no_parent;

    // compute offsets into children_ array
    memory::copy(algorithms::make_index(algorithms::child_count(parents_)), child_index_);

    std::vector<int_type> pos(parents_.size(), 0);
    for (auto i = 1u; i < parents_.size(); ++i) {
        auto p = parents_[i];
        children_[child_index_[p] + pos[p]] = i;
        ++pos[p];
    }
}

tree::size_type tree::num_children() const {
    return static_cast<size_type>(children_.size());
}

tree::size_type tree::num_children(size_t b) const {
    return child_index_[b+1] - child_index_[b];
}

tree::size_type tree::num_segments() const {
    // the number of segments/nodes is the size of the child index minus 1
    // ... except for the case of an empty tree
    auto sz = static_cast<size_type>(child_index_.size());
    return sz ? sz - 1 : 0;
}

const tree::iarray& tree::child_index() {
    return child_index_;
}

const tree::iarray& tree::children() const {
    return children_;
}

const tree::iarray& tree::parents() const {
    return parents_;
}

const tree::int_type& tree::parent(size_t b) const {
    return parents_[b];
}
tree::int_type& tree::parent(size_t b) {
    return parents_[b];
}

/// memory used to store tree (in bytes)
std::size_t tree::memory() const {
    return sizeof(int_type)*(children_.size()+child_index_.size()+parents_.size())
        + sizeof(tree);
}

void tree::init(tree::size_type nnode) {
    if (nnode) {
        auto nchild = nnode - 1;
        children_.resize(nchild);
        child_index_.resize(nnode+1);
        parents_.resize(nnode);
    }
    else {
        children_.resize(0);
        child_index_.resize(0);
        parents_.resize(0);
    }
}

// recursive helper for the depth_from_root() below
void depth_from_root(const tree& t, tree::iarray& depth, tree::int_type segment) {
    auto d = depth[t.parent(segment)] + 1;
    depth[segment] = d;
    for(auto c : t.children(segment)) {
        depth_from_root(t, depth, c);
    }
}

tree::iarray depth_from_root(const tree& t) {
    tree::iarray depth(t.num_segments());
    depth[0] = 0;
    for (auto c: t.children(0)) {
        depth_from_root(t, depth, c);
    }

    return depth;
}

} // namespace arb
