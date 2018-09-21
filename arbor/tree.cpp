#include <algorithm>
#include <cassert>
#include <numeric>
#include <queue>
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

tree::int_type tree::split_node(int_type ix) {
    using util::make_span;

    auto insert_at_p  = parents_.begin() + ix;
    auto insert_at_ci = child_index_.begin() + ix;
    auto insert_at_c  = children_.begin() + child_index_[ix];
    auto new_node_ix  = ix;

    // we first adjust the parent sructure

    // first create a new node N below the parent
    auto parent = parents_[ix];
    parents_.insert(insert_at_p, parent);
    // and attach the remining subtree below it
    parents_[ix+1] = new_node_ix;
    // shift all parents, as the indices changed when we
    // inserted a new node
    for (auto i: make_span(ix + 2, parents().size())) {
        if (parents_[i] >= new_node_ix) {
            parents_[i]++;
        }
    }

    // now we adjust the children structure

    // insert a child node for the new node N, pointing to
    // the old node A
    child_index_.insert(insert_at_ci, child_index_[ix]);
    // we will set this value later as it will be overridden
    children_.insert(insert_at_c, ~0u);
    // shift indices for all larger indices, as we inserted
    // a new element in the list
    for (auto i: make_span(ix + 1, child_index_.size())) {
        child_index_[i]++;
    }
    for (auto i: make_span(0, children_.size())) {
        if(children_[i] > new_node_ix) {
            children_[i]++;
        }
    }
    // set the children of the new node to the old subtree
    children_[child_index_[ix]] = ix + 1;

    return ix+1;
}

tree::iarray tree::select_new_root(int_type root) {
    using util::make_span;

    const auto num_nodes = parents().size();

    if(root >= num_nodes && root != no_parent) {
        throw std::domain_error(
            "root is out of bounds: root="+std::to_string(root)+", nodes="
            +std::to_string(num_nodes)
        );
    }

    // walk up to the old root and turn `parent->child` into `parent<-child`
    auto prev = no_parent;
    auto current = root;
    while (current != no_parent) {
        auto parent = parents_[current];
        parents_[current] = prev;
        prev = current;
        current = parent;
    }

    // sort the list to get min degree ordering and keep index such that we
    // can sort also the `branch_starts` array.

    // comput the depth for each node
    iarray depth (num_nodes, 0);
    // the depth when we don't count nodes that only have one child
    iarray reduced_depth (num_nodes, 0);
    // the index of the last node that passed this node on its way to the root
    // we need this to keep nodes that are part of the same reduced tree close
    // together in the final sorting.
    //
    // Instead of the left order we want the right order.
    // .-----------------------.
    // |    0            0     |
    // |   / \          / \    |
    // |  1   2        1   3   |
    // |  |   |        |   |   |
    // |  3   4        2   4   |
    // |     / \          / \  |
    // |    5  6         5  6  |
    // '-----------------------'
    //
    // we achieve this by first sorting by reduced_depth, branch_ix and depth
    // in this order. The resulting ordering satisfies minimal degree ordering.
    //
    // Using the tree above we would get the following results:
    //
    // `depth`           `reduced_depth`   `branch_ix`
    // .-----------.     .-----------.     .-----------.
    // |    0      |     |    0      |     |    6      |
    // |   / \     |     |   / \     |     |   / \     |
    // |  1   1    |     |  1   1    |     |  2   6    |
    // |  |   |    |     |  |   |    |     |  |   |    |
    // |  2   2    |     |  1   1    |     |  2   6    |
    // |     / \   |     |     / \   |     |     / \   |
    // |    3   3  |     |    2   2  |     |    5   6  |
    // '-----------'     '-----------'     '-----------'
    iarray branch_ix (num_nodes, 0);
    // we cannot use the existing `children_` array as we only updated the
    // parent structure yet
    auto new_num_children = algorithms::child_count(parents_);
    for (auto n: make_span(num_nodes)) {
        branch_ix[n] = n;
        auto prev = n;
        auto curr = parents_[n];

        // find the way to the root
        while (curr != no_parent) {
            depth[n]++;
            if (new_num_children[curr] > 1) {
                reduced_depth[n]++;
            }
            branch_ix[curr] = branch_ix[prev];
            curr = parents_[curr];
        }
    }

    // maps new indices to old indices
    iarray indices (num_nodes);
    // fill array with indices
    for (auto i: make_span(num_nodes)) {
        indices[i] = i;
    }
    // perform sort by depth index to get the permutation
    std::sort(indices.begin(), indices.end(), [&](auto i, auto j){
        if (reduced_depth[i] != reduced_depth[j]) {
            return reduced_depth[i] < reduced_depth[j];
        }
        if (branch_ix[i] != branch_ix[j]) {
            return branch_ix[i] < branch_ix[j];
        }
        return depth[i] < depth[j];
    });
    // maps old indices to new indices
    iarray indices_inv (num_nodes);
    // fill array with indices
    for (auto i: make_span(num_nodes)) {
        indices_inv[i] = i;
    }
    // perform sort
    std::sort(indices_inv.begin(), indices_inv.end(), [&](auto i, auto j){
        return indices[i] < indices[j];
    });

    // translate the parent vetor to new indices
    for (auto i: make_span(num_nodes)) {
        if (parents_[i] != no_parent) {
            parents_[i] = indices_inv[parents_[i]];
        }
    }

    iarray new_parents (num_nodes);
    for (auto i: make_span(num_nodes)) {
        new_parents[i] = parents_[indices[i]];
    }
    // parent now starts with the root, then it's children, then their
    // children, etc...

    // recompute the children array
    memory::copy(new_parents, parents_);
    memory::copy(algorithms::make_index(algorithms::child_count(parents_)), child_index_);

    std::vector<int_type> pos(parents_.size(), 0);
    for (auto i = 1u; i < parents_.size(); ++i) {
        auto p = parents_[i];
        children_[child_index_[p] + pos[p]] = i;
        ++pos[p];
    }

    return indices;
}

tree::iarray tree::minimize_depth() {
    const auto num_nodes = parents().size();
    tree::iarray seen(num_nodes, 0);

    // find the furhtest node from the root
    std::queue<tree::int_type> queue;
    queue.push(0); // start at the root node
    seen[0] = 1;
    auto front = queue.front();
    // breath first traversal
    while (!queue.empty()) {
        front = queue.front();
        queue.pop();
        // we only have to check children as we started at the root node
        auto cs = children(front);
        for (auto c: cs) {
            if (seen[c] == 0) {
                seen[c] = 1;
                queue.push(c);
            }
        }
    }

    auto u = front;

    // find the furhtest node from this node
    std::fill(seen.begin(), seen.end(), 0);
    queue.push(u);
    seen[u] = 1;
    front = queue.front();
    // breath first traversal
    while (!queue.empty()) {
        front = queue.front();
        queue.pop();
        auto cs = children(front);
        for (auto c: cs) {
            if (seen[c] == 0) {
                seen[c] = 1;
                queue.push(c);
            }
        }
        // also check the partent node!
        auto c = parent(front);
        if (c != tree::no_parent && seen[c] == 0) {
            seen[c] = 1;
            queue.push(c);
        }
    }

    auto v = front;

    // now find the middle between u and v

    // each path to the root
    tree::iarray path_to_root_u (1, u);
    tree::iarray path_to_root_v (1, v);

    auto curr = parent(u);
    while (curr != tree::no_parent) {
        path_to_root_u.push_back(curr);
        curr = parent(curr);
    }
    curr = parent(v);
    while (curr != tree::no_parent) {
        path_to_root_v.push_back(curr);
        curr = parent(curr);
    }

    // reduce the path
    auto last_together = 0;
    while (path_to_root_u.back() == path_to_root_v.back()) {
        last_together = path_to_root_u.back();
        path_to_root_u.pop_back();
        path_to_root_v.pop_back();
    }
    path_to_root_u.push_back(last_together);

    auto path_length = path_to_root_u.size() + path_to_root_v.size() - 1;

    // walk up half of the path length to find the middle node
    tree::int_type root;
    if (path_to_root_u.size() > path_to_root_v.size()) {
        root = path_to_root_u[path_length / 2];
    } else {
        root = path_to_root_v[path_length / 2];
    }

    return select_new_root(root);
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
