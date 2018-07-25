#pragma once

#include <algorithm>
#include <cassert>
#include <numeric>
#include <vector>

#include <arbor/common_types.hpp>

#include "algorithms.hpp"
#include "memory/memory.hpp"
#include "util/span.hpp"

namespace arb {

class tree {
public:
    using int_type   = cell_lid_type;
    using size_type  = cell_local_size_type;

    using iarray = std::vector<int_type>;
    static constexpr int_type no_parent = (int_type)-1;

    tree() = default;

    tree& operator=(tree&& other) {
        std::swap(child_index_, other.child_index_);
        std::swap(children_, other.children_);
        std::swap(parents_, other.parents_);
        return *this;
    }

    tree& operator=(tree const& other) {
        children_ = other.children_;
        child_index_ = other.child_index_;
        parents_ = other.child_index_;
        return *this;
    }

    // copy constructors take advantage of the assignment operators
    // defined above
    tree(tree const& other) {
        *this = other;
    }

    tree(tree&& other) {
        *this = std::move(other);
    }

    /// Create the tree from a parent index that lists the parent segment
    /// of each segment in a cell tree.
    tree(std::vector<int_type> parent_index) {
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

        memory::copy(algorithms::make_index(algorithms::child_count(parents_)), child_index_);

        std::vector<int_type> pos(parents_.size(), 0);
        for (auto i = 1u; i < parents_.size(); ++i) {
            auto p = parents_[i];
            children_[child_index_[p] + pos[p]] = i;
            ++pos[p];
        }
    }

    size_type num_children() const {
        return static_cast<size_type>(children_.size());
    }

    size_type num_children(size_t b) const {
        return child_index_[b+1] - child_index_[b];
    }

    size_type num_segments() const {
        // the number of segments/nodes is the size of the child index minus 1
        // ... except for the case of an empty tree
        auto sz = static_cast<size_type>(child_index_.size());
        return sz ? sz - 1 : 0;
    }

    /// return the child index
    const iarray& child_index() {
        return child_index_;
    }

    /// return the list of all children
    const iarray& children() const {
        return children_;
    }

    /// return the list of all children of branch i
    auto children(size_type i) const {
        const auto b = child_index_[i];
        const auto e = child_index_[i+1];
        return util::subrange_view(children_, b, e);
    }

    /// return the list of parents
    const iarray& parents() const {
        return parents_;
    }

    /// return the parent of branch b
    int_type parent(size_t b) const {
        return parents_[b];
    }
    int_type& parent(size_t b) {
        return parents_[b];
    }

    /// memory used to store tree (in bytes)
    std::size_t memory() const {
        return sizeof(int_type)*(children_.size()+child_index_.size()+parents_.size())
            + sizeof(tree);
    }

private:
    void init(size_type nnode) {
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

    // state
    iarray children_;
    iarray child_index_;
    iarray parents_;
};

template <typename C>
std::vector<tree::int_type> make_parent_index(tree const& t, C const& counts)
{
    using util::make_span;
    using int_type = tree::int_type;
    constexpr auto no_parent = tree::no_parent;

    if (!algorithms::all_positive(counts) || counts.size() != t.num_segments()) {
        throw std::domain_error(
            "make_parent_index requires one non-zero count per segment"
        );
    }
    auto index = algorithms::make_index(counts);
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
    assert(algorithms::is_minimal_degree(parent_index));

    return parent_index;
}

} // namespace arb
