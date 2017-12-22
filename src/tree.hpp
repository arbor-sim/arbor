#pragma once

#include <algorithm>
#include <cassert>
#include <numeric>
#include <vector>

#include <algorithms.hpp>
#include <common_types.hpp>
#include <memory/memory.hpp>
#include <util/span.hpp>

namespace arb {

class tree {
public:
    using int_type   = cell_lid_type;
    using size_type  = cell_local_size_type;

    using iarray = memory::host_vector<int_type>;
    using view_type  = typename iarray::view_type;
    using const_view_type = typename iarray::const_view_type;
    static constexpr int_type no_parent = (int_type)-1;

    tree() = default;

    tree& operator=(tree&& other) {
        std::swap(data_, other.data_);
        std::swap(child_index_, other.child_index_);
        std::swap(children_, other.children_);
        std::swap(parents_, other.parents_);
        return *this;
    }

    tree& operator=(tree const& other) {
        data_ = other.data_;
        set_ranges(other.num_segments());
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
        EXPECTS(parent_index.size()!=0u);

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
    const_view_type child_index() {
        return child_index_;
    }

    /// return the list of all children
    const_view_type children() const {
        return children_;
    }

    /// return the list of all children of branch b
    const_view_type children(size_type b) const {
        return children_(child_index_[b], child_index_[b+1]);
    }

    /// return the list of parents
    const_view_type parents() const {
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
        return sizeof(int_type)*data_.size() + sizeof(tree);
    }

    iarray change_root(size_t b) {
        assert(b<num_segments());

        // no need to rebalance if the root node has been requested
        if(b==0) {
            return iarray();
        }

        // create new tree with memory allocated
        tree new_tree;
        new_tree.init(num_segments());

        // add the root node
        new_tree.parents_[0] = no_parent;
        new_tree.child_index_[0] = 0;

        // allocate space for the permutation vector that
        // will represent the permutation performed on the branches
        // during the rebalancing
        iarray p(num_segments(), -1);

        // recersively rebalance the tree
        new_tree.add_children(0, b, 0, p, *this);

        // renumber the child indexes
        std::transform(
            new_tree.children_.begin(), new_tree.children_.end(),
            new_tree.children_.begin(), [&p] (int i) {return p[i];}
        );

        // copy in new data with a move because the information in
        // new_tree is not kept
        std::swap(data_, new_tree.data_);
        set_ranges(new_tree.num_segments());

        return p;
    }

private:
    void init(size_type nnode) {
        auto nchild = nnode - 1;

        data_ = iarray(nchild + (nnode + 1) + nnode);
        set_ranges(nnode);
    }

    void set_ranges(size_type nnode) {
        if (nnode) {
            auto nchild = nnode - 1;
            // data_ is partitioned as follows:
            // data_ = [children_[nchild], child_index_[nnode+1], parents_[nnode]]
            assert(data_.size() == unsigned(nchild + (nnode+1) + nnode));
            children_    = data_(0, nchild);
            child_index_ = data_(nchild, nchild+nnode+1);
            parents_     = data_(nchild+nnode+1, memory::end);

            // check that arrays have appropriate size
            // this should be moved into a unit test
            assert(children_.size()    == unsigned(nchild));
            assert(child_index_.size() == unsigned(nnode+1));
            assert(parents_.size()     == unsigned(nnode));
        }
        else {
            children_    = data_(0, 0);
            child_index_ = data_(0, 0);
            parents_     = data_(0, 0);
        }
    }

    /// Renumber the sub-tree with old_node as its root with new_node as
    /// the new index of old_node. This is a helper function for the
    /// renumbering required when a new root node is chosen to improve
    /// the balance of the tree.
    /// Optionally add the parent of old_node as a child of new_node, which
    /// will be applied recursively until the old root has been processed,
    /// which indicates that the renumbering is finished.
    ///
    /// precondition - the node new_node has already been placed in the tree
    /// precondition - all of new_node's children have been added to the tree
    ///     new_node : the new index of the node whose children are to be added
    ///                to the tree
    ///     old_node : the index of new_node in the original tree
    ///     parent_node : equals index of old_node's parent in the original tree
    ///                   should be a child of new_node
    ///                 : equals -1 if the old_node's parent is not a child of
    ///                   new_node
    ///     p : permutation vector, p[i] is the new index of node i in the old
    ///         tree
    int_type add_children(
        int_type new_node,
        int_type old_node,
        int_type parent_node,
        view_type p,
        tree const& old_tree
    )
    {
        // check for the sentinel that indicates that the old root has
        // been processed
        if (old_node==no_parent) {
            return new_node;
        }

        p[old_node] = new_node;

        // the list of the children of the original node
        auto old_children = old_tree.children(old_node);

        auto this_node = new_node;
        auto pos = child_index_[this_node];

        auto add_parent_as_child = parent_node!=no_parent && old_node>0;
        //
        // STEP 1 : add the child indexes for this_node
        //
        // first add the children of the node
        for(auto b : old_children) {
            if(b != parent_node) {
                children_[pos++] = b;
                parents_[pos] = new_node;
            }
        }
        // then add the node's parent as a child if applicable
        if(add_parent_as_child) {
            children_[pos++] = old_tree.parent(old_node);
            parents_[pos] = new_node;
        }
        child_index_[this_node+1] = pos;

        //
        // STEP 2 : recursively add each child's children
        //
        new_node++;
        for (auto b : old_children) {
            if (b != parent_node) {
                new_node = add_children(new_node, b, no_parent, p, old_tree);
            }
        }
        if (add_parent_as_child) {
            new_node =
                add_children(
                    new_node, old_tree.parent(old_node), old_node, p, old_tree
                );
        }

        return new_node;
    }

    //////////////////////////////////////////////////
    // state
    //////////////////////////////////////////////////
    iarray data_;

    // provide default parameters so that tree type can
    // be default constructed
    view_type children_   = data_(0, 0);
    view_type child_index_= data_(0, 0);
    view_type parents_    = data_(0, 0);
};

template <typename C>
std::vector<tree::int_type> make_parent_index(tree const& t, C const& counts)
{
    using util::make_span;
    using int_type = tree::int_type;
    constexpr auto no_parent = tree::no_parent;

    if (!algorithms::is_positive(counts) || counts.size() != t.num_segments()) {
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
