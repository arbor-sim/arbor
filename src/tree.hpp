#pragma once

#include <algorithm>
#include <cassert>
#include <numeric>
#include <vector>
#include <vector/include/Vector.hpp>

#include "algorithms.hpp"
#include "util.hpp"

namespace nest {
namespace mc {

class tree {
    using range = memory::Range;

    public :

    using int_type = int16_t;

    using index_type = memory::HostVector<int_type>;
    using index_view = index_type::view_type;

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
        set_ranges(other.num_nodes());
        return *this;
    }

    // copy constructors take advantage of the assignment operators
    // defined above
    tree(tree const& other)
    {
        *this = other;
    }

    tree(tree&& other)
    {
        *this = std::move(other);
    }

    /// create the tree from a parent_index
    template <typename I>
    tree(std::vector<I> const& parent_index)
    {
        // validate the inputs
        if(!algorithms::is_minimal_degree(parent_index)) {
            throw std::domain_error(
                "parent index used to build a tree did not satisfy minimal degree ordering"
            );
        }

        // n = number of compartment in cell
        auto n = parent_index.size();

        // On completion of this loop child_count[i] is the number of children
        // of compartment i compensate count for compartment 0, which has itself
        // as its own parent
        index_type child_count(n, 0);
        child_count[0] = -1;
        for(auto i : parent_index) {
            ++child_count[i];
        }

        // Find the number of branches by summing the number of children of all
        // compartments with more than 1 child.
        auto nbranches = 1 + child_count[0]; // children of the root node are counted differently
        for(auto i : range(1,n)) {
            if(child_count[i]>1) {
                nbranches += child_count[i];
            }
        }

        // allocate memory for storing the tree
        init(nbranches);

        // index of the branch for each compartment
        std::vector<I> branch_index(n);

        // Mark the parent of the root node as -1.
        // This simplifies the implementation of some algorithms on the tree
        parents_[0]=-1;

        // bcount records how many branches have been created in the loop
        auto bcount=0;

        for(auto i : range(1,n)) {
            // the branch index of the parent of compartment i
            auto parent_node = parent_index[i];
            // index of the parent of compartment i
            auto parent_branch = branch_index[parent_node];

            // if this compartments's parent
            //  - has more than one child
            //  - or is the root branch
            // this is the first compartment in a branch, so mark it as such
            if(child_count[parent_node]>1 || parent_node==0) {
                bcount++;
                branch_index[i] = bcount;
                parents_[bcount] = parent_branch;
            }
            // not the first compartment in a branch,
            // so inherit the parent's branch number
            else {
                branch_index[i] = parent_branch;
            }
        }

        child_index_(memory::all) = 0;
        // the root node has to be handled separately: all of its children must
        // be counted
        child_index_[1] = child_count[0];
        for(auto i : range(1, n)) {
            if(child_count[i]>1) {
                child_index_[branch_index[i]+1] = child_count[i];
            }
        }
        std::partial_sum(child_index_.begin(), child_index_.end(), child_index_.begin());

        // Fill in the list of children of each branch.
        // Requires some additional book keeping to keep track of how many
        // children have already been filled in for each branch.
        for(auto i : range(1, nbranches)) {
            // parents_[i] is the parent of branch i, for which i must be added
            // as a child, and child_index_[p] is the index into which the next
            // child of p is to be stored
            auto p = parents_[i];
            children_[child_index_[p]] = i;
            ++child_index_[p];
        }

        // The child index has already been calculated as a side-effect of the
        // loop above, but is shifted one value to the left, so perform a
        // rotation to the right.
        for(auto i=nbranches-1; i>=00; --i) {
            child_index_[i+1] = child_index_[i];
        }
        child_index_[0] = 0;
    }

    size_t num_children() const {
        return children_.size();
    }
    size_t num_children(size_t b) const {
        return child_index_[b+1] - child_index_[b];
    }
    size_t num_nodes() const {
        // the number of nodes is the size of the child index minus 1
        // ... except for the case of an empty tree
        auto sz = child_index_.size();
        return sz ? sz - 1 : 0;
    }

    /// return the child index
    const index_view child_index() const {
        return child_index_;
    }

    /// return the list of all children
    const index_view children() const {
        return children_;
    }

    /// return the list of all children of branch b
    const index_view children(size_t b) const {
        return children_(child_index_[b], child_index_[b+1]);
    }

    /// return the list of parents
    const index_view parents() const {
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
    size_t memory() const {
        return sizeof(int_type)*data_.size() + sizeof(tree);
    }

    index_type change_root(size_t b) {
        assert(b<num_nodes());

        // no need to rebalance if the root node has been requested
        if(b==0) {
            return index_type();
        }

        // create new tree with memory allocated
        tree new_tree;
        new_tree.init(num_nodes());

        // add the root node
        new_tree.parents_[0] = -1;
        new_tree.child_index_[0] = 0;

        // allocate space for the permutation vector that
        // will represent the permutation performed on the branches
        // during the rebalancing
        index_type p(num_nodes(), -1);

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
        set_ranges(new_tree.num_nodes());

        return p;
    }

    private :

    void init(int nnode) {
        auto nchild = nnode -1;

        data_ = index_type(nchild + (nnode + 1) + nnode);
        set_ranges(nnode);
    }

    void set_ranges(int nnode) {
        if(nnode) {
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
    int add_children(
        int new_node,
        int old_node,
        int parent_node,
        index_view p,
        tree const& old_tree
    )
    {
        // check for the senitel that indicates that the old root has
        // been processed
        if(old_node==-1) {
            return new_node;
        }

        p[old_node] = new_node;

        // the list of the children of the original node
        auto old_children = old_tree.children(old_node);

        auto this_node = new_node;
        auto pos = child_index_[this_node];

        auto add_parent_as_child = parent_node>=0 && old_node>0;
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
        for(auto b : old_children) {
            if(b != parent_node) {
                new_node = add_children(new_node, b, -1, p, old_tree);
            }
        }
        if(add_parent_as_child) {
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
    index_type data_;

    // provide default parameters so that tree type can
    // be default constructed
    index_view children_   = data_(0, 0);
    index_view child_index_= data_(0, 0);
    index_view parents_    = data_(0, 0);
};

template <typename C>
std::vector<int> make_parent_index(tree const& t, C const& counts)
{
    using range = memory::Range;

    if(   !algorithms::is_positive(counts)
        || counts.size() != t.num_nodes() )
    {
        throw std::domain_error(
            "make_parent_index requires one non-zero count per segment"
        );
    }
    auto index = algorithms::make_index(counts);
    auto num_compartments = index.back();
    std::vector<int> parent_index(num_compartments);
    auto pos = 0;
    for(int i : range(0, t.num_nodes())) {
        // get the parent of this segment
        // taking care for the case where the root node has -1 as its parent
        auto parent = t.parent(i);
        parent = parent>=0 ? parent : 0;

        // the index of the first compartment in the segment
        // is calculated differently for the root (i.e when i==parent)
        if(i!=parent) {
            parent_index[pos++] = index[parent+1]-1;
        }
        else {
            parent_index[pos++] = parent;
        }
        // number the remaining compartments in the segment consecutively
        while(pos<index[i+1]) {
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

} // namespace mc
} // namespace nest
